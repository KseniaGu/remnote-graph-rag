import re
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Any

from pydantic import ValidationError
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

from backend.configs.constants import (
    DEFAULT_MAX_MESSAGE_CHARS,
    DEFAULT_RECENT_MESSAGE_LIMIT,
    DEFAULT_SESSION_MEMORY_COLLECTION_NAME,
    DEFAULT_SESSION_MEMORY_TTL_SECONDS,
    DEFAULT_SUMMARY_MAX_CHARS,
    SESSION_MEMORY_SCHEMA_VERSION,
)
from backend.configs.storage import StorageSettings
from backend.utils.helpers import logger
from backend.workflows.presentation import Message, SessionSnapshot


def normalize_session_id(session_id: str) -> str:
    """Keeps browser-provided anonymous session IDs safe for Mongo document keys."""
    cleaned = re.sub(r"[^a-zA-Z0-9_.:-]", "_", session_id.strip())
    return cleaned[:160]


def compress_text(text: str, max_chars: int = DEFAULT_MAX_MESSAGE_CHARS) -> str:
    """Deterministically compresses long chat messages without an LLM call."""
    text = re.sub(r"\s+", " ", text.strip())
    if len(text) <= max_chars:
        return text

    if max_chars <= 80:
        return text[:max_chars]

    marker = " ... [truncated] ... "
    head_chars = max_chars * 2 // 3
    tail_chars = max(0, max_chars - head_chars - len(marker))
    tail = text[-tail_chars:].lstrip() if tail_chars else ""
    return f"{text[:head_chars].rstrip()}{marker}{tail}"[:max_chars]


def format_messages_for_memory(
    messages: list[Message],
    max_message_chars: int = DEFAULT_MAX_MESSAGE_CHARS,
) -> str:
    """Formats persisted messages into a compact prompt block."""
    chunks = []
    for message in messages:
        label = f"{message.role}:{message.agent}" if message.agent else message.role
        content = compress_text(message.content, max_message_chars)
        if content:
            chunks.append(f"- {label}: {content}")
    return "\n".join(chunks)


def build_extractive_summary(
    messages: list[Message],
    max_chars: int = DEFAULT_SUMMARY_MAX_CHARS,
) -> str:
    """Builds a cheap, deterministic summary from older turns.

    This is intentionally extractive. It gives agents access to older topics
    without adding a summarizer call to the critical path.
    """
    if not messages:
        return ""

    summary = format_messages_for_memory(messages, max_message_chars=360)
    return compress_text(summary, max_chars)


_MISSING = object()


def make_bson_safe(value: Any) -> Any:
    """Converts framework proxy wrappers into plain BSON-encodable containers."""
    wrapped = getattr(value, "__wrapped__", _MISSING)
    if wrapped is not _MISSING and wrapped is not value:
        return make_bson_safe(wrapped)

    if isinstance(value, dict):
        return {str(key): make_bson_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [make_bson_safe(item) for item in value]

    return value


def is_context_dependent_turn(text: str) -> bool:
    """Heuristic for whether older conversation memory is likely useful."""
    normalized = re.sub(r"\s+", " ", text.casefold())
    markers = (
        "previous",
        "earlier",
        "above",
        "before",
        "last answer",
        "last question",
        "my answer",
        "my response",
        "what did i",
        "continue",
        "summarize",
        "recap",
        "compare",
        "this",
        "that",
        "these",
        "those",
        "it",
        "they",
    )
    return any(re.search(rf"\b{re.escape(marker)}\b", normalized) for marker in markers)


class SessionMemoryStore:
    """MongoDB-backed chat memory independent of LangGraph checkpoints."""

    def __init__(
        self,
        settings: Any | None = None,
        collection_name: str = DEFAULT_SESSION_MEMORY_COLLECTION_NAME,
    ) -> None:
        if settings is None:
            settings = StorageSettings().checkpoint_storage
        self.settings = settings
        self.collection_name = collection_name
        self._client: Any | None = None
        self._collection: Any | None = None
        self._indexes_ready = False

    def _get_collection(self) -> Any:
        if self._collection is not None:
            return self._collection

        self._client = MongoClient(
            self.settings.uri.get_secret_value(),
            serverSelectionTimeoutMS=1500,
            connectTimeoutMS=1500,
        )
        self._client.admin.command("ping")
        self._collection = self._client[self.settings.db_name][self.collection_name]
        self._ensure_indexes()
        return self._collection

    def _ensure_indexes(self) -> None:
        if self._indexes_ready or self._collection is None:
            return
        self._collection.create_index("updated_at")
        self._collection.create_index("expires_at", expireAfterSeconds=0)
        self._indexes_ready = True

    def _read(
        self, session_id: str, *, require_available: bool = False
    ) -> tuple[SessionSnapshot, str] | None:
        safe_session_id = normalize_session_id(session_id)
        if not safe_session_id:
            return None

        try:
            doc = self._get_collection().find_one(
                {
                    "_id": safe_session_id,
                    "schema_version": SESSION_MEMORY_SCHEMA_VERSION,
                }
            )
        except Exception as e:
            if require_available:
                raise
            logger.warning(f"Session memory read failed: {e}")
            return None
        if not doc or doc.get("schema_version") != SESSION_MEMORY_SCHEMA_VERSION:
            return None

        expires_at = doc.get("expires_at")
        if not isinstance(expires_at, datetime):
            return None
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        summary = doc.get("summary")
        if expires_at <= datetime.now(UTC) or not isinstance(summary, str):
            return None
        try:
            snapshot = SessionSnapshot.model_validate(doc.get("presentation"))
        except ValidationError:
            return None
        if snapshot.session_id != safe_session_id:
            return None
        return snapshot, summary

    def get(
        self, session_id: str, *, require_available: bool = False
    ) -> SessionSnapshot | None:
        """Returns a validated session snapshot on a current, unexpired hit."""
        result = self._read(session_id, require_available=require_available)
        return result[0] if result else None

    def get_prompt_summary(self, session_id: str, latest_user_message: str) -> str:
        """Returns older-session memory only when the current turn likely needs it."""
        if not is_context_dependent_turn(latest_user_message):
            return ""

        result = self._read(session_id)
        return result[1] if result else ""

    def replace_snapshot(
        self,
        snapshot: SessionSnapshot,
        recent_limit: int = DEFAULT_RECENT_MESSAGE_LIMIT,
    ) -> bool:
        """Persists a complete snapshot with a content-only extractive summary."""
        session_id = snapshot.session_id
        now = datetime.now(UTC)
        doc = {
            "_id": session_id,
            "schema_version": SESSION_MEMORY_SCHEMA_VERSION,
            "presentation": make_bson_safe(snapshot.model_dump()),
            "summary": build_extractive_summary(snapshot.messages[:-recent_limit]),
            "message_count": len(snapshot.messages),
            "updated_at": now,
            "expires_at": now + timedelta(seconds=DEFAULT_SESSION_MEMORY_TTL_SECONDS),
        }
        try:
            self._get_collection().replace_one(
                {"_id": session_id, "cleared": {"$ne": True}}, doc, upsert=True
            )
        except DuplicateKeyError:
            # A concurrent Clear owns this retired session ID.
            return False
        except Exception as e:
            logger.warning(f"Session memory write failed: {e}")
            return False
        return True

    def delete(self, session_id: str) -> bool:
        """Removes conversation data and rejects outstanding writes to its ID."""
        safe_session_id = normalize_session_id(session_id)
        if not safe_session_id:
            return False

        try:
            self._get_collection().replace_one(
                {"_id": safe_session_id},
                {
                    "_id": safe_session_id,
                    "cleared": True,
                    "expires_at": datetime.now(UTC)
                    + timedelta(seconds=DEFAULT_SESSION_MEMORY_TTL_SECONDS),
                },
                upsert=True,
            )
        except Exception as e:
            logger.warning(f"Session memory delete failed: {e}")
            return False
        return True


@lru_cache(maxsize=1)
def get_session_memory_store() -> SessionMemoryStore:
    """Returns the process-local session memory singleton."""
    return SessionMemoryStore()
