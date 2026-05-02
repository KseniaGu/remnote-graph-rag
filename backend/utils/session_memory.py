import re
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Optional

from pymongo import MongoClient

from backend.configs.constants import SESSION_MEMORY_SCHEMA_VERSION, DEFAULT_SESSION_MEMORY_COLLECTION_NAME, \
    DEFAULT_RECENT_MESSAGE_LIMIT, DEFAULT_MAX_MESSAGE_CHARS, DEFAULT_SUMMARY_MAX_CHARS, \
    DEFAULT_SESSION_MEMORY_TTL_SECONDS
from backend.configs.storage import StorageSettings
from backend.utils.helpers import logger


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
        messages: list[dict[str, str]],
        max_message_chars: int = DEFAULT_MAX_MESSAGE_CHARS,
) -> str:
    """Formats persisted messages into a compact prompt block."""
    chunks = []
    for message in messages:
        role = message.get("role", "unknown")
        agent = message.get("agent", "")
        label = f"{role}:{agent}" if agent else role
        content = compress_text(message.get("content", ""), max_message_chars)
        if content:
            chunks.append(f"- {label}: {content}")
    return "\n".join(chunks)


def build_extractive_summary(
        messages: list[dict[str, str]],
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
            settings: Optional[Any] = None,
            collection_name: str = DEFAULT_SESSION_MEMORY_COLLECTION_NAME,
    ) -> None:
        if settings is None:
            settings = StorageSettings().checkpoint_storage
        self.settings = settings
        self.collection_name = collection_name
        self._client: Optional[Any] = None
        self._collection: Optional[Any] = None
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

    def get(self, session_id: str) -> Optional[dict[str, Any]]:
        """Returns a persisted session memory document, or None on miss/error."""
        safe_session_id = normalize_session_id(session_id)
        if not safe_session_id:
            return None

        try:
            doc = self._get_collection().find_one(
                {"_id": safe_session_id, "schema_version": SESSION_MEMORY_SCHEMA_VERSION}
            )
        except Exception as e:
            logger.warning(f"Session memory read failed: {e}")
            return None
        if not doc:
            return None

        expires_at = doc.get("expires_at")
        if expires_at is not None:
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if expires_at <= datetime.now(timezone.utc):
                return None
        return doc

    def get_prompt_summary(self, session_id: str, latest_user_message: str) -> str:
        """Returns older-session memory only when the current turn likely needs it."""
        if not is_context_dependent_turn(latest_user_message):
            return ""

        doc = self.get(session_id)
        if not doc:
            return ""
        return doc.get("summary", "")

    def replace_messages(
            self,
            session_id: str,
            messages: list[dict[str, str]],
            recent_limit: int = DEFAULT_RECENT_MESSAGE_LIMIT,
            session_history: Optional[list[dict[str, Any]]] = None,
            recent_maps: Optional[list[dict[str, Any]]] = None,
            visual_artifacts: Optional[list[dict[str, Any]]] = None,
    ) -> bool:
        """Persists browser-session messages, summary, and optional UI restore metadata."""
        safe_session_id = normalize_session_id(session_id)
        if not safe_session_id:
            return False

        now = datetime.now(timezone.utc)
        safe_messages = make_bson_safe(messages)
        older_messages = safe_messages[:-recent_limit] if len(safe_messages) > recent_limit else []
        summary = build_extractive_summary(older_messages)
        doc = {
            "_id": safe_session_id,
            "schema_version": SESSION_MEMORY_SCHEMA_VERSION,
            "messages": safe_messages,
            "summary": summary,
            "message_count": len(safe_messages),
            "updated_at": now,
            "expires_at": now + timedelta(seconds=DEFAULT_SESSION_MEMORY_TTL_SECONDS),
        }
        if session_history is not None:
            doc["session_history"] = make_bson_safe(session_history)
        if recent_maps is not None:
            doc["recent_maps"] = make_bson_safe(recent_maps)
        if visual_artifacts is not None:
            doc["visual_artifacts"] = make_bson_safe(visual_artifacts)

        try:
            self._get_collection().replace_one({"_id": safe_session_id}, doc, upsert=True)
        except Exception as e:
            logger.warning(f"Session memory write failed: {e}")
            return False
        return True

    def delete(self, session_id: str) -> bool:
        """Deletes a persisted browser-session memory document."""
        safe_session_id = normalize_session_id(session_id)
        if not safe_session_id:
            return False

        try:
            self._get_collection().delete_one({"_id": safe_session_id})
        except Exception as e:
            logger.warning(f"Session memory delete failed: {e}")
            return False
        return True


@lru_cache(maxsize=1)
def get_session_memory_store() -> SessionMemoryStore:
    """Returns the process-local session memory singleton."""
    return SessionMemoryStore()
