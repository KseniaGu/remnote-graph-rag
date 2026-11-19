import re
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Any

from pydantic import ValidationError
from pymongo import MongoClient

from backend.configs.constants import (
    CACHE_SCHEMA_VERSION,
    DEFAULT_CACHE_COLLECTION_NAME,
    DEFAULT_TTL_SECONDS,
)
from backend.utils.helpers import logger
from backend.utils.session_memory import make_bson_safe
from backend.workflows.presentation import CachedTurn


def normalize_quick_action_prompt(prompt: str) -> str:
    """Normalizes quick-action prompts for exact-match caching."""
    return re.sub(r"\s+", " ", prompt.strip()).casefold()


class QuickActionAnswerCache:
    """MongoDB-backed read-through cache for quick-action answers.

    Cache hits avoid initializing the full LangGraph workflow. The cache is intentionally
    exact-match only, so arbitrary user questions never receive a stale quick-action answer.
    """

    def __init__(
        self,
        settings: Any | None = None,
        collection_name: str = DEFAULT_CACHE_COLLECTION_NAME,
        ttl_seconds: int | None = None,
    ) -> None:
        if settings is None:
            from backend.configs.storage import StorageSettings

            settings = StorageSettings().checkpoint_storage
        self.settings = settings
        self.collection_name = collection_name
        self.ttl_seconds = self._get_ttl_seconds(ttl_seconds)
        self._client = None
        self._collection = None
        self._indexes_ready = False

    @staticmethod
    def _get_ttl_seconds(explicit_ttl: int | None) -> int | None:
        if explicit_ttl is not None:
            return explicit_ttl if explicit_ttl > 0 else None

        return DEFAULT_TTL_SECONDS

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
        if self.ttl_seconds is not None:
            self._collection.create_index("expires_at", expireAfterSeconds=0)
        self._indexes_ready = True

    def get(self, prompt: str) -> CachedTurn | None:
        """Returns cached response payload for a prompt, or None on miss/error."""
        key = normalize_quick_action_prompt(prompt)
        now = datetime.now(UTC)

        try:
            doc = self._get_collection().find_one(
                {"_id": key, "schema_version": CACHE_SCHEMA_VERSION}
            )
        except Exception as e:
            logger.warning(f"Quick-action cache read failed: {e}")
            return None

        if not doc or doc.get("schema_version") != CACHE_SCHEMA_VERSION:
            return None

        expires_at = doc.get("expires_at")
        if expires_at is not None:
            if not isinstance(expires_at, datetime):
                return None
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=UTC)
            if expires_at <= now:
                return None

        try:
            return CachedTurn.model_validate(doc.get("presentation"))
        except ValidationError:
            return None

    def set(
        self,
        prompt: str,
        turn: CachedTurn,
        ttl_seconds: int | None = None,
    ) -> bool:
        """Stores a quick-action answer. Returns False if MongoDB is unavailable."""
        if not turn.responses and not turn.artifacts:
            return False

        key = normalize_quick_action_prompt(prompt)
        now = datetime.now(UTC)
        effective_ttl = (
            self._get_ttl_seconds(ttl_seconds)
            if ttl_seconds is not None
            else self.ttl_seconds
        )
        doc = {
            "_id": key,
            "schema_version": CACHE_SCHEMA_VERSION,
            "prompt": prompt,
            "presentation": make_bson_safe(turn.model_dump()),
            "updated_at": now,
        }
        if effective_ttl is not None:
            doc["expires_at"] = now + timedelta(seconds=effective_ttl)

        try:
            self._get_collection().replace_one({"_id": key}, doc, upsert=True)
        except Exception as e:
            logger.warning(f"Quick-action cache write failed: {e}")
            return False
        return True


@lru_cache(maxsize=1)
def get_quick_action_cache() -> QuickActionAnswerCache:
    """Returns the process-local quick-action cache singleton."""
    return QuickActionAnswerCache()
