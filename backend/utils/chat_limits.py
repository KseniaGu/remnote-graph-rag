import asyncio
import contextvars
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Any

from pymongo import MongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError

from backend.configs.chat_limits import ChatLimitsSettings
from backend.configs.enums import LLMProviderType
from backend.configs.messages import (
    ERROR_CHAT_COOLDOWN,
    ERROR_GLOBAL_CHAT_BUSY,
    ERROR_OLLAMA_DAILY_LIMIT,
    ERROR_QUOTA_STORAGE_UNAVAILABLE,
    ERROR_TAVILY_DAILY_LIMIT,
    ERROR_VISITOR_ACTIVE_TURN,
    ERROR_VISITOR_DAILY_LIMIT,
    ERROR_WORKFLOW_BUDGET,
    ERROR_WORKFLOW_TIMEOUT,
)
from backend.configs.storage import StorageSettings
from backend.utils.helpers import logger


class ChatLimitError(RuntimeError):
    """Base exception for a user-safe chat restriction failure."""

    user_message = ERROR_WORKFLOW_BUDGET
    reason = "chat_limit"

    def __init__(self, user_message: str | None = None) -> None:
        super().__init__(user_message or self.user_message)
        self.user_message = user_message or self.user_message


class ChatCooldownExceeded(ChatLimitError):
    user_message = ERROR_CHAT_COOLDOWN
    reason = "visitor_cooldown"

    def __init__(self, retry_after_seconds: float) -> None:
        self.retry_after_seconds = max(0.0, retry_after_seconds)
        super().__init__()


class VisitorDailyLimitExceeded(ChatLimitError):
    user_message = ERROR_VISITOR_DAILY_LIMIT
    reason = "visitor_daily_limit"


class VisitorActiveTurnExceeded(ChatLimitError):
    user_message = ERROR_VISITOR_ACTIVE_TURN
    reason = "visitor_active_turn"


class GlobalChatBusy(ChatLimitError):
    user_message = ERROR_GLOBAL_CHAT_BUSY
    reason = "global_chat_busy"


class OllamaDailyLimitExceeded(ChatLimitError):
    user_message = ERROR_OLLAMA_DAILY_LIMIT
    reason = "ollama_daily_limit"


class TavilyLimitExceeded(ChatLimitError):
    user_message = ERROR_TAVILY_DAILY_LIMIT
    reason = "tavily_limit"


class WorkflowBudgetExceeded(ChatLimitError):
    user_message = ERROR_WORKFLOW_BUDGET
    reason = "workflow_budget"


class WorkflowTimeoutExceeded(ChatLimitError):
    user_message = ERROR_WORKFLOW_TIMEOUT
    reason = "workflow_timeout"


class QuotaStorageUnavailable(ChatLimitError):
    user_message = ERROR_QUOTA_STORAGE_UNAVAILABLE
    reason = "quota_storage_unavailable"


@dataclass
class ChatTurnContext:
    """Mutable counters scoped to one accepted chat turn."""

    visitor_id: str
    session_id: str
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    cache_hit: bool = False
    logical_llm_calls: int = 0
    llm_provider_attempts: int = 0
    tavily_logical_searches: int = 0
    tavily_attempts: int = 0
    retries_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    agents: set[str] = field(default_factory=set)


@dataclass
class ChatAdmission:
    """Resources reserved for one accepted chat turn."""

    context: ChatTurnContext
    visitor_lease_acquired: bool = False
    global_slot: int | None = None
    daily_turn_counted: bool = False


_CURRENT_CHAT_TURN: contextvars.ContextVar[ChatTurnContext | None] = (
    contextvars.ContextVar("current_chat_turn", default=None)
)


@contextmanager
def bind_chat_turn(context: ChatTurnContext | None):
    """Binds a turn context for nested LangGraph and tool tasks."""
    token = _CURRENT_CHAT_TURN.set(context)
    try:
        yield
    finally:
        _CURRENT_CHAT_TURN.reset(token)


def get_current_chat_turn() -> ChatTurnContext | None:
    """Returns the turn executing in the current async context, if any."""
    return _CURRENT_CHAT_TURN.get()


def set_current_chat_turn(context: ChatTurnContext | None) -> contextvars.Token:
    """Binds a turn and returns the token needed to restore prior context."""
    return _CURRENT_CHAT_TURN.set(context)


def reset_current_chat_turn(token: contextvars.Token) -> None:
    """Restores the context that preceded a turn binding."""
    _CURRENT_CHAT_TURN.reset(token)


def provider_status_code(exc: BaseException) -> int | None:
    """Extracts an HTTP-like status code from supported provider exceptions."""
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def is_transient_provider_error(exc: BaseException) -> bool:
    """Returns whether a provider failure is safe to retry once."""
    status = provider_status_code(exc)
    if status is not None:
        return 500 <= status <= 599
    name = type(exc).__name__.casefold()
    message = str(exc).casefold()
    return (
        "timeout" in name
        or "timeout" in message
        or "connect" in name
        or "connection" in message
    )


def truncate_text(text: str, max_chars: int) -> str:
    """Truncates ordered context while marking the omitted tail."""
    if len(text) <= max_chars:
        return text
    marker = "\n\n[Additional context truncated by the application.]"
    return text[: max(0, max_chars - len(marker))].rstrip() + marker


class ChatLimitService:
    """MongoDB-backed distributed admission, budget, and telemetry service."""

    def __init__(
        self,
        settings: ChatLimitsSettings | None = None,
        storage_settings: Any | None = None,
        state_collection: Any | None = None,
        usage_collection: Any | None = None,
    ) -> None:
        self.settings = settings or ChatLimitsSettings()
        if storage_settings is None:
            storage_settings = StorageSettings().checkpoint_storage
        self.storage_settings = storage_settings
        self._client: Any | None = None
        self._state_collection = state_collection
        self._usage_collection = usage_collection
        self._indexes_ready = (
            state_collection is not None and usage_collection is not None
        )
        self._init_lock = threading.Lock()

    def _get_collections(self) -> tuple[Any, Any]:
        if self._state_collection is not None and self._usage_collection is not None:
            self._ensure_indexes()
            return self._state_collection, self._usage_collection

        try:
            with self._init_lock:
                if self._state_collection is None or self._usage_collection is None:
                    self._client = MongoClient(
                        self.storage_settings.uri.get_secret_value(),
                        serverSelectionTimeoutMS=1500,
                        connectTimeoutMS=1500,
                        socketTimeoutMS=1500,
                    )
                    self._client.admin.command("ping")
                    database = self._client[self.storage_settings.db_name]
                    self._state_collection = database[
                        self.settings.state_collection_name
                    ]
                    self._usage_collection = database[
                        self.settings.usage_collection_name
                    ]
                self._ensure_indexes()
        except PyMongoError as exc:
            logger.error(
                "Chat quota storage initialization failed",
                error_type=type(exc).__name__,
            )
            raise QuotaStorageUnavailable from exc
        return self._state_collection, self._usage_collection

    def _ensure_indexes(self) -> None:
        if self._indexes_ready:
            return
        assert self._state_collection is not None
        assert self._usage_collection is not None
        self._state_collection.create_index("expires_at", expireAfterSeconds=0)
        self._usage_collection.create_index("expires_at", expireAfterSeconds=0)
        self._usage_collection.create_index("started_at")
        self._usage_collection.create_index([("visitor_id", 1), ("started_at", -1)])
        for slot in range(self.settings.global_active_workflows):
            self._state_collection.update_one(
                {"_id": f"workflow_slot:{slot}"},
                {"$setOnInsert": {"kind": "workflow_slot", "slot": slot}},
                upsert=True,
            )
        self._indexes_ready = True

    def ping(self) -> bool:
        """Returns whether shared quota storage is ready."""
        if not self.settings.shared_quotas_enabled:
            return True
        try:
            self._get_collections()[0].find_one({}, projection={"_id": 1})
        except ChatLimitError:
            return False
        except PyMongoError:
            return False
        return True

    @staticmethod
    def _period_keys(now: datetime) -> tuple[str, str]:
        return now.strftime("%Y-%m-%d"), now.strftime("%Y-%m")

    def _expires_at(self, now: datetime) -> datetime:
        return now + timedelta(days=self.settings.usage_event_ttl_days)

    def _conditional_increment(
        self,
        key: str,
        kind: str,
        limit: int,
        now: datetime,
        operation_id: str | None = None,
    ) -> int | None:
        state, _ = self._get_collections()
        query: dict[str, Any] = {
            "_id": key,
            "$or": [{"count": {"$lt": limit}}, {"count": {"$exists": False}}],
        }
        update: dict[str, Any] = {
            "$inc": {"count": 1},
            "$set": {
                "kind": kind,
                "updated_at": now,
                "expires_at": self._expires_at(now),
            },
            "$setOnInsert": {"created_at": now},
        }
        if operation_id is not None:
            query["operation_ids"] = {"$ne": operation_id}
            update["$addToSet"] = {"operation_ids": operation_id}
        try:
            doc = state.find_one_and_update(
                query,
                update,
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
        except DuplicateKeyError:
            if operation_id is not None:
                try:
                    existing = state.find_one(
                        {"_id": key, "operation_ids": operation_id},
                        projection={"count": 1},
                    )
                except PyMongoError as exc:
                    raise QuotaStorageUnavailable from exc
                if existing:
                    return int(existing.get("count", 0))
            return None
        except PyMongoError as exc:
            raise QuotaStorageUnavailable from exc
        return int(doc.get("count", 0)) if doc else None

    def _acquire_visitor_lease(self, context: ChatTurnContext, now: datetime) -> None:
        state, _ = self._get_collections()
        key = f"visitor_state:{context.visitor_id}"
        cooldown_cutoff = now - timedelta(
            seconds=self.settings.visitor_cooldown_seconds
        )
        lease_until = now + timedelta(seconds=self.settings.workflow_lease_seconds)
        query = {
            "_id": key,
            "$and": [
                {
                    "$or": [
                        {"active_until": {"$lte": now}},
                        {"active_until": {"$exists": False}},
                    ]
                },
                {
                    "$or": [
                        {"last_submission_at": {"$lte": cooldown_cutoff}},
                        {"last_submission_at": {"$exists": False}},
                    ]
                },
            ],
        }
        try:
            doc = state.find_one_and_update(
                query,
                {
                    "$set": {
                        "kind": "visitor_state",
                        "visitor_id": context.visitor_id,
                        "active_turn_id": context.turn_id,
                        "active_until": lease_until,
                        "last_submission_at": now,
                        "expires_at": self._expires_at(now),
                    },
                    "$setOnInsert": {"created_at": now},
                },
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
        except DuplicateKeyError:
            doc = None
        except PyMongoError as exc:
            raise QuotaStorageUnavailable from exc
        if doc:
            return

        try:
            existing = state.find_one({"_id": key}) or {}
        except PyMongoError as exc:
            raise QuotaStorageUnavailable from exc
        active_until = existing.get("active_until")
        if active_until is not None:
            if active_until.tzinfo is None:
                active_until = active_until.replace(tzinfo=UTC)
            if active_until > now:
                raise VisitorActiveTurnExceeded
        last_submission = existing.get("last_submission_at")
        if last_submission is not None:
            if last_submission.tzinfo is None:
                last_submission = last_submission.replace(tzinfo=UTC)
            retry_after = (
                self.settings.visitor_cooldown_seconds
                - (now - last_submission).total_seconds()
            )
            if retry_after > 0:
                raise ChatCooldownExceeded(retry_after)
        raise VisitorActiveTurnExceeded

    def _try_acquire_global_slot(
        self, context: ChatTurnContext, now: datetime
    ) -> int | None:
        state, _ = self._get_collections()
        lease_until = now + timedelta(seconds=self.settings.workflow_lease_seconds)
        for slot in range(self.settings.global_active_workflows):
            try:
                doc = state.find_one_and_update(
                    {
                        "_id": f"workflow_slot:{slot}",
                        "$or": [
                            {"lease_until": {"$lte": now}},
                            {"lease_until": {"$exists": False}},
                        ],
                    },
                    {
                        "$set": {
                            "lease_id": context.turn_id,
                            "visitor_id": context.visitor_id,
                            "lease_until": lease_until,
                            "updated_at": now,
                        }
                    },
                    return_document=ReturnDocument.AFTER,
                )
            except PyMongoError as exc:
                raise QuotaStorageUnavailable from exc
            if doc:
                return slot
        return None

    async def _acquire_global_slot(
        self, context: ChatTurnContext, wait_seconds: float
    ) -> int:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait_seconds
        while True:
            slot = await asyncio.to_thread(
                self._try_acquire_global_slot, context, datetime.now(UTC)
            )
            if slot is not None:
                return slot
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise GlobalChatBusy
            await asyncio.sleep(min(self.settings.global_queue_poll_seconds, remaining))

    def _release(self, admission: ChatAdmission) -> None:
        if not self.settings.shared_quotas_enabled:
            return
        try:
            state, _ = self._get_collections()
            if admission.global_slot is not None:
                state.update_one(
                    {
                        "_id": f"workflow_slot:{admission.global_slot}",
                        "lease_id": admission.context.turn_id,
                    },
                    {"$unset": {"lease_id": "", "visitor_id": "", "lease_until": ""}},
                )
            if admission.visitor_lease_acquired:
                state.update_one(
                    {
                        "_id": f"visitor_state:{admission.context.visitor_id}",
                        "active_turn_id": admission.context.turn_id,
                    },
                    {"$unset": {"active_turn_id": "", "active_until": ""}},
                )
        except (ChatLimitError, PyMongoError) as exc:
            logger.error(
                "Chat quota lease release failed", error_type=type(exc).__name__
            )

    async def admit_cached_turn(
        self, visitor_id: str, session_id: str
    ) -> ChatAdmission:
        """Admits a cached turn without consuming expensive-workflow counters."""
        context = ChatTurnContext(
            visitor_id=visitor_id, session_id=session_id, cache_hit=True
        )
        admission = ChatAdmission(context=context)
        if not self.settings.shared_quotas_enabled:
            return admission
        await asyncio.to_thread(self._acquire_visitor_lease, context, datetime.now(UTC))
        admission.visitor_lease_acquired = True
        return admission

    async def admit_expensive_turn(
        self, visitor_id: str, session_id: str, wait_seconds: float | None = None
    ) -> ChatAdmission:
        """Reserves visitor, global-concurrency, and daily-turn capacity."""
        context = ChatTurnContext(visitor_id=visitor_id, session_id=session_id)
        admission = ChatAdmission(context=context)
        if not self.settings.shared_quotas_enabled:
            return admission
        try:
            await asyncio.to_thread(
                self._acquire_visitor_lease, context, datetime.now(UTC)
            )
            admission.visitor_lease_acquired = True
            admission.global_slot = await self._acquire_global_slot(
                context,
                self.settings.global_queue_wait_seconds
                if wait_seconds is None
                else wait_seconds,
            )
            now = datetime.now(UTC)
            day, _ = self._period_keys(now)
            count = await asyncio.to_thread(
                self._conditional_increment,
                f"visitor_day:{visitor_id}:{day}",
                "visitor_day",
                self.settings.visitor_turns_per_day,
                now,
                context.turn_id,
            )
            if count is None:
                raise VisitorDailyLimitExceeded
            admission.daily_turn_counted = True
            return admission
        except Exception:
            await asyncio.to_thread(self._release, admission)
            raise

    def begin_llm_call(self, context: ChatTurnContext | None, role: str) -> None:
        """Charges one logical LLM call before any provider attempt."""
        if context is None:
            return
        if context.logical_llm_calls >= self.settings.max_logical_llm_calls_per_turn:
            raise WorkflowBudgetExceeded
        context.logical_llm_calls += 1
        context.agents.add(role)

    async def reserve_llm_attempt(
        self,
        context: ChatTurnContext | None,
        provider: LLMProviderType | str,
        role: str,
    ) -> None:
        """Reserves one LLM provider attempt before network I/O."""
        if context is None:
            return
        if context.llm_provider_attempts >= self.settings.max_llm_attempts_per_turn:
            raise WorkflowBudgetExceeded
        provider_name = (
            provider.value if isinstance(provider, LLMProviderType) else str(provider)
        )
        if (
            self.settings.shared_quotas_enabled
            and provider_name == LLMProviderType.ollama.value
        ):
            now = datetime.now(UTC)
            day, _ = self._period_keys(now)
            count = await asyncio.to_thread(
                self._conditional_increment,
                f"provider_day:ollama:{day}",
                "ollama_day",
                self.settings.ollama_attempts_per_day,
                now,
                f"{context.turn_id}:llm:{context.llm_provider_attempts + 1}",
            )
            if count is None:
                raise OllamaDailyLimitExceeded
            if count in {
                self.settings.ollama_warning_attempts,
                self.settings.ollama_critical_warning_attempts,
            }:
                logger.warning(
                    "Ollama daily attempt threshold reached",
                    attempt_count=count,
                    daily_limit=self.settings.ollama_attempts_per_day,
                )
        context.llm_provider_attempts += 1
        context.agents.add(role)

    def begin_tavily_search(self, context: ChatTurnContext | None) -> None:
        """Charges one logical Tavily search before its provider attempts."""
        if context is None:
            return
        if context.tavily_logical_searches >= self.settings.tavily_searches_per_turn:
            raise WorkflowBudgetExceeded
        context.tavily_logical_searches += 1

    async def reserve_tavily_attempt(self, context: ChatTurnContext | None) -> None:
        """Reserves daily and monthly Tavily capacity before network I/O."""
        if context is None:
            return
        if self.settings.shared_quotas_enabled:
            now = datetime.now(UTC)
            day, month = self._period_keys(now)
            operation_id = f"{context.turn_id}:tavily:{context.tavily_attempts + 1}"
            daily_count = await asyncio.to_thread(
                self._conditional_increment,
                f"provider_day:tavily:{day}",
                "tavily_day",
                self.settings.tavily_attempts_per_day,
                now,
                operation_id,
            )
            if daily_count is None:
                raise TavilyLimitExceeded
            monthly_count = await asyncio.to_thread(
                self._conditional_increment,
                f"provider_month:tavily:{month}",
                "tavily_month",
                self.settings.tavily_attempts_per_month,
                now,
                operation_id,
            )
            if monthly_count is None:
                raise TavilyLimitExceeded
        context.tavily_attempts += 1

    def claim_retry(self, context: ChatTurnContext | None) -> bool:
        """Claims the turn's single retry token."""
        if context is None:
            return True
        if context.retries_used >= self.settings.max_retries_per_turn:
            return False
        context.retries_used += 1
        return True

    @staticmethod
    def record_model_usage(context: ChatTurnContext | None, response: Any) -> None:
        """Accumulates provider token metadata without storing model content."""
        if context is None or response is None:
            return
        usage = getattr(response, "usage_metadata", None) or {}
        metadata = getattr(response, "response_metadata", None) or {}
        input_tokens = usage.get("input_tokens", metadata.get("prompt_eval_count", 0))
        output_tokens = usage.get("output_tokens", metadata.get("eval_count", 0))
        total_tokens = usage.get("total_tokens", 0)
        try:
            input_count = max(0, int(input_tokens or 0))
            output_count = max(0, int(output_tokens or 0))
            total_count = max(0, int(total_tokens or input_count + output_count))
        except (TypeError, ValueError):
            return
        context.input_tokens += input_count
        context.output_tokens += output_count
        context.total_tokens += total_count

    def _write_usage_event(self, context: ChatTurnContext, status: str) -> None:
        _, usage = self._get_collections()
        ended_at = datetime.now(UTC)
        doc = {
            "_id": context.turn_id,
            "visitor_id": context.visitor_id,
            "session_id": context.session_id,
            "started_at": context.started_at,
            "ended_at": ended_at,
            "duration_ms": int((ended_at - context.started_at).total_seconds() * 1000),
            "status": status,
            "cache_hit": context.cache_hit,
            "agents": sorted(context.agents),
            "logical_llm_calls": context.logical_llm_calls,
            "llm_provider_attempts": context.llm_provider_attempts,
            "tavily_logical_searches": context.tavily_logical_searches,
            "tavily_attempts": context.tavily_attempts,
            "retries_used": context.retries_used,
            "input_tokens": context.input_tokens,
            "output_tokens": context.output_tokens,
            "total_tokens": context.total_tokens,
            "expires_at": self._expires_at(ended_at),
        }
        usage.replace_one({"_id": context.turn_id}, doc, upsert=True)

    async def finish_turn(self, admission: ChatAdmission, status: str) -> None:
        """Records content-free telemetry and releases all held leases."""
        if self.settings.shared_quotas_enabled:
            try:
                await asyncio.to_thread(
                    self._write_usage_event, admission.context, status
                )
            except (ChatLimitError, PyMongoError) as exc:
                logger.error(
                    "Chat usage event write failed", error_type=type(exc).__name__
                )
            await asyncio.to_thread(self._release, admission)
        logger.info(
            "Chat turn completed",
            turn_id=admission.context.turn_id,
            status=status,
            cache_hit=admission.context.cache_hit,
            logical_llm_calls=admission.context.logical_llm_calls,
            llm_provider_attempts=admission.context.llm_provider_attempts,
            tavily_attempts=admission.context.tavily_attempts,
            input_tokens=admission.context.input_tokens,
            output_tokens=admission.context.output_tokens,
        )


@lru_cache(maxsize=1)
def get_chat_limit_service() -> ChatLimitService:
    """Returns the process-local client for shared MongoDB limit state."""
    return ChatLimitService()
