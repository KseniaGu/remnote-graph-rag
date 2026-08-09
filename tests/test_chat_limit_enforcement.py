import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.state import _canonical_visitor_id, _events_with_timeout
from backend.configs.chat_limits import ChatLimitsSettings
from backend.configs.enums import LLMProviderType, ModelRoleType
from backend.utils.chat_limits import (
    ChatLimitService,
    ChatTurnContext,
    OllamaDailyLimitExceeded,
    reset_current_chat_turn,
    set_current_chat_turn,
)
from backend.workflows.learner import LearnerWorkflow


def make_service() -> ChatLimitService:
    return ChatLimitService(
        settings=ChatLimitsSettings(shared_quotas_enabled=False, _env_file=None),
        storage_settings=object(),
    )


def test_two_fixed_global_slots_are_independently_acquired() -> None:
    state = Mock()
    usage = Mock()
    service = ChatLimitService(
        settings=ChatLimitsSettings(shared_quotas_enabled=True, _env_file=None),
        storage_settings=object(),
        state_collection=state,
        usage_collection=usage,
    )
    state.find_one_and_update.side_effect = [
        {"_id": "workflow_slot:0"},
        None,
        {"_id": "workflow_slot:1"},
        None,
        None,
    ]
    now = ChatTurnContext(visitor_id="clock", session_id="clock").started_at

    first = service._try_acquire_global_slot(
        ChatTurnContext(visitor_id="one", session_id="one"), now
    )
    second = service._try_acquire_global_slot(
        ChatTurnContext(visitor_id="two", session_id="two"), now
    )
    third = service._try_acquire_global_slot(
        ChatTurnContext(visitor_id="three", session_id="three"), now
    )

    assert (first, second, third) == (0, 1, None)


def test_ollama_warnings_are_logging_only_until_the_hard_cap() -> None:
    service = make_service()
    service.settings.shared_quotas_enabled = True
    counts = iter([449, 450, 540, 600, None])
    service._conditional_increment = lambda *args, **kwargs: next(counts)
    context = ChatTurnContext(visitor_id="visitor", session_id="session")

    async def reserve() -> None:
        for _ in range(4):
            await service.reserve_llm_attempt(
                context, LLMProviderType.ollama, "analyst"
            )
        with pytest.raises(OllamaDailyLimitExceeded):
            await service.reserve_llm_attempt(
                context, LLMProviderType.ollama, "analyst"
            )

    with patch("backend.utils.chat_limits.logger.warning") as warning:
        asyncio.run(reserve())

    assert warning.call_count == 2
    assert [call.kwargs["attempt_count"] for call in warning.call_args_list] == [
        450,
        540,
    ]
    assert context.llm_provider_attempts == 4


def test_model_call_uses_the_turns_only_retry_for_a_timeout() -> None:
    class FlakyModel:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, messages, config):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError
            return SimpleNamespace(
                usage_metadata={
                    "input_tokens": 4,
                    "output_tokens": 6,
                    "total_tokens": 10,
                },
                response_metadata={},
            )

    workflow = object.__new__(LearnerWorkflow)
    workflow.chat_limit_service = make_service()
    workflow.models_settings = SimpleNamespace(
        analyst=SimpleNamespace(provider=LLMProviderType.ollama)
    )
    workflow.analyst = FlakyModel()
    context = ChatTurnContext(visitor_id="visitor", session_id="session")

    async def invoke():
        token = set_current_chat_turn(context)
        try:
            with patch("backend.workflows.learner.asyncio.sleep", new=AsyncMock()):
                return await workflow.call_model([], ModelRoleType.analyst)
        finally:
            reset_current_chat_turn(token)

    response = asyncio.run(invoke())

    assert response is not None
    assert workflow.analyst.calls == 2
    assert context.logical_llm_calls == 1
    assert context.llm_provider_attempts == 2
    assert context.retries_used == 1
    assert context.total_tokens == 10


def test_workflow_timeout_closes_the_underlying_stream() -> None:
    state = {"closed": False}

    async def slow_events():
        try:
            await asyncio.sleep(0.05)
            yield object()
        finally:
            state["closed"] = True

    async def consume() -> None:
        with pytest.raises(TimeoutError):
            async for _ in _events_with_timeout(slow_events(), 0.001):
                pass

    asyncio.run(consume())
    assert state["closed"] is True


def test_browser_identifiers_are_canonical_opaque_uuids() -> None:
    value = "{12345678-1234-5678-1234-567812345678}"

    assert _canonical_visitor_id(value) == "12345678-1234-5678-1234-567812345678"
