import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from backend.configs.chat_limits import ChatLimitsSettings
from backend.configs.enums import LLMProviderType
from backend.utils.chat_limits import (
    ChatLimitService,
    ChatTurnContext,
    OllamaDailyLimitExceeded,
    WorkflowBudgetExceeded,
    get_current_chat_turn,
    is_transient_provider_error,
    reset_current_chat_turn,
    set_current_chat_turn,
    truncate_text,
)
from backend.workflows.agents.tools import WebResearchInput, deep_web_research
from backend.workflows.learner import LearnerWorkflow


def make_service(**overrides) -> ChatLimitService:
    settings = ChatLimitsSettings(
        shared_quotas_enabled=False,
        _env_file=None,
        **overrides,
    )
    return ChatLimitService(settings=settings, storage_settings=object())


def test_chat_limit_defaults_match_the_public_contract() -> None:
    settings = ChatLimitsSettings(_env_file=None)

    assert settings.visitor_turns_per_day == 50
    assert settings.visitor_cooldown_seconds == 10
    assert settings.global_active_workflows == 2
    assert settings.global_queue_wait_seconds == 5
    assert settings.workflow_timeout_seconds == 180
    assert settings.workflow_lease_seconds == 240
    assert settings.ollama_attempts_per_day == 600
    assert settings.tavily_attempts_per_day == 30
    assert settings.tavily_attempts_per_month == 500


@pytest.mark.parametrize(
    "overrides",
    [
        {"workflow_timeout_seconds": 240, "workflow_lease_seconds": 240},
        {"ollama_warning_attempts": 600},
        {"ollama_critical_warning_attempts": 601},
        {"max_logical_llm_calls_per_turn": 7, "max_llm_attempts_per_turn": 6},
    ],
)
def test_invalid_limit_combinations_are_rejected(overrides: dict) -> None:
    with pytest.raises(ValidationError):
        ChatLimitsSettings(_env_file=None, **overrides)


def test_turn_envelope_allows_five_logical_calls_and_six_attempts() -> None:
    service = make_service()
    context = ChatTurnContext(visitor_id="visitor", session_id="session")

    for index in range(5):
        service.begin_llm_call(context, f"agent-{index}")
    with pytest.raises(WorkflowBudgetExceeded):
        service.begin_llm_call(context, "sixth")

    async def reserve_attempts() -> None:
        for _ in range(6):
            await service.reserve_llm_attempt(
                context, LLMProviderType.ollama, "analyst"
            )
        with pytest.raises(WorkflowBudgetExceeded):
            await service.reserve_llm_attempt(
                context, LLMProviderType.ollama, "analyst"
            )

    asyncio.run(reserve_attempts())


def test_retry_token_is_shared_across_a_turn() -> None:
    service = make_service()
    context = ChatTurnContext(visitor_id="visitor", session_id="session")

    assert service.claim_retry(context) is True
    assert service.claim_retry(context) is False
    assert context.retries_used == 1


def test_usage_metadata_is_aggregated_without_content() -> None:
    service = make_service()
    context = ChatTurnContext(visitor_id="visitor", session_id="session")
    response = SimpleNamespace(
        content="must not be read",
        usage_metadata={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
        response_metadata={},
    )

    service.record_model_usage(context, response)

    assert (context.input_tokens, context.output_tokens, context.total_tokens) == (
        12,
        8,
        20,
    )
    assert not hasattr(context, "content")


def test_contextvars_do_not_leak_between_concurrent_turns() -> None:
    async def observe(visitor_id: str) -> str:
        context = ChatTurnContext(visitor_id=visitor_id, session_id="session")
        token = set_current_chat_turn(context)
        try:
            await asyncio.sleep(0)
            current = get_current_chat_turn()
            assert current is context
            return current.visitor_id
        finally:
            reset_current_chat_turn(token)

    async def run() -> list[str]:
        return await asyncio.gather(observe("one"), observe("two"))

    assert asyncio.run(run()) == ["one", "two"]
    assert get_current_chat_turn() is None


def test_transient_error_classifier_retries_only_connection_timeout_and_5xx() -> None:
    class HttpError(Exception):
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

    class ConnectError(Exception):
        pass

    assert is_transient_provider_error(TimeoutError()) is True
    assert is_transient_provider_error(ConnectError()) is True
    assert is_transient_provider_error(HttpError(503)) is True
    assert is_transient_provider_error(HttpError(429)) is False
    assert is_transient_provider_error(HttpError(401)) is False


def test_text_context_is_hard_capped() -> None:
    result = truncate_text("x" * 500, 100)

    assert len(result) <= 100
    assert "truncated" in result


def test_web_topic_rejects_blank_and_oversized_values() -> None:
    with pytest.raises(ValidationError):
        WebResearchInput(topic="   ")
    with pytest.raises(ValidationError):
        WebResearchInput(topic="x" * 257)


def test_tavily_tool_uses_five_results_and_caps_context() -> None:
    class FakeSearchEngine:
        def __init__(self) -> None:
            self.kwargs = None

        def search(self, **kwargs):
            self.kwargs = kwargs
            return {
                "results": [
                    {
                        "title": f"Source {index}",
                        "url": f"https://example.test/{index}",
                        "score": 0.9,
                        "content": "x" * 200,
                    }
                    for index in range(8)
                ]
            }

    engine = FakeSearchEngine()
    service = make_service(
        tavily_result_max_chars=40,
        tavily_context_max_chars=400,
    )
    tool = deep_web_research(engine, limit_service=service)
    context = ChatTurnContext(visitor_id="visitor", session_id="session")

    async def invoke() -> str:
        token = set_current_chat_turn(context)
        try:
            return await tool.ainvoke({"topic": "  current topic  "})
        finally:
            reset_current_chat_turn(token)

    result = asyncio.run(invoke())

    assert engine.kwargs == {
        "query": "current topic",
        "search_depth": "advanced",
        "max_results": 5,
        "timeout": 60.0,
    }
    assert context.tavily_logical_searches == 1
    assert context.tavily_attempts == 1
    assert len(result) <= 400
    assert "Result 6" not in result


def test_call_tools_executes_only_first_recognized_call() -> None:
    class FakeTool:
        def __init__(self) -> None:
            self.calls = []

        async def ainvoke(self, args):
            self.calls.append(args)
            return "result"

    first = FakeTool()
    second = FakeTool()
    workflow = object.__new__(LearnerWorkflow)
    workflow.tools = {"first": first, "second": second}
    response = SimpleNamespace(
        tool_calls=[
            {"name": "first", "args": {"value": 1}},
            {"name": "second", "args": {"value": 2}},
        ]
    )

    result = asyncio.run(workflow.call_tools(response))

    assert result == {"first": "result"}
    assert first.calls == [{"value": 1}]
    assert second.calls == []


def test_ollama_cap_blocks_before_the_provider_attempt_is_recorded() -> None:
    service = make_service()
    service.settings.shared_quotas_enabled = True
    service._conditional_increment = lambda *args, **kwargs: None
    context = ChatTurnContext(visitor_id="visitor", session_id="session")

    async def reserve() -> None:
        with pytest.raises(OllamaDailyLimitExceeded):
            await service.reserve_llm_attempt(
                context, LLMProviderType.ollama, "analyst"
            )

    asyncio.run(reserve())
    assert context.llm_provider_attempts == 0
