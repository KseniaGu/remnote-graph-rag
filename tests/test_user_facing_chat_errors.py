import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.configs.chat_limits import ChatLimitsSettings
from backend.configs.enums import LLMProviderType, ModelRoleType
from backend.utils.chat_errors import (
    AIRequestRejected,
    AIServiceCapacity,
    AIServiceConfiguration,
    AIServiceUnavailable,
    KnowledgeBaseUnavailable,
    WebSearchCapacity,
    WorkflowInitializationUnavailable,
)
from backend.utils.chat_limits import (
    ChatLimitService,
    ChatTurnContext,
    reset_current_chat_turn,
    set_current_chat_turn,
)
from backend.workflows.agents.tools import deep_web_research
from backend.workflows.learner import LearnerWorkflow
from backend.workflows.learner_reflex import ReflexLearnerWorkflow


def make_service() -> ChatLimitService:
    return ChatLimitService(
        settings=ChatLimitsSettings(shared_quotas_enabled=False, _env_file=None),
        storage_settings=object(),
    )


class ProviderError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"provider secret for status {status_code}")
        self.status_code = status_code


@pytest.mark.parametrize(
    ("status_code", "expected_error"),
    [
        (429, AIServiceCapacity),
        (401, AIServiceConfiguration),
        (400, AIRequestRejected),
    ],
)
def test_model_provider_status_uses_sanitized_error(
    status_code: int, expected_error: type[Exception]
) -> None:
    class FailingModel:
        async def ainvoke(self, messages, config):
            raise ProviderError(status_code)

    workflow = object.__new__(LearnerWorkflow)
    workflow.chat_limit_service = make_service()
    workflow.models_settings = SimpleNamespace(
        analyst=SimpleNamespace(provider=LLMProviderType.ollama)
    )
    workflow.analyst = FailingModel()
    context = ChatTurnContext(visitor_id="visitor", session_id="session")

    async def invoke() -> None:
        token = set_current_chat_turn(context)
        try:
            with pytest.raises(expected_error) as raised:
                await workflow.call_model([], ModelRoleType.analyst)
            assert "secret" not in str(raised.value)
        finally:
            reset_current_chat_turn(token)

    asyncio.run(invoke())


def test_exhausted_model_timeout_uses_unavailable_message() -> None:
    class FailingModel:
        async def ainvoke(self, messages, config):
            raise TimeoutError("private provider details")

    workflow = object.__new__(LearnerWorkflow)
    workflow.chat_limit_service = make_service()
    workflow.models_settings = SimpleNamespace(
        analyst=SimpleNamespace(provider=LLMProviderType.ollama)
    )
    workflow.analyst = FailingModel()
    context = ChatTurnContext(visitor_id="visitor", session_id="session")

    async def invoke() -> None:
        token = set_current_chat_turn(context)
        try:
            with (
                patch("backend.workflows.learner.asyncio.sleep", new=AsyncMock()),
                pytest.raises(AIServiceUnavailable) as raised,
            ):
                await workflow.call_model([], ModelRoleType.analyst)
            assert "private provider details" not in str(raised.value)
        finally:
            reset_current_chat_turn(token)

    asyncio.run(invoke())


def test_tavily_429_uses_web_capacity_message() -> None:
    class FailingSearch:
        def search(self, **kwargs):
            raise ProviderError(429)

    tool = deep_web_research(FailingSearch(), limit_service=make_service())
    context = ChatTurnContext(visitor_id="visitor", session_id="session")

    async def invoke() -> None:
        token = set_current_chat_turn(context)
        try:
            with pytest.raises(WebSearchCapacity) as raised:
                await tool.ainvoke({"topic": "current topic"})
            assert "secret" not in str(raised.value)
        finally:
            reset_current_chat_turn(token)

    asyncio.run(invoke())


def test_call_tools_does_not_swallow_user_facing_failures() -> None:
    class FailingTool:
        async def ainvoke(self, args):
            raise KnowledgeBaseUnavailable()

    workflow = object.__new__(LearnerWorkflow)
    workflow.tools = {"search": FailingTool()}
    response = SimpleNamespace(tool_calls=[{"name": "search", "args": {}}])

    with pytest.raises(KnowledgeBaseUnavailable):
        asyncio.run(workflow.call_tools(response))


def test_initialization_failure_hides_local_paths() -> None:
    workflow = ReflexLearnerWorkflow()
    private_path = "/Users/test/private/models/all-MiniLM-L6-v2"

    with (
        patch(
            "backend.workflows.learner_reflex.LangSmithSettings.configure",
            side_effect=FileNotFoundError(private_path),
        ),
        patch("backend.workflows.learner_reflex.logger.exception"),
        pytest.raises(WorkflowInitializationUnavailable) as raised,
    ):
        workflow._ensure_initialized()

    assert private_path not in str(raised.value)
    assert "required service or model" in raised.value.user_message
