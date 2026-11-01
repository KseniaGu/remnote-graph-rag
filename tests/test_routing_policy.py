import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from backend.utils.chat_errors import AIResponseInvalid
from backend.workflows.agents.schemas import RoutingDecision, State
from backend.workflows.agents.tools import KnowledgeSearchInput
from backend.workflows.learner import LearnerWorkflow


def test_routing_json_envelope_is_normalized():
    envelope = {
        "raw": AIMessage(
            content='{"request_scope":"out_of_scope","next_step":"__end__","reasoning":"unrelated"}'
        ),
        "parsed": None,
        "parsing_error": None,
    }
    result = LearnerWorkflow._normalize_routing_decision(envelope)
    assert result.request_scope == "out_of_scope"
    assert result.next_step == "__end__"


def test_unresolved_routing_raises():
    workflow = LearnerWorkflow.__new__(LearnerWorkflow)
    workflow.create_messages_to_pass = lambda *args: []
    workflow.call_model = AsyncMock(return_value=None)
    with pytest.raises(AIResponseInvalid):
        asyncio.run(workflow.orchestrator_node(State()))


def test_policy_required_and_named_requires_aliases():
    with pytest.raises(ValidationError):
        KnowledgeSearchInput.model_validate(
            {"queries": ["cats"], "required_topics": []}
        )
    with pytest.raises(ValidationError):
        KnowledgeSearchInput.model_validate(
            {"queries": ["cats"], "topic_policy": "named", "required_topics": []}
        )
    assert (
        KnowledgeSearchInput.model_validate(
            {"queries": ["optimizers"], "topic_policy": "broad", "required_topics": []}
        ).topic_policy
        == "broad"
    )


def test_invalid_retrieval_policy_fails_before_tools():
    workflow = LearnerWorkflow.__new__(LearnerWorkflow)
    workflow.create_messages_to_pass = lambda *args: []
    workflow.call_model = AsyncMock(
        return_value=AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_knowledge_base",
                    "args": {"queries": ["cats"], "required_topics": []},
                    "id": "call1",
                    "type": "tool_call",
                }
            ],
        )
    )
    workflow.call_tools = AsyncMock(return_value={})
    with pytest.raises(AIResponseInvalid):
        asyncio.run(workflow.retriever_node(State(request_scope="ambiguous")))
    workflow.call_tools.assert_not_awaited()


def _run_adapter_graph(message):
    import json
    from unittest.mock import Mock, patch

    import httpx
    from langchain_core.messages import HumanMessage
    from langchain_openrouter import ChatOpenRouter
    from langgraph.graph import StateGraph
    from openrouter import OpenRouter

    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "test-id",
                "model": "test-model",
                "object": "chat.completion",
                "created": 0,
                "system_fingerprint": None,
                "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 4,
                    "total_tokens": 16,
                },
            },
        )

    client = OpenRouter(
        api_key="dummy",
        retry_config=None,
        async_client=httpx.AsyncClient(
            transport=httpx.MockTransport(respond), follow_redirects=False
        ),
    )
    model = ChatOpenRouter(model="test", api_key="dummy", client=client, max_retries=0)
    workflow = LearnerWorkflow.__new__(LearnerWorkflow)
    workflow.orchestrator = model.with_structured_output(
        RoutingDecision, include_raw=True
    )
    workflow.models_settings = SimpleNamespace(
        orchestrator=SimpleNamespace(provider="openrouter")
    )
    workflow.chat_limit_service = SimpleNamespace(
        begin_llm_call=Mock(),
        reserve_llm_attempt=AsyncMock(),
        record_model_usage=Mock(),
        claim_retry=Mock(return_value=False),
    )
    workflow.create_messages_to_pass = Mock(return_value=[("human", "cats")])
    workflow.call_tools = AsyncMock()
    workflow.workflow = StateGraph(State)
    workflow._init_nodes()
    graph = workflow.workflow.compile()
    with patch(
        "socket.socket.connect", side_effect=AssertionError("network forbidden")
    ):
        try:
            result = asyncio.run(
                graph.ainvoke({"messages": [HumanMessage(content="cats")]})
            )
        except AIResponseInvalid:
            result = "invalid"
    workflow.call_tools.assert_not_awaited()
    assert len(requests) == 1
    assert workflow.chat_limit_service.record_model_usage.call_count == 1
    assert (
        workflow.chat_limit_service.record_model_usage.call_args.args[1].usage_metadata[
            "total_tokens"
        ]
        == 16
    )
    return result


@pytest.mark.parametrize(
    "message",
    [
        {
            "role": "assistant",
            "content": '{"request_scope":"out_of_scope","next_step":"__end__","reasoning":"unrelated"}',
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call1",
                    "type": "function",
                    "function": {
                        "name": "RoutingDecision",
                        "arguments": '{"request_scope":"out_of_scope","next_step":"__end__","reasoning":"unrelated"}',
                    },
                }
            ],
        },
    ],
)
def test_real_adapter_graph_accepts_tool_or_json(message):
    result = _run_adapter_graph(message)
    assert result["request_scope"] == "out_of_scope"
    assert result["next_step"] == "__end__"


@pytest.mark.parametrize(
    "message",
    [
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": '{"request_scope":'},
        {
            "role": "assistant",
            "content": '{"request_scope":"out_of_scope","next_step":"__end__"}',
        },
        {
            "role": "assistant",
            "content": '{"request_scope":"out_of_scope","next_step":"__end__","reasoning":"x","extra":1}',
        },
        {
            "role": "assistant",
            "content": '{"request_scope":"wrong","next_step":"__end__","reasoning":"x"}',
        },
        {
            "role": "assistant",
            "content": '{"request_scope":"in_scope","request_scope":"out_of_scope","next_step":"__end__","reasoning":"x"}',
        },
        {
            "role": "assistant",
            "content": '{"request_scope":"out_of_scope","next_step":"__end__","reasoning":1}',
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call1",
                    "type": "function",
                    "function": {"name": "Wrong", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call1",
                    "type": "function",
                    "function": {"name": "RoutingDecision", "arguments": "{invalid"},
                }
            ],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "a",
                    "type": "function",
                    "function": {
                        "name": "RoutingDecision",
                        "arguments": '{"request_scope":"out_of_scope","next_step":"__end__","reasoning":"x"}',
                    },
                },
                {
                    "id": "b",
                    "type": "function",
                    "function": {
                        "name": "RoutingDecision",
                        "arguments": '{"request_scope":"out_of_scope","next_step":"__end__","reasoning":"x"}',
                    },
                },
            ],
        },
        {
            "role": "assistant",
            "content": '{"request_scope":"in_scope","next_step":"retriever","reasoning":"x"}',
            "tool_calls": [
                {
                    "id": "call1",
                    "type": "function",
                    "function": {
                        "name": "RoutingDecision",
                        "arguments": '{"request_scope":"out_of_scope","next_step":"__end__","reasoning":"x"}',
                    },
                }
            ],
        },
    ],
)
def test_real_adapter_graph_rejects_invalid_envelopes(message):
    assert _run_adapter_graph(message) == "invalid"


@pytest.mark.parametrize(
    ("scope", "policy", "topics", "allowed"),
    [
        ("in_scope", "broad", [], True),
        ("ambiguous", "broad", [], False),
        ("unclassified", "broad", [], False),
        ("ambiguous", "named", [["EAGLT"]], True),
        ("in_scope", "named", [], False),
        ("in_scope", "broad", [["Adam"]], False),
    ],
)
def test_retriever_checks_policy_before_invoking_tools(scope, policy, topics, allowed):
    workflow = LearnerWorkflow.__new__(LearnerWorkflow)
    workflow.create_messages_to_pass = lambda *args: []
    workflow.call_model = AsyncMock(
        return_value=AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_knowledge_base",
                    "args": {
                        "queries": ["query"],
                        "topic_policy": policy,
                        "required_topics": topics,
                    },
                    "id": "test",
                    "type": "tool_call",
                }
            ],
        )
    )
    workflow.call_tools = AsyncMock(
        return_value={
            "search_knowledge_base": "RETRIEVER RESULTS:\n[SOURCE] (Score: 0.9) evidence"
        }
    )
    if allowed:
        result = asyncio.run(
            workflow.retriever_node(State(request_scope=scope, routing_validated=True))
        )
        assert result["retrieval_status"] == "adequate"
        workflow.call_tools.assert_awaited_once()
    else:
        with pytest.raises(AIResponseInvalid):
            asyncio.run(
                workflow.retriever_node(
                    State(request_scope=scope, routing_validated=True)
                )
            )
        workflow.call_tools.assert_not_awaited()


@pytest.mark.parametrize(
    ("query", "policy", "topics", "evidence", "expected"),
    [
        ("optimizers", "broad", [], "Adam uses adaptive moments.", True),
        ("neural networks", "broad", [], "Backpropagation computes gradients.", True),
        ("databases", "broad", [], "B-tree indexes enable lookups.", True),
        ("RetNet", "named", [["RetNet"]], "ResNet is a convolutional model.", False),
        (
            "MLA",
            "named",
            [["MLA", "Multi-Head Latent Attention"]],
            "Ordinary attention uses queries and keys.",
            False,
        ),
        (
            "RetNet",
            "named",
            [["RetNet", "Retentive Network"]],
            "Retentive Network is a sequence model.",
            True,
        ),
        (
            "RetNet versus Mamba",
            "named",
            [["RetNet"], ["Mamba"]],
            "RetNet and Mamba are sequence models.",
            True,
        ),
        ("RetNet", "named", [["RetNet"]], "A convolutional network exists.", False),
    ],
)
@pytest.mark.parametrize("mode", ["optimized", "legacy"])
def test_development_topic_cases_both_retrieval_paths(
    query, policy, topics, evidence, expected, mode
):
    from synthetic_evidence import FakeNode, FakeNodeWithScore, FakeRetriever

    from backend.configs.constants import RETRIEVAL_TOPIC_MISMATCH
    from backend.workflows.agents.tools import search_knowledge_base

    class Pipeline:
        def search(self, queries):
            return f"RETRIEVER RESULTS:\n\nQUERY: {queries[0]}\n[SOURCE] (Score: 0.90) {evidence}"

    tool = (
        search_knowledge_base(analyst_pipeline=Pipeline())
        if mode == "optimized"
        else search_knowledge_base(
            FakeRetriever({query: [FakeNodeWithScore(FakeNode(evidence), 0.9)]})
        )
    )
    result = tool.invoke(
        {"queries": [query], "topic_policy": policy, "required_topics": topics}
    )
    assert (result != RETRIEVAL_TOPIC_MISMATCH) is expected


def test_broad_search_rejects_unvalidated_in_scope():
    workflow = LearnerWorkflow.__new__(LearnerWorkflow)
    workflow.create_messages_to_pass = lambda *args: []
    workflow.call_model = AsyncMock(
        return_value=AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_knowledge_base",
                    "args": {"queries": ["optimizers"], "topic_policy": "broad"},
                    "id": "x",
                    "type": "tool_call",
                }
            ],
        )
    )
    workflow.call_tools = AsyncMock()
    with pytest.raises(AIResponseInvalid):
        asyncio.run(workflow.retriever_node(State(request_scope="in_scope")))
    workflow.call_tools.assert_not_awaited()
