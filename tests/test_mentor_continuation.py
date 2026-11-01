"""Offline contracts for an active Mentor quiz continuation."""

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from backend.configs.models import ModelSettings, _vllm_models
from backend.utils.chat_errors import AIResponseInvalid

PROMPT_DIR = Path("backend/llm/prompts/learner_workflow/orchestrator/routing")


def test_active_mentor_continuation_precedence_is_explicit():
    version = ModelSettings().orchestrator.prompt_version["routing"]
    prompt = yaml.safe_load((PROMPT_DIR / f"{version}.yaml").read_text())["system"][
        "system_instruction"
    ]
    for required in (
        "active quiz or interview",
        "stuck, asks for a hint or explanation",
        "summary alone",
        "explicit new standalone request",
        "visualization",
        "quit or cancel",
        "latest human message",
        "For non-empty Current Context reaching this slow path, choose analyst unless",
        "out-of-scope request, quit or cancel, explicit new standalone request, visualization",
    ):
        assert required in prompt


def test_active_routing_selectors_choose_v6_without_changing_graph_index(monkeypatch):
    monkeypatch.setenv("VLLM_ROUTING_URL", "https://vllm.example")
    monkeypatch.setenv("VLLM_MODEL_PATH", "test")
    monkeypatch.setenv("VLLM_GENERATION_URL", "https://vllm.example")
    for settings in (ModelSettings(), _vllm_models()):
        assert settings.orchestrator.prompt_version == {
            "graph_index": "v2",
            "routing": "v6",
        }


def _fixture_graph(decision, *, tool_call=False):
    """Use the actual structured adapter and compiled workflow with fake transport."""
    import json
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    import httpx
    from langchain_core.messages import AIMessage
    from langchain_openrouter import ChatOpenRouter
    from langgraph.graph import StateGraph
    from openrouter import OpenRouter

    from backend.configs.enums import ModelRoleType
    from backend.workflows.agents.schemas import RoutingDecision, State
    from backend.workflows.learner import LearnerWorkflow

    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "dummy",
                "model": "dummy",
                "object": "chat.completion",
                "created": 0,
                "system_fingerprint": None,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": (
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "route",
                                        "type": "function",
                                        "function": {
                                            "name": "RoutingDecision",
                                            "arguments": json.dumps(decision),
                                        },
                                    }
                                ],
                            }
                            if tool_call
                            else {
                                "role": "assistant",
                                "content": json.dumps(decision)
                                if decision is not None
                                else "",
                            }
                        ),
                    }
                ],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "total_tokens": 3,
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
    model = ChatOpenRouter(
        model="dummy",
        api_key="dummy",
        client=client,
        max_retries=0,
        disable_streaming=True,
    )
    w = LearnerWorkflow.__new__(LearnerWorkflow)
    w.orchestrator = model.with_structured_output(RoutingDecision, include_raw=True)
    w.mentor = SimpleNamespace(
        ainvoke=AsyncMock(return_value=AIMessage(content="Synthetic Mentor response"))
    )
    w.models_settings = SimpleNamespace(
        orchestrator=SimpleNamespace(provider="openrouter"),
        mentor=SimpleNamespace(provider="openrouter"),
    )
    w.chat_limit_service = SimpleNamespace(
        begin_llm_call=Mock(),
        reserve_llm_attempt=AsyncMock(),
        record_model_usage=Mock(),
        claim_retry=Mock(return_value=False),
    )
    w.call_tools = AsyncMock()

    def read_prompt(path):
        y = yaml.safe_load(Path(path).read_text())
        return y["user"], y["system"]

    w.prompts = {
        ModelRoleType.orchestrator: read_prompt(PROMPT_DIR / "v6.yaml"),
        ModelRoleType.mentor: read_prompt(
            "backend/llm/prompts/learner_workflow/mentor/v4.yaml"
        ),
    }

    async def placeholder(state):
        return {
            "messages": [
                AIMessage(
                    content="Synthetic worker result",
                    additional_kwargs={"agent": "[ANALYST]"},
                )
            ]
        }

    w.retriever_node = AsyncMock(side_effect=placeholder)
    w.researcher_node = AsyncMock(side_effect=placeholder)
    w.analyst_node = AsyncMock(side_effect=placeholder)
    w.visualizer_node = AsyncMock(side_effect=placeholder)
    w.workflow = StateGraph(State)
    w._init_nodes()
    return w, w.workflow.compile(), requests


# Each fixture specifies the model output; it tests dispatch/contract, not model quality.
MATRIX = [
    (
        "stuck",
        "I can't answer this question. Please explain this topic",
        "attention quiz",
        "mentor",
    ),
    ("hint", "Please give me a hint", "attention quiz", "mentor"),
    ("answer", "The scale reduces very large dot products", "attention quiz", "mentor"),
    ("summary only", "Please explain this question", "summary only", "mentor"),
    (
        "ColBERT guidance",
        "Explain it in a guidance mode please",
        "ColBERT quiz",
        "mentor",
    ),
    (
        "new topic",
        "Explain TCP congestion control as a new topic",
        "attention quiz",
        "retriever",
    ),
    ("visualize", "Draw a graph of Mamba", "attention quiz", "retriever"),
    ("quit", "Quit the quiz", "attention quiz", "__end__"),
    ("cancel", "Cancel this quiz", "attention quiz", "__end__"),
    (
        "out of scope",
        "Please tell me about biological cats",
        "attention quiz",
        "__end__",
    ),
    (
        "stale summary",
        "Start a new standalone topic: databases",
        "stale quiz",
        "retriever",
    ),
]


def _input_case(context, question):
    from langchain_core.messages import AIMessage, HumanMessage

    if context == "summary only":
        return [
            HumanMessage(content=question)
        ], "An attention-mechanism quiz is active in Mentor mode."
    topic = "ColBERT and MaxSim" if context == "ColBERT quiz" else "attention mechanism"
    summary = (
        "An older attention quiz is active."
        if context == "stale quiz"
        else f"An {topic} quiz is active in Mentor mode."
    )
    return [
        HumanMessage(content=f"Quiz me on {topic}"),
        AIMessage(
            content=(
                "How does scaled dot-product attention avoid overly sharp softmax scores?"
                if topic == "attention mechanism"
                else "Why can ColBERT pre-compute document token representations?"
            )
        ),
        HumanMessage(content=question),
    ], summary


@pytest.mark.parametrize("name,question,context,step", MATRIX)
def test_synthetic_adapter_graph_dispatch_and_rendered_context(
    name, question, context, step
):
    scope = "out_of_scope" if name == "out of scope" else "in_scope"
    w, graph, requests = _fixture_graph(
        {"request_scope": scope, "next_step": step, "reasoning": "synthetic"}
    )
    messages, summary = _input_case(context, question)
    with patch(
        "socket.socket.connect", side_effect=AssertionError("network forbidden")
    ):
        result = asyncio.run(
            graph.ainvoke({"messages": messages, "session_summary": summary})
        )
    assert len(requests) == 1
    prompt_messages = requests[0]["messages"]
    rendered = "\n".join(str(m.get("content", "")) for m in prompt_messages)
    assert question in rendered and summary in rendered
    assert "active quiz or interview" in rendered
    assert w.retriever_node.await_count == int(step == "retriever")
    assert not w.researcher_node.await_count and not w.analyst_node.await_count
    w.call_tools.assert_not_awaited()
    if step == "mentor":
        assert result["messages"][-1].content == "Synthetic Mentor response"
        mentor_messages = w.mentor.ainvoke.await_args.args[0]
        mentor_text = "\n".join(str(message) for message in mentor_messages)
        assert question in mentor_text and summary in mentor_text
        if context != "summary only":
            assert (
                "softmax" in mentor_text
                if context != "ColBERT quiz"
                else "ColBERT" in mentor_text
            )
    else:
        w.mentor.ainvoke.assert_not_awaited()


def test_invalid_routing_stops_and_valid_conflicting_route_is_not_overridden():
    messages, summary = _input_case(
        "attention quiz", "I can't answer this question. Please explain this topic"
    )
    w, graph, requests = _fixture_graph(None)
    with patch(
        "socket.socket.connect", side_effect=AssertionError("network forbidden")
    ):
        with pytest.raises(AIResponseInvalid):
            asyncio.run(
                graph.ainvoke({"messages": messages, "session_summary": summary})
            )
    assert len(requests) == 1
    w.mentor.ainvoke.assert_not_awaited()
    w.retriever_node.assert_not_awaited()
    w.call_tools.assert_not_awaited()
    w, graph, _ = _fixture_graph(
        {
            "request_scope": "in_scope",
            "next_step": "retriever",
            "reasoning": "synthetic conflict",
        }
    )
    with patch(
        "socket.socket.connect", side_effect=AssertionError("network forbidden")
    ):
        asyncio.run(graph.ainvoke({"messages": messages, "session_summary": summary}))
    w.retriever_node.assert_awaited_once()
    w.mentor.ainvoke.assert_not_awaited()


@pytest.mark.parametrize("tool_call", [False, True])
def test_exact_stuck_through_wrapper_and_controlled_live(monkeypatch, tool_call):
    from backend.configs.enums import WorkflowEventType
    from backend.evaluation import runtime_live
    from backend.evaluation.runtime_evaluation import RuntimeScenario
    from backend.workflows.learner_reflex import ReflexLearnerWorkflow

    decision = {
        "request_scope": "in_scope",
        "next_step": "mentor",
        "reasoning": "synthetic",
    }
    question = "I can't answer this question. Please explain this topic"
    history = [
        {"role": "user", "content": "Quiz me on attention mechanism"},
        {
            "role": "assistant",
            "content": "How does scaled dot-product attention avoid overly sharp softmax scores?",
        },
    ]
    summary = "An attention-mechanism quiz is active in Mentor mode."

    async def wrapper_events(wrapper):
        return [
            event
            async for event in wrapper.process_message(
                question, history, session_summary=summary
            )
        ]

    w, graph, requests = _fixture_graph(decision, tool_call=tool_call)
    wrapper = ReflexLearnerWorkflow()
    wrapper._graph = graph
    with patch(
        "socket.socket.connect", side_effect=AssertionError("network forbidden")
    ):
        events = asyncio.run(wrapper_events(wrapper))
    assert [
        event.data for event in events if event.type == WorkflowEventType.RESPONSE
    ] == [{"content": "Synthetic Mentor response", "agent": "mentor"}]
    assert sum(event.type == WorkflowEventType.COMPLETE for event in events) == 1
    assert len(requests) == 1
    w.retriever_node.assert_not_awaited()
    w.researcher_node.assert_not_awaited()
    w.analyst_node.assert_not_awaited()
    w.call_tools.assert_not_awaited()
    assert "softmax" in str(w.mentor.ainvoke.await_args.args[0])

    w, graph, requests = _fixture_graph(decision, tool_call=tool_call)
    wrapper = ReflexLearnerWorkflow()
    wrapper._graph = graph
    monkeypatch.setattr(runtime_live, "get_workflow", lambda: wrapper)
    scenario = RuntimeScenario.model_validate(
        {
            "schema_version": "runtime-evaluation",
            "id": "synthetic_mentor",
            "version": 1,
            "review_status": "reviewed",
            "description": "Synthetic continuation",
            "trace_ids": [],
            "input": {
                "user_message": question,
                "message_history": history,
                "session_summary": summary,
            },
            "expectations": {"allowed_worker_sequences": [["mentor"]]},
            "provenance": ["synthetic"],
        }
    )
    with patch(
        "socket.socket.connect", side_effect=AssertionError("network forbidden")
    ):
        record = asyncio.run(runtime_live.run_controlled_live_scenario(scenario))
    assert record.final_response == "Synthetic Mentor response"
    assert record.worker_sequence == ["mentor"]
    assert record.provenance["response_origin"] == "graph_message"
    assert not record.tools
    assert len(requests) == 1
    w.retriever_node.assert_not_awaited()
    w.researcher_node.assert_not_awaited()
    w.analyst_node.assert_not_awaited()
    w.call_tools.assert_not_awaited()
    assert "softmax" in str(w.mentor.ainvoke.await_args.args[0])
