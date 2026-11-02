import asyncio

import pytest
from langchain_core.messages import AIMessage
from synthetic_runtime import make_scenario
from synthetic_workflow import FakeGraph, wrapper

from backend.configs.enums import WorkflowEventType
from backend.configs.messages import FALLBACK_OUT_OF_SCOPE
from backend.evaluation import runtime_live
from backend.evaluation.runtime_evaluation import _final_response
from backend.utils.chat_errors import AIResponseInvalid
from backend.utils.chat_limits import EvaluationBudgetUnavailable


def collect(generator):
    async def run():
        return [event async for event in generator]

    return asyncio.run(run())


@pytest.mark.parametrize(
    "method", ["process_message", "stream_with_status", "stream_with_tokens"]
)
@pytest.mark.parametrize("old_agent", ["ANALYST", "MENTOR"])
def test_out_of_scope_response_is_current_turn_only_with_prior_agent_history(
    method, old_agent
):
    w = wrapper(FakeGraph(old_agent=old_agent))
    history = [
        {"role": "user", "content": "Prior question"},
        {"role": "assistant", "content": "Prior response"},
    ]
    events = collect(getattr(w, method)("cats", history))
    responses = [
        event.data for event in events if event.type == WorkflowEventType.RESPONSE
    ]
    assert responses == [{"content": FALLBACK_OUT_OF_SCOPE, "agent": "system"}]
    assert sum(event.type == WorkflowEventType.COMPLETE for event in events) == 1
    assert not any(event.type == WorkflowEventType.ERROR for event in events)


@pytest.mark.parametrize(
    "method", ["process_message", "stream_with_status", "stream_with_tokens"]
)
def test_invalid_route_is_error_without_refusal_or_success(method):
    events = collect(
        getattr(wrapper(FakeGraph(error=AIResponseInvalid())), method)("cats", [])
    )
    assert [
        event.data["reason"]
        for event in events
        if event.type == WorkflowEventType.ERROR
    ] == ["ai_response_invalid"]
    assert not any(
        event.type in (WorkflowEventType.RESPONSE, WorkflowEventType.COMPLETE)
        for event in events
    )


@pytest.mark.parametrize(
    "method", ["process_message", "stream_with_status", "stream_with_tokens"]
)
def test_current_answer_not_replaced_by_fallback(method):
    events = collect(
        getattr(wrapper(FakeGraph(current="current answer")), method)("question", [])
    )
    assert [
        event.data for event in events if event.type == WorkflowEventType.RESPONSE
    ] == [{"content": "current answer", "agent": "analyst"}]


@pytest.mark.parametrize(
    "method", ["process_message", "stream_with_status", "stream_with_tokens"]
)
def test_graph_artifact_not_replaced_by_fallback(method):
    events = collect(getattr(wrapper(FakeGraph(graph=True)), method)("question", []))
    assert sum(event.type == WorkflowEventType.VISUALIZATION for event in events) == 1
    assert not any(event.type == WorkflowEventType.RESPONSE for event in events)
    assert sum(event.type == WorkflowEventType.COMPLETE for event in events) == 1


def test_controlled_live_reconstructs_fallback_with_explicit_origin(monkeypatch):
    monkeypatch.setattr(runtime_live, "get_workflow", lambda: wrapper(FakeGraph()))
    record = asyncio.run(runtime_live.run_controlled_live_scenario(make_scenario()))
    assert record.final_response == FALLBACK_OUT_OF_SCOPE
    assert record.provenance["response_origin"] == "application_fallback_reconstruction"


@pytest.mark.parametrize(
    "error", [AIResponseInvalid(), EvaluationBudgetUnavailable("budget")]
)
def test_controlled_live_failure_never_invents_fallback(monkeypatch, error):
    monkeypatch.setattr(
        runtime_live, "get_workflow", lambda: wrapper(FakeGraph(error=error))
    )
    if isinstance(error, EvaluationBudgetUnavailable):
        with pytest.raises(EvaluationBudgetUnavailable):
            asyncio.run(runtime_live.run_controlled_live_scenario(make_scenario()))
    else:
        record = asyncio.run(runtime_live.run_controlled_live_scenario(make_scenario()))
        assert record.trace_status == "error"
        assert record.final_response == ""


def test_graph_only_import_does_not_synthesize_fallback():
    inputs = {"messages": [{"type": "human", "content": "cats"}]}
    outputs = {
        "messages": inputs["messages"],
        "request_scope": "out_of_scope",
        "next_step": "__end__",
    }
    assert _final_response(inputs, outputs) == ""


def test_controlled_live_ignores_expanded_checkpoint_history(monkeypatch):
    class CheckpointGraph(FakeGraph):
        def state(self, initial):
            state = super().state(initial)
            state["messages"].insert(
                0,
                AIMessage(
                    content="older mentor", additional_kwargs={"agent": "[MENTOR]"}
                ),
            )
            return state

    monkeypatch.setattr(
        runtime_live, "get_workflow", lambda: wrapper(CheckpointGraph())
    )
    record = asyncio.run(runtime_live.run_controlled_live_scenario(make_scenario()))
    assert record.final_response == FALLBACK_OUT_OF_SCOPE
    assert record.provenance["response_origin"] == "application_fallback_reconstruction"


def test_controlled_live_preserves_current_answer(monkeypatch):
    monkeypatch.setattr(
        runtime_live,
        "get_workflow",
        lambda: wrapper(FakeGraph(current="current answer")),
    )
    record = asyncio.run(runtime_live.run_controlled_live_scenario(make_scenario()))
    assert record.final_response == "current answer"
    assert record.provenance["response_origin"] == "graph_message"


def test_controlled_live_graph_artifact_is_not_replaced_by_refusal(monkeypatch):
    monkeypatch.setattr(
        runtime_live, "get_workflow", lambda: wrapper(FakeGraph(graph=True))
    )
    record = asyncio.run(runtime_live.run_controlled_live_scenario(make_scenario()))
    assert record.final_response == ""
    assert record.output_modality == "graph"
    assert record.provenance["response_origin"] == "unavailable"


@pytest.mark.parametrize("kind", ["scope", "answer", "graph"])
def test_status_stream_handles_real_node_deltas_with_prior_history(kind):
    class DeltaGraph(FakeGraph):
        async def astream(self, initial, config):
            if kind == "scope":
                yield {
                    "orchestrator": {
                        "request_scope": "out_of_scope",
                        "next_step": "__end__",
                    }
                }
            else:
                yield {
                    "orchestrator": {
                        "request_scope": "in_scope",
                        "next_step": "analyst",
                    }
                }
                if kind == "answer":
                    yield {
                        "analyst": {
                            "messages": [
                                AIMessage(
                                    content="current answer",
                                    additional_kwargs={"agent": "[ANALYST]"},
                                )
                            ]
                        }
                    }
                else:
                    yield {"visualizer": {"visual_artifacts": [{"nodes": []}]}}
                yield {"orchestrator": {"next_step": "__end__"}}

    history = [
        {"role": "user", "content": "Previous"},
        {"role": "assistant", "content": "Old answer"},
    ]
    events = collect(wrapper(DeltaGraph()).stream_with_status("Current", history))
    responses = [e.data for e in events if e.type == WorkflowEventType.RESPONSE]
    if kind == "scope":
        assert responses == [{"content": FALLBACK_OUT_OF_SCOPE, "agent": "system"}]
    elif kind == "answer":
        assert responses == [{"content": "current answer", "agent": "analyst"}]
    else:
        assert not responses
        assert sum(e.type == WorkflowEventType.VISUALIZATION for e in events) == 1
    assert sum(e.type == WorkflowEventType.COMPLETE for e in events) == 1
