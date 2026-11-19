import asyncio

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt
from synthetic_workflow import wrapper

from backend.configs.enums import WorkflowEventType as Kind


def event(kind, name, run_id, *, node="", parents=(), **data):
    return {
        "event": kind,
        "name": name,
        "run_id": run_id,
        "parent_ids": list(parents),
        "metadata": {"langgraph_node": node},
        "data": data,
    }


def response(text, message_id, chunk_id):
    return AIMessage(
        content=text,
        id=message_id,
        additional_kwargs={
            "agent": "[ANALYST]",
            "response_evidence": {"sources": [{"kind": "note", "chunk_id": chunk_id}]},
        },
    )


class EventGraph:
    def __init__(self, *events):
        self.events = events

    async def astream_events(self, initial, **kwargs):
        for item in self.events:
            if isinstance(item, Exception):
                raise item
            yield item


def collect(*events):
    async def run():
        stream = wrapper(EventGraph(*events)).stream_with_tokens("Question", [])
        return [item async for item in stream]

    return asyncio.run(run())


def boundaries(node, node_id, *messages):
    return (
        event("on_chain_start", node, node_id, node=node, parents=("root",)),
        event(
            "on_chain_end",
            node,
            node_id,
            node=node,
            parents=("root",),
            output={"messages": list(messages)},
        ),
    )


def token(node_id, model_id, text):
    return event(
        "on_chat_model_stream",
        "analyst",
        model_id,
        node="analyst",
        parents=("root", node_id),
        chunk=AIMessageChunk(content=text, id=model_id),
    )


def test_repeated_invocations_reconcile_individual_responses_and_evidence():
    first = response("Final answer one", "model-first", "chunk-a")
    extra = response("Another final response", "non-streamed", "chunk-b")
    last = response("Final answer two", "model-last", "chunk-c")
    start_a, end_a = boundaries("analyst", "worker-a", first, extra)
    start_b, end_b = boundaries("analyst", "worker-b", last)
    events = collect(
        event("on_chain_start", "LangGraph", "root"),
        *boundaries("orchestrator", "route-a"),
        start_a,
        token("worker-a", "abandoned-model", "Abandoned attempt"),
        token("worker-a", "model-first", "Draft one"),
        end_a,
        *boundaries("orchestrator", "route-b"),
        start_b,
        token("worker-b", "model-last", "Draft two"),
        end_b,
        event(
            "on_chain_end",
            "LangGraph",
            "root",
            output={"messages": [first, extra, last]},
        ),
    )
    starts = [item.data["stage"] for item in events if item.type == Kind.AGENT_START]
    ends = [item.data["stage"] for item in events if item.type == Kind.AGENT_END]
    assert [stage.invocation_id for stage in starts] == [
        "route-a",
        "worker-a",
        "route-b",
        "worker-b",
    ]
    assert [stage.sequence for stage in starts] == list(range(4))
    assert all(
        stage.outcome == "completed" and stage.duration_seconds >= 0 for stage in ends
    )
    assert all(
        start.start_offset == end.start_offset
        for start, end in zip(starts, ends, strict=True)
    )
    first_end = next(
        index
        for index, item in enumerate(events)
        if item.type == Kind.AGENT_END
        and item.data["stage"].invocation_id == "worker-a"
    )
    first_final = next(
        index for index, item in enumerate(events) if item.type == Kind.RESPONSE
    )
    assert first_end < first_final
    tokens = {
        item.data["chunk"]: item.data["response_id"]
        for item in events
        if item.type == Kind.TOKEN
    }
    responses = [item.data for item in events if item.type == Kind.RESPONSE]
    assert [item["content"] for item in responses] == [
        first.content,
        extra.content,
        last.content,
    ]
    assert [item["evidence"].sources[0].chunk_id for item in responses] == [
        "chunk-a",
        "chunk-b",
        "chunk-c",
    ]
    assert [item["invocation_id"] for item in responses] == [
        "worker-a",
        "worker-a",
        "worker-b",
    ]
    assert responses[0]["response_id"] == tokens["Draft one"]
    assert responses[2]["response_id"] == tokens["Draft two"]
    assert tokens["Abandoned attempt"] not in {
        item["response_id"] for item in responses
    }


def test_failed_invocation_keeps_only_previously_finalized_response():
    first = response("Confirmed", "model-first", "chunk-a")
    start_b, _ = boundaries("analyst", "worker-b")
    events = collect(
        event("on_chain_start", "LangGraph", "root"),
        *boundaries("analyst", "worker-a", first),
        start_b,
        token("worker-b", "failed-model", "Unconfirmed"),
        RuntimeError("internal failure"),
    )
    assert [item.data["content"] for item in events if item.type == Kind.RESPONSE] == [
        "Confirmed"
    ]
    assert [
        item.data["stage"].outcome for item in events if item.type == Kind.AGENT_END
    ] == ["completed", "failed"]
    assert events[-1].type == Kind.ERROR
    assert not any(item.type == Kind.COMPLETE for item in events)


def test_incomplete_stream_has_no_success_or_fallback():
    start, _ = boundaries("analyst", "worker")
    events = collect(
        event("on_chain_start", "LangGraph", "root"),
        start,
        token("unrelated", "unknown", "Unattributed"),
        event("on_chain_end", "analyst", "", node="analyst", output={}),
    )
    assert [
        item.data["stage"].outcome for item in events if item.type == Kind.AGENT_END
    ] == ["incomplete"]
    assert [item.type for item in events] == [
        Kind.AGENT_START,
        Kind.AGENT_END,
        Kind.ERROR,
    ]


def test_root_error_does_not_attribute_parallel_invocations_without_evidence():
    start_a, _ = boundaries("analyst", "worker-a")
    start_b, _ = boundaries("mentor", "worker-b")
    events = collect(
        event("on_chain_start", "LangGraph", "root"),
        start_a,
        start_b,
        RuntimeError("Which invocation failed is unknown"),
    )
    assert events[-1].data["outcome"] == "error"
    assert [
        item.data["stage"].outcome for item in events if item.type == Kind.AGENT_END
    ] == ["incomplete", "incomplete"]


def test_compiled_graph_reconciles_model_callback_stream_to_final_message():
    model = GenericFakeChatModel(messages=iter(["draft content"]))

    async def analyst(state):
        output = await model.ainvoke(state["messages"])
        output.content = "normalized content"
        output.additional_kwargs["agent"] = "[ANALYST]"
        return {"messages": [output]}

    graph = StateGraph(MessagesState)
    graph.add_node("analyst", analyst)
    graph.add_edge(START, "analyst")
    graph.add_edge("analyst", END)

    async def run():
        stream = wrapper(graph.compile()).stream_with_tokens("Question", [])
        return [item async for item in stream]

    events = asyncio.run(run())
    tokens = [item.data for item in events if item.type == Kind.TOKEN]
    [final] = [item.data for item in events if item.type == Kind.RESPONSE]
    assert "".join(item["chunk"] for item in tokens) == "draft content"
    assert {item["response_id"] for item in tokens} == {final["response_id"]}
    assert {item["invocation_id"] for item in tokens} == {final["invocation_id"]}
    assert final["content"] == "normalized content"


def test_compiled_graph_interrupt_has_no_success_or_fallback():
    async def analyst(state):
        interrupt("paused")

    graph = StateGraph(MessagesState)
    graph.add_node("analyst", analyst)
    graph.add_edge(START, "analyst")
    graph.add_edge("analyst", END)

    async def run():
        stream = wrapper(graph.compile()).stream_with_tokens("Question", [])
        return [item async for item in stream]

    events = asyncio.run(run())
    assert [item.type for item in events] == [
        Kind.AGENT_START,
        Kind.AGENT_END,
        Kind.ERROR,
    ]
    assert events[1].data["stage"].outcome == "interrupted"
    assert events[2].data["outcome"] == "interrupted"


@pytest.mark.parametrize("when", ["before", "after"])
def test_static_interrupt_is_reported_even_with_empty_interrupt_values(when):
    async def analyst(state):
        return {"messages": [response("First answer", "one", "chunk-a")]}

    graph = StateGraph(MessagesState)
    graph.add_node("analyst", analyst)
    graph.add_edge(START, "analyst")
    graph.add_edge("analyst", END)
    compile_options = {f"interrupt_{when}": ["analyst"]}

    async def run():
        stream = wrapper(graph.compile(checkpointer=InMemorySaver(), **compile_options))
        return [item async for item in stream.stream_with_tokens("Question", [])]

    events = asyncio.run(run())
    assert events[-1].data["outcome"] == "interrupted"
    assert not any(item.type == Kind.COMPLETE for item in events)
    assert [item.data["content"] for item in events if item.type == Kind.RESPONSE] == (
        ["First answer"] if when == "after" else []
    )
