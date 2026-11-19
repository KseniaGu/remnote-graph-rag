import asyncio
from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.messages import ToolMessage
from synthetic_runtime import make_scenario
from synthetic_workflow import FakeGraph, wrapper

from backend.evaluation import runtime_live
from backend.evaluation.runtime_evaluation import (
    RuntimeRecord,
    ToolObservation,
    _graph_observation,
    _tool_evaluation_output,
    load_runtime_scenarios,
    score_runtime_record,
)
from backend.evaluation.runtime_live import _live_graph_observation
from backend.workflows.presentation import (
    GraphEdge,
    GraphNode,
    GraphPayload,
    GraphToolResult,
)


@pytest.mark.parametrize("state", ["valid", "empty", "malformed", "error"])
def test_structured_graph_normalizes_like_legacy_graph(state):
    left = GraphNode(id="concept_a", label="Alpha")
    right = GraphNode(id="concept_b", label="Beta")
    payload = GraphPayload(
        artifact_id=str(uuid4()),
        title="Map",
        nodes=[left, right],
        edges=[
            GraphEdge.from_relationship(left.id, "USES", right.id, synthetic=False),
            GraphEdge.from_relationship(left.id, "EXTENDS", right.id, synthetic=False),
            GraphEdge.from_relationship(
                right.id, "RELATED_TO", left.id, synthetic=True
            ),
        ],
    )
    legacy_output = [
        [left.id, right.id],
        [(edge.source, edge.predicate, edge.target) for edge in payload.edges],
        ["map"],
    ]
    display_order = list(
        set(
            legacy_output[0] + [left.id, right.id, left.id, right.id, right.id, left.id]
        )
    )
    labels = {left.id: left.label, right.id: right.label}
    hovertexts = {
        left.id: "Alpha → EXTENDS → Beta",
        right.id: "Beta → RELATED_TO → Alpha",
    }
    legacy_artifacts = [
        {
            "data": [
                {
                    "mode": "markers+text",
                    "text": [labels[node_id] for node_id in display_order],
                },
                {
                    "mode": "markers",
                    "hovertext": [hovertexts[node_id] for node_id in display_order],
                },
            ]
        }
    ]
    raw_graph = GraphToolResult(queries=["map"], graph=payload).model_dump(mode="json")
    raw_artifacts = [payload.model_dump(mode="json")]
    if state == "empty":
        legacy_output, legacy_artifacts = [[], [], ["map"]], []
        raw_graph, raw_artifacts = {"queries": ["map"], "graph": None}, []
    elif state == "malformed":
        legacy_output, legacy_artifacts = {"bad": True}, [{}]
        raw_graph["graph"]["edges"][0]["target"] = "missing"
        raw_artifacts = [raw_graph["graph"]]
    elif state == "error":
        legacy_output, legacy_artifacts = None, []
        raw_graph, raw_artifacts = None, []

    def trace_graph(output, artifacts):
        tool = ToolObservation(run_id="graph-tool", name="get_subgraphs_to_visualize")
        run = {"_id": tool.run_id, "outputs": {"output": output}}
        return _graph_observation({"visual_artifacts": artifacts}, [tool], [run])

    expected = _live_graph_observation(legacy_output, legacy_artifacts)
    actual = _live_graph_observation(raw_graph, raw_artifacts)
    assert actual == expected
    assert trace_graph(raw_graph, raw_artifacts) == trace_graph(
        legacy_output, legacy_artifacts
    )
    scenario = next(
        case
        for case in load_runtime_scenarios(
            Path("backend/evaluation/cases/runtime/scenarios.json")
        )
        if case.id == "explicit_mamba_visualization_success"
    )
    record = RuntimeRecord(
        scenario_id=scenario.id,
        scenario_version=scenario.version,
        run_id="fixture",
        source_mode="controlled_live",
        input=scenario.input,
        graph=expected,
        output_modality="graph" if expected.artifact_count else "none",
    )
    assert score_runtime_record(
        scenario, record.model_copy(update={"graph": actual})
    ) == score_runtime_record(scenario, record)


def test_tool_evidence_projection_uses_original_kb_text():
    message = ToolMessage(
        "Original context for chunk_0123456789abcdef",
        tool_call_id="search",
        artifact={"sources": [{"chunk_id": "private-sidecar"}]},
    )
    outputs = [
        message,
        message.model_dump(),
        {"id": ["ToolMessage"], "kwargs": message.model_dump()},
    ]
    for output in outputs:
        assert (
            _tool_evaluation_output("search_knowledge_base", output) == message.content
        )


def test_live_evaluation_projects_tool_events_before_collecting_evidence(monkeypatch):
    content = "[SOURCE] chunk_0123456789abcdef"
    graph = GraphPayload(
        artifact_id=str(uuid4()),
        title="Map",
        nodes=[
            GraphNode(
                id="concept_a",
                label="Alpha",
                source_ids=["note:chunk_ffffffffffffffff"],
            )
        ],
        edges=[],
    )

    class Events(FakeGraph):
        def __init__(self, structured):
            super().__init__(current="Answer")
            self.structured = structured

        async def astream_events(self, initial, config, version):
            outputs = {
                "search_knowledge_base": ToolMessage(
                    content,
                    tool_call_id="kb",
                    artifact={"sources": ["chunk_ffffffffffffffff"]},
                )
                if self.structured
                else content,
                "get_subgraphs_to_visualize": GraphToolResult(
                    queries=["map"],
                    graph=graph,
                ).model_dump(mode="json")
                if self.structured
                else (["concept_a"], [], ["map"]),
            }
            for name, output in outputs.items():
                yield {
                    "event": "on_tool_start",
                    "name": name,
                    "run_id": name,
                    "metadata": {"langgraph_node": "retriever"},
                    "data": {"input": {"queries": ["map"]}},
                }
                yield {
                    "event": "on_tool_end",
                    "name": name,
                    "run_id": name,
                    "data": {"output": output},
                }
            async for event in super().astream_events(initial, config, version):
                yield event

    def evaluate(structured):
        monkeypatch.setattr(
            runtime_live, "get_workflow", lambda: wrapper(Events(structured))
        )
        return asyncio.run(runtime_live.run_controlled_live_scenario(make_scenario()))

    original = evaluate(False)
    structured = evaluate(True)
    assert original.tools == structured.tools
    assert original.graph == structured.graph
    assert original.usage.logical_llm_calls == structured.usage.logical_llm_calls
