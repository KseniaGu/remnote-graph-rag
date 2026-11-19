import asyncio
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app import state as app_state
from app.state import AppState
from backend.configs.constants import CACHE_SCHEMA_VERSION
from backend.workflows.agents.schemas import State
from backend.workflows.learner import LearnerWorkflow
from backend.workflows.presentation import (
    ExecutionStage,
    GraphArtifact,
    GraphPayload,
    GraphToolResult,
    Message,
    ResponseEvidence,
    ResponseSource,
    SessionSnapshot,
    TurnExecution,
)


def graph_payload(label="Attention"):
    return GraphPayload.model_validate(
        {
            "artifact_id": str(uuid4()),
            "title": label,
            "nodes": [{"id": "concept-a", "label": label}],
            "edges": [],
        }
    )


def test_live_agent_progress_tracks_observed_invocations_for_the_current_turn():
    state = AppState(_reflex_internal_init=True)
    identity = {"session_id": str(uuid4()), "turn_id": str(uuid4())}
    state.executions = [TurnExecution(**identity, origin="workflow")]
    state.is_processing = True
    inactive = dict.fromkeys(
        ("Orchestrator", "Retriever", "Researcher", "Analyst", "Mentor", "Visualizer"),
        "inactive",
    )

    def progress():
        return {agent.name: agent.outcome for agent in state.live_agent_progress}

    assert progress() == inactive
    state._processing_turn_id = identity["turn_id"]
    for index, (agent, outcome) in enumerate(
        (
            ("orchestrator", "completed"),
            ("researcher", "failed"),
            ("orchestrator", "running"),
        )
    ):
        stage = ExecutionStage(
            **identity,
            sequence=index,
            invocation_id=str(index),
            agent=agent,
            start_offset=index,
        )
        state._record_stage(stage)
        if outcome != "running":
            state._record_stage(
                stage.model_copy(
                    update={"outcome": outcome, "end_offset": index + 0.25}
                )
            )
    assert progress() == {**inactive, "Orchestrator": "running", "Researcher": "failed"}
    state._finish_execution("interrupted", 3)
    assert progress() == {
        **inactive,
        "Orchestrator": "interrupted",
        "Researcher": "failed",
    }
    assert [stage.outcome for stage in state.executions[0].stages] == [
        "completed",
        "failed",
        "interrupted",
    ]

    cache_identity = {**identity, "turn_id": str(uuid4())}
    state.executions = [
        *state.executions,
        TurnExecution(**cache_identity, origin="cache"),
    ]
    state._processing_turn_id = cache_identity["turn_id"]
    state.agent_history = ["mentor"]
    assert progress() == inactive
    assert state.processing_from_cache


def test_active_visualizer_accepts_structured_output_and_accumulates_artifacts():
    workflow = object.__new__(LearnerWorkflow)
    state = State()
    graphs = [graph_payload(), graph_payload("KV cache")]
    queries = [
        ["BERT architecture", "BERT training objective", "bidirectional attention"],
        ["KV cache", "grouped-query attention"],
    ]
    for graph, graph_queries in zip(graphs, queries, strict=True):
        tool_result = GraphToolResult(queries=graph_queries, graph=graph)
        state.context = json.dumps(
            {"get_subgraphs_to_visualize": tool_result.model_dump(mode="json")}
        )
        update = asyncio.run(workflow.visualizer_node(state))
        state = state.model_copy(update=update)
    assert [item["artifact_id"] for item in state.visual_artifacts] == [
        graph.artifact_id for graph in graphs
    ]
    assert [item["retrieval_queries"] for item in state.visual_artifacts] == queries


@pytest.mark.parametrize("text", ["[RESEARCH_COMPLETE]", "Visualization failed"])
def test_graph_evidence_text_is_not_a_routing_signal(text):
    graph = graph_payload()
    graph.nodes[0].source_ids.append("note:chunk")
    graph.sources.append(
        ResponseSource(kind="note", source_id="note:chunk", excerpt=text)
    )
    result = GraphToolResult(queries=["map"], graph=graph)
    state = State(
        context=json.dumps({"get_subgraphs_to_visualize": result.model_dump()})
    )
    assert LearnerWorkflow._deterministic_route(state) == "visualizer"


def test_optimized_tool_is_structured_by_default_in_application_workflow():
    from backend.configs.search import KnowledgeGraphSearchSettings

    workflow = object.__new__(LearnerWorkflow)
    workflow.knowledge_graph_indexer = SimpleNamespace(
        kg_search_settings=KnowledgeGraphSearchSettings()
    )
    from unittest.mock import patch

    graph = graph_payload()
    with patch("backend.workflows.learner.VisualizerRetrievalPipeline") as pipeline:
        pipeline.return_value.structured_graph.return_value = GraphToolResult(
            queries=["map"], graph=graph
        )
        tool = workflow._build_visualizer_tool()
        result = tool.invoke({"queries": ["map"]})
    assert result["graph"]["artifact_id"] == graph.artifact_id


def test_late_map_keeps_existing_selection_and_only_current_map_can_prepare_draft():
    state = AppState(_reflex_internal_init=True)
    state.session_id = str(uuid4())
    turn_id = str(uuid4())
    first, second = [
        GraphArtifact(
            **graph_payload(label).model_dump(),
            session_id=state.session_id,
            turn_id=turn_id,
            timestamp="12:00",
        )
        for label in ("Attention", "KV cache")
    ]
    state._append_maps([first])
    state.select_recent_map(first.artifact_id)
    state.open_chat()
    state._append_maps([second])
    assert state.active_view == "chat"
    assert state.selected_recent_map_id == first.artifact_id
    state.prepare_graph_explanation(second.artifact_id, "concept-a")
    assert state.graph_draft == ""
    state.open_graph()
    state.prepare_graph_explanation(first.artifact_id, "concept-a")
    assert state.active_view == "chat"
    assert state.graph_draft == "Explain Attention."
    assert not state.messages
    assert CACHE_SCHEMA_VERSION > 4


def test_graph_actions_keep_response_evidence_and_graph_source_scope_distinct():
    state = AppState(_reflex_internal_init=True)
    state.session_id = str(uuid4())
    turn = {"session_id": state.session_id, "turn_id": str(uuid4())}
    graph = GraphArtifact(**graph_payload().model_dump(), **turn, timestamp="12:00")
    graph.nodes[0].source_ids = ["note:graph"]
    graph.sources = [
        ResponseSource(kind="note", source_id="note:graph", title="Graph note")
    ]
    response_id = str(uuid4())
    state.messages = [
        Message(
            **turn,
            role="user",
            content="Attention",
            agent="",
            timestamp="12:00",
            dom_id=f"message-{uuid4().hex}",
        ),
        Message(
            **turn,
            role="assistant",
            content="Answer",
            agent="analyst",
            timestamp="12:01",
            dom_id=f"message-{uuid4().hex}",
            response_id=response_id,
            evidence=ResponseEvidence(
                concept_ids=["concept-a"],
                sources=[
                    ResponseSource(kind="note", chunk_id="missing"),
                    ResponseSource(kind="note", title="Supporting relation note"),
                    ResponseSource(
                        kind="note", title="Answer note", excerpt="Attention details."
                    ),
                ],
            ),
        ),
    ]
    state._append_maps([graph])
    assert [item.title for item in state.study_sources.notes] == ["Answer note"]
    state.select_recent_map(graph.artifact_id)
    assert state.active_view == "graph"
    state.show_graph_source(graph.artifact_id, "concept-a", "note:wrong")
    assert state.active_view == "graph"
    state.show_graph_source(graph.artifact_id, "concept-a", "note:graph")
    assert state.active_view == "chat"
    assert state.study_sources.status == "graph"
    assert [item.title for item in state.study_sources.notes] == ["Graph note"]
    state.select_response_sources(response_id)
    assert [item.title for item in state.study_sources.notes] == ["Answer note"]


def test_restored_answer_resolves_its_relationship_sources(monkeypatch):
    from backend.workflows import learner_reflex

    state = AppState(_reflex_internal_init=True)
    state.session_id = str(uuid4())
    turn = {
        "session_id": state.session_id,
        "turn_id": str(uuid4()),
        "timestamp": "12:00",
    }
    refs = ["chunk-edge", "chunk-private", "chunk-missing", "chunk-direct"]
    direct = ResponseSource(
        kind="note",
        chunk_id=refs[-1],
        excerpt="Passage sent to Analyst.",
        retrieval_score=0.81,
    )
    snapshot = SessionSnapshot(
        session_id=state.session_id,
        artifacts=[],
        messages=[
            Message(
                **turn,
                role="user",
                content="Transformer?",
                agent="",
                dom_id=f"message-{uuid4().hex}",
            ),
            Message(
                **turn,
                role="assistant",
                content="Answer",
                agent="analyst",
                dom_id=f"message-{uuid4().hex}",
                response_id=str(uuid4()),
                evidence=ResponseEvidence(
                    sources=[
                        *(
                            ResponseSource(kind="note", chunk_id=chunk)
                            for chunk in refs
                        ),
                        direct,
                    ],
                ),
            ),
        ],
    )
    reads = []

    def get_document(chunk_id, *, raise_error):
        assert raise_error is False
        reads.append(chunk_id)
        if chunk_id == "chunk-missing":
            return None
        return SimpleNamespace(
            id_=chunk_id,
            text="A note that supports the relationship.",
            metadata={
                "path": ["Architecture", "Attention"],
                "quarantined": chunk_id == "chunk-private",
                "postprocess_chunk_summary": f"Stored summary for {chunk_id}.",
            },
        )

    monkeypatch.setattr(
        app_state,
        "get_session_memory_store",
        lambda: SimpleNamespace(get=lambda *a, **kw: snapshot),
    )
    runtime = learner_reflex.ReflexLearnerWorkflow()
    monkeypatch.setattr(learner_reflex, "get_workflow", lambda: runtime)
    monkeypatch.setattr(
        learner_reflex,
        "open_document_store",
        lambda *_: SimpleNamespace(get_document=get_document),
        raising=False,
    )

    available, restored = asyncio.run(state._load_session_snapshot(state.session_id))
    assert available
    state._restore_snapshot(state.session_id, restored)
    [passage] = state.study_sources.notes
    assert passage.title == "Note 1"
    assert passage.summary == "Stored summary for chunk-direct."
    assert passage.excerpt == ""
    [support] = state.study_sources.relationship_notes
    assert support.title == "Note 2"
    assert support.path == ""
    assert support.summary == "Stored summary for chunk-edge."
    assert support.excerpt == ""
    assert reads == refs
    restored_answer = restored.messages[-1]
    assert any(
        source.excerpt == direct.excerpt for source in restored_answer.evidence.sources
    )
    assert any(
        source.retrieval_score == 0.81 for source in restored_answer.evidence.sources
    )
