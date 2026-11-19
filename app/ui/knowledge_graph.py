"""Local Reflex boundary for the Stage 4 graph workspace."""

from collections.abc import Iterable

import reflex as rx

from backend.workflows.presentation import GraphPayload, ResponseSource

GRAPH_STYLESHEET = rx.asset("knowledge_graph.css", shared=True)
GRAPH_LIBRARY = f"$/public{rx.asset('knowledge_graph.jsx', shared=True)}"
rx.asset("graph_layout.mjs", shared=True)
rx.asset("graph_camera.mjs", shared=True)


class KnowledgeGraph(rx.Component):
    """Mounts the workspace. The host supplies a stable artifact key and height."""

    library = GRAPH_LIBRARY
    tag = "KnowledgeGraph"

    graph: rx.Var[GraphPayload | None]
    sources: rx.Var[list[ResponseSource]]
    theme: rx.Var[str]
    active: rx.Var[bool]
    selection_id: rx.Var[str]
    selection_nonce: rx.Var[int]
    maps: rx.Var[list]
    can_clear_session: rx.Var[bool]
    on_prepare: rx.EventHandler[rx.event.passthrough_event_spec(str, str)]
    on_select_map: rx.EventHandler[rx.event.passthrough_event_spec(str)]


class SourceList(rx.Component):
    library = GRAPH_LIBRARY
    tag = "SourceList"

    title: rx.Var[str]


def resolve_graph_action(
    artifacts: Iterable[GraphPayload],
    artifact_id: str,
    concept_id: str,
    source_id: str | None = None,
) -> bool:
    """Checks a callback against the host's current, session-owned artifacts."""
    return any(
        graph.artifact_id == artifact_id
        and any(
            node.id == concept_id
            and (source_id is None or source_id in node.source_ids)
            for node in graph.nodes
        )
        for graph in artifacts
    )
