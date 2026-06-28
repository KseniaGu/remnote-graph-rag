#!/usr/bin/env python
"""Render final-storage concept relations as an interactive Plotly graph."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx
import plotly.graph_objects as go


DEFAULT_STORAGE_DIR = Path(
    "data/testing/final_retrieval_optimization_v7_split_prompts/"
    "optimized_pipeline_run/final_storage"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Plotly HTML graph of concept nodes and semantic relations from final_storage."
    )
    parser.add_argument(
        "--storage-dir",
        type=Path,
        default=DEFAULT_STORAGE_DIR,
        help="Directory containing property_graph_store.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path. Defaults to <storage-dir>/concept_relation_graph.html.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Network layout seed for reproducible node placement.",
    )
    parser.add_argument(
        "--layout-iterations",
        type=int,
        default=250,
        help="Number of spring-layout iterations.",
    )
    parser.add_argument(
        "--max-hover-list",
        type=int,
        default=8,
        help="Maximum list items shown in hover fields.",
    )
    parser.add_argument(
        "--include-plotlyjs",
        choices=("cdn", "inline"),
        default="cdn",
        help="Use 'inline' for a self-contained HTML file; 'cdn' keeps the file smaller.",
    )
    return parser.parse_args()


def load_property_graph(storage_dir: Path) -> dict[str, Any]:
    path = storage_dir / "property_graph_store.json"
    if not path.exists():
        raise FileNotFoundError(f"property graph store not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def is_concept_node(node_id: str, node: dict[str, Any]) -> bool:
    properties = node.get("properties") or {}
    return node_id.startswith("concept_") or bool(properties.get("postprocess_concept_id"))


def concept_label(node_id: str, node: dict[str, Any]) -> str:
    properties = node.get("properties") or {}
    return str(
        properties.get("display_name")
        or properties.get("entity_name")
        or node.get("label")
        or node.get("name")
        or node_id
    )


def short_list(values: Any, *, limit: int) -> str:
    if values is None:
        return ""
    if not isinstance(values, list):
        values = [values]
    rendered = [str(value) for value in values if value is not None and str(value)]
    if not rendered:
        return ""
    suffix = "" if len(rendered) <= limit else f"<br>... +{len(rendered) - limit} more"
    return "<br>".join(html.escape(value) for value in rendered[:limit]) + suffix


def hover_section(title: str, value: Any, *, limit: int) -> str:
    rendered = short_list(value, limit=limit)
    if not rendered:
        return ""
    return f"<br><b>{html.escape(title)}:</b><br>{rendered}"


def build_graph(data: dict[str, Any]) -> tuple[nx.MultiDiGraph, dict[str, dict[str, Any]], list[dict[str, Any]]]:
    nodes = data.get("nodes") or {}
    relations = data.get("relations") or {}
    concepts = {node_id: node for node_id, node in nodes.items() if is_concept_node(node_id, node)}

    graph = nx.MultiDiGraph()
    for node_id, node in concepts.items():
        properties = node.get("properties") or {}
        graph.add_node(
            node_id,
            label=concept_label(node_id, node),
            concept_type=str(node.get("label") or properties.get("type") or "CONCEPT"),
            properties=properties,
        )

    semantic_relations: list[dict[str, Any]] = []
    for relation_id, relation in relations.items():
        source_id = relation.get("source_id")
        target_id = relation.get("target_id")
        predicate = str(relation.get("label") or "")
        if source_id not in concepts or target_id not in concepts:
            continue
        if predicate == "MENTIONS":
            continue
        properties = relation.get("properties") or {}
        payload = {
            "id": relation_id,
            "source_id": source_id,
            "target_id": target_id,
            "predicate": predicate,
            "properties": properties,
        }
        semantic_relations.append(payload)
        graph.add_edge(source_id, target_id, key=relation_id, **payload)

    return graph, concepts, semantic_relations


def node_hover_text(
    node_id: str,
    node_data: dict[str, Any],
    *,
    degree: int,
    max_hover_list: int,
) -> str:
    properties = node_data.get("properties") or {}
    label = html.escape(str(node_data.get("label") or node_id))
    concept_type = html.escape(str(node_data.get("concept_type") or "CONCEPT"))
    return (
        f"<b>{label}</b>"
        f"<br><b>Type:</b> {concept_type}"
        f"<br><b>ID:</b> {html.escape(node_id)}"
        f"<br><b>Degree:</b> {degree}"
        f"<br><b>Source chunks:</b> {len(properties.get('source_chunk_ids') or [])}"
        f"<br><b>Salience:</b> {html.escape(str(properties.get('postprocess_max_salience', '')))}"
        f"<br><b>Resolution:</b> {html.escape(str(properties.get('postprocess_resolution_source', '')))}"
        f"<br><b>Merge status:</b> {html.escape(str(properties.get('postprocess_merge_status', '')))}"
        + hover_section("Aliases", properties.get("aliases"), limit=max_hover_list)
        + hover_section("Evidence spans", properties.get("evidence_spans"), limit=max_hover_list)
        + hover_section("Source chunk IDs", properties.get("source_chunk_ids"), limit=max_hover_list)
    )


def edge_hover_text(
    relation: dict[str, Any],
    labels_by_id: dict[str, str],
    *,
    max_hover_list: int,
) -> str:
    properties = relation.get("properties") or {}
    source_label = labels_by_id.get(relation["source_id"], relation["source_id"])
    target_label = labels_by_id.get(relation["target_id"], relation["target_id"])
    return (
        f"<b>{html.escape(source_label)}</b>"
        f" -> <b>{html.escape(str(relation['predicate']))}</b>"
        f" -> <b>{html.escape(target_label)}</b>"
        f"<br><b>Relation ID:</b> {html.escape(str(relation['id']))}"
        f"<br><b>Family:</b> {html.escape(str(properties.get('predicate_family', '')))}"
        f"<br><b>Confidence:</b> {html.escape(str(properties.get('max_confidence', '')))}"
        f"<br><b>Generality:</b> {html.escape(str(properties.get('max_generality_score', '')))}"
        f"<br><b>Retrieval usefulness:</b> {html.escape(str(properties.get('max_retrieval_usefulness', '')))}"
        f"<br><b>Visualization usefulness:</b> {html.escape(str(properties.get('max_visualization_usefulness', '')))}"
        + hover_section("Relation phrases", properties.get("relation_phrases"), limit=max_hover_list)
        + hover_section("Evidence spans", properties.get("evidence_spans"), limit=max_hover_list)
        + hover_section("Evidence chunk IDs", properties.get("evidence_chunk_ids"), limit=max_hover_list)
    )


def make_figure(
    graph: nx.MultiDiGraph,
    relations: list[dict[str, Any]],
    *,
    seed: int,
    layout_iterations: int,
    max_hover_list: int,
) -> go.Figure:
    if graph.number_of_nodes() == 0:
        raise ValueError("No concept nodes found in property graph store.")

    undirected = nx.Graph()
    undirected.add_nodes_from(graph.nodes)
    for source_id, target_id in graph.edges():
        if source_id == target_id:
            continue
        undirected.add_edge(source_id, target_id)

    positions = nx.spring_layout(
        undirected if undirected.number_of_edges() else graph,
        seed=seed,
        iterations=layout_iterations,
        k=None,
    )

    labels_by_id = {node_id: data["label"] for node_id, data in graph.nodes(data=True)}

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_mid_x: list[float] = []
    edge_mid_y: list[float] = []
    edge_hover: list[str] = []
    edge_label_text: list[str] = []
    for relation in relations:
        source = relation["source_id"]
        target = relation["target_id"]
        if source not in positions or target not in positions:
            continue
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_mid_x.append((x0 + x1) / 2)
        edge_mid_y.append((y0 + y1) / 2)
        edge_hover.append(edge_hover_text(relation, labels_by_id, max_hover_list=max_hover_list))
        edge_label_text.append(str(relation["predicate"]))

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line={"width": 0.7, "color": "rgba(90, 100, 120, 0.35)"},
        hoverinfo="skip",
        name="relations",
    )
    edge_hover_trace = go.Scatter(
        x=edge_mid_x,
        y=edge_mid_y,
        mode="markers",
        marker={"size": 8, "color": "rgba(0, 0, 0, 0)"},
        text=edge_hover,
        hoverinfo="text",
        name="relation details",
    )

    type_counts = Counter(data.get("concept_type", "CONCEPT") for _, data in graph.nodes(data=True))
    type_order = [item for item, _ in type_counts.most_common()]
    palette = [
        "#2563eb",
        "#16a34a",
        "#dc2626",
        "#9333ea",
        "#ca8a04",
        "#0891b2",
        "#db2777",
        "#475569",
        "#ea580c",
        "#4f46e5",
    ]
    color_by_type = {concept_type: palette[index % len(palette)] for index, concept_type in enumerate(type_order)}

    node_traces: list[go.Scatter] = []
    for concept_type in type_order:
        xs: list[float] = []
        ys: list[float] = []
        texts: list[str] = []
        hovers: list[str] = []
        sizes: list[float] = []
        for node_id, data in graph.nodes(data=True):
            if data.get("concept_type") != concept_type:
                continue
            x, y = positions[node_id]
            degree = graph.degree(node_id)
            properties = data.get("properties") or {}
            source_chunk_count = len(properties.get("source_chunk_ids") or [])
            xs.append(x)
            ys.append(y)
            texts.append(str(data["label"]))
            hovers.append(node_hover_text(node_id, data, degree=degree, max_hover_list=max_hover_list))
            sizes.append(9 + min(24, degree * 2.5 + source_chunk_count))
        node_traces.append(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                text=texts,
                textposition="top center",
                textfont={"size": 9},
                hoverinfo="text",
                hovertext=hovers,
                marker={
                    "size": sizes,
                    "color": color_by_type[concept_type],
                    "line": {"width": 0.8, "color": "white"},
                    "opacity": 0.88,
                },
                name=f"{concept_type} ({len(xs)})",
            )
        )

    figure = go.Figure(data=[edge_trace, edge_hover_trace, *node_traces])
    figure.update_layout(
        title={
            "text": (
                "Final Storage Concept Graph"
                f"<br><sup>{graph.number_of_nodes()} concepts, {len(relations)} semantic relations</sup>"
            ),
            "x": 0.01,
        },
        showlegend=True,
        hovermode="closest",
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
        plot_bgcolor="white",
        xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
        yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
        legend={"itemsizing": "constant"},
    )
    return figure


def main() -> int:
    args = parse_args()
    storage_dir = args.storage_dir.expanduser().resolve()
    output = (args.output or storage_dir / "concept_relation_graph.html").expanduser().resolve()

    data = load_property_graph(storage_dir)
    graph, _, relations = build_graph(data)
    figure = make_figure(
        graph,
        relations,
        seed=args.seed,
        layout_iterations=args.layout_iterations,
        max_hover_list=args.max_hover_list,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        str(output),
        include_plotlyjs=args.include_plotlyjs,
        full_html=True,
        config={"responsive": True, "displaylogo": False},
    )
    print(f"Wrote {output}")
    print(f"Concept nodes: {graph.number_of_nodes()}")
    print(f"Semantic relations: {len(relations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
