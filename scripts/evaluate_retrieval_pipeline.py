"""Run local retrieval benchmarks and debugging for Graph RAG retrieval.

By default this script scores reviewed benchmark cases from
evals/retrieval/benchmark_cases.jsonl against local production storage. Pass
--debug-default-queries to run the older exploratory query set without scoring.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.evaluation.retrieval_benchmark import (
    ActualEvidence,
    BenchmarkValidationError,
    CaseResult,
    EvidenceCatalog,
    ReferenceValidationReport,
    actual_evidence_from_analyst_result,
    actual_evidence_from_visualizer_result,
    load_benchmark_cases,
    render_markdown_summary,
    score_case,
    summarize_results,
    validate_benchmark_references,
)


DEFAULT_RUN_ROOT = ROOT_DIR / "data" / "production" / "full_optimized_pipeline_run"
DEFAULT_STORAGE_DIR = ROOT_DIR / "storage"
DEFAULT_RAW_DATA_DIR = ROOT_DIR / "data" / "raw" / "AI Research"
DEFAULT_EMBEDDER_DIR = ROOT_DIR / "models" / "all-MiniLM-L6-v2"
DEFAULT_OUTPUT_ROOT = DEFAULT_RUN_ROOT / "retrieval_eval"
DEFAULT_BENCHMARK_FILE = ROOT_DIR / "evals" / "retrieval" / "benchmark_cases.jsonl"

DEFAULT_ANALYST_QUERIES = [
    "Text Classification",
    "text classification datasets",
    "logistic regression for text classification",
    "Naive Bayes vs Logistic Regression",
    "CLIP training objective",
    "Kaggle agents / ReAct",
]

DEFAULT_VISUALIZER_QUERY_SETS = [
    ["Text Classification Methods"],
    ["Text Classification Datasets"],
    ["Naive Bayes", "Logistic Regression"],
    ["CLIP Training Objective"],
    ["CLIP architecture"],
    ["Kaggle agents", "ReAct orchestration"],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate local Graph RAG retrieval outputs against reviewed benchmark cases.",
    )
    parser.add_argument(
        "--benchmark-file",
        type=Path,
        default=DEFAULT_BENCHMARK_FILE,
        help="JSONL benchmark file. Ignored when --debug-default-queries is used.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=None,
        help="Benchmark case id to run. Repeat for multiple cases.",
    )
    parser.add_argument(
        "--debug-default-queries",
        action="store_true",
        help="Run the exploratory default/ad hoc queries instead of scoring benchmark cases.",
    )
    parser.add_argument(
        "--fail-on-threshold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit with code 1 when any benchmark case fails.",
    )
    parser.add_argument(
        "--validate-references-only",
        action="store_true",
        help="Validate benchmark evidence IDs against storage and exit without loading the index.",
    )
    parser.add_argument(
        "--allow-invalid-benchmark-references",
        action="store_true",
        help="Continue retrieval even if benchmark evidence IDs are stale or unavailable.",
    )
    parser.add_argument(
        "--storage-dir",
        type=Path,
        default=DEFAULT_STORAGE_DIR,
        help="Local final_storage directory to load.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for outputs. Defaults to a timestamped directory under retrieval_eval.",
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=DEFAULT_RAW_DATA_DIR,
        help="Raw data directory recorded in PathSettings.",
    )
    parser.add_argument(
        "--parsed-pdfs-dir",
        type=Path,
        default=DEFAULT_RUN_ROOT / "parsed_pdfs",
        help="Parsed PDFs directory recorded in PathSettings.",
    )
    parser.add_argument(
        "--parsed-images-dir",
        type=Path,
        default=DEFAULT_RUN_ROOT / "parsed_images",
        help="Parsed images directory recorded in PathSettings.",
    )
    parser.add_argument(
        "--parsed-texts-dir",
        type=Path,
        default=DEFAULT_RUN_ROOT / "parsed_texts",
        help="Parsed external text directory recorded in PathSettings.",
    )
    parser.add_argument(
        "--embedder-model-path",
        type=Path,
        default=None,
        help="Local HuggingFace embedder path. Defaults to EMBEDDER_MODEL_PATH, then models/all-MiniLM-L6-v2.",
    )
    parser.add_argument(
        "--analyst-reranker-mode",
        choices=["config", "disabled", "sentence_transformers", "ollama_llm_rerank"],
        default="config",
        help="Override Analyst reranker mode. 'config' uses KnowledgeGraphSearchSettings defaults.",
    )
    parser.add_argument(
        "--mode",
        choices=["both", "analyst", "visualizer"],
        default="both",
        help="Which retrieval path to run.",
    )
    parser.add_argument(
        "--analyst-query",
        action="append",
        default=None,
        help="Analyst query to run. Repeat for multiple queries.",
    )
    parser.add_argument(
        "--visualizer-query",
        action="append",
        default=None,
        help=(
            "Visualizer query set to run. Repeat for multiple plots. "
            "Use 'query one||query two' for a multi-query visualization request."
        ),
    )
    parser.add_argument(
        "--include-legacy-analyst",
        action="store_true",
        help="Also run the legacy VectorContextRetriever Analyst formatting path.",
    )
    parser.add_argument(
        "--include-legacy-visualizer",
        action="store_true",
        help="Also run the legacy VectorContextRetriever Visualizer formatting path.",
    )
    parser.add_argument(
        "--no-render-html",
        action="store_false",
        dest="render_html",
        help="Do not write Plotly HTML files for Visualizer outputs.",
    )
    parser.set_defaults(render_html=True)
    parser.add_argument(
        "--render-png",
        action="store_true",
        help="Write PNG images for Visualizer outputs. Requires Plotly image export support, usually kaleido.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open Visualizer figures interactively with Plotly show().",
    )
    return parser.parse_args()


def timestamp_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_output_dir(output_dir: Path | None) -> Path:
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    root = DEFAULT_OUTPUT_ROOT / timestamp_id()
    if not root.exists():
        root.mkdir(parents=True, exist_ok=False)
        return root

    for suffix in range(1, 1000):
        candidate = DEFAULT_OUTPUT_ROOT / f"{root.name}-{suffix:03d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
    raise RuntimeError(f"Could not create a unique output directory under {DEFAULT_OUTPUT_ROOT}")


def resolve_embedder_path(requested_path: Path | None) -> Path:
    if requested_path is not None:
        return requested_path.expanduser().resolve()

    from backend.configs.models import ModelSettings

    configured_path = Path(ModelSettings().embedder.model_path).expanduser()
    if configured_path.exists():
        return configured_path.resolve()

    if DEFAULT_EMBEDDER_DIR.exists():
        return DEFAULT_EMBEDDER_DIR.resolve()

    return configured_path


def make_query_sets(raw_query_sets: list[str] | None) -> list[list[str]]:
    if not raw_query_sets:
        return DEFAULT_VISUALIZER_QUERY_SETS

    query_sets: list[list[str]] = []
    for raw_query_set in raw_query_sets:
        queries = [part.strip() for part in raw_query_set.split("||") if part.strip()]
        if queries:
            query_sets.append(queries)
    return query_sets


def make_path_settings(args: argparse.Namespace) -> PathSettings:
    from backend.configs.paths import PathSettings

    return PathSettings(
        raw_data_dir=args.raw_data_dir,
        parsed_pdfs_dir=args.parsed_pdfs_dir,
        parsed_images_dir=args.parsed_images_dir,
        parsed_texts_dir=args.parsed_texts_dir,
        local_storage_dir=args.storage_dir,
    )


def make_search_settings(args: argparse.Namespace) -> KnowledgeGraphSearchSettings:
    from backend.configs.search import KnowledgeGraphSearchSettings

    if args.analyst_reranker_mode == "config":
        return KnowledgeGraphSearchSettings()
    return KnowledgeGraphSearchSettings(analyst_reranker_mode=args.analyst_reranker_mode)


def load_indexer(args: argparse.Namespace) -> tuple[KnowledgeGraphIndexer, Path]:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    from backend.configs.storage import LocalStorageSettings, StorageSettings
    from backend.knowledge_graph.indexer import KnowledgeGraphIndexer
    from backend.knowledge_graph.storage import KnowledgeGraphStorage

    storage_dir = args.storage_dir.expanduser().resolve()
    if not storage_dir.exists():
        raise FileNotFoundError(f"Storage directory does not exist: {storage_dir}")

    embedder_path = resolve_embedder_path(args.embedder_model_path)
    if not embedder_path.exists():
        raise FileNotFoundError(
            f"Local embedder path does not exist: {embedder_path}. "
            "Pass --embedder-model-path or set EMBEDDER_MODEL_PATH."
        )

    args.storage_dir = storage_dir
    path_settings = make_path_settings(args)
    local_storage = LocalStorageSettings(storage_path=path_settings.local_storage_dir)
    storage_settings = StorageSettings(
        document_storage=local_storage,
        index_storage=local_storage,
        vector_storage=local_storage,
        property_graph_storage=local_storage,
    )
    kg_storage = KnowledgeGraphStorage(path_settings, storage_settings)
    embedder = HuggingFaceEmbedding(
        str(embedder_path),
        trust_remote_code=True,
        embed_batch_size=5,
        local_files_only=True,
    )
    indexer = KnowledgeGraphIndexer(
        kg_storage.storage_context,
        path_settings,
        storage_settings.document_storage.storage_type,
        make_search_settings(args),
        embedder,
        None,
    )
    indexer.load_index()
    return indexer, embedder_path


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def load_storage_reference_data(storage_dir: Path) -> dict[str, Any]:
    storage_dir = storage_dir.expanduser().resolve()
    source_metadata_by_id = load_source_metadata_by_id(storage_dir / "docstore.json")
    source_status_by_id = load_source_status_by_id(storage_dir)
    return {
        "source_metadata_by_id": source_metadata_by_id,
        "source_status_by_id": source_status_by_id,
        "embedded_source_ids": load_embedded_source_ids(storage_dir / "default__vector_store.json"),
        "graph_triplets": load_graph_triplets(storage_dir / "property_graph_store.json"),
    }


def load_source_metadata_by_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    source_metadata_by_id: dict[str, dict[str, Any]] = {}
    for node_id, record in (data.get("docstore/data") or {}).items():
        node_data = record.get("__data__", {}) if isinstance(record, dict) else {}
        metadata = node_data.get("metadata") or {}
        if not isinstance(metadata, dict):
            continue
        if metadata.get("docstore_node_kind") == "postprocessed_embedding_passage":
            continue
        source_id = str(metadata.get("chunk_id") or node_data.get("id_") or node_id)
        if source_id.startswith("chunk_"):
            source_metadata_by_id[source_id] = metadata
    return source_metadata_by_id


def load_source_status_by_id(storage_dir: Path) -> dict[str, dict[str, Any]]:
    postprocess_dir = resolve_postprocess_dir(storage_dir)
    if postprocess_dir is None:
        return {}
    statuses: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(postprocess_dir / "llm_postprocess_decisions.jsonl"):
        chunk_id = row.get("chunk_id")
        if not chunk_id:
            continue
        statuses[str(chunk_id)] = {
            "postprocess_action": row.get("action"),
            "postprocess_issue_types": row.get("issue_types") or [],
            "postprocess_warnings": row.get("warnings") or [],
        }
    return statuses


def resolve_postprocess_dir(storage_dir: Path) -> Path | None:
    manifest_path = storage_dir / "postprocessed_graph_storage_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    postprocess_dir = manifest.get("postprocess_dir")
    if not postprocess_dir:
        return None
    path = Path(str(postprocess_dir))
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def load_embedded_source_ids(path: Path) -> set[str] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    embedded_ids: set[str] = set()
    for source_id in (data.get("embedding_dict") or {}):
        source_id = str(source_id)
        if source_id.startswith("chunk_") and "::passage_" not in source_id:
            embedded_ids.add(source_id)

    metadata_dict = data.get("metadata_dict") or {}
    if isinstance(metadata_dict, dict):
        for metadata in metadata_dict.values():
            if not isinstance(metadata, dict):
                continue
            parent_id = str(metadata.get("parent_chunk_id") or metadata.get("chunk_id") or "")
            if parent_id.startswith("chunk_"):
                embedded_ids.add(parent_id)
    return embedded_ids


def load_graph_triplets(path: Path) -> set[tuple[str, str, str]] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    graph_triplets: set[tuple[str, str, str]] = set()
    for triplet in data.get("triplets") or []:
        if isinstance(triplet, (list, tuple)) and len(triplet) == 3:
            subject_id, predicate, object_id = triplet
            graph_triplets.add((str(subject_id), str(predicate), str(object_id)))
    return graph_triplets


def load_benchmark_inputs(args: argparse.Namespace) -> tuple[list[Any], EvidenceCatalog]:
    case_ids = set(args.case_id) if args.case_id else None
    cases = load_benchmark_cases(args.benchmark_file, mode=args.mode, case_ids=case_ids)
    catalog = EvidenceCatalog.from_storage_dir(args.storage_dir.expanduser().resolve(), root_dir=ROOT_DIR)
    return cases, catalog


def validate_benchmark_inputs(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    cases: list[Any] | None = None,
    catalog: EvidenceCatalog | None = None,
) -> tuple[list[Any], EvidenceCatalog, ReferenceValidationReport]:
    loaded_cases, loaded_catalog = load_benchmark_inputs(args) if cases is None or catalog is None else (cases, catalog)
    storage_reference_data = load_storage_reference_data(args.storage_dir)
    report = validate_benchmark_references(loaded_cases, loaded_catalog, **storage_reference_data)
    write_json(output_dir / "benchmark_reference_report.json", report.model_dump(exclude_none=True))
    return loaded_cases, loaded_catalog, report


def summarize_reference_report(report: ReferenceValidationReport) -> str:
    if report.passed:
        return "Benchmark reference validation passed."
    issues = []
    issue_fields = {
        "missing source chunks": report.missing_source_chunk_ids,
        "source chunks without embeddings": report.source_chunk_ids_not_embedded,
        "disabled source chunks": report.disabled_source_chunk_ids,
        "quarantined source chunks": report.quarantined_source_chunk_ids,
        "missing concepts": report.missing_concept_ids,
        "missing relations": report.missing_relation_ids,
        "relations missing graph triplets": report.relation_ids_missing_graph_triplet,
    }
    for label, values in issue_fields.items():
        if values:
            issues.append(f"{label}: {len(values)}")
    return "Benchmark reference validation failed: " + ", ".join(issues)


def summarize_search_settings(settings: Any) -> dict[str, Any]:
    payload = settings.model_dump(mode="json") if hasattr(settings, "model_dump") else dict(settings)
    selected_fields = [
        "analyst_retrieval_mode",
        "visualizer_retrieval_mode",
        "analyst_source_candidate_k",
        "analyst_source_final_k",
        "analyst_source_min_relative_score",
        "analyst_source_min_raw_margin",
        "analyst_source_min_keep",
        "analyst_source_max_per_path",
        "analyst_source_exact_topic_boost",
        "analyst_source_fill_min_score",
        "analyst_source_fill_min_relative_score",
        "analyst_relation_final_k",
        "analyst_relation_min_relative_score",
        "analyst_relation_min_raw_margin",
        "analyst_relation_seed_extra_k",
        "analyst_relation_seed_min_score",
        "analyst_context_max_chars",
        "analyst_graph_depth",
        "analyst_graph_relation_limit",
        "analyst_reranker_mode",
        "analyst_source_rerank_candidate_k",
        "analyst_source_rerank_max_chars",
        "analyst_relation_reranker_enabled",
        "analyst_relation_rerank_candidate_k",
        "analyst_relation_rerank_max_chars",
        "analyst_relation_require_source_evidence",
        "visualizer_anchor_top_k",
        "visualizer_anchor_min_score",
        "visualizer_source_candidate_k",
        "visualizer_concept_candidate_k",
        "visualizer_max_nodes",
        "visualizer_max_edges",
        "visualizer_min_nodes",
        "visualizer_max_edges_per_node",
        "visualizer_graph_depth",
        "visualizer_allow_synthetic_edges",
        "visualizer_synthetic_edge_limit",
        "visualizer_anchor_source_filter",
        "visualizer_include_isolated_nodes",
        "visualizer_synthetic_edge_label",
        "visualizer_show_chunks",
        "visualizer_denied_relation_labels",
    ]
    return {field: payload.get(field) for field in selected_fields if field in payload}


def summarize_reranker_settings() -> dict[str, Any]:
    from backend.configs.models import RerankerSettings

    settings = RerankerSettings()
    payload = settings.model_dump(mode="json")
    model_path = Path(str(payload.get("model_path", ""))).expanduser()
    payload["model_path_exists"] = model_path.exists()
    return payload


def safe_slug(text: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    slug = re.sub(r"_+", "_", slug).strip("._-")
    return (slug or "query")[:max_len]


def count_marker_lines(output: str, marker: str) -> int:
    return sum(1 for line in output.splitlines() if line.startswith(marker))


def summarize_analyst_output(query: str, output: str, pipeline: str) -> dict[str, Any]:
    return {
        "pipeline": pipeline,
        "query": query,
        "chars": len(output),
        "source_count": count_marker_lines(output, "[SOURCE]"),
        "source_path_count": count_marker_lines(output, "[SOURCE PATH]"),
        "relation_count": count_marker_lines(output, "[RELATION]"),
        "has_no_results_sentinel": output.strip() == "No relevant information found.",
        "contains_mentions_relation": "-> MENTIONS ->" in output,
        "output": output,
    }


def run_analyst(
    indexer: KnowledgeGraphIndexer,
    queries: list[str],
    output_dir: Path,
    *,
    include_legacy: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    blocks: list[str] = []

    from backend.workflows.agents.analyst_retrieval import AnalystRetrievalPipeline
    from backend.workflows.agents.tools import search_knowledge_base

    current_tool = search_knowledge_base(analyst_pipeline=AnalystRetrievalPipeline(indexer))
    analyst_runs: list[tuple[str, Any]] = [("current", current_tool)]

    if include_legacy:
        legacy_retriever = indexer.get_retriever(indexer.kg_search_settings.retriever_params)
        analyst_runs.append(("legacy_vector_context", search_knowledge_base(legacy_retriever, None)))

    splitter = "\n" + "-" * 99 + "\n"
    for pipeline_name, tool in analyst_runs:
        for idx, query in enumerate(queries, start=1):
            output = tool.invoke({"queries": [query]})
            rows.append(summarize_analyst_output(query, output, pipeline_name))
            blocks.append(f"{pipeline_name.upper()} {idx} EXAMPLE RESULTS:{splitter}{output}{splitter}")

    (output_dir / "analyst_results.txt").write_text("\n".join(blocks), encoding="utf-8")
    write_jsonl(output_dir / "analyst_results.jsonl", rows)
    return rows


def graph_metrics(nodes: list[str], triplets: list[tuple[str, str, str]]) -> dict[str, Any]:
    node_set = set(nodes)
    degree: dict[str, int] = {node: 0 for node in nodes}
    adjacency: dict[str, set[str]] = {node: set() for node in nodes}
    dangling_edges = 0
    generic_edges = 0
    relation_counts = Counter()

    for subject, predicate, object_ in triplets:
        relation_counts[predicate] += 1
        if predicate == "RELATED_TO":
            generic_edges += 1
        if subject not in node_set or object_ not in node_set:
            dangling_edges += 1
            continue
        degree[subject] += 1
        degree[object_] += 1
        adjacency[subject].add(object_)
        adjacency[object_].add(subject)

    visited: set[str] = set()
    component_sizes: list[int] = []
    for node in nodes:
        if node in visited:
            continue
        queue: deque[str] = deque([node])
        visited.add(node)
        size = 0
        while queue:
            current = queue.popleft()
            size += 1
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        component_sizes.append(size)

    isolated_nodes = [node for node, count in degree.items() if count == 0]
    return {
        "node_count": len(nodes),
        "edge_count": len(triplets),
        "dangling_edge_count": dangling_edges,
        "generic_edge_count": generic_edges,
        "isolated_node_count": len(isolated_nodes),
        "isolated_nodes": isolated_nodes,
        "component_count": len(component_sizes),
        "component_sizes": sorted(component_sizes, reverse=True),
        "max_degree": max(degree.values(), default=0),
        "degree_by_node": degree,
        "relation_count_by_label": dict(sorted(relation_counts.items())),
    }


def describe_graph_nodes(
    indexer: KnowledgeGraphIndexer,
    nodes: list[str],
    triplets: list[tuple[str, str, str]],
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    degree_by_node = metrics.get("degree_by_node", {})
    try:
        graph = indexer.generate_nx_graph_from(nodes, triplets)
    except Exception as exc:
        return [
            {
                "id": node,
                "label": node,
                "hover": node,
                "degree": degree_by_node.get(node, 0),
                "error": repr(exc),
            }
            for node in nodes
        ]

    details: list[dict[str, Any]] = []
    for node in nodes:
        data = graph.nodes[node] if node in graph.nodes else {}
        details.append(
            {
                "id": node,
                "label": data.get("text", node),
                "hover": data.get("hover", data.get("text", node)),
                "degree": degree_by_node.get(node, 0),
            }
        )
    return details


def render_visualization(
    indexer: KnowledgeGraphIndexer,
    nodes: list[str],
    triplets: list[tuple[str, str, str]],
    queries: list[str],
    graph_dir: Path,
    *,
    pipeline_name: str,
    render_html: bool,
    render_png: bool,
    show: bool,
) -> dict[str, str]:
    from backend.configs.constants import TITLE_MAX_LENGTH

    if not nodes and not triplets:
        return {}

    title = " & ".join(queries)
    title = (title[:TITLE_MAX_LENGTH] + "...") if len(title) > TITLE_MAX_LENGTH else title
    figure = indexer.get_graph_visualization(nodes, triplets, title=title.title())
    output_paths: dict[str, str] = {}
    basename = f"{pipeline_name}_{safe_slug('_'.join(queries))}"

    if render_html:
        html_path = graph_dir / f"{basename}.html"
        figure.write_html(str(html_path))
        output_paths["html"] = str(html_path)

    if render_png:
        png_path = graph_dir / f"{basename}.png"
        try:
            figure.write_image(str(png_path))
            output_paths["png"] = str(png_path)
        except Exception as exc:
            output_paths["png_error"] = repr(exc)

    if show:
        figure.show()

    return output_paths


def run_visualizer(
    indexer: KnowledgeGraphIndexer,
    query_sets: list[list[str]],
    output_dir: Path,
    *,
    include_legacy: bool,
    render_html: bool,
    render_png: bool,
    show: bool,
) -> list[dict[str, Any]]:
    graph_dir = output_dir / "visualizer_graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    from backend.workflows.agents.tools import get_subgraphs_to_visualize
    from backend.workflows.agents.visualizer_retrieval import VisualizerRetrievalPipeline

    current_tool = get_subgraphs_to_visualize(visualizer_pipeline=VisualizerRetrievalPipeline(indexer))
    visualizer_runs: list[tuple[str, Any]] = [("current", current_tool)]

    if include_legacy:
        legacy_retriever = indexer.get_retriever(indexer.kg_search_settings.visualizer_retriever_params)
        visualizer_runs.append(("legacy_vector_context", get_subgraphs_to_visualize(legacy_retriever)))

    for pipeline_name, tool in visualizer_runs:
        for queries in query_sets:
            nodes, triplets, returned_queries = tool.invoke({"queries": queries})
            metrics = graph_metrics(nodes, triplets)
            node_details = describe_graph_nodes(indexer, nodes, triplets, metrics)
            render_paths = render_visualization(
                indexer,
                nodes,
                triplets,
                returned_queries,
                graph_dir,
                pipeline_name=pipeline_name,
                render_html=render_html,
                render_png=render_png,
                show=show,
            )
            rows.append(
                {
                    "pipeline": pipeline_name,
                    "queries": queries,
                    "returned_queries": returned_queries,
                    "nodes": nodes,
                    "node_details": node_details,
                    "triplets": triplets,
                    "metrics": metrics,
                    "render_paths": render_paths,
                }
            )

    write_json(output_dir / "visualizer_results.json", rows)
    return rows


def merge_actual_evidence(items: list[ActualEvidence]) -> ActualEvidence:
    source_ids: list[str] = []
    source_paths: list[str] = []
    concept_ids: list[str] = []
    concept_labels: list[str] = []
    relation_ids: list[str] = []
    relations = []
    graph_metrics: dict[str, Any] = {}

    for item in items:
        source_ids.extend(item.source_chunk_ids_ranked)
        source_paths.extend(item.source_paths)
        concept_ids.extend(item.concept_ids)
        concept_labels.extend(item.concept_labels)
        relation_ids.extend(item.relation_ids)
        relations.extend(item.relations)
        graph_metrics.update(item.graph_metrics)

    return ActualEvidence(
        source_chunk_ids_ranked=list(dict.fromkeys(source_ids)),
        source_paths=list(dict.fromkeys(source_paths)),
        concept_ids=list(dict.fromkeys(concept_ids)),
        concept_labels=list(dict.fromkeys(concept_labels)),
        relation_ids=list(dict.fromkeys(relation_ids)),
        relations=relations,
        graph_metrics=graph_metrics,
    )


def run_benchmark(
    indexer: KnowledgeGraphIndexer,
    args: argparse.Namespace,
    output_dir: Path,
    *,
    embedder_path: Path,
    cases: list[Any] | None = None,
    catalog: EvidenceCatalog | None = None,
    reference_report: ReferenceValidationReport | None = None,
) -> list[CaseResult]:
    from backend.workflows.agents.analyst_retrieval import AnalystRetrievalPipeline
    from backend.workflows.agents.visualizer_retrieval import VisualizerRetrievalPipeline

    if cases is None or catalog is None:
        cases, catalog = load_benchmark_inputs(args)
    analyst_pipeline = AnalystRetrievalPipeline(indexer)
    visualizer_pipeline = VisualizerRetrievalPipeline(indexer)

    results: list[CaseResult] = []
    actual_rows: list[dict[str, Any]] = []

    for case in cases:
        if case.mode == "analyst":
            query_actuals = [
                actual_evidence_from_analyst_result(
                    analyst_pipeline._search_one(query),  # noqa: SLF001 - benchmark needs structured candidates.
                    catalog,
                )
                for query in case.queries
            ]
            actual = merge_actual_evidence(query_actuals)
        else:
            nodes, triplets, returned_queries = visualizer_pipeline.visualize(case.queries)
            metrics = graph_metrics(nodes, triplets)
            node_details = describe_graph_nodes(indexer, nodes, triplets, metrics)
            actual = actual_evidence_from_visualizer_result(
                nodes=nodes,
                triplets=triplets,
                node_details=node_details,
                metrics=metrics,
                catalog=catalog,
            )
            actual.graph_metrics["returned_queries"] = returned_queries

        result = score_case(case, actual, catalog)
        results.append(result)
        actual_rows.append(
            {
                "case_id": case.id,
                "mode": case.mode,
                "queries": case.queries,
                "actual_evidence": actual.compact(),
            }
        )

    summary = summarize_results(results)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "storage_dir": str(args.storage_dir),
        "output_dir": str(output_dir),
        "embedder_model_path": str(embedder_path),
        "benchmark_file": str(args.benchmark_file),
        "mode": args.mode,
        "case_ids": args.case_id or [],
        "requested_analyst_reranker_mode": args.analyst_reranker_mode,
        "analyst_reranker_mode": indexer.kg_search_settings.analyst_reranker_mode,
        "effective_search_settings": summarize_search_settings(indexer.kg_search_settings),
        "reranker_settings": summarize_reranker_settings(),
        "benchmark_reference_report": reference_report.model_dump(exclude_none=True) if reference_report else None,
        "summary": summary,
    }

    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "case_results.jsonl", [result.model_dump(exclude_none=True) for result in results])
    write_jsonl(output_dir / "actual_evidence.jsonl", actual_rows)
    (output_dir / "summary.md").write_text(render_markdown_summary(summary, results), encoding="utf-8")
    return results


def build_manifest(
    args: argparse.Namespace,
    output_dir: Path,
    embedder_path: Path,
    analyst_queries: list[str],
    visualizer_query_sets: list[list[str]],
    analyst_rows: list[dict[str, Any]],
    visualizer_rows: list[dict[str, Any]],
    *,
    effective_search_settings: dict[str, Any],
    reranker_settings: dict[str, Any],
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "storage_dir": str(args.storage_dir),
        "output_dir": str(output_dir),
        "embedder_model_path": str(embedder_path),
        "mode": args.mode,
        "requested_analyst_reranker_mode": args.analyst_reranker_mode,
        "analyst_reranker_mode": effective_search_settings.get("analyst_reranker_mode"),
        "effective_search_settings": effective_search_settings,
        "reranker_settings": reranker_settings,
        "include_legacy_analyst": args.include_legacy_analyst,
        "include_legacy_visualizer": args.include_legacy_visualizer,
        "render_html": args.render_html,
        "render_png": args.render_png,
        "analyst_queries": analyst_queries,
        "visualizer_query_sets": visualizer_query_sets,
        "analyst_summary": [
            {
                "pipeline": row["pipeline"],
                "query": row["query"],
                "chars": row["chars"],
                "source_count": row["source_count"],
                "relation_count": row["relation_count"],
                "has_no_results_sentinel": row["has_no_results_sentinel"],
                "contains_mentions_relation": row["contains_mentions_relation"],
            }
            for row in analyst_rows
        ],
        "visualizer_summary": [
            {
                "pipeline": row["pipeline"],
                "queries": row["queries"],
                "node_count": row["metrics"]["node_count"],
                "edge_count": row["metrics"]["edge_count"],
                "component_count": row["metrics"]["component_count"],
                "isolated_node_count": row["metrics"]["isolated_node_count"],
                "generic_edge_count": row["metrics"]["generic_edge_count"],
                "dangling_edge_count": row["metrics"]["dangling_edge_count"],
                "render_paths": row["render_paths"],
            }
            for row in visualizer_rows
        ],
    }


def main() -> int:
    args = parse_args()
    try:
        output_dir = resolve_output_dir(args.output_dir)
        analyst_queries = args.analyst_query or DEFAULT_ANALYST_QUERIES
        visualizer_query_sets = make_query_sets(args.visualizer_query)

        print(f"Writing outputs to: {output_dir}")
        benchmark_cases: list[Any] | None = None
        benchmark_catalog: EvidenceCatalog | None = None
        reference_report: ReferenceValidationReport | None = None
        if not args.debug_default_queries:
            benchmark_cases, benchmark_catalog, reference_report = validate_benchmark_inputs(args, output_dir)
            print(summarize_reference_report(reference_report))
            print(f"- Reference report: {output_dir / 'benchmark_reference_report.json'}")
            if args.validate_references_only:
                return 0 if reference_report.passed else 2
            if not reference_report.passed and not args.allow_invalid_benchmark_references:
                print(
                    "Benchmark references are invalid. "
                    "Fix the benchmark file or pass --allow-invalid-benchmark-references to continue.",
                    file=sys.stderr,
                )
                return 2

        print(f"Loading local storage from: {args.storage_dir}")
        indexer, embedder_path = load_indexer(args)
        print(f"Using embedder: {embedder_path}")

        if not args.debug_default_queries:
            results = run_benchmark(
                indexer,
                args,
                output_dir,
                embedder_path=embedder_path,
                cases=benchmark_cases,
                catalog=benchmark_catalog,
                reference_report=reference_report,
            )
            summary = summarize_results(results)
            print("Retrieval benchmark complete.")
            print(f"- Summary: {output_dir / 'summary.md'}")
            print(f"- Case results: {output_dir / 'case_results.jsonl'}")
            print(f"- Actual evidence: {output_dir / 'actual_evidence.jsonl'}")
            print(f"- Passed: {summary['passed_count']}/{summary['case_count']}")
            if summary["failed_count"]:
                print(f"- Failed cases: {', '.join(summary['failed_case_ids'])}")
                return 1 if args.fail_on_threshold else 0
            return 0

        analyst_rows: list[dict[str, Any]] = []
        visualizer_rows: list[dict[str, Any]] = []

        if args.mode in {"both", "analyst"}:
            analyst_rows = run_analyst(
                indexer,
                analyst_queries,
                output_dir,
                include_legacy=args.include_legacy_analyst,
            )

        if args.mode in {"both", "visualizer"}:
            visualizer_rows = run_visualizer(
                indexer,
                visualizer_query_sets,
                output_dir,
                include_legacy=args.include_legacy_visualizer,
                render_html=args.render_html,
                render_png=args.render_png,
                show=args.show,
            )

        manifest = build_manifest(
            args,
            output_dir,
            embedder_path,
            analyst_queries,
            visualizer_query_sets,
            analyst_rows,
            visualizer_rows,
            effective_search_settings=summarize_search_settings(indexer.kg_search_settings),
            reranker_settings=summarize_reranker_settings(),
        )
        write_json(output_dir / "manifest.json", manifest)

        print("Retrieval debug run complete.")
        print(f"- Manifest: {output_dir / 'manifest.json'}")
        if analyst_rows:
            print(f"- Analyst text: {output_dir / 'analyst_results.txt'}")
            print(f"- Analyst JSONL: {output_dir / 'analyst_results.jsonl'}")
        if visualizer_rows:
            print(f"- Visualizer JSON: {output_dir / 'visualizer_results.json'}")
            if args.render_html or args.render_png:
                print(f"- Visualizer graphs: {output_dir / 'visualizer_graphs'}")
        return 0
    except (BenchmarkValidationError, FileNotFoundError) as exc:
        print(f"Retrieval evaluation setup failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
