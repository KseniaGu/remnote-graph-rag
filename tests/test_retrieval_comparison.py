import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from synthetic_analyst import (
    FakeTextNode,
    FakeVectorStore,
    RaisingReranker,
    RecordingReranker,
    make_indexer,
    make_metadata,
)
from synthetic_retrieval_case import make_case_payload

from backend.configs.search import KnowledgeGraphSearchSettings
from backend.evaluation.evaluation_reporting import _build_scorecard, _retrieval_metrics
from backend.evaluation.retrieval_benchmark import (
    ActualEvidence,
    BenchmarkCase,
    EvidenceCatalog,
    score_case,
)
from scripts.evaluate_retrieval_pipeline import load_indexer, parse_args, run_benchmark


def test_source_only_cli_requires_disabled_reranker(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval",
            "--mode",
            "analyst",
            "--analyst-variant",
            "source_vector_only",
            "--analyst-reranker-mode",
            "disabled",
        ],
    )
    assert parse_args().analyst_variant == "source_vector_only"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval",
            "--analyst-variant",
            "source_vector_only",
            "--analyst-reranker-mode",
            "sentence_transformers",
        ],
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_source_only_scores_graph_fields_as_unavailable():
    case = BenchmarkCase.from_raw(make_case_payload())
    result = score_case(
        case,
        ActualEvidence(source_chunk_ids_ranked=["chunk_1"]),
        variant="source_vector_only",
    )
    assert result.scores["concept_recall"] is None
    assert result.scores["relation_recall"] is None
    assert result.scores["dangling_edge_count"] is None
    assert result.scores["chunk_node_count"] is None
    assert result.scores["context_precision_at_10"] is None
    assert result.scores["forbidden_evidence_count"] is None
    assert result.scores["source_forbidden_evidence_count"] == 0
    assert result.passed is None
    assert result.model_dump(exclude_none=True)["passed"] is None
    assert result.model_dump(exclude_none=True)["scores"]["concept_recall"] is None


def test_scorecard_does_not_count_incomplete_as_failure_or_graph_zero():
    rows = [
        {
            "passed": None,
            "scores": {
                "forbidden_evidence_count": None,
                "dangling_edge_count": None,
                "chunk_node_count": None,
            },
        }
    ]
    metrics = _retrieval_metrics(rows)
    assert metrics["case_pass_rate"] is None
    assert metrics["incomplete_case_count"] == 1
    assert metrics["forbidden_evidence_count"] is None
    assert metrics["forbidden_evidence_rate"] is None
    assert metrics["dangling_edge_count"] is None
    assert metrics["chunk_node_count"] is None


def test_mixed_comparison_arms_are_not_pooled(tmp_path):
    manifests = []
    for name, variant, reranker in (
        ("a", "source_vector_only", "disabled"),
        ("b", "optimized", "disabled"),
    ):
        run_dir = tmp_path / name
        run_dir.mkdir()
        (run_dir / "case_results.jsonl").write_text(
            '{"case_id":"case", "passed":true, "scores":{"retrieval_adequate":1}}\n'
        )
        manifests.append(
            {
                "_run_dir": str(run_dir),
                "run_id": name,
                "run_kind": "retrieval",
                "status": "completed",
                "invocation": {
                    "analyst_variant": variant,
                    "analyst_reranker_mode": reranker,
                },
            }
        )
    scorecard = _build_scorecard("fixture", manifests, manifests)
    assert scorecard["retrieval_quality"]["status"] == "N/A"
    assert "mixed" in scorecard["retrieval_quality"]["reason"].lower()


def test_source_only_loader_never_initializes_graph_index(tmp_path):
    for name in ("docstore.json", "default__vector_store.json"):
        (tmp_path / name).write_text("{}")
    args = SimpleNamespace(
        storage_dir=tmp_path,
        embedder_model_path=tmp_path,
        analyst_variant="source_vector_only",
        analyst_reranker_mode="disabled",
    )
    with (
        patch(
            "scripts.evaluate_retrieval_pipeline.make_path_settings",
            return_value=SimpleNamespace(local_storage_dir=tmp_path),
        ),
        patch(
            "llama_index.embeddings.huggingface.HuggingFaceEmbedding",
            return_value=object(),
        ),
        patch(
            "llama_index.core.storage.docstore.SimpleDocumentStore.from_persist_dir",
            return_value=object(),
        ),
        patch(
            "llama_index.core.vector_stores.SimpleVectorStore.from_persist_dir",
            return_value=object(),
        ),
        patch(
            "backend.knowledge_graph.storage.KnowledgeGraphStorage",
            side_effect=AssertionError("graph storage initialized"),
        ),
        patch(
            "backend.knowledge_graph.indexer.KnowledgeGraphIndexer.load_index",
            side_effect=AssertionError("graph index initialized"),
        ),
    ):
        indexer, _ = load_indexer(args)
    assert indexer.index is None
    assert not hasattr(indexer.storage_context, "property_graph_store")


def test_source_only_case_artifacts_are_scored_and_timed_without_graph(
    tmp_path, monkeypatch
):
    class GraphForbidden:
        def __init__(self, vector):
            self.vector_store = vector

        @property
        def property_graph_store(self):
            raise AssertionError("graph access forbidden")

    node = FakeTextNode("chunk_good", "Adam uses adaptive moments.", make_metadata())
    indexer = make_indexer(
        FakeVectorStore(["chunk_good"]),
        {"chunk_good": node},
        settings=KnowledgeGraphSearchSettings(analyst_reranker_mode="disabled"),
    )
    indexer.index = GraphForbidden(indexer.storage_context.vector_store)
    case = BenchmarkCase.from_raw(
        make_case_payload(
            id="source_case",
            queries=["Adam"],
            expected_evidence={"source_chunk_ids": {"required_all": ["chunk_good"]}},
            forbidden_evidence={},
            thresholds={"evidence_chunk_recall": 1.0},
        )
    )
    args = SimpleNamespace(
        analyst_variant="source_vector_only",
        visualizer_variant="config",
        analyst_reranker_mode="disabled",
        storage_dir=tmp_path,
        benchmark_file=tmp_path / "cases",
        mode="analyst",
        case_id=None,
    )
    times = iter([10.0, 10.25])
    monkeypatch.setattr(
        "scripts.evaluate_retrieval_pipeline.time.perf_counter", lambda: next(times)
    )
    results = run_benchmark(
        indexer,
        args,
        tmp_path,
        embedder_path=tmp_path,
        cases=[case],
        catalog=EvidenceCatalog(),
    )
    assert results[0].passed is None
    row = json.loads((tmp_path / "case_results.jsonl").read_text())
    assert row["scores"]["dangling_edge_count"] is None
    assert row["observability"]["retrieval_seconds"] == 0.25
    assert row["observability"]["embedding_operations"] == 1
    assert row["observability"]["reranking_operations"] == 0
    assert row["observability"]["effective_reranker_mode"] == "disabled"


def test_source_only_error_remains_error_with_partial_observability(
    tmp_path, monkeypatch
):
    class BrokenVector:
        def query(self, _query):
            raise RuntimeError("fixture failure")

    indexer = make_indexer(
        BrokenVector(),
        {},
        settings=KnowledgeGraphSearchSettings(analyst_reranker_mode="disabled"),
    )
    case = BenchmarkCase.from_raw(
        make_case_payload(
            id="error_case",
            queries=["Adam"],
            expected_evidence={},
            forbidden_evidence={},
            thresholds={},
        )
    )
    args = SimpleNamespace(
        analyst_variant="source_vector_only",
        visualizer_variant="config",
        analyst_reranker_mode="disabled",
        storage_dir=tmp_path,
        benchmark_file=tmp_path / "cases",
        mode="analyst",
        case_id=None,
    )
    times = iter([1.0, 1.5])
    monkeypatch.setattr(
        "scripts.evaluate_retrieval_pipeline.time.perf_counter", lambda: next(times)
    )
    results = run_benchmark(
        indexer,
        args,
        tmp_path,
        embedder_path=tmp_path,
        cases=[case],
        catalog=EvidenceCatalog(),
    )
    assert results[0].passed is False
    assert results[0].actual_evidence["retrieval_error"] == "RuntimeError"
    row = json.loads((tmp_path / "case_results.jsonl").read_text())
    assert row["observability"]["retrieval_seconds"] == 0.5
    assert row["observability"]["embedding_operations"] == 1
    assert row["observability"]["candidate_count"] is None
    assert row["observability"]["comparison_measurement_valid"] is False
    assert row["observability"]["comparison_measurement_reason"] == "retrieval failed"


@pytest.mark.parametrize(
    ("mode", "reranker", "expected", "operations", "failures"),
    [
        ("disabled", None, "disabled", 0, 0),
        ("sentence_transformers", RecordingReranker(), "sentence_transformers", 1, 0),
        ("sentence_transformers", RaisingReranker(), "fallback", 1, 1),
    ],
)
def test_optimized_case_exposes_actual_reranker_disposition(
    tmp_path, mode, reranker, expected, operations, failures
):
    node = FakeTextNode("chunk_good", "Adam uses adaptive moments.", make_metadata())
    indexer = make_indexer(
        FakeVectorStore(["chunk_good"]),
        {"chunk_good": node},
        settings=KnowledgeGraphSearchSettings(analyst_reranker_mode=mode),
    )
    case = BenchmarkCase.from_raw(
        make_case_payload(
            id="case",
            queries=["Adam"],
            expected_evidence={"source_chunk_ids": {"required_all": ["chunk_good"]}},
            forbidden_evidence={},
            thresholds={},
        )
    )
    args = SimpleNamespace(
        analyst_variant="optimized",
        visualizer_variant="config",
        analyst_reranker_mode=mode,
        storage_dir=tmp_path,
        benchmark_file=tmp_path / "cases",
        mode="analyst",
        case_id=None,
    )
    with patch(
        "backend.workflows.agents.analyst_retrieval.AnalystRetrievalPipeline._create_configured_reranker",
        return_value=reranker,
    ):
        result = run_benchmark(
            indexer,
            args,
            tmp_path,
            embedder_path=tmp_path,
            cases=[case],
            catalog=EvidenceCatalog(),
        )[0]
    assert result.observability["requested_reranker_mode"] == mode
    assert result.observability["effective_reranker_mode"] == expected
    assert result.observability["reranking_operations"] == operations
    assert result.observability["reranking_failures"] == failures
    assert result.observability["comparison_measurement_valid"] is (
        expected != "fallback"
    )
    assert result.observability["comparison_measurement_reason"] == (
        "reranker fallback" if expected == "fallback" else None
    )


def test_existing_loader_initializes_storage_before_embedder_and_index(tmp_path):
    order = []
    args = SimpleNamespace(
        storage_dir=tmp_path,
        embedder_model_path=tmp_path,
        analyst_variant="optimized",
        analyst_reranker_mode="disabled",
    )
    storage = SimpleNamespace(storage_context=object())
    indexer = SimpleNamespace(load_index=lambda: order.append("load"))
    with (
        patch(
            "scripts.evaluate_retrieval_pipeline.make_path_settings",
            return_value=SimpleNamespace(local_storage_dir=tmp_path),
        ),
        patch(
            "backend.knowledge_graph.storage.KnowledgeGraphStorage",
            side_effect=lambda *a: (order.append("storage"), storage)[1],
        ),
        patch(
            "llama_index.embeddings.huggingface.HuggingFaceEmbedding",
            side_effect=lambda *a, **k: (order.append("embedder"), object())[1],
        ),
        patch(
            "backend.knowledge_graph.indexer.KnowledgeGraphIndexer",
            side_effect=lambda *a: (order.append("index"), indexer)[1],
        ),
    ):
        load_indexer(args)
    assert order == ["storage", "embedder", "index", "load"]


def test_main_source_only_relevance_review_does_not_access_graph(tmp_path, monkeypatch):
    from scripts import evaluate_retrieval_pipeline as cli

    class NoGraph:
        @property
        def property_graph_store(self):
            raise AssertionError("graph access forbidden")

    indexer = SimpleNamespace(
        index=NoGraph(), storage_context=NoGraph(), kg_search_settings=SimpleNamespace()
    )
    review = tmp_path / "review.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval",
            "--mode",
            "analyst",
            "--analyst-variant",
            "source_vector_only",
            "--analyst-reranker-mode",
            "disabled",
            "--output-dir",
            str(tmp_path / "out"),
            "--write-relevance-review",
            str(review),
        ],
    )
    monkeypatch.setattr(
        cli,
        "validate_benchmark_inputs",
        lambda *a: ([], EvidenceCatalog(), SimpleNamespace(passed=True)),
    )
    monkeypatch.setattr(cli, "summarize_reference_report", lambda report: "ok")
    monkeypatch.setattr(cli, "load_indexer", lambda args: (indexer, tmp_path))
    monkeypatch.setattr(cli, "run_benchmark", lambda *a, **k: [])
    monkeypatch.setattr(cli, "storage_snapshot_identity", lambda path: "fixture")
    monkeypatch.setattr(
        cli, "render_context_relevance_review", lambda *a, **k: "review"
    )
    monkeypatch.setattr(cli, "register_completed_run", lambda *a, **k: None)
    assert cli.main() == 0
    assert review.read_text() == "review"


def test_reranker_fallback_cannot_be_pooled_with_success(tmp_path):
    manifests = []
    for name, effective in (
        ("success", "sentence_transformers"),
        ("fallback", "fallback"),
    ):
        run_dir = tmp_path / name
        run_dir.mkdir()
        (run_dir / "case_results.jsonl").write_text(
            json.dumps(
                {
                    "passed": True,
                    "scores": {},
                    "observability": {"effective_reranker_mode": effective},
                }
            )
            + "\n"
        )
        manifests.append(
            {
                "_run_dir": str(run_dir),
                "run_id": name,
                "run_kind": "retrieval",
                "status": "completed",
                "invocation": {
                    "analyst_variant": "optimized",
                    "analyst_reranker_mode": "sentence_transformers",
                },
            }
        )
    quality = _build_scorecard("fixture", manifests, manifests)["retrieval_quality"]
    assert quality["status"] == "N/A"
    assert quality["fallback_case_count"] == 1


def test_requested_reranker_not_invoked_is_invalid_comparison(tmp_path):
    indexer = make_indexer(
        FakeVectorStore([]),
        {},
        settings=KnowledgeGraphSearchSettings(
            analyst_reranker_mode="sentence_transformers"
        ),
    )
    case = BenchmarkCase.from_raw(
        make_case_payload(
            id="empty",
            queries=["Adam"],
            expected_evidence={},
            forbidden_evidence={},
            thresholds={},
        )
    )
    args = SimpleNamespace(
        analyst_variant="optimized",
        visualizer_variant="config",
        analyst_reranker_mode="sentence_transformers",
        storage_dir=tmp_path,
        benchmark_file=tmp_path / "cases",
        mode="analyst",
        case_id=None,
    )
    with patch(
        "backend.workflows.agents.analyst_retrieval.AnalystRetrievalPipeline._create_configured_reranker",
        return_value=RecordingReranker(),
    ):
        result = run_benchmark(
            indexer,
            args,
            tmp_path,
            embedder_path=tmp_path,
            cases=[case],
            catalog=EvidenceCatalog(),
        )[0]
    assert result.observability["effective_reranker_mode"] == "not_invoked"
    assert result.observability["comparison_measurement_valid"] is False
    assert (
        result.observability["comparison_measurement_reason"] == "reranker not_invoked"
    )


def test_partial_reranking_excludes_scorecard_aggregation(tmp_path):
    (tmp_path / "case_results.jsonl").write_text(
        json.dumps(
            {
                "passed": True,
                "scores": {},
                "observability": {
                    "effective_reranker_mode": "partial",
                    "comparison_measurement_valid": False,
                },
            }
        )
        + "\n"
    )
    manifest = {
        "_run_dir": str(tmp_path),
        "run_id": "partial",
        "run_kind": "retrieval",
        "status": "completed",
        "invocation": {
            "analyst_variant": "optimized",
            "analyst_reranker_mode": "sentence_transformers",
        },
    }
    quality = _build_scorecard("fixture", [manifest], [manifest])["retrieval_quality"]
    assert quality["status"] == "N/A"
    assert quality["invalid_comparison_case_count"] == 1
