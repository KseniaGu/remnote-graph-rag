import json
import tempfile
import unittest
from pathlib import Path

from backend.evaluation.retrieval_benchmark import (
    ActualEvidence,
    BenchmarkValidationError,
    EvidenceCatalog,
    RelationCatalogEntry,
    RelationEvidence,
    actual_evidence_from_legacy_results,
    load_benchmark_cases,
    score_case,
    summarize_results,
    validate_benchmark_references,
)
from backend.workflows.agents.retrieval_evidence import (
    NormalizedMetadata,
    QueryEvidenceResult,
)
from backend.workflows.agents.retrieval_evidence import (
    RelationEvidence as RuntimeRelationEvidence,
)


def make_case_payload(**overrides):
    payload = {
        "schema_version": "retrieval-benchmark-v0",
        "id": "case_1",
        "review_status": "candidate_needs_review",
        "mode": "analyst",
        "queries": ["query"],
        "seeded_from": [],
        "expected_evidence": {
            "source_chunk_ids": {
                "required_all": ["chunk_1"],
                "required_any": ["chunk_2"],
            },
            "concept_labels": {"required_any": ["CLIP"]},
            "required_answer_points": ["not scored"],
        },
        "forbidden_evidence": {
            "source_chunk_ids": [],
            "concept_ids": [],
            "concept_labels": [],
            "relations": [{"predicate": "MENTIONS"}],
        },
        "thresholds": {
            "evidence_chunk_recall": 0.5,
            "concept_recall": 1.0,
            "forbidden_evidence_count": 0,
        },
        "review_notes": None,
    }
    payload.update(overrides)
    return payload


class RetrievalBenchmarkTests(unittest.TestCase):
    def test_load_benchmark_cases_filters_mode_and_case_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            rows = [
                make_case_payload(id="analyst_case", mode="analyst"),
                make_case_payload(id="visualizer_case", mode="visualizer"),
            ]
            path.write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
            )

            cases = load_benchmark_cases(
                path, mode="analyst", case_ids={"analyst_case"}
            )

        self.assertEqual(["analyst_case"], [case.id for case in cases])

    def test_load_benchmark_cases_rejects_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            row = make_case_payload(id="duplicate")
            path.write_text(json.dumps(row) + "\n" + json.dumps(row), encoding="utf-8")

            with self.assertRaises(BenchmarkValidationError):
                load_benchmark_cases(path)

    def test_score_case_uses_shared_recall_and_forbidden_relations(
        self,
    ) -> None:
        case = load_case(
            make_case_payload(
                expected_evidence={
                    "source_chunk_ids": {"required_all": ["chunk_2"]},
                    "concept_labels": {"required_all": ["CLIP"]},
                    "required_answer_points": ["ignored"],
                },
                thresholds={
                    "evidence_chunk_recall": 1.0,
                    "concept_recall": 1.0,
                    "forbidden_evidence_count": 0,
                },
            )
        )
        actual = ActualEvidence(
            source_chunk_ids_ranked=["chunk_1", "chunk_2"],
            concept_labels=["CLIP"],
            relations=[
                RelationEvidence(
                    predicate="MENTIONS", subject_label="chunk", object_label="CLIP"
                )
            ],
        )

        result = score_case(case, actual)

        self.assertFalse(result.passed)
        self.assertEqual(1.0, result.scores["evidence_chunk_recall"])
        self.assertNotIn("source_mrr_at_6", result.scores)
        self.assertEqual(1, result.scores["forbidden_evidence_count"])
        self.assertIn("relations", result.forbidden_hits)

    def test_relation_id_matches_runtime_relation_by_catalog_triplet_and_labels(
        self,
    ) -> None:
        case = load_case(
            make_case_payload(
                expected_evidence={
                    "relation_ids": {"required_all": ["rel_compare"]},
                    "relations": {
                        "required_all": [
                            {
                                "subject_id": "concept_nb",
                                "predicate": "COMPARES_TO",
                                "object_id": "concept_lr",
                                "evidence_chunk_ids": ["chunk_compare"],
                            }
                        ]
                    },
                    "required_answer_points": [],
                },
                thresholds={"relation_recall": 1.0, "forbidden_evidence_count": 0},
            )
        )
        catalog = EvidenceCatalog(
            concept_labels_by_id={
                "concept_nb": {"Naive Bayes"},
                "concept_lr": {"Logistic regression"},
            },
            relation_by_id={
                "rel_compare": RelationCatalogEntry(
                    relation_id="rel_compare",
                    subject_id="concept_nb",
                    predicate="COMPARES_TO",
                    object_id="concept_lr",
                    evidence_chunk_ids=["chunk_compare"],
                )
            },
        )
        actual = ActualEvidence(
            relations=[
                RelationEvidence(
                    subject_label="Naive Bayes",
                    predicate="COMPARES_TO",
                    object_label="Logistic regression",
                    evidence_chunk_ids=["chunk_compare"],
                )
            ]
        )

        result = score_case(case, actual, catalog)

        self.assertTrue(result.passed)
        self.assertEqual(1.0, result.scores["relation_recall"])

    def test_legacy_adapter_preserves_stable_graph_and_edge_evidence_ids(
        self,
    ) -> None:
        runtime_relation = RuntimeRelationEvidence(
            rank=1,
            subject="CLIP",
            predicate="USES",
            object="Contrastive Objective",
            subject_id="concept_clip",
            relation_id="rel_uses",
            object_id="concept_objective",
            raw_relation="CLIP -> USES -> Contrastive Objective",
            metadata=NormalizedMetadata(),
            evidence_chunk_ids=["chunk_grounding"],
        )

        actual = actual_evidence_from_legacy_results(
            [QueryEvidenceResult(query="CLIP", items=[runtime_relation])]
        )

        self.assertEqual(["chunk_grounding"], actual.source_chunk_ids_ranked)
        self.assertEqual(["concept_clip", "concept_objective"], actual.concept_ids)
        self.assertEqual(["rel_uses"], actual.relation_ids)
        self.assertEqual("concept_clip", actual.relations[0].subject_id)

    def test_visualizer_shape_thresholds_count_chunk_nodes_and_dangling_edges(
        self,
    ) -> None:
        case = load_case(
            make_case_payload(
                mode="visualizer",
                expected_evidence={"required_answer_points": []},
                thresholds={
                    "max_chunk_nodes": 0,
                    "max_dangling_edges": 0,
                    "forbidden_evidence_count": 0,
                },
            )
        )
        actual = ActualEvidence(
            concept_ids=["concept_clip", "chunk_leaked"],
            graph_metrics={"dangling_edge_count": 1},
        )

        result = score_case(case, actual)

        self.assertFalse(result.passed)
        self.assertEqual(1, result.scores["chunk_node_count"])
        self.assertEqual(1, result.scores["dangling_edge_count"])

    def test_summary_counts_failed_cases(self) -> None:
        passing_case = load_case(
            make_case_payload(
                expected_evidence={"required_answer_points": []}, thresholds={}
            )
        )
        failing_case = load_case(
            make_case_payload(
                id="failing",
                expected_evidence={
                    "source_chunk_ids": {"required_all": ["missing"]},
                    "required_answer_points": [],
                },
                thresholds={},
            )
        )

        summary = summarize_results(
            [
                score_case(passing_case, ActualEvidence()),
                score_case(failing_case, ActualEvidence()),
            ]
        )

        self.assertEqual(2, summary["case_count"])
        self.assertEqual(1, summary["passed_count"])
        self.assertEqual(["failing"], summary["failed_case_ids"])

    def test_reference_validation_reports_stale_and_unavailable_evidence_ids(
        self,
    ) -> None:
        case = load_case(
            make_case_payload(
                expected_evidence={
                    "source_chunk_ids": {
                        "required_all": ["chunk_1"],
                        "required_any": ["chunk_2"],
                    },
                    "concept_ids": {
                        "required_all": ["concept_ok"],
                        "required_any": ["concept_missing"],
                    },
                    "relation_ids": {
                        "required_all": ["rel_ok"],
                        "required_any": ["rel_missing"],
                    },
                    "relations": {
                        "required_any": [
                            {
                                "subject_id": "concept_ok",
                                "predicate": "REL",
                                "object_id": "concept_missing",
                                "evidence_chunk_ids": ["chunk_3"],
                            }
                        ]
                    },
                    "required_answer_points": [],
                },
                thresholds={},
            )
        )
        catalog = EvidenceCatalog(
            concept_labels_by_id={"concept_ok": {"Concept OK"}},
            relation_by_id={
                "rel_ok": RelationCatalogEntry(
                    relation_id="rel_ok",
                    subject_id="concept_ok",
                    predicate="REL",
                    object_id="concept_missing",
                    evidence_chunk_ids=["chunk_3"],
                )
            },
        )

        report = validate_benchmark_references(
            [case],
            catalog,
            source_metadata_by_id={
                "chunk_1": {},
                "chunk_2": {"retrieval_enabled": False},
                "chunk_3": {"postprocess_action": "discard"},
            },
            embedded_source_ids={"chunk_1"},
            graph_triplets={("concept_ok", "REL", "concept_other")},
        )

        self.assertFalse(report.passed)
        self.assertEqual(["chunk_2", "chunk_3"], report.source_chunk_ids_not_embedded)
        self.assertEqual(["chunk_2", "chunk_3"], report.disabled_source_chunk_ids)
        self.assertEqual(["chunk_3"], report.quarantined_source_chunk_ids)
        self.assertEqual(["concept_missing"], report.missing_concept_ids)
        self.assertEqual(["rel_missing"], report.missing_relation_ids)
        self.assertEqual(
            "rel_ok", report.relation_ids_missing_graph_triplet[0]["relation_id"]
        )

    def test_reference_validation_allows_cleaned_keep_actions(self) -> None:
        case = load_case(
            make_case_payload(
                expected_evidence={
                    "source_chunk_ids": {"required_all": ["chunk_1"]},
                    "required_answer_points": [],
                },
                thresholds={},
            )
        )

        report = validate_benchmark_references(
            [case],
            EvidenceCatalog(),
            source_metadata_by_id={
                "chunk_1": {"postprocess_action": "keep_with_cleaned_text"}
            },
            embedded_source_ids={"chunk_1"},
        )

        self.assertTrue(report.passed)


def load_case(payload):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cases.jsonl"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return load_benchmark_cases(path)[0]
