import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import backend.evaluation.evaluation_reporting as evaluation_reporting
from backend.evaluation.evaluation_reporting import (
    allocate_run_dir,
    build_evaluation_reports,
    current_evaluation_fingerprint,
    publish_evaluation_report,
    register_completed_run,
    register_failed_run,
)
from scripts.build_evaluation_report import main as report_main


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def write_identity_fixture(root: Path) -> None:
    files = {
        "pyproject.toml": "[project]\nname = 'fixture'\n",
        "uv.lock": "version = 1\n",
        "evals/runtime/scenarios.json": "[]\n",
        "evals/retrieval/benchmark_cases.jsonl": "{}\n",
        "scripts/evaluate_runtime_workflow.py": "RUNTIME = 1\n",
        "scripts/evaluate_retrieval_pipeline.py": "RETRIEVAL = 1\n",
        "scripts/build_evaluation_report.py": "REPORT = 1\n",
        "backend/app.py": "APPLICATION = 1\n",
        "backend/llm/prompts/judge.txt": "prompt-v1\n",
        "storage/application_manifest.json": '{"snapshot": 1}\n',
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


class EvaluationReportingTests(unittest.TestCase):
    def test_latest_deterministic_run_replaces_snapshot_but_history_is_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            invocation = {"mode": "analyst", "case_ids": ["case"]}
            first = allocate_run_dir("retrieval", root)
            write_jsonl(
                first / "case_results.jsonl",
                [{"case_id": "case", "passed": True, "scores": {}}],
            )
            first_manifest = register_completed_run(
                first,
                run_kind="retrieval",
                invocation=invocation,
                configuration={"source_revision": "fixture-revision", "executor": "a"},
                evaluation_root=root,
            )
            second = allocate_run_dir("retrieval", root)
            write_jsonl(
                second / "case_results.jsonl",
                [{"case_id": "case", "passed": False, "scores": {}}],
            )
            second_manifest = register_completed_run(
                second,
                run_kind="retrieval",
                invocation=invocation,
                configuration={"source_revision": "fixture-revision", "executor": "b"},
                evaluation_root=root,
            )
            failed = allocate_run_dir("retrieval", root)
            register_failed_run(
                failed,
                run_kind="retrieval",
                invocation=invocation,
                error="fixture failure",
                configuration={"source_revision": "fixture-revision"},
                evaluation_root=root,
            )

            self.assertEqual(
                first_manifest["evaluation_fingerprint"],
                second_manifest["evaluation_fingerprint"],
            )
            scorecard = json.loads((root / "latest.json").read_text(encoding="utf-8"))
            history = json.loads(
                (
                    root
                    / "fingerprints"
                    / first_manifest["evaluation_fingerprint"]
                    / "history.json"
                ).read_text(encoding="utf-8")
            )

        self.assertEqual(3, scorecard["history_run_count"])
        self.assertEqual(1, scorecard["selected_run_count"])
        self.assertEqual(0.0, scorecard["retrieval_quality"]["case_pass_rate"])
        self.assertEqual(3, len(history["runs"]))
        self.assertNotIn("overall_score", scorecard)
        self.assertNotIn("overall_status", scorecard)
        self.assertIn("metric_definitions", scorecard)
        self.assertIn(
            "task_success_rate",
            scorecard["metric_definitions"]["Runtime task behavior"],
        )
        self.assertEqual(
            [second_manifest["run_id"]],
            scorecard["retrieval_quality"]["source_run_ids"],
        )

    def test_live_repetitions_accumulate_within_one_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_ids = []
            for repetition, passed in ((1, True), (2, False)):
                run_dir = allocate_run_dir("live", root)
                write_jsonl(
                    run_dir / "case_results.jsonl",
                    [
                        {
                            "scenario_id": "repeated_case",
                            "run_id": f"record-{repetition}",
                            "passed": passed,
                            "checks": [],
                            "metrics": {},
                            "failure_types": [],
                        }
                    ],
                )
                write_jsonl(
                    run_dir / "runtime_records.jsonl",
                    [
                        {
                            "scenario_id": "repeated_case",
                            "worker_sequence": ["retriever", "analyst"],
                            "evidence_ids": ["chunk_a"],
                        }
                    ],
                )
                manifest = register_completed_run(
                    run_dir,
                    run_kind="live",
                    invocation={"case_ids": ["repeated_case"], "repetitions": 1},
                    configuration={"source_revision": "fixture-revision"},
                    evaluation_root=root,
                )
                run_ids.append(manifest["run_id"])

            scorecard = json.loads((root / "latest.json").read_text(encoding="utf-8"))

        self.assertEqual(2, scorecard["runtime_task_behavior"]["run_count"])
        self.assertEqual(0.5, scorecard["runtime_task_behavior"]["task_success_rate"])
        self.assertEqual(2, scorecard["reliability"]["repetition_count"])
        self.assertEqual(1.0, scorecard["reliability"]["route_consistency"]["mean"])
        self.assertEqual(
            sorted(run_ids), scorecard["runtime_task_behavior"]["source_run_ids"]
        )

    def test_judges_accumulate_only_with_matching_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for score in (True, False):
                run_dir = allocate_run_dir("judge", root)
                write_jsonl(
                    run_dir / "judge_results.jsonl",
                    [
                        {
                            "dimension": "analyst_usefulness",
                            "status": "success",
                            "score": score,
                            "provider_call_attempted": True,
                            "judge_provider": "fixture-provider",
                            "judge_model": "fixture-model",
                            "rubric_version": "v1",
                        }
                    ],
                )
                register_completed_run(
                    run_dir,
                    run_kind="judge",
                    invocation={"dimension": "analyst_usefulness"},
                    configuration={"source_revision": "fixture-revision"},
                    evaluation_root=root,
                )

            scorecard = json.loads((root / "latest.json").read_text(encoding="utf-8"))

        metric = scorecard["optional_semantic_quality"][
            "analyst_usefulness [fixture-provider/fixture-model; v1]"
        ]
        self.assertEqual(2, metric["sample_count"])
        self.assertEqual(2, metric["result_count"])
        self.assertEqual(2, metric["successful_result_count"])
        self.assertEqual(0, metric["error_count"])
        self.assertEqual(2, metric["provider_call_attempt_count"])
        self.assertEqual(0.5, metric["pass_rate"])

    def test_failed_new_fingerprint_does_not_replace_root_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            completed = allocate_run_dir("retrieval", root)
            write_jsonl(completed / "case_results.jsonl", [])
            manifest = register_completed_run(
                completed,
                run_kind="retrieval",
                invocation={"mode": "analyst"},
                configuration={"source_revision": "revision-a"},
                evaluation_root=root,
            )
            failed = allocate_run_dir("retrieval", root)
            register_failed_run(
                failed,
                run_kind="retrieval",
                invocation={"mode": "analyst"},
                error="fixture failure",
                configuration={"source_revision": "revision-b"},
                evaluation_root=root,
            )

            latest = json.loads((root / "latest.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["evaluation_fingerprint"], latest["fingerprint"])

    def test_reported_source_revisions_do_not_split_content_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fingerprints = set()
            for revision in ("revision-a", "revision-b"):
                run_dir = allocate_run_dir("retrieval", root)
                write_jsonl(run_dir / "case_results.jsonl", [])
                manifest = register_completed_run(
                    run_dir,
                    run_kind="retrieval",
                    invocation={"mode": "analyst"},
                    configuration={"source_revision": revision},
                    evaluation_root=root,
                )
                fingerprints.add(manifest["evaluation_fingerprint"])

            directories = list((root / "fingerprints").iterdir())

        self.assertEqual(1, len(fingerprints))
        self.assertEqual(1, len(directories))

    def test_publish_writes_curated_snapshot_without_raw_run_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            evaluation_root = base / "evaluation"
            reports_root = base / "reports"
            run_dir = allocate_run_dir("retrieval", evaluation_root)
            write_jsonl(
                run_dir / "case_results.jsonl",
                [{"case_id": "case", "passed": True, "scores": {}}],
            )
            write_jsonl(
                run_dir / "actual_evidence.jsonl",
                [{"case_id": "case", "excerpt": "private fixture evidence"}],
            )
            manifest = register_completed_run(
                run_dir,
                run_kind="retrieval",
                invocation={"mode": "analyst"},
                configuration={"source_revision": "fixture-revision"},
                evaluation_root=evaluation_root,
            )

            first = publish_evaluation_report(
                evaluation_root,
                reports_root,
                fingerprint=manifest["evaluation_fingerprint"],
            )
            second = publish_evaluation_report(
                evaluation_root,
                reports_root,
                fingerprint=manifest["evaluation_fingerprint"],
            )

            latest = json.loads(
                (reports_root / "latest.json").read_text(encoding="utf-8")
            )
            provenance = json.loads(
                (reports_root / "latest_provenance.json").read_text(encoding="utf-8")
            )
            history_directories = list((reports_root / "history").iterdir())

        self.assertEqual(manifest["evaluation_fingerprint"], latest["fingerprint"])
        self.assertEqual(first["snapshot"], second["snapshot"])
        self.assertEqual(1, len(history_directories))
        self.assertEqual(1, provenance["contributing_run_count"])
        self.assertFalse((reports_root / "actual_evidence.jsonl").exists())
        self.assertIn("retrieved evidence text", provenance["excluded_content"])

    def test_versioned_report_schema_family_history_remains_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = allocate_run_dir("retrieval", root)
            write_jsonl(run_dir / "case_results.jsonl", [])
            manifest = register_completed_run(
                run_dir,
                run_kind="retrieval",
                invocation={"mode": "analyst"},
                evaluation_root=root,
            )
            manifest["report_schema_version"] = (
                f"{evaluation_reporting.REPORT_SCHEMA_VERSION}-v{2}"
            )
            (run_dir / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            reports = build_evaluation_reports(root)

        self.assertIn(manifest["evaluation_fingerprint"], reports)
        self.assertEqual(
            evaluation_reporting.REPORT_SCHEMA_VERSION,
            reports[manifest["evaluation_fingerprint"]]["schema_version"],
        )

    def test_rebuilding_an_exact_fingerprint_does_not_repoint_root_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fingerprints = []
            for revision in ("revision-a", "revision-b"):
                run_dir = allocate_run_dir("retrieval", root)
                write_jsonl(run_dir / "case_results.jsonl", [])
                manifest = register_completed_run(
                    run_dir,
                    run_kind="retrieval",
                    invocation={"mode": "analyst"},
                    configuration={"source_revision": revision},
                    evaluation_root=root,
                )
                fingerprints.append(manifest["evaluation_fingerprint"])

            latest_before = json.loads(
                (root / "latest.json").read_text(encoding="utf-8")
            )
            build_evaluation_reports(root, fingerprint=fingerprints[0])
            latest_after = json.loads(
                (root / "latest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(fingerprints[1], latest_before["fingerprint"])
        self.assertEqual(latest_before["fingerprint"], latest_after["fingerprint"])

    def test_publishing_reports_does_not_change_content_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            repository_root = Path(tmp)
            write_identity_fixture(repository_root)
            evaluation_root = repository_root / "data" / "evaluation"
            reports_root = repository_root / "reports" / "evaluation"
            with patch.object(evaluation_reporting, "ROOT", repository_root):
                before = current_evaluation_fingerprint()
                run_dir = allocate_run_dir("retrieval", evaluation_root)
                write_jsonl(
                    run_dir / "case_results.jsonl",
                    [
                        {
                            "case_id": "case",
                            "mode": "analyst",
                            "passed": True,
                            "scores": {},
                        }
                    ],
                )
                manifest = register_completed_run(
                    run_dir,
                    run_kind="retrieval",
                    invocation={
                        "mode": "both",
                        "analyst_variant": "legacy_vector_context",
                        "visualizer_variant": "optimized",
                    },
                    evaluation_root=evaluation_root,
                )
                publish_evaluation_report(
                    evaluation_root,
                    reports_root,
                    fingerprint=manifest["evaluation_fingerprint"],
                )
                (reports_root / ".DS_Store").write_text("metadata", encoding="utf-8")
                after = current_evaluation_fingerprint()

        self.assertEqual(before, after)

    def test_every_relevant_identity_category_changes_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            repository_root = Path(tmp)
            write_identity_fixture(repository_root)
            relevant_files = [
                repository_root / "backend" / "app.py",
                repository_root / "backend" / "llm" / "prompts" / "judge.txt",
                repository_root / "evals" / "runtime" / "scenarios.json",
                repository_root / "evals" / "retrieval" / "benchmark_cases.jsonl",
                repository_root / "pyproject.toml",
                repository_root / "uv.lock",
                repository_root / "storage" / "application_manifest.json",
            ]
            with patch.object(evaluation_reporting, "ROOT", repository_root):
                previous = current_evaluation_fingerprint()
                for index, path in enumerate(relevant_files, start=1):
                    path.write_text(
                        path.read_text(encoding="utf-8") + f"change-{index}\n",
                        encoding="utf-8",
                    )
                    observed = current_evaluation_fingerprint()
                    self.assertNotEqual(previous, observed, path)
                    previous = observed
                (repository_root / "backend" / ".DS_Store").write_text(
                    "irrelevant", encoding="utf-8"
                )
                self.assertEqual(previous, current_evaluation_fingerprint())

    def test_unrun_semantic_dimensions_are_visible_as_na(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = allocate_run_dir("judge", root)
            write_jsonl(
                run_dir / "judge_results.jsonl",
                [
                    {
                        "dimension": "mentor_pedagogy",
                        "status": "skipped",
                        "reason": "required Mentor worker was not reached",
                        "provider_call_attempted": False,
                        "judge_provider": "fixture",
                        "judge_model": "fixture-model",
                        "rubric_version": "v1",
                    },
                    {
                        "dimension": "claim_faithfulness",
                        "status": "error",
                        "reason": "SECRET raw provider body",
                        "provider_call_attempted": True,
                        "parser_classification": "malformed_json",
                        "error_type": "OutputParserException",
                        "judge_provider": "fixture",
                        "judge_model": "fixture-model",
                        "rubric_version": "v1",
                    },
                ],
            )
            register_completed_run(
                run_dir,
                run_kind="judge",
                invocation={"dimensions": ["mentor_pedagogy", "claim_faithfulness"]},
                evaluation_root=root,
            )
            scorecard_text = (root / "latest.json").read_text(encoding="utf-8")
            scorecard = json.loads(scorecard_text)

        dimensions = scorecard["optional_semantic_quality"]["dimensions"]
        self.assertEqual("N/A", dimensions["graph_usefulness"]["status"])
        self.assertEqual(0, dimensions["graph_usefulness"]["result_count"])
        self.assertEqual(1, dimensions["mentor_pedagogy"]["skipped_count"])
        self.assertEqual(0, dimensions["mentor_pedagogy"]["error_count"])
        mentor_identity = scorecard["optional_semantic_quality"][
            "mentor_pedagogy [fixture/fixture-model; v1]"
        ]
        self.assertIsNone(mentor_identity["pass_rate"])
        self.assertEqual(1, mentor_identity["skipped_result_count"])
        self.assertNotIn("SECRET raw provider body", scorecard_text)
        self.assertIn(
            "parser=malformed_json",
            scorecard["optional_semantic_quality"][
                "claim_faithfulness [fixture/fixture-model; v1]"
            ]["error_reasons"][0],
        )

    def test_default_publication_rejects_implicit_incomplete_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "evaluation"
            reports = Path(tmp) / "reports"
            run_dir = allocate_run_dir("retrieval", root)
            write_jsonl(run_dir / "case_results.jsonl", [])
            register_completed_run(
                run_dir,
                run_kind="retrieval",
                invocation={"mode": "analyst"},
                evaluation_root=root,
            )

            with self.assertRaisesRegex(ValueError, "requires --fingerprint"):
                publish_evaluation_report(root, reports)

    def test_one_fingerprint_can_account_for_a_complete_campaign(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenarios = json.loads(
                (
                    evaluation_reporting.ROOT / "evals" / "runtime" / "scenarios.json"
                ).read_text(encoding="utf-8")
            )

            retrieval = allocate_run_dir("retrieval", root)
            write_jsonl(
                retrieval / "case_results.jsonl",
                [
                    {
                        "case_id": "analyst_case",
                        "mode": "analyst",
                        "passed": True,
                        "scores": {
                            "context_precision_at_10": 0.5,
                            "context_recall": 0.5,
                        },
                    }
                ],
            )
            register_completed_run(
                retrieval,
                run_kind="retrieval",
                invocation={
                    "mode": "both",
                    "analyst_variant": "legacy_vector_context",
                    "visualizer_variant": "optimized",
                },
                evaluation_root=root,
            )

            offline_records = []
            offline_results = []
            for scenario in scenarios:
                for repetition, _ in enumerate(scenario.get("trace_ids", []), start=1):
                    offline_records.append(
                        {
                            "scenario_id": scenario["id"],
                            "repetition": repetition,
                            "provenance": {
                                "source_configuration_status": "needs verification"
                            },
                        }
                    )
                    offline_results.append(
                        {
                            "scenario_id": scenario["id"],
                            "run_id": f"offline-{scenario['id']}-{repetition}",
                            "passed": True,
                            "checks": [],
                            "metrics": {},
                            "failure_types": [],
                        }
                    )
            offline = allocate_run_dir("offline", root)
            write_jsonl(offline / "runtime_records.jsonl", offline_records)
            write_jsonl(offline / "case_results.jsonl", offline_results)
            register_completed_run(
                offline,
                run_kind="offline",
                invocation={"framework": "none", "trace_export": "fixture.json"},
                evaluation_root=root,
            )

            live_records = []
            live_results = []
            for scenario in scenarios:
                repetitions = (
                    3
                    if scenario["id"]
                    == "researcher_structured_output_truncation_reliability"
                    else 1
                )
                for repetition in range(1, repetitions + 1):
                    live_records.append(
                        {
                            "scenario_id": scenario["id"],
                            "repetition": repetition,
                            "provenance": {"source_configuration_status": "verified"},
                        }
                    )
                    live_results.append(
                        {
                            "scenario_id": scenario["id"],
                            "run_id": f"live-{scenario['id']}-{repetition}",
                            "passed": True,
                            "checks": [],
                            "metrics": {},
                            "failure_types": [],
                        }
                    )
            live = allocate_run_dir("live", root)
            write_jsonl(live / "runtime_records.jsonl", live_records)
            write_jsonl(live / "case_results.jsonl", live_results)
            register_completed_run(
                live,
                run_kind="live",
                invocation={"case_ids": [item["id"] for item in scenarios]},
                evaluation_root=root,
            )

            judge = allocate_run_dir("judge", root)
            write_jsonl(
                judge / "judge_results.jsonl",
                [
                    {
                        "dimension": dimension,
                        "status": "skipped",
                        "reason": "fixture prerequisite unavailable",
                        "provider_call_attempted": False,
                        "judge_provider": "fixture",
                        "judge_model": "fixture-model",
                        "rubric_version": "v1",
                    }
                    for dimension in evaluation_reporting.EXPECTED_SEMANTIC_DIMENSIONS
                ],
            )
            manifest = register_completed_run(
                judge,
                run_kind="judge",
                invocation={
                    "dimensions": list(
                        evaluation_reporting.EXPECTED_SEMANTIC_DIMENSIONS
                    )
                },
                evaluation_root=root,
            )
            scorecard = json.loads((root / "latest.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["evaluation_fingerprint"], scorecard["fingerprint"])
        self.assertTrue(scorecard["coverage"]["complete"])
        self.assertEqual([], scorecard["coverage"]["missing_requirements"])
        self.assertEqual(
            {"offline": len(offline_results), "live": len(live_results)},
            scorecard["runtime_task_behavior"]["sample_counts"],
        )

    def test_markdown_groups_nested_metrics_and_avoids_raw_json(self):
        scorecard = {
            "fingerprint": "fixture",
            "generated_at": "2026-08-23T00:00:00Z",
            "history_run_count": 3,
            "selected_run_count": 3,
            "coverage": {
                "complete": False,
                "missing_requirements": ["retrieval_relevance_labels"],
                "requirements": [],
            },
            "retrieval_quality": {
                "source_run_ids": ["retrieval-run"],
                "case_pass_rate": 0.5,
                "diagnostics": {"answer_point_recall": 0.75},
            },
            "runtime_task_behavior": {
                "source_run_ids": ["runtime-run"],
                "sample_counts": {"offline": 2, "live": 3},
                "functional_task_success_rate": 0.6,
                "path_efficiency_ratio": {
                    "count": 5,
                    "mean": 0.8,
                    "median": 1.0,
                    "p95": 1.0,
                },
            },
            "reliability": {
                "source_run_ids": ["runtime-run"],
                "route_consistency": {
                    "mean": 0.9,
                    "by_scenario": {
                        "researcher_structured_output_truncation_reliability": 0.8
                    },
                },
                "evidence_set_jaccard_stability": {
                    "mean": 0.7,
                    "by_scenario": {"mentor_stuck_continuation": 0.5},
                },
                "failure_type_frequency": {"invalid_retriever_output": 2},
            },
            "efficiency": {
                "source_run_ids": ["runtime-run"],
                "input_tokens": {
                    "count": 5,
                    "mean": 1000.0,
                    "median": 900.0,
                    "p95": 1500.0,
                },
                "retry_rate": 0.2,
            },
            "optional_semantic_quality": {
                "source_run_ids": ["judge-run"],
                "dimensions": {
                    "claim_faithfulness": {
                        "status": "N/A",
                        "reason": "not run",
                        "result_count": 0,
                        "successful_count": 0,
                        "skipped_count": 0,
                        "error_count": 0,
                        "provider_attempted_count": 0,
                    }
                },
            },
            "diagnostics": {
                "failed_runtime_checks": [],
                "failure_type_frequency": {"invalid_retriever_output": 2},
                "agentevals": [],
                "runtime_mode_breakdown": {
                    "live": {
                        "sample_count": 3,
                        "passed_count": 2,
                        "failed_count": 1,
                        "failure_type_frequency": {"invalid_retriever_output": 1},
                    }
                },
                "provenance": {"source_configuration_status_counts": {"verified": 3}},
                "run_artifacts": [],
            },
            "metric_definitions": {},
        }

        markdown = evaluation_reporting._render_scorecard(scorecard)

        self.assertIn("### Route consistency", markdown)
        self.assertIn(
            "| `researcher_structured_output_truncation_reliability` | 80.0% |",
            markdown,
        )
        self.assertNotIn(
            "route_consistency.by_scenario.researcher_structured_output_truncation_reliability",
            markdown,
        )
        self.assertIn("| Measurement | Samples | Mean | Median | P95 |", markdown)
        self.assertIn("| Case pass rate | 50.0% |", markdown)
        self.assertIn("| Invalid retriever output | 2 |", markdown)
        self.assertNotIn('{"invalid_retriever_output": 2}', markdown)
        self.assertIn("<details><summary>Source runs (1)</summary>", markdown)

    def test_cli_verifies_explicit_campaign_run_fingerprints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_ids = []
            for kind in ("retrieval", "live"):
                run_dir = allocate_run_dir(kind, root)
                write_jsonl(run_dir / "case_results.jsonl", [])
                manifest = register_completed_run(
                    run_dir,
                    run_kind=kind,
                    invocation={"fixture": kind},
                    evaluation_root=root,
                )
                run_ids.append(manifest["run_id"])

            result = report_main(
                [
                    "--evaluation-root",
                    str(root),
                    "--verify-run-fingerprints",
                    "--run-id",
                    run_ids[0],
                    "--run-id",
                    run_ids[1],
                ]
            )

        self.assertEqual(0, result)


if __name__ == "__main__":
    unittest.main()
