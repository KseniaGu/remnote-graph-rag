import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from langchain_core.exceptions import OutputParserException

from backend.evaluation.runtime_evaluation import (
    RuntimeRecord,
    RuntimeScenario,
    evaluate_with_openevals_judge,
    load_runtime_records,
    load_runtime_scenarios,
    normalize_trace_export,
    normalize_trace_runs,
    score_runtime_record,
    summarize_runtime_results,
    write_runtime_artifacts,
)
from backend.evaluation.runtime_live import _live_error_code, _live_graph_observation
from scripts.evaluate_runtime_workflow import main


class FakeStructuredJudge:
    def __init__(self, envelope=None, error=None):
        self.envelope = envelope
        self.error = error
        self.method = None
        self.schema = None
        self.messages = []

    def with_structured_output(self, schema, *, method, include_raw):
        self.schema = schema
        self.method = method
        self.include_raw = include_raw
        return self

    def invoke(self, messages):
        self.messages = messages
        if self.error is not None:
            raise self.error
        return self.envelope


class ProviderFailure(RuntimeError):
    status_code = 500


def make_scenario(**overrides):
    payload = {
        "schema_version": "runtime-evaluation",
        "id": "runtime_case",
        "version": 1,
        "review_status": "reviewed",
        "description": "Synthetic runtime contract.",
        "trace_ids": ["trace_1"],
        "input": {"user_message": "Explain Mamba"},
        "expectations": {
            "allowed_worker_sequences": [["retriever", "analyst"]],
            "required_agents": ["retriever", "analyst"],
            "required_tools": ["search_knowledge_base"],
            "request_scope": "in_scope",
            "allowed_retrieval_statuses": ["adequate"],
            "retrieval_outcome": "adequate",
            "sources_exhausted": False,
            "requested_modality": "text",
            "final_response_required": True,
            "forbidden_failure_types": ["provider_error"],
            "budgets": {
                "max_worker_steps": 2,
                "max_provider_attempts": 3,
                "max_tavily_searches": 0,
            },
        },
        "provenance": ["synthetic"],
    }
    payload.update(overrides)
    return RuntimeScenario.model_validate(payload)


def llm_run(*, stop_reason="stop", content="ok", tool_calls=None):
    return {
        "_id": "llm_1",
        "trace_id": "trace_1",
        "parent_run_id": "wrapper_1",
        "name": "ChatOllama",
        "run_type": "llm",
        "status": "success",
        "start_time": "2026-01-01T00:00:03Z",
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "extra": {
            "metadata": {
                "langgraph_node": "retriever",
                "ls_provider": "ollama",
                "ls_model_name": "model",
            }
        },
        "outputs": {
            "generations": [
                [
                    {
                        "message": {
                            "kwargs": {
                                "content": content,
                                "tool_calls": tool_calls or [],
                                "response_metadata": {"done_reason": stop_reason},
                            }
                        }
                    }
                ]
            ]
        },
    }


def root_run(*, retrieval_status="adequate", sources_exhausted=False):
    human = {"type": "human", "kwargs": {"content": "Explain Mamba"}}
    answer = {
        "type": "ai",
        "kwargs": {
            "content": "Grounded answer.",
            "additional_kwargs": {"agent": "[ANALYST]"},
        },
    }
    return [
        {
            "_id": "root_1",
            "trace_id": "trace_1",
            "parent_run_id": None,
            "name": "LangGraph",
            "run_type": "chain",
            "status": "success",
            "start_time": "2026-01-01T00:00:00Z",
            "latency": 1.25,
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "inputs": {"messages": [human]},
            "outputs": {
                "messages": [human, answer],
                "request_scope": "in_scope",
                "sources_exhausted": sources_exhausted,
                "next_step": "__end__",
                "visual_artifacts": [],
            },
            "extra": {
                "metadata": {
                    "thread_id": "thread_1",
                    "revision_id": "fixture-revision",
                }
            },
        },
        {
            "_id": "orchestrator_1",
            "trace_id": "trace_1",
            "parent_run_id": "root_1",
            "name": "orchestrator",
            "run_type": "chain",
            "status": "success",
            "start_time": "2026-01-01T00:00:01Z",
            "outputs": {"next_step": "retriever"},
        },
        {
            "_id": "retriever_1",
            "trace_id": "trace_1",
            "parent_run_id": "root_1",
            "name": "retriever",
            "run_type": "chain",
            "status": "success",
            "start_time": "2026-01-01T00:00:02Z",
            "outputs": {"retrieval_status": retrieval_status},
        },
        llm_run(),
        {
            "_id": "tool_1",
            "trace_id": "trace_1",
            "parent_run_id": "retriever_1",
            "name": "search_knowledge_base",
            "run_type": "tool",
            "status": "success",
            "start_time": "2026-01-01T00:00:04Z",
            "inputs": {"queries": ["Mamba"], "api_key": "never-export"},
            "outputs": {"output": [{"id": "chunk_mamba"}]},
            "extra": {"metadata": {"langgraph_node": "retriever"}},
        },
        {
            "_id": "analyst_1",
            "trace_id": "trace_1",
            "parent_run_id": "root_1",
            "name": "analyst",
            "run_type": "chain",
            "status": "success",
            "start_time": "2026-01-01T00:00:05Z",
            "outputs": {},
        },
    ]


class RuntimeEvaluationTests(unittest.TestCase):
    def test_scenario_loader_rejects_duplicate_ids(self):
        row = make_scenario().model_dump(mode="json")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scenarios.json"
            path.write_text(json.dumps([row, row]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_runtime_scenarios(path)

    def test_versioned_runtime_schema_family_normalizes_to_current_name(self):
        scenario_payload = make_scenario().model_dump(mode="json")
        scenario_payload["schema_version"] = (
            f"{scenario_payload['schema_version']}-v{2}"
        )
        scenario = RuntimeScenario.model_validate(scenario_payload)
        record_payload = normalize_trace_runs(root_run(), scenario).model_dump(
            mode="json"
        )
        record_payload["schema_version"] = f"{record_payload['schema_version']}-v{2}"
        record = RuntimeRecord.model_validate(record_payload)

        self.assertEqual("runtime-evaluation", scenario.schema_version)
        self.assertEqual("runtime-evaluation", record.schema_version)

    def test_scenario_rejects_allowed_route_above_worker_budget(self):
        payload = make_scenario().model_dump(mode="json")
        payload["expectations"]["allowed_worker_sequences"] = [
            ["retriever", "researcher", "analyst"]
        ]
        payload["expectations"]["budgets"]["max_worker_steps"] = 2
        with self.assertRaisesRegex(ValueError, "max_worker_steps"):
            RuntimeScenario.model_validate(payload)

    def test_trace_normalizer_rejects_missing_scenario_trace(self):
        with self.assertRaisesRegex(ValueError, "missing scenario traces"):
            normalize_trace_export([], [make_scenario()])

    def test_normalizer_captures_routes_evidence_usage_and_redacts_secrets(self):
        record = normalize_trace_runs(root_run(), make_scenario())

        self.assertEqual(["retriever", "analyst"], record.worker_sequence)
        self.assertEqual(["retriever"], record.route_decisions)
        self.assertEqual(["chunk_mamba"], record.evidence_ids)
        self.assertEqual("[REDACTED]", record.tools[0].arguments["api_key"])
        self.assertEqual("adequate", record.retrieval_outcome)
        self.assertEqual(1, record.usage.provider_attempts)
        self.assertTrue(record.terminated)

    def test_runtime_record_rejects_evidence_above_item_bound(self):
        record = normalize_trace_runs(root_run(), make_scenario())
        payload = record.model_dump(mode="json")
        payload["bounded_evidence"] = [
            {
                "evidence_id": f"chunk_{index}",
                "kind": "local_chunk",
                "excerpt": "bounded",
            }
            for index in range(11)
        ]

        with self.assertRaisesRegex(ValueError, "item limit"):
            RuntimeRecord.model_validate(payload)

    def test_live_failure_classifier_uses_specific_categories(self):
        cases = {
            "structured_output_parse_error": ValueError("OutputParser validation"),
            "timeout": TimeoutError("Ollama timeout"),
            "recursion_limit": RuntimeError("GraphRecursionError"),
            "storage_error": RuntimeError("Pinecone storage unavailable"),
            "tool_error": RuntimeError("Tavily tool failed"),
            "provider_error": RuntimeError("Ollama provider unavailable"),
            "workflow_error": RuntimeError("unexpected workflow state"),
        }
        for expected, error in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(expected, _live_error_code(error))

    def test_truncated_retriever_is_not_classified_as_true_empty(self):
        runs = root_run(retrieval_status="no_results")
        runs[-3] = llm_run(stop_reason="length", content="")
        runs = [run for run in runs if run.get("run_type") != "tool"]

        record = normalize_trace_runs(runs, make_scenario())

        self.assertEqual("invalid_model_output", record.retrieval_outcome)
        self.assertIn(
            "invalid_retriever_output", {item.code for item in record.failures}
        )
        self.assertIn("model_output_truncated", {item.code for item in record.failures})

    def test_empty_retrieval_after_local_tool_is_true_empty(self):
        record = normalize_trace_runs(
            root_run(retrieval_status="no_results", sources_exhausted=True),
            make_scenario(),
        )

        self.assertEqual("true_empty", record.retrieval_outcome)
        self.assertIn("sources_exhausted", {item.code for item in record.failures})

    def test_deterministic_scorer_reports_route_and_budget_independently(self):
        scenario = make_scenario()
        record = normalize_trace_runs(root_run(), scenario)
        record.worker_sequence = ["retriever", "researcher", "analyst"]
        record.usage.provider_attempts = 4

        result = score_runtime_record(scenario, record)
        failed = {
            (item.dimension, item.name)
            for item in result.checks
            if item.status == "fail"
        }

        self.assertFalse(result.passed)
        self.assertIn(("routing", "worker_sequence"), failed)
        self.assertIn(("budget", "provider_attempts"), failed)

    def test_provider_budget_is_report_only(self):
        scenario = make_scenario()
        record = normalize_trace_runs(root_run(), scenario)
        record.usage.provider_attempts = 4

        result = score_runtime_record(scenario, record)
        provider_check = next(
            item for item in result.checks if item.name == "provider_attempts"
        )

        self.assertTrue(result.passed)
        self.assertEqual("fail", provider_check.status)
        self.assertFalse(provider_check.gating)

    def test_basic_loop_detector_is_gating(self):
        scenario = make_scenario()
        record = normalize_trace_runs(root_run(), scenario)
        record.action_signatures = ["retriever:search:query", "retriever:search:query"]

        result = score_runtime_record(scenario, record)

        self.assertFalse(result.passed)
        self.assertTrue(result.metrics["loop_detected"])
        self.assertEqual(1, result.metrics["repeated_action_count"])

    def test_basic_loop_detector_ignores_extra_tools_in_one_worker_step(self):
        scenario = make_scenario()
        record = normalize_trace_runs(root_run(), scenario)
        record.action_signatures.append(record.action_signatures[0])

        result = score_runtime_record(scenario, record)

        self.assertFalse(result.metrics["loop_detected"])

    def test_claim_faithfulness_aggregates_structured_claims(self):
        scenario = make_scenario()
        scenario.expectations.judge_dimensions = ["claim_faithfulness"]
        record = normalize_trace_runs(root_run(), scenario)

        def fake_factory(**kwargs):
            def evaluator(**payload):
                return {
                    "claims": [
                        {
                            "claim": "Supported.",
                            "verdict": "supported",
                            "evidence_ids": ["chunk_mamba"],
                            "reason": "Direct support.",
                        },
                        {
                            "claim": "Partial.",
                            "verdict": "partial",
                            "evidence_ids": ["chunk_mamba"],
                            "reason": "Incomplete support.",
                        },
                        {
                            "claim": "Unsupported.",
                            "verdict": "unsupported",
                            "evidence_ids": [],
                            "reason": "No support.",
                        },
                    ]
                }

            return evaluator

        result = evaluate_with_openevals_judge(
            scenario,
            record,
            "claim_faithfulness",
            judge=object(),
            judge_provider="fixture",
            judge_model="fixture-model",
            evaluator_factory=fake_factory,
        )

        self.assertEqual("success", result.status)
        self.assertEqual(1, result.supported_claims)
        self.assertEqual(1, result.partial_claims)
        self.assertEqual(1, result.unsupported_claims)
        self.assertEqual(0.5, result.grounded_claim_rate)

    def test_claim_judge_captures_success_diagnostics_and_json_mode_schema(self):
        scenario = make_scenario()
        scenario.expectations.judge_dimensions = ["claim_faithfulness"]
        record = normalize_trace_runs(root_run(), scenario)
        raw = SimpleNamespace(
            content='{"claims":[{"claim":"Grounded.","verdict":"supported"}]}',
            response_metadata={"status_code": 200, "done_reason": "stop"},
        )
        judge = FakeStructuredJudge(
            {
                "raw": raw,
                "parsed": {
                    "claims": [
                        {
                            "claim": "Grounded.",
                            "verdict": "supported",
                            "evidence_ids": ["chunk_mamba"],
                            "reason": "Direct.",
                        }
                    ]
                },
                "parsing_error": None,
            }
        )

        result = evaluate_with_openevals_judge(
            scenario,
            record,
            "claim_faithfulness",
            judge=judge,
            judge_provider="fixture",
            judge_model="fixture-model",
            configured_output_token_limit=4096,
            structured_output_method="json_mode",
        )

        self.assertEqual("success", result.status)
        self.assertEqual("completed", result.transport_status)
        self.assertEqual(200, result.http_status_code)
        self.assertTrue(result.output_present)
        self.assertGreater(result.output_size_chars or 0, 0)
        self.assertEqual("success", result.parser_classification)
        self.assertFalse(result.confirmed_truncation)
        self.assertEqual(4096, result.configured_output_token_limit)
        self.assertEqual("json_mode", judge.method)
        self.assertIn("Return exactly one JSON object", judge.messages[0].content)
        self.assertIn('"claims"', judge.messages[0].content)

    def test_claim_judge_classifies_provider_failure_without_raw_content(self):
        scenario = make_scenario()
        scenario.expectations.judge_dimensions = ["claim_faithfulness"]
        record = normalize_trace_runs(root_run(), scenario)

        result = evaluate_with_openevals_judge(
            scenario,
            record,
            "claim_faithfulness",
            judge=FakeStructuredJudge(error=ProviderFailure("private failure")),
            judge_provider="fixture",
            judge_model="fixture-model",
            configured_output_token_limit=4096,
        )

        self.assertEqual("error", result.status)
        self.assertEqual("failed", result.transport_status)
        self.assertEqual(500, result.http_status_code)
        self.assertEqual("provider_failure", result.parser_classification)
        self.assertNotIn("private failure", result.model_dump_json())

    def test_claim_judge_classifies_malformed_json(self):
        scenario = make_scenario()
        scenario.expectations.judge_dimensions = ["claim_faithfulness"]
        record = normalize_trace_runs(root_run(), scenario)
        try:
            json.loads("{")
        except json.JSONDecodeError as cause:
            parsing_error = OutputParserException(
                "private malformed output", llm_output="SECRET"
            )
            parsing_error.__cause__ = cause
        raw = SimpleNamespace(content="{", response_metadata={"done_reason": "stop"})
        judge = FakeStructuredJudge(
            {"raw": raw, "parsed": None, "parsing_error": parsing_error}
        )

        result = evaluate_with_openevals_judge(
            scenario,
            record,
            "claim_faithfulness",
            judge=judge,
            judge_provider="fixture",
            judge_model="fixture-model",
            configured_output_token_limit=4096,
        )

        self.assertEqual("error", result.status)
        self.assertEqual("completed", result.transport_status)
        self.assertEqual("malformed_json", result.parser_classification)
        self.assertFalse(result.confirmed_truncation)
        self.assertNotIn("SECRET", result.model_dump_json())

    def test_claim_judge_classifies_schema_validation_failure(self):
        scenario = make_scenario()
        scenario.expectations.judge_dimensions = ["claim_faithfulness"]
        record = normalize_trace_runs(root_run(), scenario)
        raw = SimpleNamespace(content='{"claims":[]}', response_metadata={})
        judge = FakeStructuredJudge(
            {
                "raw": raw,
                "parsed": {
                    "claims": [
                        {
                            "claim": "Invalid verdict.",
                            "verdict": "unknown",
                            "evidence_ids": [],
                        }
                    ]
                },
                "parsing_error": None,
            }
        )

        result = evaluate_with_openevals_judge(
            scenario,
            record,
            "claim_faithfulness",
            judge=judge,
            judge_provider="fixture",
            judge_model="fixture-model",
            configured_output_token_limit=4096,
        )

        self.assertEqual("error", result.status)
        self.assertEqual("schema_validation_failure", result.parser_classification)
        self.assertEqual("completed", result.transport_status)

    def test_claim_judge_records_confirmed_truncation_from_stop_reason(self):
        scenario = make_scenario()
        scenario.expectations.judge_dimensions = ["claim_faithfulness"]
        record = normalize_trace_runs(root_run(), scenario)
        parsing_error = OutputParserException("private truncated output")
        raw = SimpleNamespace(
            content='{"claims":[',
            response_metadata={"status_code": 200, "done_reason": "length"},
        )
        judge = FakeStructuredJudge(
            {"raw": raw, "parsed": None, "parsing_error": parsing_error}
        )

        result = evaluate_with_openevals_judge(
            scenario,
            record,
            "claim_faithfulness",
            judge=judge,
            judge_provider="fixture",
            judge_model="fixture-model",
            configured_output_token_limit=4096,
        )

        self.assertEqual("error", result.status)
        self.assertEqual("length", result.stop_reason)
        self.assertTrue(result.confirmed_truncation)
        self.assertEqual(200, result.http_status_code)

    def test_reliability_summary_keeps_route_and_evidence_stability_separate(self):
        scenario = make_scenario()
        first = normalize_trace_runs(root_run(), scenario, repetition=1)
        second = first.model_copy(deep=True)
        second.run_id = "trace_2"
        second.repetition = 2
        second.evidence_ids = ["chunk_other"]
        results = [
            score_runtime_record(scenario, first),
            score_runtime_record(scenario, second),
        ]

        summary = summarize_runtime_results([first, second], results)
        reliability = summary["reliability"][scenario.id]

        self.assertEqual(1.0, reliability["route_consistency"])
        self.assertEqual(0.0, reliability["evidence_set_stability"])

    def test_artifacts_declare_zero_offline_provider_calls(self):
        scenario = make_scenario()
        record = normalize_trace_runs(root_run(), scenario)
        result = score_runtime_record(scenario, record)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            write_runtime_artifacts(output, [record], [result], source="fixture")
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            summary = (output / "summary.md").read_text(encoding="utf-8")

        self.assertEqual(0, manifest["provider_calls_made_by_offline_scoring"])
        self.assertFalse(manifest["judge_enabled"])
        self.assertEqual(["fixture-revision"], manifest["source_revisions"])
        self.assertIn("Judge calls: 0", summary)

    def test_record_loader_round_trips_written_jsonl(self):
        scenario = make_scenario()
        record = normalize_trace_runs(root_run(), scenario)
        result = score_runtime_record(scenario, record)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            write_runtime_artifacts(output, [record], [result], source="fixture")
            loaded = load_runtime_records(output / "runtime_records.jsonl")

        self.assertEqual([record], loaded)

    def test_semantic_judge_adapter_is_dimension_scoped_and_inspectable(self):
        scenario = make_scenario()
        scenario.expectations.judge_dimensions = ["analyst_usefulness"]
        record = normalize_trace_runs(root_run(), scenario)
        observed = {}

        def fake_factory(**kwargs):
            observed.update(kwargs)

            def evaluator(**payload):
                observed.update(payload)
                return {
                    "key": "analyst_usefulness",
                    "score": True,
                    "comment": "Direct and useful.",
                }

            return evaluator

        result = evaluate_with_openevals_judge(
            scenario,
            record,
            "analyst_usefulness",
            judge=object(),
            judge_provider="fixture",
            judge_model="fixture-model",
            configured_output_token_limit=512,
            evaluator_factory=fake_factory,
        )

        self.assertEqual("success", result.status)
        self.assertTrue(result.score)
        self.assertEqual("runtime-semantic-v1", result.rubric_version)
        self.assertEqual("Grounded answer.", observed["outputs"]["response"])
        self.assertFalse(observed["continuous"])
        self.assertIn("required_output_schema", observed["prompt"])
        self.assertEqual(512, result.configured_output_token_limit)
        self.assertEqual("completed", result.transport_status)
        self.assertEqual("success", result.parser_classification)

    def test_semantic_judge_rejects_unobservable_grounding_dimension(self):
        scenario = make_scenario()
        scenario.expectations.judge_dimensions = ["claim_faithfulness"]
        record = normalize_trace_runs(root_run(), scenario)
        record.bounded_evidence = []
        record.evidence_capture_status = "unavailable"

        result = evaluate_with_openevals_judge(
            scenario,
            record,
            "claim_faithfulness",
            judge=object(),
            judge_provider="fixture",
            judge_model="fixture-model",
            evaluator_factory=lambda **kwargs: None,
        )

        self.assertEqual("skipped", result.status)
        self.assertEqual("not_attempted", result.transport_status)
        self.assertIn("bounded evidence", result.reason)

    def test_semantic_judge_records_safe_parser_diagnostics_and_call_attempt(self):
        scenario = make_scenario()
        scenario.expectations.judge_dimensions = ["analyst_usefulness"]
        record = normalize_trace_runs(root_run(), scenario)

        def fake_factory(**_kwargs):
            def evaluator(**_payload):
                try:
                    json.loads("{")
                except json.JSONDecodeError as cause:
                    raise OutputParserException(
                        "private provider payload", llm_output="SECRET"
                    ) from cause

            return evaluator

        result = evaluate_with_openevals_judge(
            scenario,
            record,
            "analyst_usefulness",
            judge=object(),
            judge_provider="fixture",
            judge_model="fixture-model",
            evaluator_factory=fake_factory,
        )

        self.assertEqual("error", result.status)
        self.assertTrue(result.provider_call_attempted)
        self.assertEqual("OutputParserException", result.error_type)
        self.assertIn("JSONDecodeError", result.error_cause_types)
        serialized = result.model_dump_json()
        self.assertNotIn("private provider payload", serialized)
        self.assertNotIn("SECRET", serialized)

    def test_semantic_judge_rejects_wrong_final_worker_without_calling_provider(self):
        scenario = make_scenario()
        scenario.expectations.judge_dimensions = ["mentor_pedagogy"]
        record = normalize_trace_runs(root_run(), scenario)
        factory_calls = []

        def fake_factory(**kwargs):
            factory_calls.append(kwargs)
            raise AssertionError("judge factory must not run")

        result = evaluate_with_openevals_judge(
            scenario,
            record,
            "mentor_pedagogy",
            judge=object(),
            judge_provider="fixture",
            judge_model="fixture-model",
            evaluator_factory=fake_factory,
        )

        self.assertEqual("skipped", result.status)
        self.assertFalse(result.provider_call_attempted)
        self.assertEqual("PrerequisiteUnavailable", result.error_type)
        self.assertIn("required final worker 'mentor'", result.reason)
        self.assertEqual([], factory_calls)

    def test_live_graph_observation_uses_captured_artifact_labels(self):
        artifacts = [
            {
                "data": [
                    {
                        "mode": "markers",
                        "hovertext": ["Mamba → IS_A → Sequence model"],
                    },
                    {
                        "mode": "markers+text",
                        "text": ["Mamba", "Sequence model"],
                    },
                ]
            }
        ]

        graph = _live_graph_observation(
            (
                ["concept_mamba", "concept_sequence_model"],
                [("concept_mamba", "IS_A", "concept_sequence_model")],
                ["Mamba"],
            ),
            artifacts,
        )

        self.assertEqual("captured", graph.capture_status)
        self.assertEqual({"Mamba", "Sequence model"}, set(graph.node_labels.values()))
        self.assertEqual([("Mamba", "IS_A", "Sequence model")], graph.labeled_edges)

    def test_live_cli_requires_explicit_provider_confirmation(self):
        self.assertEqual(
            2,
            main(["live", "--case-id", "out_of_scope_biological_cats"]),
        )

    def test_judge_cli_requires_explicit_provider_confirmation(self):
        self.assertEqual(
            2,
            main(
                [
                    "judge",
                    "--records",
                    "does-not-need-to-exist.jsonl",
                    "--case-id",
                    "local_transformer_answer_success",
                    "--dimension",
                    "analyst_usefulness",
                ]
            ),
        )

    def test_live_cli_caps_initial_repetitions_before_provider_calls(self):
        self.assertEqual(
            2,
            main(
                [
                    "live",
                    "--case-id",
                    "out_of_scope_biological_cats",
                    "--repetitions",
                    "4",
                    "--confirm-provider-calls",
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
