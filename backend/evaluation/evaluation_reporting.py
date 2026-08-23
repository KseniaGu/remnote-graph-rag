"""Shared immutable history and scorecards for Graph RAG evaluation runs."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import statistics
import subprocess
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVALUATION_ROOT = ROOT / "data" / "evaluation"
DEFAULT_REPORTS_ROOT = ROOT / "reports" / "evaluation"
REPORT_SCHEMA_VERSION = "graph-rag-evaluation"
DETERMINISTIC_RUN_KINDS = {"retrieval", "offline"}
RUNTIME_RUN_KINDS = {"offline", "live"}
HASHED_RELATIVE_PATHS = (
    "pyproject.toml",
    "uv.lock",
    "evals/runtime/scenarios.json",
    "evals/retrieval/benchmark_cases.jsonl",
    "scripts/evaluate_runtime_workflow.py",
    "scripts/evaluate_retrieval_pipeline.py",
    "scripts/build_evaluation_report.py",
)
HASHED_TREE_RELATIVE_PATHS = ("backend",)
IGNORED_IDENTITY_PARTS = {
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "__pycache__",
    "reports",
}
EXPECTED_SEMANTIC_DIMENSIONS = (
    "claim_faithfulness",
    "analyst_usefulness",
    "mentor_pedagogy",
    "conversational_continuity",
    "graph_usefulness",
)
REQUIRED_LIVE_REPETITIONS = {"researcher_structured_output_truncation_reliability": 3}


def is_evaluation_report_schema(value: Any) -> bool:
    """Checks the current schema name and compatible immutable predecessors."""
    observed = str(value or "")
    version_prefix = f"{REPORT_SCHEMA_VERSION}-v"
    return observed == REPORT_SCHEMA_VERSION or (
        observed.startswith(version_prefix)
        and observed.removeprefix(version_prefix).isdigit()
    )


SCORECARD_METRIC_DEFINITIONS = {
    "Retrieval quality": {
        "case_pass_rate": "Share of retrieval cases satisfying every applicable benchmark contract.",
        "evidence_chunk_recall": "Required source chunk IDs returned divided by required source chunk IDs.",
        "concept_recall": "Required graph concepts returned divided by required graph concepts.",
        "relation_recall": "Required graph relations returned divided by required graph relations.",
        "forbidden_evidence_count/rate": "Observed regression-backed forbidden evidence items and the share of cases containing any.",
        "retrieval_adequacy_rate/error_rate": "Share reported adequate and mean retrieval error count across benchmark cases.",
        "dangling_edge_count/chunk_node_count": "Structural graph diagnostics for dangling edges and leaked chunk nodes.",
        "context_precision_at_10": "Weighted reviewed relevance in the first ten ranked Analyst source slots divided by ten.",
        "context_recall": "Reviewed required answer points supported by returned Analyst evidence divided by all required answer points.",
        "concept/relation diagnostics": "Separate ID, label, and relation-spec recall used to explain aggregate misses.",
    },
    "Runtime task behavior": {
        "task_success_rate": "Runs passing every applicable gating contract divided by evaluated runs.",
        "routing_correctness": "Share of applicable runs whose worker sequence matches an allowed route.",
        "required/forbidden_agent_compliance": "Share of applicable runs containing every required agent or avoiding every forbidden agent.",
        "path_efficiency_ratio": "Shortest allowed completed worker path divided by actual worker steps, capped at one; incomplete runs are N/A.",
        "tool_selection_correctness": "Share of applicable runs satisfying required and forbidden tool contracts.",
        "tool_argument_validity": "Share of observed required tool calls satisfying every argument constraint.",
        "one_tool_per_worker": "Share of applicable worker steps that comply with the one-tool boundary.",
        "retrieval_status_correctness": "Share of applicable runs with an allowed local retrieval status and outcome.",
        "local_to_web_fallback_correctness": "Share of applicable runs that escalate, or avoid escalation, according to the scenario contract.",
        "unnecessary_web_rate": "Tavily-using runs divided by scenarios that explicitly forbid web use.",
        "source_exhaustion_correctness": "Share of applicable runs whose source-exhaustion state matches the contract.",
        "final_response/modality/termination rates": "Shares satisfying final-response presence, requested output modality, and workflow termination contracts.",
        "graph_contract_rate": "Share of graph-applicable runs passing all structural and required-anchor checks.",
    },
    "Reliability": {
        "repetition_count": "Number of runtime observations contributing to this fingerprint.",
        "pass_rate and Wilson interval": "Functional pass proportion with sample count represented by a 95% Wilson interval.",
        "route_consistency": "Per repeated scenario, fraction following its most common worker route.",
        "evidence_set_jaccard_stability": "Mean pairwise Jaccard overlap of evidence IDs for repetitions of the same scenario.",
        "looping_rate": "Runs with an identical worker/action signature repeated after Orchestrator return divided by applicable runs.",
        "output_limit_hit_rate": "Runs observing a provider output-limit stop or classified truncation divided by runtime runs.",
        "failure_type_frequency": "Counts of classified provider, parser, tool, storage, timeout, truncation, recursion, and workflow failures.",
    },
    "Efficiency": {
        "worker_steps": "Distribution of non-Orchestrator worker executions per run.",
        "logical_llm_calls": "Distribution of logical model operations before provider-level retries.",
        "provider_attempts": "Distribution of actual provider attempts, including retries.",
        "retries/retry_rate": "Retry counts and total retries divided by total provider attempts.",
        "tavily_searches": "Distribution of Tavily tool calls per run.",
        "input/output/total_tokens": "Provider-reported token distributions; unavailable observations are omitted, not zero-filled.",
        "tokens_per_successful_run": "Total tokens across evaluated runs divided by functionally successful runs; N/A when none succeed.",
        "latency_seconds": "End-to-end time-to-resolution distribution with mean, median, and p95.",
        "gating": "False means efficiency limits are reported but do not initially fail functional task success.",
    },
    "Optional semantic quality": {
        "claim_faithfulness": "Counts supported, partially supported, and unsupported factual claims against captured bounded evidence.",
        "grounded_claim_rate": "Supported claims plus half-weighted partial claims divided by all classified factual claims.",
        "analyst_usefulness": "Boolean judge rate for direct, clear, non-substitutive Analyst answers.",
        "mentor_pedagogy": "Boolean judge rate for learner-aware hints and appropriate instructional next steps.",
        "conversational_continuity": "Boolean judge rate for preserving topic and interaction mode from recent state.",
        "graph_usefulness": "Boolean judge rate for relevant, useful labeled graph relationships.",
        "judge_execution_diagnostics": "Per-dimension result, success, error, and actual provider-attempt counts with privacy-safe failure reasons.",
    },
}


def allocate_run_dir(run_kind: str, root: Path = DEFAULT_EVALUATION_ROOT) -> Path:
    """Allocates a unique run directory without overwriting history."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_{run_kind}_{uuid.uuid4().hex[:8]}"
    return root / "runs" / run_id


def register_completed_run(
    output_dir: Path,
    *,
    run_kind: str,
    invocation: dict[str, Any],
    configuration: dict[str, Any] | None = None,
    evaluation_root: Path = DEFAULT_EVALUATION_ROOT,
) -> dict[str, Any]:
    """Finalizes a run manifest and refreshes its fingerprint scorecard."""
    manifest_path = output_dir / "manifest.json"
    existing = _read_json(manifest_path) if manifest_path.exists() else {}
    snapshot = build_evaluation_snapshot(configuration or {})
    fingerprint = _hash_json(snapshot["fingerprint_input"])
    invocation_key = _hash_json(
        {
            "run_kind": run_kind,
            "invocation": invocation,
            "fingerprint": fingerprint,
        }
    )
    created_at = str(existing.get("created_at") or datetime.now(UTC).isoformat())
    manifest = {
        **existing,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "run_id": output_dir.name,
        "run_kind": run_kind,
        "status": "completed",
        "created_at": created_at,
        "evaluation_fingerprint": fingerprint,
        "invocation_key": invocation_key,
        "invocation": _redact(invocation),
        "configuration_snapshot": snapshot,
        "artifacts": sorted(
            path.name for path in output_dir.iterdir() if path.is_file()
        ),
    }
    _write_json(manifest_path, manifest)
    build_evaluation_reports(evaluation_root)
    return manifest


def register_failed_run(
    output_dir: Path,
    *,
    run_kind: str,
    invocation: dict[str, Any],
    error: str,
    configuration: dict[str, Any] | None = None,
    evaluation_root: Path = DEFAULT_EVALUATION_ROOT,
) -> dict[str, Any]:
    """Records an inspectable failed run without replacing completed snapshots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = build_evaluation_snapshot(configuration or {})
    fingerprint = _hash_json(snapshot["fingerprint_input"])
    manifest = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "run_id": output_dir.name,
        "run_kind": run_kind,
        "status": "failed",
        "created_at": datetime.now(UTC).isoformat(),
        "evaluation_fingerprint": fingerprint,
        "invocation_key": _hash_json(
            {"run_kind": run_kind, "invocation": invocation, "fingerprint": fingerprint}
        ),
        "invocation": _redact(invocation),
        "configuration_snapshot": snapshot,
        "error": _safe_error(error),
        "artifacts": sorted(
            path.name for path in output_dir.iterdir() if path.is_file()
        ),
    }
    _write_json(output_dir / "manifest.json", manifest)
    build_evaluation_reports(evaluation_root)
    return manifest


def build_evaluation_snapshot(configuration: dict[str, Any]) -> dict[str, Any]:
    """Builds a stable, secret-free fingerprint input."""
    revision, _ = _git_revision()
    prompt_root = ROOT / "backend" / "llm" / "prompts"
    storage_manifests = sorted((ROOT / "storage").rglob("*manifest*.json"))
    from backend.configs.models import get_model_settings
    from backend.configs.search import KnowledgeGraphSearchSettings
    from backend.configs.storage import StorageSettings

    application_configuration = {
        "models": get_model_settings().model_dump(mode="json"),
        "search": KnowledgeGraphSearchSettings().model_dump(mode="json"),
        "storage": StorageSettings().model_dump(mode="json"),
    }
    fingerprint_input = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "file_hashes": {
            relative: _hash_file(ROOT / relative)
            for relative in HASHED_RELATIVE_PATHS
            if (ROOT / relative).exists()
        },
        "tree_hashes": {
            relative: _hash_tree(ROOT / relative)
            for relative in HASHED_TREE_RELATIVE_PATHS
        },
        "prompt_tree_hash": _hash_tree(prompt_root),
        "storage_manifest_hashes": {
            str(path.relative_to(ROOT)): _hash_file(path) for path in storage_manifests
        },
        "application_configuration": _redact(application_configuration),
    }
    return {
        "fingerprint_input": fingerprint_input,
        "repository_state": {
            "revision": revision,
            "reported_source_revision": configuration.get("source_revision"),
        },
        "executor_configuration": _redact(configuration),
    }


def current_evaluation_fingerprint(
    configuration: dict[str, Any] | None = None,
) -> str:
    """Returns the fingerprint for the current repository and effective settings."""
    snapshot = build_evaluation_snapshot(configuration or {})
    return _hash_json(snapshot["fingerprint_input"])


def build_evaluation_reports(
    evaluation_root: Path = DEFAULT_EVALUATION_ROOT,
    *,
    fingerprint: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Rebuilds per-fingerprint scorecards from evaluation run manifests."""
    run_root = evaluation_root / "runs"
    manifests = []
    if run_root.exists():
        for path in sorted(run_root.glob("*/manifest.json")):
            manifest = _read_json(path)
            if is_evaluation_report_schema(manifest.get("report_schema_version")):
                manifest["_run_dir"] = str(path.parent)
                manifests.append(manifest)

    by_fingerprint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for manifest in manifests:
        key = str(manifest.get("evaluation_fingerprint") or "")
        if key and (fingerprint is None or key == fingerprint):
            by_fingerprint[key].append(manifest)

    reports: dict[str, dict[str, Any]] = {}
    latest_key: str | None = None
    latest_time = ""
    for key, items in by_fingerprint.items():
        selected = _select_current_runs(items)
        scorecard = _build_scorecard(key, items, selected)
        destination = evaluation_root / "fingerprints" / key
        destination.mkdir(parents=True, exist_ok=True)
        _write_json(destination / "scorecard.json", scorecard)
        (destination / "scorecard.md").write_text(
            _render_scorecard(scorecard), encoding="utf-8"
        )
        _write_json(
            destination / "history.json",
            {
                "fingerprint": key,
                "runs": [
                    _history_item(item)
                    for item in sorted(
                        items, key=lambda value: str(value.get("created_at") or "")
                    )
                ],
            },
        )
        reports[key] = scorecard
        completed_times = [
            str(item.get("created_at") or "")
            for item in items
            if item.get("status") == "completed"
        ]
        if completed_times:
            newest = max(completed_times)
            if newest >= latest_time:
                latest_time = newest
                latest_key = key

    if latest_key is not None and fingerprint is None:
        latest = reports[latest_key]
        _write_json(evaluation_root / "latest.json", latest)
        (evaluation_root / "latest.md").write_text(
            _render_scorecard(latest), encoding="utf-8"
        )
    return reports


def publish_evaluation_report(
    evaluation_root: Path = DEFAULT_EVALUATION_ROOT,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    *,
    fingerprint: str | None = None,
    allow_incomplete: bool = False,
    confirm_complete: bool = False,
) -> dict[str, Path | str | int]:
    """Publishes one sanitized, commit-friendly scorecard snapshot.

    Raw run artifacts remain under the ignored evaluation root. Publication copies
    only the selected scorecard and a bounded provenance summary, keeping the
    repository report inspectable without exposing captured evidence or traces.
    """
    if fingerprint is None and not (allow_incomplete or confirm_complete):
        raise ValueError(
            "Publication requires --fingerprint, --confirm-complete, or "
            "--allow-incomplete; newest-fingerprint publication is not implicit."
        )
    reports = build_evaluation_reports(evaluation_root, fingerprint=fingerprint)
    selected_fingerprint = fingerprint or _latest_report_fingerprint(evaluation_root)
    if not selected_fingerprint or selected_fingerprint not in reports:
        raise ValueError("No completed evaluation fingerprint is available to publish.")

    scorecard = reports[selected_fingerprint]
    if not scorecard.get("selected_run_count"):
        raise ValueError("The selected fingerprint has no completed contributing runs.")
    complete = bool(scorecard.get("coverage", {}).get("complete"))
    if confirm_complete and not complete:
        missing = scorecard.get("coverage", {}).get("missing_requirements", [])
        raise ValueError(
            "The selected fingerprint is incomplete: " + ", ".join(map(str, missing))
        )
    if fingerprint is None and not allow_incomplete and not complete:
        raise ValueError(
            "The newest fingerprint is incomplete; use --fingerprint or "
            "--allow-incomplete."
        )

    source_dir = evaluation_root / "fingerprints" / selected_fingerprint
    created_dates = _fingerprint_run_dates(source_dir / "history.json")
    snapshot_date = (
        min(created_dates).date().isoformat()
        if created_dates
        else datetime.now(UTC).date().isoformat()
    )
    snapshot_name = f"{snapshot_date}_{selected_fingerprint[:12]}"
    history_dir = reports_root / "history" / snapshot_name
    history_dir.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(source_dir / "scorecard.md", history_dir / "scorecard.md")
    shutil.copyfile(source_dir / "scorecard.json", history_dir / "scorecard.json")
    shutil.copyfile(source_dir / "scorecard.md", reports_root / "latest.md")
    shutil.copyfile(source_dir / "scorecard.json", reports_root / "latest.json")

    provenance = _publication_provenance(
        evaluation_root,
        scorecard,
        snapshot_name=snapshot_name,
    )
    _write_json(history_dir / "provenance.json", provenance)
    _write_json(reports_root / "latest_provenance.json", provenance)
    _write_publication_index(reports_root)
    return {
        "fingerprint": selected_fingerprint,
        "snapshot": snapshot_name,
        "contributing_run_count": len(scorecard.get("contributing_run_ids", [])),
        "latest_markdown": reports_root / "latest.md",
        "history_directory": history_dir,
    }


def _select_current_runs(
    manifests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    deterministic: dict[str, dict[str, Any]] = {}
    accumulated: list[dict[str, Any]] = []
    for manifest in sorted(
        manifests, key=lambda value: str(value.get("created_at") or "")
    ):
        if manifest.get("status") != "completed":
            continue
        if manifest.get("run_kind") in DETERMINISTIC_RUN_KINDS:
            deterministic[str(manifest.get("invocation_key"))] = manifest
        else:
            accumulated.append(manifest)
    return [*deterministic.values(), *accumulated]


def _build_scorecard(
    fingerprint: str,
    history: list[dict[str, Any]],
    selected: list[dict[str, Any]],
) -> dict[str, Any]:
    retrieval_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    runtime_records: list[dict[str, Any]] = []
    judge_rows: list[dict[str, Any]] = []
    framework_rows: list[dict[str, Any]] = []

    for manifest in selected:
        run_dir = Path(str(manifest["_run_dir"]))
        kind = manifest.get("run_kind")
        run_id = str(manifest.get("run_id"))
        if kind == "retrieval":
            retrieval_rows.extend(
                _tag_source_run(
                    _read_jsonl(run_dir / "case_results.jsonl"), run_id, "retrieval"
                )
            )
        elif kind in RUNTIME_RUN_KINDS:
            runtime_rows.extend(
                _tag_source_run(
                    _read_jsonl(run_dir / "case_results.jsonl"), run_id, str(kind)
                )
            )
            runtime_records.extend(
                _tag_source_run(
                    _read_jsonl(run_dir / "runtime_records.jsonl"), run_id, str(kind)
                )
            )
            framework = _read_json(run_dir / "framework_results.json")
            if isinstance(framework, list):
                framework_rows.extend(_tag_source_run(framework, run_id, str(kind)))
        elif kind == "judge":
            judge_rows.extend(
                _tag_source_run(
                    _read_jsonl(run_dir / "judge_results.jsonl"), run_id, "judge"
                )
            )

    semantic_quality = _semantic_metrics(judge_rows)
    coverage = _coverage_metrics(
        selected, retrieval_rows, runtime_records, judge_rows, framework_rows
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "generated_at": datetime.now(UTC).isoformat(),
        "history_run_count": len(history),
        "selected_run_count": len(selected),
        "contributing_run_ids": [str(item.get("run_id")) for item in selected],
        "metric_definitions": SCORECARD_METRIC_DEFINITIONS,
        "retrieval_quality": _retrieval_metrics(retrieval_rows),
        "runtime_task_behavior": _runtime_task_metrics(runtime_rows),
        "reliability": _reliability_metrics(runtime_rows, runtime_records),
        "efficiency": _efficiency_metrics(runtime_rows, runtime_records),
        "optional_semantic_quality": semantic_quality,
        "coverage": coverage,
        "diagnostics": {
            "failed_runtime_checks": _failed_checks(runtime_rows),
            "failure_type_frequency": dict(
                Counter(
                    failure
                    for row in runtime_rows
                    for failure in row.get("failure_types", [])
                )
            ),
            "runtime_mode_breakdown": _runtime_mode_breakdown(runtime_rows),
            "provenance": _provenance_diagnostics(selected, runtime_records),
            "agentevals": framework_rows,
            "run_artifacts": [
                {
                    "run_id": str(item.get("run_id")),
                    "run_kind": str(item.get("run_kind")),
                    "run_directory": _display_path(Path(str(item.get("_run_dir")))),
                    "artifacts": item.get("artifacts", []),
                }
                for item in selected
            ],
        },
    }


def _retrieval_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "N/A", "reason": "no retrieval run for this fingerprint"}
    scores = [row.get("scores", {}) for row in rows]
    return {
        "source_run_ids": _source_run_ids(rows),
        "case_count": len(rows),
        "case_pass_rate": _mean([bool(row.get("passed")) for row in rows]),
        "evidence_chunk_recall": _metric_mean(scores, "evidence_chunk_recall"),
        "concept_recall": _metric_mean(scores, "concept_recall"),
        "relation_recall": _metric_mean(scores, "relation_recall"),
        "forbidden_evidence_count": _metric_sum(scores, "forbidden_evidence_count"),
        "forbidden_evidence_rate": _mean(
            [float(score.get("forbidden_evidence_count", 0)) > 0 for score in scores]
        ),
        "retrieval_adequacy_rate": _metric_mean(scores, "retrieval_adequate"),
        "retrieval_error_rate": _metric_mean(scores, "retrieval_error_count"),
        "dangling_edge_count": _metric_sum(scores, "dangling_edge_count"),
        "chunk_node_count": _metric_sum(scores, "chunk_node_count"),
        "context_precision_at_10": _optional_metric_mean(
            scores,
            "context_precision_at_10",
            "relevance labels not reviewed",
        ),
        "context_recall": _optional_metric_mean(
            scores,
            "context_recall",
            "relevance labels not reviewed",
        ),
        "diagnostics": {
            name: _metric_mean([row.get("diagnostics", {}) for row in rows], name)
            for name in (
                "concept_id_recall",
                "concept_label_recall",
                "relation_id_recall",
                "relation_spec_recall",
            )
        },
    }


def _runtime_task_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "N/A", "reason": "no runtime run for this fingerprint"}
    checks = [check for row in rows for check in row.get("checks", [])]
    metrics = [row.get("metrics", {}) for row in rows]
    return {
        "source_run_ids": _source_run_ids(rows),
        "run_count": len(rows),
        "sample_counts": {
            "offline": sum(row.get("_source_run_kind") == "offline" for row in rows),
            "live": sum(row.get("_source_run_kind") == "live" for row in rows),
        },
        "task_success_rate": _mean([bool(row.get("passed")) for row in rows]),
        "routing_correctness": _check_rate(checks, "routing", "worker_sequence"),
        "required_agent_compliance": _check_rate(checks, "routing", "required_agents"),
        "forbidden_agent_compliance": _check_rate(
            checks, "routing", "forbidden_agents"
        ),
        "path_efficiency_ratio": _values_summary(
            _numeric(metrics, "path_efficiency_ratio")
        ),
        "tool_selection_correctness": _mean_optional(
            [item.get("tool_selection_correct") for item in metrics]
        ),
        "tool_argument_validity": _mean_optional(
            [item.get("tool_argument_valid") for item in metrics]
        ),
        "one_tool_per_worker": _check_rate(checks, "tools", "one_tool_per_worker"),
        "retrieval_status_correctness": _check_rate(
            checks, "retrieval", "retrieval_status"
        ),
        "local_to_web_fallback_correctness": _mean_optional(
            [item.get("fallback_correct") for item in metrics]
        ),
        "source_exhaustion_correctness": _check_rate(
            checks, "fallback", "sources_exhausted"
        ),
        "unnecessary_web_rate": _mean_optional(
            [item.get("unnecessary_web") for item in metrics]
        ),
        "final_response_rate": _check_rate(checks, "answer", "final_response_present"),
        "modality_correctness": _check_rate(checks, "modality", "requested_modality"),
        "termination_rate": _check_rate(checks, "termination", "terminated"),
        "graph_contract_rate": _run_dimension_rate(rows, "graph"),
    }


def _reliability_metrics(
    rows: list[dict[str, Any]], records: list[dict[str, Any]]
) -> dict[str, Any]:
    if not rows:
        return {"status": "N/A", "reason": "no runtime run for this fingerprint"}
    passed = sum(bool(row.get("passed")) for row in rows)
    records_by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_scenario[str(record.get("scenario_id"))].append(record)

    route_consistency: dict[str, float] = {}
    evidence_stability: dict[str, float] = {}
    for scenario_id, scenario_records in records_by_scenario.items():
        if len(scenario_records) < 2:
            continue
        routes = [
            tuple(record.get("worker_sequence", [])) for record in scenario_records
        ]
        route_consistency[scenario_id] = max(Counter(routes).values()) / len(routes)
        evidence = [set(record.get("evidence_ids", [])) for record in scenario_records]
        overlaps = [
            len(left & right) / len(left | right) if left or right else 1.0
            for index, left in enumerate(evidence)
            for right in evidence[index + 1 :]
        ]
        if overlaps:
            evidence_stability[scenario_id] = statistics.fmean(overlaps)

    return {
        "source_run_ids": _source_run_ids(rows),
        "repetition_count": len(rows),
        "pass_rate": passed / len(rows),
        "pass_rate_wilson_95": list(_wilson(passed, len(rows))),
        "route_consistency": (
            {
                "mean": statistics.fmean(route_consistency.values()),
                "by_scenario": route_consistency,
            }
            if route_consistency
            else {"status": "N/A", "reason": "no repeated scenario runs"}
        ),
        "evidence_set_jaccard_stability": (
            {
                "mean": statistics.fmean(evidence_stability.values()),
                "by_scenario": evidence_stability,
            }
            if evidence_stability
            else {"status": "N/A", "reason": "no repeated scenario pairs"}
        ),
        "looping_rate": _mean(
            [bool(row.get("metrics", {}).get("loop_detected")) for row in rows]
        ),
        "output_limit_hit_rate": _mean(
            [bool(row.get("metrics", {}).get("output_limit_hit")) for row in rows]
        ),
        "failure_type_frequency": dict(
            Counter(failure for row in rows for failure in row.get("failure_types", []))
        ),
    }


def _efficiency_metrics(
    rows: list[dict[str, Any]], records: list[dict[str, Any]]
) -> dict[str, Any]:
    if not rows:
        return {"status": "N/A", "reason": "no runtime run for this fingerprint"}
    metrics = [row.get("metrics", {}) for row in rows]
    successful_count = sum(bool(row.get("passed")) for row in rows)
    all_tokens = sum(_numeric(metrics, "total_tokens"))
    attempts = sum(_numeric(metrics, "provider_attempts"))
    retries = sum(_numeric(metrics, "retries"))
    return (
        {
            "source_run_ids": _source_run_ids(rows),
        }
        | {
            name: _values_summary(_numeric(metrics, name))
            for name in (
                "worker_steps",
                "logical_llm_calls",
                "provider_attempts",
                "retries",
                "tavily_searches",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "latency_seconds",
            )
        }
        | {
            "retry_rate": retries / attempts if attempts else None,
            "tokens_per_successful_run": (
                all_tokens / successful_count if successful_count else None
            ),
            "gating": False,
        }
    )


def _coverage_metrics(
    selected: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
    runtime_records: list[dict[str, Any]],
    judge_rows: list[dict[str, Any]],
    framework_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Accounts for every required full-campaign contribution."""
    contract = _scenario_campaign_contract()
    invocations = [
        item.get("invocation") if isinstance(item.get("invocation"), dict) else {}
        for item in selected
    ]
    retrieval_baseline = any(
        item.get("run_kind") == "retrieval"
        and invocation.get("mode") == "both"
        and invocation.get("analyst_variant") == "legacy_vector_context"
        and invocation.get("visualizer_variant") == "optimized"
        for item, invocation in zip(selected, invocations, strict=True)
    )
    analyst_rows = [row for row in retrieval_rows if row.get("mode") == "analyst"]
    relevance_complete = bool(analyst_rows) and all(
        isinstance(row.get("scores", {}).get(metric), (int, float))
        for row in analyst_rows
        for metric in ("context_precision_at_10", "context_recall")
    )

    offline_counts = Counter(
        str(row.get("scenario_id"))
        for row in runtime_records
        if row.get("_source_run_kind") == "offline"
    )
    live_counts = Counter(
        str(row.get("scenario_id"))
        for row in runtime_records
        if row.get("_source_run_kind") == "live"
    )
    expected_offline = {
        scenario_id: values["offline_repetitions"]
        for scenario_id, values in contract.items()
        if values["offline_repetitions"] > 0
    }
    expected_live = {
        scenario_id: values["live_repetitions"]
        for scenario_id, values in contract.items()
    }
    offline_complete = bool(expected_offline) and all(
        offline_counts[scenario_id] >= count
        for scenario_id, count in expected_offline.items()
    )
    live_complete = bool(expected_live) and all(
        live_counts[scenario_id] >= count
        for scenario_id, count in expected_live.items()
    )

    semantic_coverage = {}
    for dimension in EXPECTED_SEMANTIC_DIMENSIONS:
        items = [row for row in judge_rows if row.get("dimension") == dimension]
        semantic_coverage[dimension] = {
            "result_count": len(items),
            "successful_count": sum(item.get("status") == "success" for item in items),
            "skipped_count": sum(item.get("status") == "skipped" for item in items),
            "error_count": sum(item.get("status") == "error" for item in items),
            "provider_attempted_count": sum(
                bool(item.get("provider_call_attempted")) for item in items
            ),
            "accounted": bool(items),
        }
    semantic_complete = all(item["accounted"] for item in semantic_coverage.values())

    agentevals_requested = any(
        invocation.get("framework") == "agentevals" for invocation in invocations
    )
    provenance_complete = all(
        isinstance(item.get("invocation"), dict)
        and isinstance(item.get("configuration_snapshot"), dict)
        for item in selected
    ) and bool(selected)
    requirements = [
        _coverage_requirement(
            "retrieval_baseline",
            retrieval_baseline,
            bool(retrieval_rows),
            "mode=both; Analyst=legacy_vector_context; Visualizer=optimized",
        ),
        _coverage_requirement(
            "retrieval_relevance_labels",
            relevance_complete,
            len(analyst_rows),
            "reviewed Context Precision@10 and Context Recall for every Analyst case",
        ),
        _coverage_requirement(
            "offline_runtime",
            offline_complete,
            dict(offline_counts),
            expected_offline,
        ),
        _coverage_requirement(
            "reviewed_live_suite",
            live_complete,
            dict(live_counts),
            expected_live,
        ),
        _coverage_requirement(
            "semantic_dimensions",
            semantic_complete,
            semantic_coverage,
            list(EXPECTED_SEMANTIC_DIMENSIONS),
        ),
        _coverage_requirement(
            "agentevals_when_requested",
            not agentevals_requested or bool(framework_rows),
            len(framework_rows),
            "results required only when --framework agentevals was requested",
        ),
        _coverage_requirement(
            "invocation_and_configuration_provenance",
            provenance_complete,
            len(selected),
            "every contributing run has invocation and configuration snapshots",
        ),
    ]
    missing = [item["id"] for item in requirements if not item["complete"]]
    return {
        "complete": not missing,
        "missing_requirements": missing,
        "requirements": requirements,
        "semantic_dimension_coverage": semantic_coverage,
        "offline_sample_count": sum(offline_counts.values()),
        "live_sample_count": sum(live_counts.values()),
        "agentevals_requested": agentevals_requested,
        "agentevals_result_count": len(framework_rows),
    }


def _coverage_requirement(
    identifier: str, complete: bool, observed: Any, expected: Any
) -> dict[str, Any]:
    return {
        "id": identifier,
        "complete": complete,
        "observed": observed,
        "expected": expected,
    }


def _scenario_campaign_contract() -> dict[str, dict[str, int]]:
    payload = _read_json(ROOT / "evals" / "runtime" / "scenarios.json")
    if not isinstance(payload, list):
        return {}
    return {
        str(item["id"]): {
            "offline_repetitions": len(item.get("trace_ids", [])),
            "live_repetitions": REQUIRED_LIVE_REPETITIONS.get(str(item["id"]), 1),
        }
        for item in payload
        if isinstance(item, dict)
        and item.get("id")
        and item.get("review_status") == "reviewed"
    }


def _provenance_diagnostics(
    selected: list[dict[str, Any]], records: list[dict[str, Any]]
) -> dict[str, Any]:
    offline = [
        record for record in records if record.get("_source_run_kind") == "offline"
    ]
    return {
        "manifest_count": len(selected),
        "invocation_present_count": sum(
            isinstance(item.get("invocation"), dict) for item in selected
        ),
        "configuration_snapshot_present_count": sum(
            isinstance(item.get("configuration_snapshot"), dict) for item in selected
        ),
        "runtime_source_mode_counts": dict(
            Counter(str(item.get("_source_run_kind")) for item in records)
        ),
        "offline_trace_configuration_status_counts": dict(
            Counter(
                str(
                    item.get("provenance", {}).get("source_configuration_status")
                    or "needs verification"
                )
                for item in offline
            )
        ),
        "offline_runs_needing_configuration_verification": sorted(
            {
                str(item.get("_source_run_id"))
                for item in offline
                if item.get("provenance", {}).get("source_configuration_status")
                != "verified"
            }
        ),
    }


def _runtime_mode_breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for run_kind in ("offline", "live"):
        selected = [row for row in rows if row.get("_source_run_kind") == run_kind]
        if not selected:
            output[run_kind] = {"sample_count": 0, "passed_count": 0, "failed_count": 0}
            continue
        failures = Counter(
            failure for row in selected for failure in row.get("failure_types", [])
        )
        passed = sum(bool(row.get("passed")) for row in selected)
        output[run_kind] = {
            "sample_count": len(selected),
            "passed_count": passed,
            "failed_count": len(selected) - passed,
            "failure_type_frequency": dict(failures),
        }
    return output


def _public_judge_reason(item: dict[str, Any]) -> str:
    if item.get("status") == "error":
        parser = str(item.get("parser_classification") or "not_recorded")
        error_type = str(item.get("error_type") or "not_recorded")
        causes = ", ".join(map(str, item.get("error_cause_types", []))) or "none"
        return (
            f"parser={parser}; error_type={error_type}; "
            f"cause_types={causes}; raw output omitted"
        )
    return _safe_error(str(item.get("reason") or ""))


def _semantic_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_identity: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        identity = (
            str(row.get("dimension")),
            str(row.get("judge_provider")),
            str(row.get("judge_model")),
            str(row.get("rubric_version")),
        )
        by_identity[identity].append(row)
    output: dict[str, Any] = {
        "source_run_ids": _source_run_ids(rows),
        "dimensions": {},
    }
    for identity, items in sorted(by_identity.items()):
        dimension, provider, model, rubric = identity
        metric_key = f"{dimension} [{provider}/{model}; {rubric}]"
        successful = [item for item in items if item.get("status") == "success"]
        skipped = [item for item in items if item.get("status") == "skipped"]
        errors = [item for item in items if item.get("status") == "error"]
        common = {
            "result_count": len(items),
            "successful_result_count": len(successful),
            "skipped_result_count": len(skipped),
            "error_count": len(errors),
            "provider_call_attempt_count": sum(
                bool(item.get("provider_call_attempted")) for item in items
            ),
            "skipped_reasons": sorted({_public_judge_reason(item) for item in skipped}),
            "status": "available" if successful else "N/A",
            "error_reasons": sorted({_public_judge_reason(item) for item in errors}),
            "confirmed_truncation_count": sum(
                bool(item.get("confirmed_truncation")) for item in items
            ),
        }
        if dimension == "claim_faithfulness":
            supported = sum(
                int(item.get("supported_claims") or 0) for item in successful
            )
            partial = sum(int(item.get("partial_claims") or 0) for item in successful)
            unsupported = sum(
                int(item.get("unsupported_claims") or 0) for item in successful
            )
            total = supported + partial + unsupported
            output[metric_key] = {
                "dimension": dimension,
                "judge_provider": provider,
                "judge_model": model,
                "rubric_version": rubric,
                **common,
                "supported_claims": supported,
                "partial_claims": partial,
                "unsupported_claims": unsupported,
                "grounded_claim_rate": (
                    (supported + 0.5 * partial) / total if total else None
                ),
                "sample_count": len(successful),
            }
        else:
            values = [
                bool(item.get("score"))
                for item in successful
                if item.get("score") is not None
            ]
            output[metric_key] = {
                "dimension": dimension,
                "judge_provider": provider,
                "judge_model": model,
                "rubric_version": rubric,
                **common,
                "pass_rate": _mean_optional(values),
                "sample_count": len(values),
            }
    for dimension in EXPECTED_SEMANTIC_DIMENSIONS:
        items = [row for row in rows if row.get("dimension") == dimension]
        successful = [item for item in items if item.get("status") == "success"]
        skipped = [item for item in items if item.get("status") == "skipped"]
        errors = [item for item in items if item.get("status") == "error"]
        if successful:
            reason = ""
        elif errors:
            reason = "all attempted scores unavailable because judge execution failed"
        elif skipped:
            reason = "semantic prerequisites were unavailable"
        else:
            reason = "dimension was not run for this fingerprint"
        output["dimensions"][dimension] = {
            "status": "available" if successful else "N/A",
            "reason": reason,
            "result_count": len(items),
            "successful_count": len(successful),
            "skipped_count": len(skipped),
            "error_count": len(errors),
            "provider_attempted_count": sum(
                bool(item.get("provider_call_attempted")) for item in items
            ),
            "parser_classification_counts": dict(
                Counter(
                    str(item.get("parser_classification") or "not_recorded")
                    for item in items
                )
            ),
            "confirmed_truncation_count": sum(
                bool(item.get("confirmed_truncation")) for item in items
            ),
        }
    return output


_METRIC_LABELS = {
    "agentevals": "AgentEvals",
    "case_count": "Cases",
    "concept_id_recall": "Concept ID recall",
    "context_precision_at_10": "Context Precision@10",
    "context_recall": "Context Recall",
    "evidence_set_jaccard_stability": "Evidence-set Jaccard stability",
    "failed_runtime_checks": "Failed runtime checks",
    "pass_rate_wilson_95": "95% confidence interval",
    "provider_attempted_count": "Provider attempts",
    "route_consistency": "Route consistency",
    "run_count": "Runs",
    "sample_count": "Samples",
    "source_run_ids": "Source runs",
}


def _markdown_text(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _metric_label(name: str) -> str:
    if name in _METRIC_LABELS:
        return _METRIC_LABELS[name]
    label = name.replace("_", " ")
    for original, replacement in (
        ("llm", "LLM"),
        ("id", "ID"),
        ("p95", "P95"),
        ("tavily", "Tavily"),
    ):
        label = " ".join(
            replacement if word.lower() == original else word for word in label.split()
        )
    return label[:1].upper() + label[1:]


def _is_percentage_metric(name: str) -> bool:
    return name.endswith(
        ("_rate", "_recall", "_precision", "_correctness", "_compliance", "_ratio")
    ) or any(
        part in name
        for part in (
            "consistency",
            "grounded_claim",
            "one_tool_per_worker",
            "precision",
            "stability",
            "validity",
        )
    )


def _metric_value(name: str, value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if _is_percentage_metric(name):
            return f"{value:.1%}"
        if value.is_integer():
            return f"{int(value):,}"
        return f"{value:,.2f}"
    if isinstance(value, list):
        if name == "pass_rate_wilson_95" and len(value) == 2:
            return f"{value[0]:.1%}–{value[1]:.1%}"
        return ", ".join(_markdown_text(item) for item in value) or "None"
    if isinstance(value, dict):
        if value.get("status") == "N/A":
            reason = value.get("reason")
            return "N/A" + (f" — {_markdown_text(reason)}" if reason else "")
        return "See breakdown below"
    return _markdown_text(value)


def _append_table(
    lines: list[str], headers: tuple[str, ...], rows: list[tuple[str, ...]]
) -> None:
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)


def _render_source_runs(lines: list[str], run_ids: list[str]) -> None:
    if not run_ids:
        return
    lines.extend(
        [
            f"<details><summary>Source runs ({len(run_ids)})</summary>",
            "",
            *[f"- `{_markdown_text(run_id)}`" for run_id in run_ids],
            "",
            "</details>",
            "",
        ]
    )


def _render_metric_table(
    lines: list[str], value: dict[str, Any], *, excluded: set[str] | None = None
) -> None:
    excluded = excluded or set()
    rows = [
        (_metric_label(name), _metric_value(name, observed))
        for name, observed in value.items()
        if name not in excluded
        and not (isinstance(observed, dict) and observed.get("status") != "N/A")
    ]
    if rows:
        _append_table(lines, ("Metric", "Result"), rows)


def _render_mapping(
    lines: list[str], title: str, mapping: dict[str, Any], value_name: str
) -> None:
    lines.extend(["", f"### {title}", ""])
    _append_table(
        lines,
        ("Case", "Result"),
        [
            (f"`{_markdown_text(name)}`", _metric_value(value_name, observed))
            for name, observed in mapping.items()
        ],
    )


def _render_failure_types(lines: list[str], failures: dict[str, Any]) -> None:
    lines.extend(["", "### Failure types", ""])
    if not failures:
        lines.append("None.")
        return
    _append_table(
        lines,
        ("Failure", "Count"),
        [
            (_metric_label(name), _metric_value("count", count))
            for name, count in failures.items()
        ],
    )


def _render_distributions(lines: list[str], value: dict[str, Any]) -> None:
    distributions = {
        name: observed
        for name, observed in value.items()
        if isinstance(observed, dict)
        and observed
        and set(observed).issubset({"count", "mean", "median", "p95"})
    }
    if not distributions:
        return
    lines.extend(["", "### Distributions", ""])
    _append_table(
        lines,
        ("Measurement", "Samples", "Mean", "Median", "P95"),
        [
            (
                _metric_label(name),
                _metric_value("count", observed.get("count")),
                _metric_value(name, observed.get("mean")),
                _metric_value(name, observed.get("median")),
                _metric_value(name, observed.get("p95")),
            )
            for name, observed in distributions.items()
        ],
    )


def _coverage_summary(value: Any) -> str:
    if isinstance(value, dict):
        return f"{len(value)} entries — see breakdown below"
    if isinstance(value, list):
        return f"{len(value)} items"
    return _metric_value("coverage", value)


def _render_coverage(lines: list[str], coverage: dict[str, Any]) -> None:
    complete = bool(coverage.get("complete"))
    lines.append(
        f"**Campaign completeness:** {'Complete' if complete else 'Incomplete'}"
    )
    missing = coverage.get("missing_requirements", [])
    if missing:
        lines.append(
            "**Missing requirements:** "
            + ", ".join(_metric_label(str(item)) for item in missing)
        )
    else:
        lines.append("**Missing requirements:** None")
    lines.append("")
    _append_table(
        lines,
        ("Requirement", "Status", "Observed", "Expected"),
        [
            (
                _metric_label(str(item.get("id"))),
                "Complete" if item.get("complete") else "Missing",
                _coverage_summary(item.get("observed")),
                _coverage_summary(item.get("expected")),
            )
            for item in coverage.get("requirements", [])
        ],
    )
    for item in coverage.get("requirements", []):
        observed = item.get("observed")
        expected = item.get("expected")
        if not isinstance(observed, dict) and not isinstance(expected, dict):
            continue
        if item.get("id") == "semantic_dimensions" and isinstance(observed, dict):
            lines.extend(["", "### Semantic dimensions", ""])
            expected_dimensions = set(expected or [])
            _append_table(
                lines,
                (
                    "Dimension",
                    "Results",
                    "Success",
                    "Skipped",
                    "Errors",
                    "Provider attempts",
                    "Accounted",
                    "Expected",
                ),
                [
                    (
                        _metric_label(name),
                        _metric_value("count", result.get("result_count", 0)),
                        _metric_value("count", result.get("successful_count", 0)),
                        _metric_value("count", result.get("skipped_count", 0)),
                        _metric_value("count", result.get("error_count", 0)),
                        _metric_value(
                            "count", result.get("provider_attempted_count", 0)
                        ),
                        _metric_value("accounted", result.get("accounted", False)),
                        "Yes" if name in expected_dimensions else "No",
                    )
                    for name, result in observed.items()
                ],
            )
            continue
        names = list(
            dict.fromkeys(
                [
                    *(list(observed or {}) if isinstance(observed, dict) else []),
                    *(list(expected or {}) if isinstance(expected, dict) else []),
                ]
            )
        )
        lines.extend(["", f"### {_metric_label(str(item.get('id')))}", ""])
        _append_table(
            lines,
            ("Item", "Observed", "Expected"),
            [
                (
                    f"`{_markdown_text(name)}`",
                    _metric_value(str(name), (observed or {}).get(name))
                    if isinstance(observed, dict)
                    else _coverage_summary(observed),
                    _metric_value(str(name), (expected or {}).get(name))
                    if isinstance(expected, dict)
                    else _coverage_summary(expected),
                )
                for name in names
            ],
        )


def _render_semantic_quality(lines: list[str], value: dict[str, Any]) -> None:
    _render_source_runs(lines, value.get("source_run_ids", []))
    _append_table(
        lines,
        (
            "Dimension",
            "Status",
            "Results",
            "Success",
            "Skipped",
            "Errors",
            "Provider attempts",
            "Reason",
        ),
        [
            (
                _metric_label(dimension),
                str(item.get("status")),
                _metric_value("count", item.get("result_count", 0)),
                _metric_value("count", item.get("successful_count", 0)),
                _metric_value("count", item.get("skipped_count", 0)),
                _metric_value("count", item.get("error_count", 0)),
                _metric_value("count", item.get("provider_attempted_count", 0)),
                _markdown_text(item.get("reason") or "—"),
            )
            for dimension, item in value.get("dimensions", {}).items()
        ],
    )
    identities = [
        item
        for key, item in value.items()
        if key not in {"dimensions", "source_run_ids"} and isinstance(item, dict)
    ]
    if identities:
        lines.extend(["", "### Judge execution identities", ""])
        _append_table(
            lines,
            (
                "Dimension",
                "Provider/model",
                "Rubric",
                "Success",
                "Skipped",
                "Errors",
                "Truncations",
            ),
            [
                (
                    _metric_label(str(item.get("dimension"))),
                    f"{item.get('judge_provider')}/{item.get('judge_model')}",
                    _markdown_text(item.get("rubric_version")),
                    _metric_value("count", item.get("successful_result_count", 0)),
                    _metric_value("count", item.get("skipped_result_count", 0)),
                    _metric_value("count", item.get("error_count", 0)),
                    _metric_value("count", item.get("confirmed_truncation_count", 0)),
                )
                for item in identities
            ],
        )


def _render_retrieval_quality(lines: list[str], value: dict[str, Any]) -> None:
    _render_source_runs(lines, value.get("source_run_ids", []))
    _render_metric_table(lines, value, excluded={"source_run_ids", "diagnostics"})
    diagnostics = value.get("diagnostics")
    if isinstance(diagnostics, dict):
        lines.extend(["", "### Retrieval diagnostics", ""])
        _render_metric_table(lines, diagnostics)


def _render_runtime_behavior(lines: list[str], value: dict[str, Any]) -> None:
    _render_source_runs(lines, value.get("source_run_ids", []))
    _render_metric_table(
        lines,
        value,
        excluded={"source_run_ids", "sample_counts", "path_efficiency_ratio"},
    )
    sample_counts = value.get("sample_counts")
    if isinstance(sample_counts, dict):
        lines.extend(["", "### Samples by mode", ""])
        _append_table(
            lines,
            ("Mode", "Samples"),
            [
                (_metric_label(mode), _metric_value("count", count))
                for mode, count in sample_counts.items()
            ],
        )
    path_efficiency = value.get("path_efficiency_ratio")
    if isinstance(path_efficiency, dict):
        _render_distributions(lines, {"path_efficiency_ratio": path_efficiency})


def _render_reliability(lines: list[str], value: dict[str, Any]) -> None:
    _render_source_runs(lines, value.get("source_run_ids", []))
    excluded = {
        "source_run_ids",
        "route_consistency",
        "evidence_set_jaccard_stability",
        "failure_type_frequency",
    }
    _render_metric_table(lines, value, excluded=excluded)
    for name in ("route_consistency", "evidence_set_jaccard_stability"):
        result = value.get(name)
        if not isinstance(result, dict):
            continue
        title = _metric_label(name)
        lines.extend(["", f"### {title}", ""])
        if "mean" in result:
            lines.append(f"**Mean:** {_metric_value(name, result.get('mean'))}")
        cases = result.get("by_scenario")
        if isinstance(cases, dict):
            lines.append("")
            _append_table(
                lines,
                ("Case", "Result"),
                [
                    (f"`{_markdown_text(case)}`", _metric_value(name, observed))
                    for case, observed in cases.items()
                ],
            )
    _render_failure_types(lines, value.get("failure_type_frequency", {}))


def _render_efficiency(lines: list[str], value: dict[str, Any]) -> None:
    _render_source_runs(lines, value.get("source_run_ids", []))
    _render_metric_table(lines, value, excluded={"source_run_ids"})
    _render_distributions(lines, value)


def _render_structured_details(
    lines: list[str], title: str, value: Any, *, level: int = 3
) -> None:
    heading = "#" * min(level, 6)
    lines.extend(["", f"{heading} {_metric_label(title)}", ""])
    if isinstance(value, dict):
        scalars = {
            name: observed
            for name, observed in value.items()
            if not isinstance(observed, (dict, list))
            or (isinstance(observed, dict) and observed.get("status") == "N/A")
        }
        if scalars:
            _render_metric_table(lines, scalars)
        for name, observed in value.items():
            if isinstance(observed, dict) and observed.get("status") != "N/A":
                _render_structured_details(lines, name, observed, level=level + 1)
            elif isinstance(observed, list):
                _render_structured_details(lines, name, observed, level=level + 1)
    elif isinstance(value, list):
        if not value:
            lines.append("None.")
        else:
            lines.extend(f"- `{_markdown_text(item)}`" for item in value)
    else:
        lines.append(_metric_value(title, value))


def _render_diagnostics(lines: list[str], value: dict[str, Any]) -> None:
    failed = value.get("failed_runtime_checks", [])
    framework = value.get("agentevals", [])
    _append_table(
        lines,
        ("Diagnostic", "Count"),
        [
            ("Failed runtime checks", _metric_value("count", len(failed))),
            ("AgentEvals rows", _metric_value("count", len(framework))),
        ],
    )
    _render_failure_types(lines, value.get("failure_type_frequency", {}))

    modes = value.get("runtime_mode_breakdown", {})
    lines.extend(["", "### Runtime modes", ""])
    _append_table(
        lines,
        ("Mode", "Samples", "Passed", "Failed"),
        [
            (
                _metric_label(mode),
                _metric_value("count", item.get("sample_count", 0)),
                _metric_value("count", item.get("passed_count", 0)),
                _metric_value("count", item.get("failed_count", 0)),
            )
            for mode, item in modes.items()
        ],
    )
    mode_failures = [
        (_metric_label(mode), _metric_label(name), _metric_value("count", count))
        for mode, item in modes.items()
        for name, count in item.get("failure_type_frequency", {}).items()
    ]
    if mode_failures:
        lines.extend(["", "### Runtime failure types by mode", ""])
        _append_table(lines, ("Mode", "Failure", "Count"), mode_failures)

    provenance = value.get("provenance")
    if isinstance(provenance, dict):
        _render_structured_details(lines, "provenance diagnostics", provenance)

    artifacts = value.get("run_artifacts", [])
    if artifacts:
        lines.extend(
            [
                "",
                f"<details><summary>Run artifacts ({len(artifacts)})</summary>",
                "",
            ]
        )
        _append_table(
            lines,
            ("Run", "Kind", "Directory"),
            [
                (
                    f"`{_markdown_text(item.get('run_id'))}`",
                    _metric_label(str(item.get("run_kind"))),
                    f"`{_markdown_text(item.get('run_directory'))}`",
                )
                for item in artifacts
            ],
        )
        lines.extend(["", "</details>"])

    if failed:
        lines.extend(
            [
                "",
                f"<details><summary>Failed runtime checks ({len(failed)})</summary>",
                "",
            ]
        )
        _append_table(
            lines,
            ("Scenario", "Dimension", "Check", "Gating", "Reason"),
            [
                (
                    f"`{_markdown_text(item.get('scenario_id'))}`",
                    _metric_label(str(item.get("dimension"))),
                    _metric_label(str(item.get("name"))),
                    _metric_value("gating", item.get("gating")),
                    _markdown_text(item.get("reason") or "—"),
                )
                for item in failed
            ],
        )
        lines.extend(["", "</details>"])


def _render_scorecard(scorecard: dict[str, Any]) -> str:
    lines = [
        "# Graph RAG Evaluation Scorecard",
        "",
        f"- Fingerprint: `{scorecard['fingerprint']}`",
        f"- Generated: {scorecard['generated_at']}",
        f"- Historical runs: {scorecard['history_run_count']}",
        f"- Contributing current runs: {scorecard['selected_run_count']}",
        "",
        "This report intentionally has no overall score or overall status.",
    ]
    sections = (
        ("Campaign coverage", "coverage", _render_coverage),
        ("Retrieval quality", "retrieval_quality", _render_retrieval_quality),
        ("Runtime task behavior", "runtime_task_behavior", _render_runtime_behavior),
        ("Reliability", "reliability", _render_reliability),
        ("Efficiency", "efficiency", _render_efficiency),
        (
            "Optional semantic quality",
            "optional_semantic_quality",
            _render_semantic_quality,
        ),
        ("Diagnostics", "diagnostics", _render_diagnostics),
    )
    for title, key, renderer in sections:
        lines.extend(["", f"## {title}", ""])
        value = scorecard[key]
        if isinstance(value, dict) and value.get("status") == "N/A":
            lines.append(f"N/A — {value.get('reason', 'not available')}.")
        else:
            renderer(lines, value)

    lines.extend(["", "## Metric definitions", ""])
    lines.extend(["<details><summary>Show metric definitions</summary>", ""])
    for group, definitions in scorecard["metric_definitions"].items():
        lines.extend([f"### {group}", ""])
        _append_table(
            lines,
            ("Measurement", "Definition"),
            [
                (_metric_label(metric), definition)
                for metric, definition in definitions.items()
            ],
        )
        lines.append("")
    lines.extend(
        [
            "</details>",
            "",
            "## Metric interpretation",
            "",
            "- Functional task success excludes non-gating token and latency limits.",
            "- Context Precision@10 and Context Recall remain N/A until human labels are reviewed.",
            "- Live repetitions accumulate; deterministic invocations use their latest successful run.",
            "- Optional semantic judges never change deterministic task success.",
            "",
        ]
    )
    return "\n".join(lines)


def _tag_source_run(
    rows: list[dict[str, Any]], run_id: str, run_kind: str
) -> list[dict[str, Any]]:
    return [
        {**row, "_source_run_id": run_id, "_source_run_kind": run_kind} for row in rows
    ]


def _source_run_ids(rows: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {str(row.get("_source_run_id")) for row in rows if row.get("_source_run_id")}
    )


def _check_rate(
    checks: list[dict[str, Any]], dimension: str, name: str
) -> float | None:
    selected = [
        check
        for check in checks
        if check.get("dimension") == dimension and check.get("name") == name
    ]
    return _mean_optional([check.get("status") == "pass" for check in selected])


def _dimension_rate(checks: list[dict[str, Any]], dimension: str) -> float | None:
    selected = [check for check in checks if check.get("dimension") == dimension]
    return _mean_optional([check.get("status") == "pass" for check in selected])


def _run_dimension_rate(rows: list[dict[str, Any]], dimension: str) -> float | None:
    applicable = []
    for row in rows:
        checks = [
            check
            for check in row.get("checks", [])
            if check.get("dimension") == dimension
            and check.get("status") not in {"not_applicable", "not_observed"}
        ]
        if checks:
            applicable.append(all(check.get("status") == "pass" for check in checks))
    return _mean_optional(applicable)


def _failed_checks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "scenario_id": row.get("scenario_id"),
            "run_id": row.get("run_id"),
            "dimension": check.get("dimension"),
            "name": check.get("name"),
            "gating": check.get("gating", True),
            "reason": check.get("reason"),
        }
        for row in rows
        for check in row.get("checks", [])
        if check.get("status") == "fail"
    ]


def _optional_metric_mean(
    rows: list[dict[str, Any]], name: str, reason: str
) -> float | dict[str, str]:
    values = _numeric(rows, name)
    return statistics.fmean(values) if values else {"status": "N/A", "reason": reason}


def _metric_mean(rows: list[dict[str, Any]], name: str) -> float | None:
    values = _numeric(rows, name)
    return statistics.fmean(values) if values else None


def _metric_sum(rows: list[dict[str, Any]], name: str) -> float:
    return sum(_numeric(rows, name))


def _numeric(rows: Iterable[dict[str, Any]], name: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return values


def _values_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)],
    }


def _mean(values: Iterable[bool | int | float]) -> float:
    numeric = [float(value) for value in values]
    return statistics.fmean(numeric) if numeric else 0.0


def _mean_optional(values: Iterable[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return statistics.fmean(numeric) if numeric else None


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    z = 1.96
    p = successes / total
    denominator = 1 + z**2 / total
    centre = p + z**2 / (2 * total)
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
    return (
        max(0.0, (centre - margin) / denominator),
        min(1.0, (centre + margin) / denominator),
    )


def _history_item(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        key: manifest.get(key)
        for key in (
            "run_id",
            "run_kind",
            "status",
            "created_at",
            "invocation_key",
            "runtime_provider_attempts",
            "runtime_tavily_attempts",
            "judge_provider_calls",
        )
    }


def _latest_report_fingerprint(evaluation_root: Path) -> str | None:
    latest = _read_json(evaluation_root / "latest.json")
    value = latest.get("fingerprint") if isinstance(latest, dict) else None
    return str(value) if value else None


def _contributing_manifests(
    evaluation_root: Path,
    scorecard: dict[str, Any],
) -> list[dict[str, Any]]:
    run_ids = {str(value) for value in scorecard.get("contributing_run_ids", [])}
    manifests = []
    for run_id in sorted(run_ids):
        path = evaluation_root / "runs" / run_id / "manifest.json"
        manifest = _read_json(path)
        if isinstance(manifest, dict) and manifest.get("run_id") == run_id:
            manifests.append(manifest)
    return manifests


def _fingerprint_run_dates(history_path: Path) -> list[datetime]:
    history = _read_json(history_path)
    dates = []
    for manifest in history.get("runs", []) if isinstance(history, dict) else []:
        try:
            dates.append(datetime.fromisoformat(str(manifest.get("created_at"))))
        except (TypeError, ValueError):
            continue
    return dates


def _publication_provenance(
    evaluation_root: Path,
    scorecard: dict[str, Any],
    *,
    snapshot_name: str,
) -> dict[str, Any]:
    manifests = _contributing_manifests(evaluation_root, scorecard)
    runs = [
        {
            key: _portable_value(manifest.get(key))
            for key in (
                "run_id",
                "run_kind",
                "status",
                "created_at",
                "invocation_key",
                "invocation",
                "configuration_snapshot",
                "runtime_provider_attempts",
                "runtime_tavily_attempts",
                "judge_provider_calls",
                "artifacts",
            )
        }
        for manifest in manifests
    ]
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "snapshot": snapshot_name,
        "fingerprint": scorecard["fingerprint"],
        "scorecard_generated_at": scorecard["generated_at"],
        "published_at": datetime.now(UTC).isoformat(),
        "contributing_run_count": len(runs),
        "contributing_runs": runs,
        "excluded_content": [
            "raw trace exports",
            "runtime records",
            "retrieved evidence text",
            "tool outputs",
            "credentials and secrets",
        ],
    }


def _write_publication_index(reports_root: Path) -> None:
    rows = []
    for path in sorted((reports_root / "history").glob("*/provenance.json")):
        provenance = _read_json(path)
        if not isinstance(provenance, dict):
            continue
        snapshot = path.parent.name
        rows.append(
            (
                str(provenance.get("published_at") or ""),
                snapshot,
                str(provenance.get("fingerprint") or ""),
                int(provenance.get("contributing_run_count") or 0),
            )
        )
    lines = [
        "# Published evaluation reports",
        "",
        "Each row is one configuration fingerprint. Republishing the same "
        "fingerprint updates its stable history snapshot and the latest aliases.",
        "",
        "| Published | Fingerprint | Runs | Scorecard | Provenance |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for published_at, snapshot, fingerprint, run_count in sorted(rows, reverse=True):
        relative = f"history/{snapshot}"
        lines.append(
            f"| {published_at} | `{fingerprint}` | {run_count} | "
            f"[Markdown]({relative}/scorecard.md) / "
            f"[JSON]({relative}/scorecard.json) | "
            f"[JSON]({relative}/provenance.json) |"
        )
    lines.append("")
    (reports_root / "index.md").write_text("\n".join(lines), encoding="utf-8")


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _portable_value(value: Any) -> Any:
    if isinstance(value, dict):
        portable = {str(key): _portable_value(item) for key, item in value.items()}
        for key in ("schema_version", "report_schema_version"):
            if is_evaluation_report_schema(portable.get(key)):
                portable[key] = REPORT_SCHEMA_VERSION
        return portable
    if isinstance(value, list):
        return [_portable_value(item) for item in value]
    if isinstance(value, str):
        root = str(ROOT.resolve())
        return value.replace(root, ".")
    return value


def _git_revision() -> tuple[str, str]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tracked_diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        untracked_output = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        digest = hashlib.sha256(tracked_diff)
        for relative in sorted(
            line.strip() for line in untracked_output.splitlines() if line.strip()
        ):
            path = ROOT / relative
            if not path.is_file():
                continue
            digest.update(relative.encode())
            digest.update(path.read_bytes())
        return revision, digest.hexdigest()[:16]
    except (OSError, subprocess.CalledProcessError):
        return "unknown", "unknown"


def _hash_tree(path: Path) -> str:
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    for item in sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and _is_identity_file(candidate.relative_to(path))
    ):
        digest.update(str(item.relative_to(path)).encode())
        digest.update(item.read_bytes())
    return digest.hexdigest()


def _is_identity_file(relative: Path) -> bool:
    if relative.name == ".DS_Store":
        return False
    if any(part in IGNORED_IDENTITY_PARTS for part in relative.parts):
        return False
    return relative.suffix not in {".pyc", ".pyo", ".tmp"}


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    )
    return hashlib.sha256(encoded.encode()).hexdigest()[:24]


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if any(
                part in str(key).casefold()
                for part in (
                    "api_key",
                    "authorization",
                    "password",
                    "secret",
                    "access_token",
                    "bearer",
                    "credential",
                )
            ):
                result[str(key)] = "[REDACTED]"
            else:
                result[str(key)] = _redact(item)
        return result
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _safe_error(value: str) -> str:
    return str(value).replace("\n", " ")[:1000]


def _read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)
