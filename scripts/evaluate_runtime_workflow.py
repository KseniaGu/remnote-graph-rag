"""Evaluate the online learner workflow from exported traces or controlled live runs."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.configs.search import KnowledgeGraphSearchSettings  # noqa: E402
from backend.evaluation.evaluation_reporting import (  # noqa: E402
    DEFAULT_EVALUATION_ROOT,
    _hash_tree,
    allocate_run_dir,
    build_evaluation_snapshot,
    current_evaluation_fingerprint,
    is_evaluation_report_schema,
    register_completed_run,
    register_failed_run,
)
from backend.evaluation.portfolio_budget import (  # noqa: E402
    EvaluationBudget,
    EvaluationBudgetUnavailable,
    bind_evaluation_budget,
    campaign_ledger,
)
from backend.evaluation.runtime_evaluation import (  # noqa: E402
    SEMANTIC_JUDGE_OUTPUT_TOKEN_LIMITS,
    SEMANTIC_JUDGE_RUBRICS,
    RuntimeRecord,
    SemanticJudgeResult,
    evaluate_with_agentevals,
    evaluate_with_openevals_judge,
    load_runtime_records,
    load_runtime_scenarios,
    load_trace_export,
    normalize_trace_export,
    score_runtime_record,
    write_runtime_artifacts,
)
from backend.evaluation.runtime_live import run_controlled_live_scenario  # noqa: E402

DEFAULT_SCENARIOS = ROOT / "evals" / "runtime" / "scenarios.json"
DEFAULT_TRACE_EXPORT = ROOT / "exports" / "langsmith_traces_20260816T233652Z.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline-first evaluation of the production Graph RAG workflow."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    offline = subparsers.add_parser(
        "offline", help="Normalize and score an existing LangSmith JSON export."
    )
    _add_common_args(offline)
    offline.add_argument("--trace-export", type=Path, default=DEFAULT_TRACE_EXPORT)
    offline.add_argument(
        "--framework",
        choices=("none", "agentevals"),
        default="none",
        help="Optional non-gating trajectory cross-check (default: none).",
    )

    live = subparsers.add_parser(
        "live", help="Explicitly invoke selected cases through the compiled workflow."
    )
    _add_common_args(live)
    _add_budget_args(live)
    live.add_argument(
        "--case-id",
        action="append",
        required=True,
        help="Scenario ID to run; repeat for additional cases.",
    )
    repetitions = live.add_mutually_exclusive_group()
    repetitions.add_argument("--repetitions", type=int, default=1)
    repetitions.add_argument("--repetition", type=int, help="Run only this repetition.")
    live.add_argument(
        "--capture-diagnostics",
        action="store_true",
        help="Save private bounded live callback diagnostics.",
    )
    live.add_argument(
        "--confirm-provider-calls",
        action="store_true",
        help="Required acknowledgement that the command can call configured providers.",
    )
    live.add_argument(
        "--allow-tavily",
        action="store_true",
        help="Allow cases whose contract requires deep_web_research.",
    )
    live.add_argument(
        "--pre-dispatch-receipt",
        type=Path,
        help="Write a receipt only for a failure before final runtime admission.",
    )
    live.add_argument(
        "--require-registered-quota",
        action="store_true",
        help="Require a previously capture-bound quota baseline under the campaign lock.",
    )
    live.add_argument(
        "--require-final-runtime-admission",
        action="store_true",
        help="Require six-attempt final runtime admission under the campaign lock.",
    )
    live.add_argument(
        "--allow-expanded-live-suite",
        action="store_true",
        help="Permit more than three total live runs after explicit quota review.",
    )
    live.add_argument(
        "--framework",
        choices=("none", "agentevals"),
        default="none",
    )

    judge = subparsers.add_parser(
        "judge", help="Run explicitly selected semantic checks over saved records."
    )
    _add_common_args(judge)
    _add_budget_args(judge)
    judge.add_argument("--judge-provider", choices=("openrouter",))
    judge.add_argument("--judge-model")
    judge.add_argument(
        "--fixed-judge-pilot",
        action="store_true",
        help="Use the approved fixed-model, Phala-only evaluation pilot configuration.",
    )
    source = judge.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-run", help="Exact evaluation run ID, or latest.")
    source.add_argument(
        "--records", type=Path, help="Legacy runtime_records.jsonl path."
    )
    judge.add_argument("--case-id", action="append", required=True)
    judge.add_argument(
        "--dimension",
        action="append",
        required=True,
        choices=tuple(sorted(SEMANTIC_JUDGE_RUBRICS)),
    )
    judge.add_argument("--repetition", type=int, default=1)
    judge.add_argument(
        "--structured-output-method",
        choices=("function_calling", "json_schema", "json_mode"),
        default="function_calling",
        help="Judge structured-output transport (default: function_calling).",
    )
    judge.add_argument(
        "--confirm-provider-calls",
        action="store_true",
        help="Required acknowledgement that each selected dimension calls a judge.",
    )
    judge.add_argument(
        "--allow-expanded-judge-suite",
        action="store_true",
        help="Permit more than three judge calls after explicit quota review.",
    )
    return parser.parse_args(argv)


def _add_budget_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--budget-stage", choices=("B0", "final"))
    parser.add_argument(
        "--budget-ledger",
        type=Path,
        help="Existing absolute original canonical ledger; required for restored sources. Aliases must resolve to the approved binding.",
    )
    parser.add_argument(
        "--provider-quota",
        type=Path,
        help="Quota JSON no older than 15 minutes on this UTC day; supplied limits can only tighten the fixed 50-attempt daily ceiling and 20-request headroom.",
    )


def _evaluation_session(args: argparse.Namespace, output_dir: Path) -> EvaluationBudget:
    if (output_dir / "manifest.json").exists():
        args._active_output_dir = None
        raise ValueError(
            "evaluation output already has a manifest; resume into a new output directory"
        )
    if not args.budget_stage or not args.provider_quota:
        raise ValueError("live/judge requires --budget-stage and --provider-quota")
    ledger = campaign_ledger(ROOT, override=args.budget_ledger)
    quota = json.loads(args.provider_quota.read_text(encoding="utf-8"))
    from backend.configs.paths import PathSettings
    from backend.utils.chat_limits import get_chat_limit_service

    paths = PathSettings()
    configuration = {
        "evaluation_fingerprint": current_evaluation_fingerprint(),
        "application_configuration": build_evaluation_snapshot({})["fingerprint_input"][
            "application_configuration"
        ],
        "chat_limits": get_chat_limit_service().settings.model_dump(mode="json"),
        "paths": paths.model_dump(mode="json"),
        "selected_prompt_tree_hash": _hash_tree(paths.prompts_dir),
        "selected_storage_manifest_hashes": {
            str(path.relative_to(paths.local_storage_dir)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(paths.local_storage_dir.rglob("*manifest*.json"))
            if path.is_file()
        },
    }
    budget = EvaluationBudget(
        ledger,
        stage=args.budget_stage,
        run_id=output_dir.name,
        quota=quota,
        configuration=configuration,
    )
    budget.allow_tavily = bool(getattr(args, "allow_tavily", False))
    budget.require_registered_observation = bool(
        getattr(args, "require_registered_quota", False)
        or getattr(args, "fixed_judge_pilot", False)
    )
    args._budget = budget
    return budget


def _save_job(output_dir: Path, job_id: str, result) -> Path:
    job_dir = output_dir / "jobs"
    job_dir.mkdir(parents=True, exist_ok=True)
    artifact = job_dir / (hashlib.sha256(job_id.encode()).hexdigest() + ".json")
    with artifact.open("x", encoding="utf-8") as handle:
        handle.write(result.model_dump_json(indent=2))
        handle.flush()
        os.fsync(handle.fileno())
    return artifact


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for outputs. By default, allocate an immutable timestamped "
            "directory under <evaluation-root>/runs."
        ),
    )
    parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION_ROOT)


def _default_output_dir(command: str, evaluation_root: Path) -> Path:
    return allocate_run_dir(command, evaluation_root)


def run_offline(args: argparse.Namespace) -> int:
    output_dir = args.output_dir or _default_output_dir("offline", args.evaluation_root)
    args._active_output_dir = output_dir
    scenarios = load_runtime_scenarios(args.scenarios)
    runs = load_trace_export(args.trace_export)
    records = normalize_trace_export(runs, scenarios)
    scenario_by_id = {scenario.id: scenario for scenario in scenarios}
    results = [
        score_runtime_record(scenario_by_id[record.scenario_id], record)
        for record in records
    ]
    framework_results = None
    if args.framework == "agentevals":
        framework_results = [
            evaluate_with_agentevals(scenario_by_id[record.scenario_id], record)
            for record in records
        ]
    output_dir = args._active_output_dir
    summary = write_runtime_artifacts(
        output_dir,
        records,
        results,
        source=str(args.trace_export),
        framework_results=framework_results,
    )
    register_completed_run(
        output_dir,
        run_kind="offline",
        invocation={
            "trace_export": str(args.trace_export),
            "scenarios": str(args.scenarios),
            "framework": args.framework,
        },
        configuration={
            "trace_source_configuration_status": sorted(
                {
                    str(
                        record.provenance.get("source_configuration_status")
                        or "needs verification"
                    )
                    for record in records
                }
            ),
            "scenario_versions": {item.id: item.version for item in scenarios},
        },
        evaluation_root=args.evaluation_root,
    )
    print(
        f"Offline evaluation wrote {len(records)} records to {output_dir} "
        f"({summary['passed_count']} pass, {summary['failed_count']} fail; provider calls: 0)."
    )
    return 0


def _live_repetitions(args: argparse.Namespace) -> list[int]:
    if args.repetition is not None:
        if args.repetition <= 0:
            raise ValueError("--repetition must be positive")
        return [args.repetition]
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be positive")
    return list(range(1, args.repetitions + 1))


async def run_live(args: argparse.Namespace) -> int:
    args._final_admitted = False
    if not args.confirm_provider_calls:
        raise ValueError("live evaluation requires --confirm-provider-calls")
    repetitions = _live_repetitions(args)

    scenarios = load_runtime_scenarios(args.scenarios)
    scenario_by_id = {scenario.id: scenario for scenario in scenarios}
    unknown = sorted(set(args.case_id) - set(scenario_by_id))
    if unknown:
        raise ValueError(f"unknown case IDs: {', '.join(unknown)}")
    selected = [scenario_by_id[case_id] for case_id in dict.fromkeys(args.case_id)]
    total_runs = len(selected) * len(repetitions)
    if total_runs > 3 and not args.allow_expanded_live_suite:
        raise ValueError(
            "initial live validation is capped at three total runs; use "
            "--allow-expanded-live-suite only after quota review and approval"
        )
    if getattr(args, "require_final_runtime_admission", False) and (
        args.budget_stage != "final" or total_runs != 1
    ):
        raise ValueError("final runtime admission requires exactly one final-stage run")
    tavily_cases = [
        scenario.id
        for scenario in selected
        if "deep_web_research" in scenario.expectations.required_tools
    ]
    if tavily_cases and not args.allow_tavily:
        raise ValueError(
            "Tavily is disabled; add --allow-tavily for: " + ", ".join(tavily_cases)
        )

    search_settings = KnowledgeGraphSearchSettings()
    if search_settings.analyst_retrieval_mode != "legacy_vector_context":
        raise ValueError(
            "runtime evaluation requires legacy_vector_context Analyst mode"
        )
    if search_settings.visualizer_retrieval_mode != "optimized":
        raise ValueError("runtime evaluation requires optimized Visualizer mode")

    output_dir = args.output_dir or _default_output_dir("live", args.evaluation_root)
    args._active_output_dir = output_dir

    with (
        _evaluation_session(args, output_dir) as budget,
        bind_evaluation_budget(budget),
    ):
        if getattr(args, "require_final_runtime_admission", False):
            budget.require_final_runtime_admission()
            args._final_admitted = True
        max_expected_provider_calls = sum(
            scenario.expectations.budgets.max_provider_attempts or 6
            for scenario in selected
        ) * len(repetitions)
        print(
            "Controlled live plan: "
            f"cases={[scenario.id for scenario in selected]}, repetitions={repetitions}, "
            f"maximum configured provider attempts={max_expected_provider_calls}, "
            f"Tavily={'enabled' if args.allow_tavily else 'disabled'}, judge=disabled."
        )

        records = []
        publication_jobs = []
        for scenario in selected:
            for repetition in repetitions:
                identity = f"live:{scenario.id}:v{scenario.version}:r{repetition}"
                contract = {
                    "scenario": scenario.model_dump(mode="json"),
                    "configuration": budget.configuration,
                }
                saved = budget.begin_job(
                    identity,
                    contract_digest=hashlib.sha256(
                        json.dumps(contract, sort_keys=True).encode()
                    ).hexdigest(),
                )
                if saved is not None and budget.is_published(identity):
                    continue
                if saved is not None:
                    record = RuntimeRecord.model_validate_json(
                        saved.read_text(encoding="utf-8")
                    )
                else:
                    diagnostic_args = (
                        {
                            "diagnostics_path": output_dir
                            / "diagnostics"
                            / f"{scenario.id}-r{repetition}.jsonl"
                        }
                        if args.capture_diagnostics
                        else {}
                    )
                    record = await run_controlled_live_scenario(
                        scenario, repetition=repetition, **diagnostic_args
                    )
                    record.provenance["evaluation_budget"] = budget.provenance()
                    budget.complete_job(
                        identity, _save_job(output_dir, identity, record)
                    )
                records.append(record)
                publication_jobs.append(identity)
        if not records:
            print(
                "Selected live jobs are already published; no calls or new report contributions."
            )
            return 0
        results = [
            score_runtime_record(scenario_by_id[record.scenario_id], record)
            for record in records
        ]
        framework_results = None
        if args.framework == "agentevals":
            framework_results = [
                evaluate_with_agentevals(scenario_by_id[record.scenario_id], record)
                for record in records
            ]
        output_dir = args._active_output_dir
        budget.prepare_publication(publication_jobs, output_dir / "manifest.json")
        summary = write_runtime_artifacts(
            output_dir,
            records,
            results,
            source="controlled_live",
            framework_results=framework_results,
        )
        register_completed_run(
            output_dir,
            run_kind="live",
            invocation={
                "case_ids": [item.id for item in selected],
                "repetitions": args.repetitions,
                "selected_repetitions": repetitions,
                "framework": args.framework,
                "allow_tavily": args.allow_tavily,
            },
            configuration={
                "analyst_retrieval_mode": search_settings.analyst_retrieval_mode,
                "visualizer_retrieval_mode": search_settings.visualizer_retrieval_mode,
                "scenario_versions": {item.id: item.version for item in selected},
                "evaluation_budget": budget.provenance(),
            },
            evaluation_root=args.evaluation_root,
        )
        budget.finish_publication(publication_jobs)
        actual_attempts = (
            sum(record.usage.provider_attempts for record in records)
            if all(record.usage.provider_attempts is not None for record in records)
            else "unknown"
        )
        actual_searches = sum(record.usage.tavily_searches for record in records)
        print(
            f"Live evaluation wrote {len(records)} records to {output_dir} "
            f"({summary['passed_count']} pass, {summary['failed_count']} fail; "
            f"provider attempts in published observations: {actual_attempts}; Tavily attempts: {actual_searches}; judge calls: 0)."
        )
        return 0


def _resolve_source_records(
    source_run: str | None,
    records_path: Path | None,
    evaluation_root: Path,
    case_ids: list[str],
) -> Path:
    if records_path is not None:
        return records_path
    run_root = evaluation_root / "runs"
    if source_run != "latest":
        candidate = run_root / str(source_run) / "runtime_records.jsonl"
        if not candidate.is_file():
            raise ValueError(f"runtime source run not found: {source_run}")
        return candidate
    current_fingerprint = current_evaluation_fingerprint()
    candidates: list[tuple[str, Path]] = []
    for manifest_path in run_root.glob("*/manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        candidate = manifest_path.parent / "runtime_records.jsonl"
        if (
            not is_evaluation_report_schema(manifest.get("report_schema_version"))
            or manifest.get("evaluation_fingerprint") != current_fingerprint
            or manifest.get("status") != "completed"
            or manifest.get("run_kind") not in {"offline", "live"}
            or not candidate.is_file()
        ):
            continue
        observed = {record.scenario_id for record in load_runtime_records(candidate)}
        if set(case_ids).issubset(observed):
            candidates.append((str(manifest.get("created_at") or ""), candidate))
    if not candidates:
        raise ValueError("no completed runtime run contains every selected case")
    return max(candidates, key=lambda item: item[0])[1]


def _configure_pilot_judge(judge, dimension: str):
    """Applies the fixed judge settings without changing production model settings."""
    return judge.model_copy(
        update={
            "temperature": 0.0,
            "max_tokens": 8192 if dimension == "claim_faithfulness" else 512,
            "reasoning": {"effort": "none"},
            "openrouter_provider": {
                "only": ["Phala"],
                "allow_fallbacks": False,
                "require_parameters": True,
            },
        }
    )


def _pilot_result_safe(result: SemanticJudgeResult) -> bool:
    return (
        result.status == "success"
        and not result.confirmed_truncation
        and result.observed_model
        in {
            "xingchen-agi/xing4.0-29b-a4b:free",
            "xingchen-agi/xing4.0-29b-a4b-20261015",
        }
        and result.observed_provider.casefold() == "phala"
    )


def _pilot_retryable(result: SemanticJudgeResult) -> bool:
    if result.status != "error" or not result.provider_call_attempted:
        return False
    if result.confirmed_truncation or result.parser_classification in {
        "malformed_json",
        "schema_validation_failure",
        "structured_output_parse_failure",
    }:
        return False
    if result.observed_model and result.observed_model not in {
        "xingchen-agi/xing4.0-29b-a4b:free",
        "xingchen-agi/xing4.0-29b-a4b-20261015",
    }:
        return False
    if result.observed_provider and result.observed_provider.casefold() != "phala":
        return False
    if result.http_status_code is not None:
        return (
            result.transport_status == "completed"
            and result.parser_classification == "provider_failure"
            and 500 <= result.http_status_code <= 599
        )
    return result.error_type in {
        "TimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "ConnectionError",
        "APIConnectionError",
        "APITimeoutError",
    }


def _pilot_observations_match(
    result: SemanticJudgeResult, budget: EvaluationBudget
) -> bool:
    """Compare every observed identity in an assessment, including failed attempts."""
    observed: dict[str, str] = {}
    for attempt in [*result.prior_attempts, result.model_dump(mode="json")]:
        if not isinstance(attempt, dict):
            return False
        model = attempt.get("observed_model", "")
        provider = attempt.get("observed_provider", "")
        if not isinstance(model, str) or not isinstance(provider, str):
            return False
        if model and model not in {
            "xingchen-agi/xing4.0-29b-a4b:free",
            "xingchen-agi/xing4.0-29b-a4b-20261015",
        }:
            return False
        if provider and provider.casefold() != "phala":
            return False
        if not budget.pilot_retry_identity_matches(model, provider):
            return False
        for key, value in (("model", model), ("provider", provider)):
            if value and key in observed and observed[key] != value:
                return False
            if value:
                observed[key] = value
    return True


def _combine_pilot_attempts(result: SemanticJudgeResult, failed: SemanticJudgeResult):
    result.prior_attempts = [failed.model_dump(mode="json")]
    duplicate_id = bool(
        failed.observed_request_id
        and failed.observed_request_id == result.observed_request_id
    )
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        parts = [item.usage.token_coverage.get(field, {}) for item in (failed, result)]
        observed = sum(part.get("observed", 0) for part in parts)
        subtotals = [
            part.get("subtotal") for part in parts if part.get("subtotal") is not None
        ]
        subtotal = sum(subtotals) if subtotals else None
        complete = not duplicate_id and all(
            part.get("complete") is True for part in parts
        )
        if duplicate_id:
            observed = 0
            subtotal = None
        result.usage.token_coverage[field] = {
            "observed": observed,
            "attempted": 2,
            "subtotal": subtotal,
            "complete": complete,
        }
        setattr(result.usage, field, subtotal if complete else None)
    result.usage.provider_attempts = 2
    result.usage.retries = 1


def run_judge(args: argparse.Namespace) -> int:
    if not args.confirm_provider_calls:
        raise ValueError("judge evaluation requires --confirm-provider-calls")
    if args.repetition <= 0:
        raise ValueError("--repetition must be positive")

    output_dir = args.output_dir or _default_output_dir("judge", args.evaluation_root)
    args._active_output_dir = output_dir

    scenarios = load_runtime_scenarios(args.scenarios)
    scenario_by_id = {scenario.id: scenario for scenario in scenarios}
    unknown = sorted(set(args.case_id) - set(scenario_by_id))
    if unknown:
        raise ValueError(f"unknown case IDs: {', '.join(unknown)}")
    records_path = _resolve_source_records(
        args.source_run,
        args.records,
        args.evaluation_root,
        list(dict.fromkeys(args.case_id)),
    )
    records_source_sha256 = hashlib.sha256(records_path.read_bytes()).hexdigest()
    records = load_runtime_records(records_path)
    selected_records = []
    for case_id in dict.fromkeys(args.case_id):
        matches = [
            record
            for record in records
            if record.scenario_id == case_id and record.repetition == args.repetition
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one record for {case_id!r} repetition "
                f"{args.repetition}, found {len(matches)}"
            )
        selected_records.append(matches[0])

    dimensions = list(dict.fromkeys(args.dimension))
    jobs = [
        (scenario_by_id[record.scenario_id], record, dimension)
        for record in selected_records
        for dimension in dimensions
    ]
    invalid = [
        f"{scenario.id}:{dimension}"
        for scenario, _, dimension in jobs
        if dimension not in scenario.expectations.judge_dimensions
    ]
    if invalid:
        raise ValueError(
            "scenario does not opt into requested judge dimension: "
            + ", ".join(invalid)
        )
    if len(jobs) > 3 and not args.allow_expanded_judge_suite:
        raise ValueError(
            "initial judge validation is capped at three calls; use "
            "--allow-expanded-judge-suite only after quota review and approval"
        )

    if not args.judge_provider or not args.judge_model:
        raise ValueError("judge requires separate --judge-provider and --judge-model")
    if args.fixed_judge_pilot and (
        args.judge_provider != "openrouter"
        or args.judge_model != "xingchen-agi/xing4.0-29b-a4b:free"
        or args.structured_output_method != "json_schema"
        or not set(dimensions) <= {"claim_faithfulness", "graph_usefulness"}
    ):
        raise ValueError(
            "fixed judge pilot requires its approved model, JSON Schema, and dimensions"
        )
    from backend.configs.enums import ModelRoleType
    from backend.configs.models import OpenRouterSettings
    from backend.workflows.agents.factory import AgentsFactory

    settings = OpenRouterSettings(
        role=ModelRoleType.analyst, model_name=args.judge_model, temperature=0.0
    )
    provider_value = getattr(settings.provider, "value", settings.provider)
    provider_name = str(provider_value)
    model_name = str(settings.model_name)
    output_limits = {
        dimension: (
            8192
            if args.fixed_judge_pilot and dimension == "claim_faithfulness"
            else SEMANTIC_JUDGE_OUTPUT_TOKEN_LIMITS[dimension]
        )
        for dimension in dimensions
    }

    print(
        "Semantic judge plan: "
        f"jobs={[(scenario.id, dimension) for scenario, _, dimension in jobs]}, "
        f"provider={provider_name}, model={model_name}, "
        f"structured_output={args.structured_output_method}, "
        f"output_limits={output_limits}, shared_pilot_retries={1 if args.fixed_judge_pilot else 0}."
    )
    with (
        _evaluation_session(args, output_dir) as budget,
        bind_evaluation_budget(budget),
    ):
        budget.fixed_judge_pilot = args.fixed_judge_pilot
        if budget.is_registered_fixed_judge_quota() and not args.fixed_judge_pilot:
            raise EvaluationBudgetUnavailable(
                "registered fixed judge quota requires fixed pilot profile"
            )
        if args.fixed_judge_pilot and budget.quota.get("model") != model_name:
            raise EvaluationBudgetUnavailable(
                "fixed judge model differs from quota observation"
            )
        if args.fixed_judge_pilot:
            for job_id, artifact in budget.completed_judge_artifacts():
                saved_result = SemanticJudgeResult.model_validate_json(
                    artifact.read_text(encoding="utf-8")
                )
                if (
                    saved_result.judge_model != model_name
                    or saved_result.judge_provider != provider_name
                ):
                    continue
                if not budget.pilot_job_profile(job_id, saved_result.dimension):
                    continue
                if saved_result.status == "skipped":
                    continue
                if not _pilot_result_safe(
                    saved_result
                ) or not _pilot_observations_match(saved_result, budget):
                    raise EvaluationBudgetUnavailable(
                        "fixed judge pilot found an unsafe saved result"
                    )
                budget.match_pilot_identity(
                    saved_result.observed_model, saved_result.observed_provider
                )
        judge_results = []
        publication_jobs = []
        publication_records = {}
        for scenario, record, dimension in jobs:
            record_hash = hashlib.sha256(
                json.dumps(
                    record.model_dump(mode="json"),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            contract = {
                "scenario": scenario.model_dump(mode="json"),
                "record_sha256": record_hash,
                "source_records_file_sha256": records_source_sha256,
                "dimension": dimension,
                "provider": provider_name,
                "model": model_name,
                "method": args.structured_output_method,
                "judge_profile": "fixed-pilot-v1"
                if args.fixed_judge_pilot
                else "standard-v2",
                "configuration": budget.configuration,
            }
            contract_digest = hashlib.sha256(
                json.dumps(contract, sort_keys=True).encode()
            ).hexdigest()
            identity = f"judge:semantic-v2:{contract_digest}"
            saved = budget.begin_job(
                identity,
                contract_digest=contract_digest,
                require_setup_recovery=(
                    args.fixed_judge_pilot
                    and scenario.id == "contextual_mamba_visualization"
                    and dimension == "graph_usefulness"
                    and records_source_sha256
                    == "efa5b77ae9ed40d12220a014f2529dc83c00857817d61fb2f7484c1b1c7a1d73"
                ),
            )
            if saved is not None:
                prior_result = SemanticJudgeResult.model_validate_json(
                    saved.read_text(encoding="utf-8")
                )
                if (
                    args.fixed_judge_pilot
                    and prior_result.status != "skipped"
                    and (
                        not _pilot_result_safe(prior_result)
                        or not _pilot_observations_match(prior_result, budget)
                    )
                ):
                    raise EvaluationBudgetUnavailable(
                        "fixed judge pilot stopped on an unsafe saved result"
                    )
                if args.fixed_judge_pilot and prior_result.status == "success":
                    budget.match_pilot_identity(
                        prior_result.observed_model, prior_result.observed_provider
                    )
                if budget.is_published(identity):
                    continue
                judge_results.append(prior_result)
                publication_jobs.append(identity)
                publication_records[record.run_id] = record
                continue
            updates = {"temperature": 0.0}
            output_limit = output_limits[dimension]
            if "num_predict" in settings.__class__.model_fields:
                updates["num_predict"] = output_limit
            if "max_tokens" in settings.__class__.model_fields:
                updates["max_tokens"] = output_limit
            judge_settings = settings.model_copy(update=updates)
            judge = AgentsFactory.get_llm_by_role(judge_settings)
            if args.fixed_judge_pilot:
                judge = _configure_pilot_judge(judge, dimension)
            result = evaluate_with_openevals_judge(
                scenario,
                record,
                dimension,
                judge=judge,
                judge_provider=provider_name,
                judge_model=model_name,
                configured_output_token_limit=output_limit,
                structured_output_method=args.structured_output_method,
                strict_structured_output=args.fixed_judge_pilot,
            )
            if (
                args.fixed_judge_pilot
                and _pilot_retryable(result)
                and budget.pilot_retry_identity_matches(
                    result.observed_model, result.observed_provider
                )
            ):
                if budget.claim_pilot_retry(result.model_dump(mode="json")):
                    failed = result
                    result = evaluate_with_openevals_judge(
                        scenario,
                        record,
                        dimension,
                        judge=judge,
                        judge_provider=provider_name,
                        judge_model=model_name,
                        configured_output_token_limit=output_limit,
                        structured_output_method=args.structured_output_method,
                        strict_structured_output=True,
                    )
                    _combine_pilot_attempts(result, failed)
            budget.complete_job(identity, _save_job(output_dir, identity, result))
            judge_results.append(result)
            publication_jobs.append(identity)
            publication_records[record.run_id] = record
            if (
                args.fixed_judge_pilot
                and result.status != "skipped"
                and (
                    not _pilot_result_safe(result)
                    or not _pilot_observations_match(result, budget)
                )
            ):
                raise EvaluationBudgetUnavailable(
                    "fixed judge pilot stopped: error, truncation or missing/unexpected serving identity"
                )
            if args.fixed_judge_pilot and result.status == "success":
                budget.match_pilot_identity(
                    result.observed_model, result.observed_provider
                )
            if result.http_status_code == 429:
                raise EvaluationBudgetUnavailable(
                    "judge HTTP 429 stopped the batch; refresh quota before resuming"
                )
        if not judge_results:
            print(
                "Selected judge jobs are already published; no calls or new report contributions."
            )
            return 0
        published_records = list(publication_records.values())
        scenario_results = [
            score_runtime_record(scenario_by_id[record.scenario_id], record)
            for record in published_records
        ]
        attempted_calls = sum(
            (result.usage.provider_attempts or 0) for result in judge_results
        )
        output_dir = args._active_output_dir
        budget.prepare_publication(publication_jobs, output_dir / "manifest.json")
        write_runtime_artifacts(
            output_dir,
            published_records,
            scenario_results,
            source=str(records_path),
            judge_results=judge_results,
            judge_provider_calls=attempted_calls,
        )
        register_completed_run(
            output_dir,
            run_kind="judge",
            invocation={
                "source_run": args.source_run,
                "records": str(records_path),
                "source_records_file_sha256": records_source_sha256,
                "case_ids": list(dict.fromkeys(args.case_id)),
                "dimensions": dimensions,
                "repetition": args.repetition,
                "structured_output_method": args.structured_output_method,
                "fixed_judge_pilot": args.fixed_judge_pilot,
            },
            configuration={
                "judge_provider": provider_name,
                "judge_model": model_name,
                "judge_temperature": 0,
                "judge_output_token_limits": output_limits,
                "judge_structured_output_method": args.structured_output_method,
                "judge_rubric_versions": dict.fromkeys(
                    dimensions, "runtime-semantic-v2"
                ),
                "judge_profile": "fixed-pilot-v1"
                if args.fixed_judge_pilot
                else "standard-v2",
                "judge_provider_policy": {
                    "only": ["Phala"],
                    "allow_fallbacks": False,
                    "require_parameters": True,
                }
                if args.fixed_judge_pilot
                else None,
                "judge_reasoning": {"effort": "none"}
                if args.fixed_judge_pilot
                else None,
                "judge_strict_schema": args.fixed_judge_pilot,
                "judge_include_raw": args.fixed_judge_pilot,
                "evaluation_budget": budget.provenance(),
            },
            evaluation_root=args.evaluation_root,
        )
        budget.finish_publication(publication_jobs)
        succeeded = sum(result.status == "success" for result in judge_results)
        skipped = sum(result.status == "skipped" for result in judge_results)
        errors = sum(result.status == "error" for result in judge_results)
        print(
            f"Judge evaluation wrote {len(judge_results)} results to {output_dir} "
            f"({succeeded} successful, {skipped} skipped, {errors} error; "
            f"provider attempts in published results: {attempted_calls})."
        )
        return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "offline":
            return run_offline(args)
        if args.command == "judge":
            return run_judge(args)
        return asyncio.run(run_live(args))
    except (OSError, ValueError, RuntimeError) as exc:
        output_dir = getattr(args, "_active_output_dir", None)
        manifest_path = output_dir / "manifest.json" if output_dir is not None else None
        completed = (
            manifest_path is not None
            and manifest_path.is_file()
            and json.loads(manifest_path.read_text(encoding="utf-8")).get("status")
            == "completed"
        )
        if output_dir is not None and not completed:
            register_failed_run(
                output_dir,
                run_kind=args.command,
                invocation={"argv": list(argv) if argv is not None else sys.argv[1:]},
                error=f"{type(exc).__name__}: {exc}",
                configuration={
                    "operational_outcome": getattr(exc, "reason", "evaluation_failed"),
                    "evaluation_budget": args._budget.provenance()
                    if hasattr(args, "_budget")
                    else None,
                },
                evaluation_root=args.evaluation_root,
            )
        receipt = getattr(args, "pre_dispatch_receipt", None)
        if (
            receipt is not None
            and getattr(args, "require_final_runtime_admission", False)
            and not getattr(args, "_final_admitted", True)
        ):
            with receipt.open("x", encoding="utf-8") as handle:
                json.dump(
                    {"dispatch_started": False, "error_type": type(exc).__name__},
                    handle,
                    sort_keys=True,
                )
                handle.flush()
                os.fsync(handle.fileno())
        print(f"Runtime evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
