"""Evaluate the online learner workflow from exported traces or controlled live runs."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.configs.search import KnowledgeGraphSearchSettings  # noqa: E402
from backend.evaluation.evaluation_reporting import (  # noqa: E402
    DEFAULT_EVALUATION_ROOT,
    allocate_run_dir,
    current_evaluation_fingerprint,
    is_evaluation_report_schema,
    register_completed_run,
    register_failed_run,
)
from backend.evaluation.request_guard import EvaluationGuard  # noqa: E402
from backend.evaluation.runtime_evaluation import (  # noqa: E402
    SEMANTIC_JUDGE_OUTPUT_TOKEN_LIMITS,
    SEMANTIC_JUDGE_RUBRICS,
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
from backend.utils.chat_limits import (  # noqa: E402
    EvaluationBudgetUnavailable,
    bind_evaluation_budget,
)

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
        "--max-web-attempts",
        type=int,
        default=0,
        help="Maximum web provider attempts in this run; requires --allow-tavily.",
    )
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
    parser.add_argument(
        "--provider-quota",
        type=Path,
        help="Required fresh quota observation JSON for the configured provider and model.",
    )
    parser.add_argument(
        "--max-provider-attempts",
        type=int,
        help="Required maximum provider attempts in this run, including retries.",
    )


def _evaluation_session(args: argparse.Namespace, output_dir: Path) -> EvaluationGuard:
    if (output_dir / "manifest.json").exists() or (
        output_dir / "partial_judge_results.jsonl"
    ).exists():
        args._active_output_dir = None
        raise ValueError("evaluation output already has a manifest")
    if args.provider_quota is None or args.max_provider_attempts is None:
        raise ValueError(
            "live/judge requires --provider-quota and --max-provider-attempts"
        )
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    ledger = Path(common).parent / "data/evaluation/request_guard/ledger.json"
    guard = EvaluationGuard(
        ledger,
        quota=json.loads(args.provider_quota.read_text(encoding="utf-8")),
        max_attempts=args.max_provider_attempts,
        max_web_attempts=getattr(args, "max_web_attempts", 0),
    )
    guard.allow_tavily = (
        bool(getattr(args, "allow_tavily", False))
        and getattr(args, "max_web_attempts", 0) > 0
    )
    args._budget = guard
    return guard


@asynccontextmanager
async def _live_session(args: argparse.Namespace, output_dir: Path):
    async with _evaluation_session(args, output_dir) as guard:
        with bind_evaluation_budget(guard):
            yield guard


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

    async with _live_session(args, output_dir) as budget:
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
        for scenario in selected:
            for repetition in repetitions:
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
                record.provenance["evaluation_guard"] = budget.provenance()
                records.append(record)
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
                "evaluation_guard": budget.provenance(),
            },
            evaluation_root=args.evaluation_root,
        )
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
        dimension: SEMANTIC_JUDGE_OUTPUT_TOKEN_LIMITS[dimension]
        for dimension in dimensions
    }

    print(
        "Semantic judge plan: "
        f"jobs={[(scenario.id, dimension) for scenario, _, dimension in jobs]}, "
        f"provider={provider_name}, model={model_name}, "
        f"structured_output={args.structured_output_method}, "
        f"output_limits={output_limits}, retries=0."
    )
    with (
        _evaluation_session(args, output_dir) as budget,
        bind_evaluation_budget(budget),
    ):
        judge_results = []
        for scenario, record, dimension in jobs:
            judge_settings = settings.model_copy(
                update={"temperature": 0.0, "max_tokens": output_limits[dimension]}
            )
            judge = AgentsFactory.get_llm_by_role(judge_settings)
            result = evaluate_with_openevals_judge(
                scenario,
                record,
                dimension,
                judge=judge,
                judge_provider=provider_name,
                judge_model=model_name,
                configured_output_token_limit=output_limits[dimension],
                structured_output_method=args.structured_output_method,
            )
            judge_results.append(result)
            output_dir.mkdir(parents=True, exist_ok=True)
            fd = os.open(
                output_dir / "partial_judge_results.jsonl",
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o600,
            )
            with os.fdopen(fd, "a", encoding="utf-8") as handle:
                handle.write(result.model_dump_json() + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            if result.http_status_code == 429:
                raise EvaluationBudgetUnavailable("judge HTTP 429 stopped the batch")
        published_records = selected_records
        scenario_results = [
            score_runtime_record(scenario_by_id[record.scenario_id], record)
            for record in published_records
        ]
        attempted_calls = sum(
            (result.usage.provider_attempts or 0) for result in judge_results
        )
        output_dir = args._active_output_dir
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
                "evaluation_guard": budget.provenance(),
            },
            evaluation_root=args.evaluation_root,
        )
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
        print(f"Runtime evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
