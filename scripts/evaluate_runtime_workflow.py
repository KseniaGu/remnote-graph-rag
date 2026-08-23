"""Evaluate the online learner workflow from exported traces or controlled live runs."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
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
    live.add_argument(
        "--case-id",
        action="append",
        required=True,
        help="Scenario ID to run; repeat for additional cases.",
    )
    live.add_argument("--repetitions", type=int, default=1)
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


async def run_live(args: argparse.Namespace) -> int:
    if not args.confirm_provider_calls:
        raise ValueError("live evaluation requires --confirm-provider-calls")
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be positive")

    scenarios = load_runtime_scenarios(args.scenarios)
    scenario_by_id = {scenario.id: scenario for scenario in scenarios}
    unknown = sorted(set(args.case_id) - set(scenario_by_id))
    if unknown:
        raise ValueError(f"unknown case IDs: {', '.join(unknown)}")
    selected = [scenario_by_id[case_id] for case_id in dict.fromkeys(args.case_id)]
    total_runs = len(selected) * args.repetitions
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

    max_expected_provider_calls = (
        sum(
            scenario.expectations.budgets.max_provider_attempts or 6
            for scenario in selected
        )
        * args.repetitions
    )
    print(
        "Controlled live plan: "
        f"cases={[scenario.id for scenario in selected]}, repetitions={args.repetitions}, "
        f"maximum configured provider attempts={max_expected_provider_calls}, "
        f"Tavily={'enabled' if args.allow_tavily else 'disabled'}, judge=disabled."
    )

    records = []
    for scenario in selected:
        for repetition in range(1, args.repetitions + 1):
            records.append(  # noqa: PERF401 - await each paid run sequentially
                await run_controlled_live_scenario(
                    scenario,
                    repetition=repetition,
                )
            )
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
            "framework": args.framework,
            "allow_tavily": args.allow_tavily,
        },
        configuration={
            "analyst_retrieval_mode": search_settings.analyst_retrieval_mode,
            "visualizer_retrieval_mode": search_settings.visualizer_retrieval_mode,
            "scenario_versions": {item.id: item.version for item in selected},
        },
        evaluation_root=args.evaluation_root,
    )
    actual_attempts = sum(record.usage.provider_attempts for record in records)
    actual_searches = sum(record.usage.tavily_searches for record in records)
    print(
        f"Live evaluation wrote {len(records)} records to {output_dir} "
        f"({summary['passed_count']} pass, {summary['failed_count']} fail; "
        f"provider attempts: {actual_attempts}; Tavily attempts: {actual_searches}; judge calls: 0)."
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

    from backend.configs.models import get_model_settings
    from backend.workflows.agents.factory import AgentsFactory

    settings = get_model_settings().analyst
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
    judge_results = []
    for scenario, record, dimension in jobs:
        updates = {"temperature": 0.0}
        output_limit = SEMANTIC_JUDGE_OUTPUT_TOKEN_LIMITS[dimension]
        if "num_predict" in settings.__class__.model_fields:
            updates["num_predict"] = output_limit
        if "max_tokens" in settings.__class__.model_fields:
            updates["max_tokens"] = output_limit
        judge_settings = settings.model_copy(update=updates)
        judge = AgentsFactory.get_llm_by_role(judge_settings)
        judge_results.append(
            evaluate_with_openevals_judge(
                scenario,
                record,
                dimension,
                judge=judge,
                judge_provider=provider_name,
                judge_model=model_name,
                configured_output_token_limit=output_limit,
                structured_output_method=args.structured_output_method,
            )
        )
    scenario_results = [
        score_runtime_record(scenario_by_id[record.scenario_id], record)
        for record in selected_records
    ]
    attempted_calls = sum(result.provider_call_attempted for result in judge_results)
    output_dir = args._active_output_dir
    write_runtime_artifacts(
        output_dir,
        selected_records,
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
            "judge_rubric_versions": dict.fromkeys(dimensions, "v1"),
        },
        evaluation_root=args.evaluation_root,
    )
    succeeded = sum(result.status == "success" for result in judge_results)
    skipped = sum(result.status == "skipped" for result in judge_results)
    errors = sum(result.status == "error" for result in judge_results)
    print(
        f"Judge evaluation wrote {len(judge_results)} results to {output_dir} "
        f"({succeeded} successful, {skipped} skipped, {errors} error; "
        f"provider calls attempted: {attempted_calls})."
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
        if output_dir is not None:
            register_failed_run(
                output_dir,
                run_kind=args.command,
                invocation={"argv": list(argv) if argv is not None else sys.argv[1:]},
                error=f"{type(exc).__name__}: {exc}",
                evaluation_root=args.evaluation_root,
            )
        print(f"Runtime evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
