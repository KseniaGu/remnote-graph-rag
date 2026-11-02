"""Aggregate saved system-evaluation scores without rescoring observations."""

from collections import Counter
from typing import Any

from backend.evaluation.evaluation_reporting import (
    _runtime_task_metrics,
    _values_summary,
)


def aggregate_scored_runs(groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Summarize recorded checks separately for each original evaluator identity."""
    result = {}
    for fingerprint, rows in groups.items():
        statuses: dict[str, Counter] = {}
        for row in rows:
            for check in row.get("checks", []):
                name = f"{check['dimension']}.{check['name']}"
                statuses.setdefault(name, Counter())[check["status"]] += 1
        checks = {
            name: {
                "passed": counts["pass"],
                "eligible": counts["pass"] + counts["fail"],
                "not_observed": counts["not_observed"],
            }
            for name, counts in sorted(statuses.items())
        }
        efficiency = {}
        for name in (
            "worker_steps",
            "logical_llm_calls",
            "provider_attempts",
            "retries",
            "tavily_searches",
            "latency_seconds",
            "input_tokens",
            "output_tokens",
            "total_tokens",
        ):
            values = [row.get("metrics", {}).get(name) for row in rows]
            values = [float(value) for value in values if type(value) in (int, float)]
            summary = _values_summary(values)
            efficiency[name] = {
                "available": summary["count"],
                "mean": summary["mean"],
                "median": summary["median"],
                "p95": summary["p95"],
            }
        reliability = {}
        for name in ("loop_detected", "output_limit_hit"):
            values = [row.get("metrics", {}).get(name) for row in rows]
            observed = [value for value in values if type(value) is bool]
            reliability[name] = {"observed": len(observed), "count": sum(observed)}
        result[fingerprint] = {
            "scored": len(rows),
            "reliability": reliability,
            "passed": sum(row["passed"] is True for row in rows),
            "checks": checks,
            "recorded_rates": _runtime_task_metrics(rows),
            "failure_types": dict(
                Counter(
                    failure for row in rows for failure in row.get("failure_types", [])
                )
            ),
            "efficiency": efficiency,
        }
    return result


def derive_saved_runs(run_dirs):
    """Aggregate original scored runs and retain hashes of each input artifact."""
    import hashlib
    import json
    from pathlib import Path

    groups = {}
    sources = []
    seen = set()
    for run_dir in run_dirs:
        folder = Path(run_dir)
        manifest_bytes = (folder / "manifest.json").read_bytes()
        score_bytes = (folder / "case_results.jsonl").read_bytes()
        record_bytes = (folder / "runtime_records.jsonl").read_bytes()
        manifest = json.loads(manifest_bytes)
        rows = [json.loads(line) for line in score_bytes.splitlines() if line.strip()]
        records = [
            json.loads(line) for line in record_bytes.splitlines() if line.strip()
        ]
        fingerprint = manifest.get("evaluation_fingerprint")
        if (
            manifest.get("status") != "completed"
            or manifest.get("run_kind") != "live"
            or not fingerprint
            or len(rows) != 1
            or len(records) != 1
        ):
            raise ValueError("expected one completed live scored record per run")
        row, record = rows[0], records[0]
        identity = tuple(
            row.get(key)
            for key in ("run_id", "scenario_id", "repetition", "scenario_version")
        )
        if (
            None in identity
            or identity
            != tuple(
                record.get(key)
                for key in ("run_id", "scenario_id", "repetition", "scenario_version")
            )
            or identity in seen
        ):
            raise ValueError("scored and runtime records must match uniquely")
        seen.add(identity)
        groups.setdefault(fingerprint, []).append({**row, "_source_run_kind": "live"})
        sources.append(
            {
                "fingerprint": fingerprint,
                "scenario_id": row["scenario_id"],
                "repetition": row["repetition"],
                "run_id": row["run_id"],
                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "score_sha256": hashlib.sha256(score_bytes).hexdigest(),
                "record_sha256": hashlib.sha256(record_bytes).hexdigest(),
            }
        )
    helper = Path(__file__).with_name("evaluation_reporting.py")
    return {
        "schema": "saved-runtime-aggregation-v1",
        "derivation_code_sha256": {
            "system_report": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "evaluation_reporting": hashlib.sha256(helper.read_bytes()).hexdigest(),
        },
        "method": "Aggregated saved case-result checks and recorded metrics using evaluation_reporting; no rescoring. Eligible check counts exclude not-observed and not-applicable checks.",
        "sources": sources,
        "groups": aggregate_scored_runs(groups),
    }


def _render_system_report(report):
    august = report["august"]
    october = report["october"]
    runtime = october["runtime"]
    lines = [
        "# System evaluation: historical editions",
        "",
        "August and October are separate observations with different configurations and evaluator identities. They are not a controlled before/after comparison.",
        "",
        "## August evaluation",
        "",
        f"- August 22, 2026 UTC. Fingerprint: `{august['fingerprint']}`. The original scorecard contains {august['mode_breakdown']['offline']['sample_count']} imported-trace observations and {august['mode_breakdown']['live']['sample_count']} live executions.",
        f"- Imported traces: {august['mode_breakdown']['offline']['passed_count']} passed / {august['mode_breakdown']['offline']['sample_count']} scored; live: {august['mode_breakdown']['live']['passed_count']} passed / {august['mode_breakdown']['live']['sample_count']} scored.",
        "- The imported trace configuration requires verification. The following historical scorecard rates combine the August sources only; metric-specific eligibility counts were not recorded.",
        "- The trace's source configuration is not established by the recorded live-configuration snapshot.",
        "",
        "### Recorded runtime behavior",
        "",
        "| Metric | Recorded value | Population |",
        "| --- | ---: | --- |",
    ]
    configuration = august["live_configuration"]
    roles = configuration["roles"]
    routing = roles["orchestrator"]["prompt_version"]
    lines.insert(
        8,
        f"- Recorded live prompt selections: Orchestrator routing {routing['routing']} / graph-index {routing['graph_index']}, Retriever {roles['retriever']['prompt_version']}, Researcher {roles['researcher']['prompt_version']}, Analyst {roles['analyst']['prompt_version']}, Mentor {roles['mentor']['prompt_version']}. Analyst retrieval: {configuration['analyst_retrieval_mode']}; Visualizer: {configuration['visualizer_retrieval_mode']}.",
    )
    lines.insert(
        9,
        f"- Recorded live models: Orchestrator {roles['orchestrator'].get('model_name', 'unknown')}; Retriever {roles['retriever'].get('model_name', 'unknown')}; Analyst {roles['analyst'].get('model_name', 'unknown')}; Mentor {roles['mentor'].get('model_name', 'unknown')}. Provider and remaining role details are in the unchanged provenance.",
    )
    population = august["runtime_task_behavior"]["run_count"]
    for name, value in august["runtime_task_behavior"].items():
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and name != "run_count"
        ):
            lines.append(
                f"| {name.replace('_', ' ')} | {value:.2%} | {population} total; eligible count not recorded |"
            )
    lines += ["", "### Retrieval, reliability, efficiency, and judge", ""]
    retrieval = august["retrieval_quality"]
    lines.append(
        f"- Retrieval benchmark: {retrieval['case_count']} cases; recorded case pass rate {retrieval['case_pass_rate']:.2%}. Metric-specific eligibility counts are unavailable in the saved scorecard."
    )
    for name in (
        "evidence_chunk_recall",
        "concept_recall",
        "relation_recall",
        "context_precision_at_10",
        "context_recall",
    ):
        value = retrieval.get(name)
        lines.append(
            f"- {name.replace('_', ' ')}: {value:.2%} (eligible count not recorded)."
            if isinstance(value, (int, float))
            else f"- {name.replace('_', ' ')}: N/A; not recorded."
        )
    lines.append(
        f"- Reliability: {august['reliability']['repetition_count']} recorded observations; failure counts: {august['reliability']['failure_type_frequency']}."
    )
    for name in (
        "logical_llm_calls",
        "provider_attempts",
        "retries",
        "tavily_searches",
        "latency_seconds",
        "total_tokens",
    ):
        value = august["efficiency"].get(name, {})
        if isinstance(value, dict) and value.get("mean") is not None:
            lines.append(
                f"- {name.replace('_', ' ')}: mean {value['mean']:.2f} over {value['count']} recorded observations."
            )
        else:
            lines.append(f"- {name.replace('_', ' ')}: N/A; no recorded distribution.")
    lines += [
        "- Historical token counters lack modern field-level availability evidence and are not complete billing totals.",
        "- Judge quality: N/A; saved dimensions contain errors or skipped prerequisites, with zero successful scores.",
        "- Retrieval relevance-label coverage was incomplete. See the [unchanged August scorecard](../2026-08-22_f8d6f3b6f3e0/scorecard.md) and [provenance](../2026-08-22_f8d6f3b6f3e0/provenance.json) for recorded definitions and configurations.",
        "",
        "## October evaluation",
        "",
        f"- Live system evaluation, October 30–31, 2026 UTC: {runtime['planned']} planned executions across {runtime['scenario_count']} scenarios; {runtime['completed']} completed, {runtime['passed']} passed, {runtime['failed']} failed and {runtime['interrupted']} interrupted/unscored.",
        "- Interrupted observations have no scored record. No result here measures later revisions. Budget checks are report-only and do not determine deterministic pass/fail.",
    ]
    lines.extend(f"- {line}" for line in runtime.get("details", []))
    lines += ["", "### Metrics by original evaluator fingerprint", ""]
    all_checks = sorted(
        {name for group in october["groups"].values() for name in group["checks"]}
    )
    for group in runtime["fingerprint_groups"]:
        fingerprint = group["fingerprint"]
        observed = october["groups"][fingerprint]
        lines += [
            f"#### `{fingerprint}`",
            "",
            f"Scored {observed['scored']} of {group['observations']} planned observations; passed {observed['passed']} / {observed['scored']}. Other observations are interrupted and unscored.",
            "",
            "| Recorded check | Pass / eligible | Not observed |",
            "| --- | ---: | ---: |",
        ]
        for name in all_checks:
            counts = observed["checks"].get(name)
            if counts is None:
                lines.append(f"| {name} | N/A — no check recorded | 0 |")
                continue
            eligible = counts["eligible"]
            value = (
                f"{counts['passed']} / {eligible}"
                if eligible
                else "N/A — no observed eligible checks"
            )
            lines.append(f"| {name} | {value} | {counts['not_observed']} |")
        lines += [
            "",
            "| Efficiency metric | Mean | Available / scored |",
            "| --- | ---: | ---: |",
        ]
        for name, value in observed["efficiency"].items():
            average = (
                f"{value['mean']:.2f}"
                if value["mean"] is not None
                else "N/A — not captured in scored records"
            )
            lines.append(
                f"| {name.replace('_', ' ')} | {average} | {value['available']} / {observed['scored']} |"
            )
        lines += [
            "",
            f"Recorded classified failure counts: {observed['failure_types'] or 'none'} (check failures are shown separately above).",
        ]
        for name, count in observed.get("reliability", {}).items():
            lines.append(
                f"- {name.replace('_', ' ')}: {count['count']} / {count['observed']} observed."
                if count["observed"]
                else f"- {name.replace('_', ' ')}: N/A — no observations."
            )
        lines.append("")
    judge = october["judge"]
    lines += [
        "### Answer and graph quality",
        "",
        f"**Unavailable.** {judge['eligible_inputs']} eligible inputs; {judge['logical_attempted']} HTTP 429 attempt; {judge['successful_scored']} successful scores; {judge['unstarted']} unstarted. Judge score and serving identity are unavailable.",
        f"Judge evaluator fingerprint: `{judge['evaluator_fingerprint']}`; configured model `{judge.get('configured_judge_model', 'unknown')}` with {judge.get('configured_endpoint', 'unknown')} routing. No judge rerun was performed.",
        "",
        "### Retrieval and limitations",
        "",
        "October system retrieval-benchmark quality: N/A — no separate system-benchmark scorecard was saved for these live runs. Retrieval status and tool checks above describe workflow contracts, not evidence precision.",
        "October aggregate token usage: N/A — all 12 completed records lack complete token totals. Missing usage is not zero. The small sample, interrupted executions, unresolved graph focus, and unverified deployment/vector parity limit conclusions.",
        "The separate [retrieval optimization experiment](../../experiments/retrieval-comparison/report.md) reports its own cohorts, denominators and timing; its scores are not system-evaluation results.",
        "",
        "## Provenance and derivation",
        "",
        "October metrics aggregate original saved case-result checks and recorded metrics, grouped by original evaluator fingerprint; the source hashes and derivation are in [provenance](./provenance.json). No historical observation was rescored.",
        "August values are copied from its preserved published scorecard, without recomputation. Execution status, retrieval support and generated-answer quality remain separate.",
        "",
    ]
    return "\n".join(lines)


def publish_system_evaluation(
    august_scorecard_path,
    august_provenance_path,
    runtime_path,
    derived_path,
    judge_path,
    *,
    reports_root,
):
    """Publish one immutable consolidated system-report edition from saved sources."""
    import hashlib
    import json
    import shutil
    from pathlib import Path

    from backend.evaluation.evaluation_reporting import (
        _refresh_publication_aliases,
        _write_json,
        _write_publication_index,
    )

    paths = [
        Path(path)
        for path in (
            august_scorecard_path,
            august_provenance_path,
            runtime_path,
            derived_path,
            judge_path,
        )
    ]
    august, original_provenance, runtime_input, derived, judge = (
        json.loads(path.read_text(encoding="utf-8")) for path in paths
    )
    runtime = runtime_input["runtime"]
    groups = derived["groups"]
    if (
        set(groups) != set(runtime["fingerprints"])
        or any(
            groups[item["fingerprint"]]["scored"] != item["completed"]
            for item in runtime["fingerprint_groups"]
        )
        or sum(group["passed"] for group in groups.values()) != runtime["passed"]
    ):
        raise ValueError("derived metrics do not match the recorded October population")
    if august["fingerprint"] != original_provenance["fingerprint"]:
        raise ValueError("August scorecard and provenance identities differ")
    if (
        judge.get("judge_score") is not None
        or judge.get("successful_scored") != 0
        or judge.get("logical_attempted") != judge.get("error_http_429")
    ):
        raise ValueError("system edition requires the saved unavailable judge status")
    august_data = {
        key: august[key]
        for key in (
            "fingerprint",
            "retrieval_quality",
            "runtime_task_behavior",
            "reliability",
            "efficiency",
            "optional_semantic_quality",
            "coverage",
        )
    }
    august_data["mode_breakdown"] = august["diagnostics"]["runtime_mode_breakdown"]
    live = next(
        item
        for item in original_provenance["contributing_runs"]
        if item["run_kind"] == "live"
    )
    config = live["configuration_snapshot"]["fingerprint_input"][
        "application_configuration"
    ]
    models = config["models"]
    august_data["live_configuration"] = {
        "roles": {
            role: {
                key: models[role].get(key)
                for key in ("model_name", "provider", "prompt_version")
                if key in models[role]
            }
            for role in ("orchestrator", "retriever", "researcher", "analyst", "mentor")
        },
        "analyst_retrieval_mode": config["search"]["analyst_retrieval_mode"],
        "visualizer_retrieval_mode": config["search"]["visualizer_retrieval_mode"],
    }
    report = {
        "publication_kind": "system-evaluation",
        "observation_end": runtime_input["observation_end"],
        "august": august_data,
        "october": {"runtime": runtime, "groups": groups, "judge": judge},
    }
    hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in zip(
            (
                "august_scorecard",
                "august_provenance",
                "october_runtime",
                "october_derived",
                "october_judge",
            ),
            paths,
            strict=True,
        )
    }
    snapshot = "system-evaluation-october"
    history_dir = Path(reports_root) / "history" / snapshot
    provenance_path = history_dir / "provenance.json"
    if provenance_path.exists():
        old = json.loads(provenance_path.read_text())
        if (
            old.get("input_hashes") != hashes
            or json.loads((history_dir / "report.json").read_text()) != report
        ):
            raise ValueError(
                "system edition already exists with different source inputs"
            )
    else:
        history_dir.mkdir(parents=True, exist_ok=True)
        _write_json(history_dir / "report.json", report)
        (history_dir / "report.md").write_text(
            _render_system_report(report), encoding="utf-8"
        )
        shutil.copyfile(paths[3], history_dir / "october-metrics.json")
        _write_json(
            provenance_path,
            {
                "publication_kind": "system-evaluation",
                "publication_id": snapshot,
                "observation_end": runtime_input["observation_end"],
                "august_fingerprint": august["fingerprint"],
                "october_fingerprints": runtime["fingerprints"],
                "judge_fingerprint": judge["evaluator_fingerprint"],
                "input_hashes": hashes,
                "october_sources": derived["sources"],
                "derivation": derived["method"],
                "derivation_code_sha256": derived.get("derivation_code_sha256"),
                "historical_source": "reports/evaluation/history/2026-08-22_f8d6f3b6f3e0/scorecard.json",
            },
        )
    _refresh_publication_aliases(Path(reports_root))
    _write_publication_index(Path(reports_root))
    return {
        "snapshot": snapshot,
        "history_directory": history_dir,
        "latest_markdown": Path(reports_root) / "latest.md",
    }
