"""Aggregate saved system-evaluation scores without rescoring observations."""

from collections import Counter
from typing import Any

from backend.evaluation.evaluation_reporting import (
    _reliability_metrics,
    _runtime_task_metrics,
    _values_summary,
)


def aggregate_scored_runs(groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Summarize recorded checks separately for each original evaluator identity."""
    result = {}
    for fingerprint, rows in groups.items():
        statuses: dict[str, Counter] = {}
        gating: dict[str, Counter] = {}
        for row in rows:
            for check in row.get("checks", []):
                name = f"{check['dimension']}.{check['name']}"
                statuses.setdefault(name, Counter())[check["status"]] += 1
                if type(check.get("gating")) is bool:
                    gating.setdefault(name, Counter())[
                        str(check["gating"]).lower()
                    ] += 1
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
        metric_checks = {
            "routing_correctness": "routing.worker_sequence",
            "required_agent_compliance": "routing.required_agents",
            "forbidden_agent_compliance": "routing.forbidden_agents",
            "one_tool_per_worker": "tools.one_tool_per_worker",
            "retrieval_status_correctness": "retrieval.retrieval_status",
            "source_exhaustion_correctness": "fallback.sources_exhausted",
            "final_response_rate": "answer.final_response_present",
            "modality_correctness": "modality.requested_modality",
            "termination_rate": "termination.terminated",
        }
        metric_fields = {
            "tool_selection_correctness": "tool_selection_correct",
            "tool_argument_validity": "tool_argument_valid",
            "local_to_web_fallback_correctness": "fallback_correct",
            "unnecessary_web_rate": "unnecessary_web",
        }
        populations = {
            name: sum(statuses.get(check, {}).values())
            for name, check in metric_checks.items()
        }
        populations.update(
            {
                name: sum(row.get("metrics", {}).get(field) is not None for row in rows)
                for name, field in metric_fields.items()
            }
        )
        populations["task_success_rate"] = len(rows)
        populations["graph_contract_rate"] = sum(
            any(
                check.get("dimension") == "graph"
                and check.get("status") not in {"not_applicable", "not_observed"}
                for check in row.get("checks", [])
            )
            for row in rows
        )
        result[fingerprint] = {
            "rate_populations": populations,
            "scored": len(rows),
            "reliability": reliability,
            "passed": sum(row["passed"] is True for row in rows),
            "checks": checks,
            "gating": {name: dict(counts) for name, counts in gating.items()},
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
    records_by_group = {}
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
        records_by_group.setdefault(fingerprint, []).append(record)
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
    aggregates = aggregate_scored_runs(groups)
    for fingerprint, rows in groups.items():
        aggregates[fingerprint]["recorded_reliability"] = _reliability_metrics(
            rows, records_by_group[fingerprint]
        )
    return {
        "schema": "saved-runtime-aggregation-v1",
        "derivation_code_sha256": {
            "system_report": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "evaluation_reporting": hashlib.sha256(helper.read_bytes()).hexdigest(),
        },
        "method": "Aggregated saved case-result checks and recorded metrics using evaluation_reporting; no rescoring. Eligible check counts exclude not-observed and not-applicable checks.",
        "sources": sources,
        "groups": aggregates,
    }


def _render_system_report(report):
    from backend.evaluation.evaluation_reporting import _render_scorecard

    previous = report["reference"]
    current = report["current"]
    runtime, judge = current["runtime"], current["judge"]
    # The scorecard renderer consumes the original saved values; it does not rescore.
    original = _render_scorecard(previous["scorecard"])
    original = "\n".join(
        "#" + line if line.startswith("#") else line for line in original.splitlines()
    )
    lines = [
        "# System evaluation: consolidated report",
        "",
        "The editions below have distinct populations, configurations and evaluator identities. They are not a controlled before/after comparison.",
        "",
        f"## Reference evaluation — {previous['observation_end']} UTC",
        "",
        f"Imported traces: {previous['mode_breakdown']['offline']['sample_count']}; live executions: {previous['mode_breakdown']['live']['sample_count']}. The imported-trace configuration is not established by the recorded live configuration.",
        "",
        original,
        "",
        f"## Current evaluation — {report['observation_end']} UTC",
        "",
        f"{runtime['planned']} planned executions across {runtime['scenario_count']} scenarios; {runtime['completed']} completed, {runtime['passed']} passed, {runtime['failed']} failed and {runtime['interrupted']} interrupted/unscored.",
        "Worker-step and Tavily-search bounds can gate deterministic success; logical calls, provider attempts, retries, token usage and latency bounds are report-only. Saved outcomes are unchanged.",
    ]
    lines.extend(f"- {line}" for line in runtime.get("details", []))
    lines += ["", "### Coverage and runtime behavior", ""]
    for group in runtime["fingerprint_groups"]:
        fingerprint = group["fingerprint"]
        observed = current["groups"][fingerprint]
        lines += [
            f"#### Evaluator `{fingerprint}`",
            "",
            f"{observed['scored']} / {group['observations']} planned observations scored; {observed['passed']} / {observed['scored']} passed.",
            "",
            "| Runtime summary | Result | Eligible observations/checks |",
            "| --- | ---: | ---: |",
        ]
        for name, count in observed.get("rate_populations", {}).items():
            rate = observed["recorded_rates"].get(name)
            display = (
                f"{rate:.1%}" if rate is not None else "N/A — no eligible observations"
            )
            lines.append(f"| {name.replace('_', ' ').title()} | {display} | {count} |")
        lines += [
            "",
            "<details><summary>Recorded contract checks and gating</summary>",
            "",
            "| Workflow metric | Passed / eligible | Not observed | Gating metadata |",
            "| --- | ---: | ---: | --- |",
        ]
        for name, counts in observed["checks"].items():
            value = (
                f"{counts['passed']} / {counts['eligible']}"
                if counts["eligible"]
                else "N/A — no observed eligible checks"
            )
            gating = observed.get("gating", {}).get(name, {})
            gate_label = (
                ", ".join(f"{key}: {count}" for key, count in gating.items())
                or "Not recorded"
            )
            lines.append(
                f"| {name.replace('.', ': ').replace('_', ' ')} | {value} | {counts['not_observed']} | {gate_label} |"
            )
        lines += [
            "",
            "</details>",
            "",
            "##### Efficiency distributions",
            "",
            "| Metric | Available / scored | Mean | Median | P95 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for name, value in observed["efficiency"].items():

            def fmt(key, distribution=value):
                number = distribution.get(key)
                return f"{number:.2f}" if number is not None else "N/A — not captured"

            lines.append(
                f"| {name.replace('_', ' ')} | {value['available']} / {observed['scored']} | {fmt('mean')} | {fmt('median')} | {fmt('p95')} |"
            )
        lines += [
            "",
            "##### Reliability",
            "",
            "| Metric | Result | Population |",
            "| --- | --- | --- |",
        ]
        for name, count in observed.get("reliability", {}).items():
            result = str(count["count"]) if count["observed"] else "N/A — not observed"
            lines.append(
                f"| {name.replace('_', ' ')} | {result} | {count['observed']} observed |"
            )
        lines.append(
            f"| Classified failures | {observed['failure_types'] or 'None recorded'} | {observed['scored']} scored |"
        )
        reliability = observed.get("recorded_reliability", {})
        for name in ("route_consistency", "evidence_set_jaccard_stability"):
            value = reliability.get(name, {})
            if value.get("mean") is not None:
                result = f"{value['mean']:.3f}"
                population = f"{len(value.get('by_scenario', {}))} repeated scenarios"
            else:
                result = f"N/A — {value.get('reason', 'not recorded')}"
                population = "No eligible repeated scenarios"
            lines.append(f"| {name.replace('_', ' ')} | {result} | {population} |")
        lines.append("")
    lines += [
        "Evidence-set Jaccard uses the established convention that two empty evidence sets have similarity 1; this does not establish evidence quality.",
        "",
        "### Retrieval quality and diagnostics",
        "",
        "| Metric | Result | Population |",
        "| --- | --- | --- |",
        "| Evidence recall, precision and graph retrieval quality | N/A — no system retrieval benchmark was saved for these executions | Not measured |",
        "",
        "Workflow retrieval checks above do not measure evidence precision. The [retrieval optimization experiment](../../experiments/retrieval-comparison/report.md) has separate cohorts and results.",
        "",
        "### Answer and graph quality",
        "",
        "| Dimension | Score | Status | Coverage |",
        "| --- | ---: | --- | --- |",
    ]
    for name, item in judge.get("dimensions", {}).items():
        score = item.get("score")
        display = "N/A" if score is None else str(score)
        coverage = item.get("coverage", "Not recorded")
        lines.append(
            f"| {name.replace('_', ' ').title()} | {display} | {item.get('status', 'Unavailable')} | {coverage} |"
        )
    if not judge.get("dimensions"):
        lines.append(
            "| Dimension details | N/A | Unavailable | Not supplied in saved judge status |"
        )
    lines += [
        "",
        f"Eligible inputs: {judge['eligible_inputs']}; attempts: {judge['logical_attempted']}; successful scores: {judge['successful_scored']}; unstarted: {judge['unstarted']}; HTTP 429 errors: {judge.get('error_http_429', 0)}. Missing scores are not zero.",
        f"Judge evaluator: `{judge['evaluator_fingerprint']}`; configured model: `{judge.get('configured_judge_model', 'unknown')}`; observed serving identity: {judge.get('observed_serving_identity') or 'unknown'}.",
        "",
        "### Diagnostics and interpretation",
        "",
        "Unavailable usage remains N/A; deterministic compliance does not establish evidence support or answer quality. See [provenance](./provenance.json) for original identities and source hashes. No historical observation was rescored.",
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
        _publication_entries,
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
        raise ValueError("derived metrics do not match the recorded runtime population")
    if august["fingerprint"] != original_provenance["fingerprint"]:
        raise ValueError("reference scorecard and provenance identities differ")
    counts = ("eligible_inputs", "logical_attempted", "successful_scored", "unstarted")
    if any(type(judge.get(key)) is not int or judge[key] < 0 for key in counts):
        raise ValueError("judge counts must be nonnegative integers")
    if (
        judge["successful_scored"] > judge["logical_attempted"]
        or judge["logical_attempted"]
        + judge["unstarted"]
        + judge.get("skipped", 0)
        + judge.get("uncertain_dispatch", 0)
        != judge["eligible_inputs"]
    ):
        raise ValueError("inconsistent judge counts")
    if not isinstance(judge.get("dimensions", {}), dict):
        raise ValueError("judge dimensions must be a mapping")
    dates = [
        str(item.get("created_at", ""))[:10]
        for item in original_provenance["contributing_runs"]
        if item.get("created_at")
    ]
    reference_date = max(dates) if dates else str(august["generated_at"])[:10]
    report = {
        "publication_kind": "system-evaluation",
        "observation_end": runtime_input["observation_end"],
        "reference": {
            "observation_end": reference_date,
            "scorecard": august,
            "mode_breakdown": august["diagnostics"]["runtime_mode_breakdown"],
        },
        "current": {"runtime": runtime, "groups": groups, "judge": judge},
    }
    hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in zip(
            (
                "reference_scorecard",
                "reference_provenance",
                "runtime",
                "derived",
                "judge",
            ),
            paths,
            strict=True,
        )
    }
    import re

    source_id = runtime_input["publication_id"]
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", source_id):
        raise ValueError("publication identity must be a safe filename")
    revision = runtime_input.get("publication_revision", 2)
    if type(revision) is not int or revision < 1:
        raise ValueError("publication revision must be a positive integer")
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[
        :12
    ]
    snapshot = f"system-evaluation-v2-{source_id}-{digest}"
    history_dir = Path(reports_root) / "history" / snapshot
    for date, folder, previous in _publication_entries(Path(reports_root)):
        if (
            date == runtime_input["observation_end"][:10]
            and previous.get("publication_kind") == "system-evaluation"
            and previous.get("publication_revision", 0) == revision
            and folder.name != snapshot
        ):
            raise ValueError(
                "same-day system edition requires a distinct publication_revision"
            )
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
        shutil.copyfile(paths[3], history_dir / "runtime-metrics.json")
        _write_json(
            provenance_path,
            {
                "publication_kind": "system-evaluation",
                "publication_id": snapshot,
                "observation_end": runtime_input["observation_end"],
                "publication_revision": revision,
                "reference_fingerprint": august["fingerprint"],
                "runtime_fingerprints": runtime["fingerprints"],
                "judge_fingerprint": judge["evaluator_fingerprint"],
                "input_hashes": hashes,
                "runtime_sources": derived["sources"],
                "derivation": derived["method"],
                "derivation_code_sha256": derived.get("derivation_code_sha256"),
                "reference_source": str(paths[0]),
            },
        )
    _refresh_publication_aliases(Path(reports_root))
    _write_publication_index(Path(reports_root))
    return {
        "snapshot": snapshot,
        "history_directory": history_dir,
        "latest_markdown": Path(reports_root) / "latest.md",
    }
