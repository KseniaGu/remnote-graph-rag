from backend.evaluation.system_report import aggregate_scored_runs


def test_scored_groups_keep_fingerprints_and_eligible_denominators():
    def row(passed, status):
        return {
            "passed": passed,
            "scenario_id": "case",
            "run_id": str(passed),
            "repetition": 1,
            "checks": [
                {"dimension": "routing", "name": "worker_sequence", "status": status}
            ],
            "failure_types": [],
            "metrics": {"provider_attempts": 2, "retries": 0, "total_tokens": None},
        }

    groups = aggregate_scored_runs(
        {
            "first": [row(True, "pass"), row(False, "fail")],
            "second": [row(True, "pass")],
        }
    )
    assert groups["first"]["scored"] == 2
    assert groups["first"]["checks"]["routing.worker_sequence"] == {
        "passed": 1,
        "eligible": 2,
        "not_observed": 0,
    }
    assert groups["second"]["checks"]["routing.worker_sequence"]["eligible"] == 1
    assert groups["first"]["efficiency"]["total_tokens"]["available"] == 0
    assert groups["first"]["efficiency"]["total_tokens"]["mean"] is None


def test_derivation_rejects_unmatched_runtime_and_keeps_source_hashes(tmp_path):
    import json

    import pytest

    from backend.evaluation.system_report import derive_saved_runs

    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(
        json.dumps(
            {"status": "completed", "evaluation_fingerprint": "f1", "run_kind": "live"}
        )
    )
    (run / "case_results.jsonl").write_text(
        json.dumps(
            {
                "run_id": "one",
                "scenario_id": "case",
                "repetition": 1,
                "scenario_version": 1,
                "passed": True,
                "checks": [],
                "failure_types": [],
                "metrics": {},
            }
        )
        + "\n"
    )
    (run / "runtime_records.jsonl").write_text(
        json.dumps(
            {
                "run_id": "two",
                "scenario_id": "case",
                "repetition": 1,
                "scenario_version": 1,
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="match"):
        derive_saved_runs([run])
    (run / "runtime_records.jsonl").write_text(
        json.dumps(
            {
                "run_id": "one",
                "scenario_id": "case",
                "repetition": 1,
                "scenario_version": 1,
            }
        )
        + "\n"
    )
    result = derive_saved_runs([run])
    assert result["groups"]["f1"]["scored"] == 1
    assert len(result["sources"]) == 1
    assert set(result["sources"][0]) == {
        "fingerprint",
        "scenario_id",
        "repetition",
        "run_id",
        "manifest_sha256",
        "score_sha256",
        "record_sha256",
    }


def test_system_publication_keeps_periods_separate_and_preserves_newer_alias(tmp_path):
    import json
    from pathlib import Path

    from backend.evaluation.system_report import publish_system_evaluation

    old = Path("reports/evaluation/history/2026-08-22_f8d6f3b6f3e0")
    october = Path("reports/evaluation/history/controlled-observations")
    derived = {
        "schema": "saved-runtime-aggregation-v1",
        "method": "synthetic",
        "sources": [],
        "groups": {
            item["fingerprint"]: {
                "scored": item["completed"],
                "passed": passed,
                "checks": {},
                "recorded_rates": {},
                "failure_types": {},
                "efficiency": {},
            }
            for item, passed in zip(
                json.loads((october / "runtime.json").read_text())["runtime"][
                    "fingerprint_groups"
                ],
                (6, 2, 2),
                strict=True,
            )
        },
    }
    aggregate = tmp_path / "derived.json"
    aggregate.write_text(json.dumps(derived))
    reports = tmp_path / "reports"
    publication = publish_system_evaluation(
        old / "scorecard.json",
        old / "provenance.json",
        october / "runtime.json",
        aggregate,
        october / "judge.json",
        reports_root=reports,
    )
    output = json.loads((reports / "latest.json").read_text())
    text = (reports / "latest.md").read_text()
    assert "2026-08-22" in text and "2026-10-31" in text
    assert output["reference"]["mode_breakdown"]["offline"]["sample_count"] == 12
    assert output["reference"]["mode_breakdown"]["live"]["sample_count"] == 15
    assert output["current"]["runtime"]["planned"] == 15
    assert output["current"]["judge"]["judge_score"] is None
    assert len(output["current"]["groups"]) == 3
    from backend.evaluation.evaluation_reporting import _render_scorecard

    original = _render_scorecard(json.loads((old / "scorecard.json").read_text()))
    original = "\n".join(
        "#" + line if line.startswith("#") else line for line in original.splitlines()
    )
    assert original in text
    assert "Campaign coverage" in text and "Metric definitions" in text
    assert "Retrieval diagnostics" in text and "Median" in text and "P95" in text
    assert "Worker-step and Tavily-search bounds can gate" in text
    before = (reports / "latest.json").read_bytes()
    publish_system_evaluation(
        old / "scorecard.json",
        old / "provenance.json",
        october / "runtime.json",
        aggregate,
        october / "judge.json",
        reports_root=reports,
    )
    assert (reports / "latest.json").read_bytes() == before
    saved_metrics = publication["history_directory"] / "runtime-metrics.json"
    publish_system_evaluation(
        old / "scorecard.json",
        old / "provenance.json",
        october / "runtime.json",
        saved_metrics,
        october / "judge.json",
        reports_root=reports,
    )

    future_runtime = json.loads((october / "runtime.json").read_text())
    future_runtime["publication_id"] = "future-observations"
    future_runtime["observation_end"] = "2026-11-15"
    runtime_path = tmp_path / "future-runtime.json"
    same_day = {**future_runtime, "observation_end": "2026-10-31"}
    runtime_path.write_text(json.dumps(same_day))
    import pytest

    with pytest.raises(ValueError, match="publication_revision"):
        publish_system_evaluation(
            old / "scorecard.json",
            old / "provenance.json",
            runtime_path,
            aggregate,
            october / "judge.json",
            reports_root=reports,
        )
    same_day["publication_revision"] = 3
    runtime_path.write_text(json.dumps(same_day))
    same_day_publication = publish_system_evaluation(
        old / "scorecard.json",
        old / "provenance.json",
        runtime_path,
        aggregate,
        october / "judge.json",
        reports_root=reports,
    )
    assert (
        json.loads((reports / "latest_provenance.json").read_text())["publication_id"]
        == same_day_publication["snapshot"]
    )
    runtime_path.write_text(json.dumps(future_runtime))
    future_judge = json.loads((october / "judge.json").read_text())
    future_judge.update(
        {
            "logical_attempted": 1,
            "successful_scored": 1,
            "error_http_429": 0,
            "unstarted": 3,
            "judge_score": None,
            "dimensions": {
                "graph_usefulness": {
                    "score": 0.75,
                    "status": "Scored",
                    "coverage": "1 / 1",
                }
            },
        }
    )
    judge_path = tmp_path / "future-judge.json"
    judge_path.write_text(json.dumps(future_judge))
    future = publish_system_evaluation(
        old / "scorecard.json",
        old / "provenance.json",
        runtime_path,
        aggregate,
        judge_path,
        reports_root=reports,
    )
    assert future["snapshot"] != publication["snapshot"]
    assert (publication["history_directory"] / "report.json").read_bytes() == before
    assert (
        "| Graph Usefulness | 0.75 | Scored | 1 / 1 |"
        in (reports / "latest.md").read_text()
    )
    assert (
        json.loads((reports / "latest.json").read_text())["current"]["judge"][
            "dimensions"
        ]["graph_usefulness"]["score"]
        == 0.75
    )


def test_optimization_publication_cannot_replace_latest_system_report(tmp_path):
    import json

    from backend.evaluation.evaluation_reporting import _refresh_publication_aliases

    reports = tmp_path / "reports"
    system = reports / "history" / "system"
    system.mkdir(parents=True)
    (system / "provenance.json").write_text(
        json.dumps(
            {"publication_kind": "system-evaluation", "observation_end": "2026-10-31"}
        )
    )
    (system / "report.md").write_text("System evaluation")
    (system / "report.json").write_text("{}")
    _refresh_publication_aliases(reports)
    experiment = reports / "history" / "experiment"
    experiment.mkdir()
    (experiment / "provenance.json").write_text(
        json.dumps(
            {
                "publication_kind": "optimization-experiment",
                "observation_end": "2026-11-01",
            }
        )
    )
    _refresh_publication_aliases(reports)
    assert (reports / "latest.md").read_text() == "System evaluation"


def test_older_scorecard_cannot_replace_latest_system_report(tmp_path):
    import json

    from backend.evaluation.evaluation_reporting import _refresh_publication_aliases

    reports = tmp_path / "reports"
    for name, kind, date in (
        ("system", "system-evaluation", "2026-10-31"),
        ("older", "scorecard", "2026-08-22"),
    ):
        snapshot = reports / "history" / name
        snapshot.mkdir(parents=True)
        (snapshot / "provenance.json").write_text(
            json.dumps({"publication_kind": kind, "observation_end": date})
        )
        (snapshot / "report.md").write_text(name)
        (snapshot / "report.json").write_text("{}")
    _refresh_publication_aliases(reports)
    assert (reports / "latest.md").read_text() == "system"


def test_saved_gating_metadata_is_preserved():
    row = {
        "passed": False,
        "checks": [
            {
                "dimension": "budget",
                "name": "worker_steps",
                "status": "fail",
                "gating": True,
            },
            {
                "dimension": "budget",
                "name": "tavily_searches",
                "status": "pass",
                "gating": True,
            },
            {
                "dimension": "budget",
                "name": "latency_seconds",
                "status": "fail",
                "gating": False,
            },
        ],
        "failure_types": [],
        "metrics": {},
    }
    group = aggregate_scored_runs({"original": [row]})["original"]
    assert group["gating"]["budget.worker_steps"] == {"true": 1}
    assert group["gating"]["budget.tavily_searches"] == {"true": 1}
    assert group["gating"]["budget.latency_seconds"] == {"false": 1}
    assert group["passed"] == 0
