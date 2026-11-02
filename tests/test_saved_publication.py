import json
from pathlib import Path

from backend.evaluation.evaluation_reporting import publish_saved_evaluation


def test_saved_publication_keeps_identities_and_is_idempotent(tmp_path):
    runtime = {
        "publication_id": "synthetic-observations",
        "observation_end": "2026-10-31",
        "runtime": {
            "planned": 3,
            "completed": 2,
            "passed": 1,
            "failed": 1,
            "interrupted": 1,
            "scenario_count": 2,
            "fingerprints": ["runtime-a", "runtime-b"],
        },
    }
    retrieval = {
        "coverage": {"precision_observations": 4, "recall_available": 3},
        "provenance": {"accepted_source_tree": "retrieval-tree"},
    }
    judge = {
        "eligible_inputs": 2,
        "logical_attempted": 1,
        "successful_scored": 0,
        "error_http_429": 1,
        "unstarted": 1,
        "judge_score": None,
        "evaluator_fingerprint": "judge-fingerprint",
    }
    inputs = []
    for name, value in (
        ("runtime", runtime),
        ("retrieval", retrieval),
        ("judge", judge),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value))
        inputs.append(path)
    reports = tmp_path / "reports"
    first = publish_saved_evaluation(*inputs, reports_root=reports)
    original = (reports / "latest.json").read_bytes()
    second = publish_saved_evaluation(*inputs, reports_root=reports)
    score = json.loads(original)
    provenance = json.loads((reports / "latest_provenance.json").read_text())
    assert score["runtime"]["fingerprints"] == ["runtime-a", "runtime-b"]
    assert score["judge"]["judge_score"] is None
    assert provenance["runtime_fingerprints"] == ["runtime-a", "runtime-b"]
    assert provenance["judge_evaluator_fingerprint"] == "judge-fingerprint"
    assert "1 / 2" in (reports / "latest.md").read_text()
    assert original == (reports / "latest.json").read_bytes()
    assert first["snapshot"] == second["snapshot"]
    assert (reports / "history" / first["snapshot"] / "report.md").is_file()
    assert Path(first["history_directory"]).is_relative_to(reports)


def test_older_scorecard_publication_cannot_replace_newer_saved_report(
    tmp_path, monkeypatch
):
    from backend.evaluation import evaluation_reporting as reporting

    reports = tmp_path / "reports"
    snapshot = reports / "history" / "newer"
    snapshot.mkdir(parents=True)
    newer = {
        "publication_kind": "saved-observations",
        "publication_id": "newer",
        "observation_end": "2026-10-31",
        "runtime": {},
    }
    (snapshot / "provenance.json").write_text(json.dumps(newer))
    (snapshot / "report.json").write_text(
        json.dumps(
            {
                "observation_end": "2026-10-31",
                "runtime": {
                    "planned": 2,
                    "completed": 1,
                    "passed": 1,
                    "failed": 0,
                    "interrupted": 1,
                    "scenario_count": 1,
                    "fingerprints": ["a"],
                },
                "retrieval": {
                    "coverage": {"precision_observations": 1, "recall_available": 1}
                },
                "judge": {
                    "eligible_inputs": 1,
                    "logical_attempted": 0,
                    "successful_scored": 0,
                    "error_http_429": 0,
                    "unstarted": 1,
                    "judge_score": None,
                    "evaluator_fingerprint": "j",
                },
            }
        )
    )
    (snapshot / "report.md").write_text("# Newer report\n")
    reporting._refresh_publication_aliases(reports)
    before = [
        (reports / name).read_bytes()
        for name in ("latest.md", "latest.json", "latest_provenance.json")
    ]
    root = tmp_path / "evaluation"
    fingerprint = "old-fingerprint"
    folder = root / "fingerprints" / fingerprint
    folder.mkdir(parents=True)
    scorecard = {
        "fingerprint": fingerprint,
        "generated_at": "2026-08-22T00:00:00Z",
        "selected_run_count": 1,
        "contributing_run_ids": [],
        "coverage": {"complete": False},
    }
    (folder / "scorecard.md").write_text("old report")
    (folder / "scorecard.json").write_text(json.dumps(scorecard))
    (folder / "history.json").write_text(
        json.dumps({"runs": [{"created_at": "2026-08-22T00:00:00+00:00"}]})
    )
    monkeypatch.setattr(
        reporting,
        "build_evaluation_reports",
        lambda *args, **kwargs: {fingerprint: scorecard},
    )
    reporting.publish_evaluation_report(root, reports, fingerprint=fingerprint)
    assert before == [
        (reports / name).read_bytes()
        for name in ("latest.md", "latest.json", "latest_provenance.json")
    ]
    assert (
        reports / "history" / "2026-08-22_old-fingerpr" / "scorecard.md"
    ).read_text() == "old report"


def test_saved_cli_publishes_consistent_private_safe_outputs(tmp_path):
    from scripts.build_evaluation_report import main

    source = Path("reports/evaluation/history/controlled-observations")
    reports = tmp_path / "reports"
    args = [
        "--publish",
        "--reports-root",
        str(reports),
        "--saved-runtime",
        str(source / "runtime.json"),
        "--saved-retrieval",
        str(source / "retrieval.json"),
        "--saved-judge",
        str(source / "judge.json"),
        "--saved-report",
        str(source / "source-report.md"),
    ]
    assert main(args) == 0
    result = json.loads((reports / "latest.json").read_text())
    provenance = json.loads((reports / "latest_provenance.json").read_text())
    markdown = (reports / "latest.md").read_text()
    assert result["runtime"]["completed"] == 12
    assert result["runtime"]["passed"] == 10
    assert result["runtime"]["interrupted"] == 3
    assert result["retrieval"]["coverage"]["precision_observations"] == 176
    assert result["retrieval"]["coverage"]["recall_available"] == 145
    assert result["judge"]["judge_score"] is None
    assert provenance["runtime_fingerprints"] == result["runtime"]["fingerprints"]
    assert provenance["retrieval_provenance"] == result["retrieval"]["provenance"]
    assert (
        provenance["judge_evaluator_fingerprint"]
        == result["judge"]["evaluator_fingerprint"]
    )
    assert (
        "10 / 12" in markdown
        and "176 observations" in markdown
        and "score: N/A" in markdown
    )
    assert not any(path.name == "diagnostics" for path in reports.rglob("*"))
    previous = (reports / "latest.json").read_bytes()
    assert main(args) == 0
    assert (reports / "latest.json").read_bytes() == previous


def test_existing_publication_rejects_changed_source_without_rewriting(tmp_path):
    import pytest

    source = Path("reports/evaluation/history/controlled-observations")
    reports = tmp_path / "reports"
    runtime = tmp_path / "runtime.json"
    runtime.write_bytes((source / "runtime.json").read_bytes())
    args = (runtime, source / "retrieval.json", source / "judge.json")
    publish_saved_evaluation(*args, reports_root=reports)
    prior = (reports / "latest.json").read_bytes()
    value = json.loads(runtime.read_text())
    value["limitations"].append("A different source.")
    runtime.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="different inputs"):
        publish_saved_evaluation(*args, reports_root=reports)
    assert (reports / "latest.json").read_bytes() == prior


def test_rejects_impossible_counts_before_publishing(tmp_path):
    import pytest

    source = Path("reports/evaluation/history/controlled-observations")
    runtime = json.loads((source / "runtime.json").read_text())
    runtime["runtime"]["planned"] = -15
    runtime["runtime"]["completed"] = -12
    runtime["runtime"]["passed"] = -10
    runtime["runtime"]["failed"] = -2
    runtime["runtime"]["interrupted"] = -3
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(runtime))
    with pytest.raises(ValueError):
        publish_saved_evaluation(
            path,
            source / "retrieval.json",
            source / "judge.json",
            reports_root=tmp_path / "reports",
        )
    assert not (tmp_path / "reports").exists()
