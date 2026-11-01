"""Availability-aware token reporting with synthetic runtime and judge records."""

from backend.evaluation.evaluation_reporting import (
    _efficiency_metrics,
    _semantic_metrics,
)


def _usage(
    input_, output, total, observed, attempted, *, complete, method="event_unique_v1"
):
    return {
        "usage_method": method,
        "provider_attempts": attempted,
        "input_tokens": input_,
        "output_tokens": output,
        "total_tokens": total,
        "token_coverage": {
            name: {
                "observed": observed,
                "attempted": attempted,
                "subtotal": value,
                "complete": complete,
            }
            for name, value in (
                ("input_tokens", input_),
                ("output_tokens", output),
                ("total_tokens", total),
            )
        },
    }


def _event_record(usage, *event_ids):
    return {"usage": usage, "providers": [{"run_id": value} for value in event_ids]}


def test_runtime_efficiency_excludes_partial_and_legacy_but_includes_zero():
    complete = _usage(4, 6, 10, 1, 1, complete=True)
    zero = _usage(0, 0, 0, 1, 1, complete=True)
    partial = _usage(None, None, None, 1, 2, complete=False)
    partial["token_coverage"]["total_tokens"]["subtotal"] = 7
    legacy = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    records = [
        _event_record(complete, "one"),
        _event_record(zero, "two"),
        _event_record(partial, "three"),
        {"usage": legacy},
    ]
    rows = [
        {"passed": passed, "metrics": {"total_tokens": token}}
        for passed, token in [(True, 10), (True, 0), (False, 7), (False, 0)]
    ]
    for index, (row, record) in enumerate(zip(rows, records, strict=True)):
        identity = {
            "_source_run_id": "run",
            "run_id": str(index),
            "scenario_id": "case",
            "repetition": index,
        }
        row.update(identity)
        record.update(identity)
    result = _efficiency_metrics(rows, records)
    assert result["total_tokens"]["count"] == 2
    assert result["total_tokens"]["mean"] == 5
    coverage = result["token_coverage"]["total_tokens"]
    assert coverage["complete_runs"] == 2
    assert coverage["eligible_runs"] == 4
    assert coverage["observed_attempts"] == 3
    assert coverage["attempted"] is None  # legacy attempt denominator unknown
    assert coverage["known_attempted"] == 4
    assert coverage["partial_observed_subtotal"] == 7
    assert result["tokens_per_successful_run"] is None


def test_complete_zero_counts_and_all_complete_cost_ratio():
    records = [
        _event_record(_usage(0, 0, 0, 1, 1, complete=True), "one"),
        _event_record(_usage(4, 6, 10, 1, 1, complete=True), "two"),
    ]
    rows = [{"passed": True, "metrics": {}}, {"passed": False, "metrics": {}}]
    for index, (row, record) in enumerate(zip(rows, records, strict=True)):
        identity = {
            "_source_run_id": "run",
            "run_id": str(index),
            "scenario_id": "case",
            "repetition": index,
        }
        row.update(identity)
        record.update(identity)
    result = _efficiency_metrics(rows, records)
    assert result["total_tokens"]["mean"] == 5
    assert result["tokens_per_successful_run"] == 10
    records[1]["run_id"] = "mismatch"
    assert _efficiency_metrics(rows, records)["tokens_per_successful_run"] is None
    assert result["token_coverage"]["total_tokens"]["attempted"] == 2


def test_judge_usage_is_separate_skips_do_not_add_attempts():
    base = {
        "dimension": "graph_usefulness",
        "judge_provider": "fixture",
        "judge_model": "fixture",
        "rubric_version": "v1",
    }
    rows = [
        base
        | {
            "status": "success",
            "score": True,
            "provider_call_attempted": True,
            "observed_request_id": "judge-one",
            "usage": _usage(1, 2, 3, 1, 1, complete=True),
        },
        base
        | {
            "status": "error",
            "provider_call_attempted": True,
            "usage": _usage(None, None, None, 0, 1, complete=False),
        },
        base
        | {
            "status": "skipped",
            "provider_call_attempted": False,
            "usage": _usage(None, None, None, 0, 0, complete=False),
        },
    ]
    result = _semantic_metrics(rows)["judge_usage"]
    assert result["eligible_runs"] == 2
    assert result["skipped_count"] == 1
    assert result["total_tokens"]["mean"] == 3
    assert result["token_coverage"]["total_tokens"]["observed_attempts"] == 1
    assert result["token_coverage"]["total_tokens"]["attempted"] == 2
    assert result["token_coverage"]["total_tokens"]["complete_runs"] == 1


def test_missing_record_and_field_specific_coverage_are_not_hidden():
    partial = _usage(None, None, None, 0, 2, complete=False)
    partial["input_tokens"] = 4
    partial["token_coverage"]["input_tokens"] = {
        "observed": 2,
        "attempted": 2,
        "subtotal": 4,
        "complete": True,
    }
    partial["token_coverage"]["output_tokens"] = {
        "observed": 1,
        "attempted": 2,
        "subtotal": 0,
        "complete": False,
    }
    identity = {
        "_source_run_id": "run",
        "run_id": "1",
        "scenario_id": "case",
        "repetition": 1,
    }
    missing = {
        "_source_run_id": "run",
        "run_id": "2",
        "scenario_id": "case",
        "repetition": 2,
    }
    result = _efficiency_metrics(
        [identity | {"passed": False}, missing | {"passed": False}],
        [identity | _event_record(partial, "one", "two")],
    )
    assert result["input_tokens"]["mean"] == 4
    assert result["output_tokens"]["mean"] is None
    assert result["token_coverage"]["input_tokens"]["eligible_runs"] is None
    assert result["unpaired_row_count"] == 1
    assert result["token_coverage"]["output_tokens"]["partial_observed_subtotal"] == 0
    assert result["token_coverage"]["total_tokens"]["attempted"] is None
    assert result["tokens_per_successful_run"] is None


def test_markdown_exposes_separate_coverage_and_unavailable_denominators():
    from backend.evaluation.evaluation_reporting import (
        _render_efficiency,
        _render_semantic_quality,
    )

    records = [{"usage": _usage(0, 0, 0, 1, 1, complete=True)}, {"usage": {}}]
    lines = []
    _render_efficiency(
        lines, _efficiency_metrics([{"passed": True}, {"passed": False}], records)
    )
    _render_semantic_quality(lines, _semantic_metrics([]))
    markdown = "\n".join(lines)
    assert "Runtime token coverage" in markdown
    assert "Judge token coverage" in markdown
    assert "Unknown attempt runs" in markdown
    assert "Complete runs" in markdown
    assert "Eligible runs" in markdown
    assert "N/A" in markdown


def test_failed_unobserved_manifest_prevents_complete_cohort_claim():
    usage = _usage(4, 6, 10, 1, 1, complete=True)
    identity = {
        "_source_run_id": "run",
        "run_id": "1",
        "scenario_id": "case",
        "repetition": 1,
    }
    records = [identity | {"usage": usage}]
    rows = [identity | {"passed": True, "metrics": {}}]
    result = _efficiency_metrics(rows, records, unobserved_manifests=1)
    coverage = result["token_coverage"]["total_tokens"]
    assert coverage["eligible_runs"] is None
    assert coverage["known_eligible_runs"] == 1
    assert coverage["unobserved_manifests"] == 1
    assert coverage["attempted"] is None
    assert result["tokens_per_successful_run"] is None


def test_only_failed_manifest_has_unknown_usage_coverage_not_empty_success():
    result = _efficiency_metrics([], [], unobserved_manifests=1)
    assert result["token_coverage"]["total_tokens"]["eligible_runs"] is None
    assert result["token_coverage"]["total_tokens"]["attempted"] is None
    assert result["token_coverage"]["total_tokens"]["unobserved_manifests"] == 1
    assert result["tokens_per_successful_run"] is None


def test_judge_missing_manifest_has_unknown_coverage():
    coverage = _semantic_metrics([], unobserved_manifests=1)["judge_usage"][
        "token_coverage"
    ]["total_tokens"]
    assert coverage["eligible_runs"] is None
    assert coverage["attempted"] is None
    assert coverage["unobserved_manifests"] == 1


def test_scorecard_carries_failed_and_empty_manifest_availability(tmp_path):
    import json

    from backend.evaluation.evaluation_reporting import _build_scorecard

    run = tmp_path / "complete"
    run.mkdir()
    identity = {"run_id": "observation", "scenario_id": "case", "repetition": 1}
    (run / "case_results.jsonl").write_text(
        json.dumps(identity | {"passed": True, "checks": [], "metrics": {}}) + "\n"
    )
    (run / "runtime_records.jsonl").write_text(
        json.dumps(identity | {"usage": _usage(4, 6, 10, 1, 1, complete=True)}) + "\n"
    )
    valid = {
        "run_id": "valid",
        "run_kind": "live",
        "status": "completed",
        "_run_dir": str(run),
    }
    failed = {
        "run_id": "failed",
        "run_kind": "live",
        "status": "failed",
        "_run_dir": str(tmp_path / "failed"),
    }
    empty = {
        "run_id": "empty",
        "run_kind": "judge",
        "status": "completed",
        "_run_dir": str(tmp_path / "empty"),
    }
    result = _build_scorecard("fixture", [valid, failed, empty], [valid, empty])
    assert (
        result["efficiency"]["token_coverage"]["total_tokens"]["unobserved_manifests"]
        == 1
    )
    assert result["efficiency"]["tokens_per_successful_run"] is None
    assert (
        result["optional_semantic_quality"]["judge_usage"]["token_coverage"][
            "total_tokens"
        ]["unobserved_manifests"]
        == 1
    )


def test_scorecard_shows_coverage_even_when_no_scored_runtime_record():
    from backend.evaluation.evaluation_reporting import _render_scorecard

    result = _efficiency_metrics([], [], unobserved_manifests=1)
    scorecard = {
        "fingerprint": "fixture",
        "generated_at": "",
        "history_run_count": 1,
        "selected_run_count": 0,
        "coverage": {"complete": False, "missing_requirements": [], "requirements": []},
        "retrieval_quality": {"status": "N/A"},
        "runtime_task_behavior": {"status": "N/A"},
        "reliability": {"status": "N/A"},
        "efficiency": result,
        "optional_semantic_quality": _semantic_metrics([]),
        "diagnostics": {},
        "metric_definitions": {},
    }
    markdown = _render_scorecard(scorecard)
    assert "Runtime token coverage" in markdown
    assert "Unobserved manifests" in markdown


def test_legacy_positive_live_attempts_remain_lower_bound_not_token_completeness():
    legacy = {
        "source_mode": "controlled_live",
        "usage": {"provider_attempts": 3, "total_tokens": 0},
    }
    result = _token_summary_for_test(legacy)
    assert result["known_attempted"] == 3
    assert result["attempted"] is None
    assert result["complete_runs"] == 0
    legacy["usage"]["provider_attempts"] = 0
    assert _token_summary_for_test(legacy)["known_attempted"] == 0
    assert _token_summary_for_test(legacy)["attempted"] is None


def _token_summary_for_test(record):
    from backend.evaluation.evaluation_reporting import _token_usage_summary

    return _token_usage_summary([record])["token_coverage"]["total_tokens"]


def test_unpaired_or_duplicate_records_never_form_complete_cohort():
    identity = {
        "_source_run_id": "run",
        "run_id": "r1",
        "scenario_id": "case",
        "repetition": 1,
    }
    row = identity | {"passed": True, "metrics": {}}
    record = identity | _event_record(_usage(4, 6, 10, 1, 1, complete=True), "one")
    bad = record | {"run_id": "different"}
    for records in ([bad], [record, record]):
        result = _efficiency_metrics([row], records)
        assert result["total_tokens"]["mean"] is None
        assert result["token_coverage"]["total_tokens"]["eligible_runs"] is None
        assert result["token_coverage"]["total_tokens"]["attempted"] is None
        assert result["tokens_per_successful_run"] is None
        assert result["unpaired_usage"]["total_tokens"]["mean"] == (
            10 if len(records) == 1 else None
        )


def test_unscored_records_are_unpaired_and_not_complete_cohort():
    record = _event_record(_usage(4, 6, 10, 1, 1, complete=True), "one")
    result = _efficiency_metrics([], [record])
    assert result["total_tokens"]["mean"] is None
    assert result["token_coverage"]["total_tokens"]["eligible_runs"] is None
    assert result["token_coverage"]["total_tokens"]["pairing_incomplete"] is True
    assert result["unpaired_usage"]["total_tokens"]["mean"] == 10


def _paired_event_record(index, event_id, *, attempts=1, coverage_attempts=1, total=10):
    identity = {
        "_source_run_id": f"source-{index}",
        "run_id": f"r-{index}",
        "scenario_id": "case",
        "repetition": index,
    }
    usage = _usage(4, 6, total, coverage_attempts, coverage_attempts, complete=True)
    usage["provider_attempts"] = attempts
    usage["retries"] = 0
    record = identity | {
        "usage": usage,
        "providers": [
            {
                "run_id": event_id,
                "input_tokens": 4,
                "output_tokens": 6,
                "total_tokens": total,
            }
        ],
    }
    row = identity | {
        "passed": True,
        "metrics": {"provider_attempts": attempts, "retries": 0},
    }
    return row, record


def test_runtime_duplicate_event_ids_across_records_do_not_count_twice():
    rows, records = zip(
        _paired_event_record(1, "same"), _paired_event_record(2, "same"), strict=True
    )
    result = _efficiency_metrics(list(rows), list(records))
    assert result["total_tokens"]["mean"] is None
    assert result["tokens_per_successful_run"] is None
    assert result["token_coverage"]["total_tokens"]["identity_incomplete"] is True


def test_missing_event_id_and_inconsistent_attempts_never_complete():
    row, record = _paired_event_record(1, "")
    result = _efficiency_metrics([row], [record])
    assert result["total_tokens"]["mean"] is None
    assert result["tokens_per_successful_run"] is None
    row, record = _paired_event_record(1, "evt", attempts=4)
    result = _efficiency_metrics([row], [record])
    assert result["total_tokens"]["mean"] is None
    assert result["tokens_per_successful_run"] is None


def test_different_event_ids_with_equal_values_remain_distinct():
    rows, records = zip(
        _paired_event_record(1, "a"), _paired_event_record(2, "b"), strict=True
    )
    result = _efficiency_metrics(list(rows), list(records))
    assert result["total_tokens"]["count"] == 2
    assert result["total_tokens"]["mean"] == 10
    assert result["token_coverage"]["total_tokens"]["attempted"] == 2


def test_retry_rate_unknown_if_any_paired_run_has_unknown_counters():
    row1, record1 = _paired_event_record(1, "a")
    row2, record2 = _paired_event_record(2, "b")
    row1["metrics"]["retries"] = 1
    row2["metrics"]["provider_attempts"] = None
    row2["metrics"]["retries"] = None
    result = _efficiency_metrics([row1, row2], [record1, record2])
    assert result["retry_rate"] is None
    assert result["retry_observed_attempts"] == 1
    assert result["retry_observed_retries"] == 1


def test_judge_duplicate_or_missing_request_ids_never_complete():
    base = {
        "dimension": "graph_usefulness",
        "judge_provider": "fixture",
        "judge_model": "fixture",
        "rubric_version": "v1",
        "status": "success",
        "provider_call_attempted": True,
        "usage": _usage(1, 2, 3, 1, 1, complete=True) | {"provider_attempts": 1},
    }
    duplicate = _semantic_metrics(
        [base | {"observed_request_id": "same"}, base | {"observed_request_id": "same"}]
    )["judge_usage"]
    assert duplicate["total_tokens"]["mean"] is None
    missing = _semantic_metrics([base])["judge_usage"]
    assert missing["total_tokens"]["mean"] is None


def test_unpaired_record_duplicate_event_never_promotes_paired_coverage():
    row, record = _paired_event_record(1, "shared")
    _, unpaired = _paired_event_record(2, "shared")
    result = _efficiency_metrics([row], [record, unpaired])
    assert result["total_tokens"]["mean"] is None
    assert result["unpaired_usage"]["total_tokens"]["mean"] is None
    assert result["token_coverage"]["total_tokens"]["identity_incomplete"] is True


def test_retry_rate_rejects_disagreement_between_case_and_record_counters():
    row, record = _paired_event_record(1, "a")
    row["metrics"]["provider_attempts"] = 1
    row["metrics"]["retries"] = 1
    record["usage"]["provider_attempts"] = 4
    record["usage"]["retries"] = 0
    result = _efficiency_metrics([row], [record])
    assert result["retry_rate"] is None
    assert result["retry_counter_mismatch_runs"] == 1


def test_cross_record_duplicate_recovers_unique_event_subtotal_once():
    row1, record1 = _paired_event_record(1, "shared")
    row2, record2 = _paired_event_record(2, "shared")
    record1["providers"].append(
        {"run_id": "unique-a", "input_tokens": 2, "output_tokens": 3, "total_tokens": 5}
    )
    record2["providers"].append(
        {"run_id": "unique-b", "input_tokens": 3, "output_tokens": 4, "total_tokens": 7}
    )
    for record in (record1, record2):
        record["usage"]["provider_attempts"] = 2
        for entry in record["usage"]["token_coverage"].values():
            entry.update(attempted=2, observed=2)
    result = _efficiency_metrics([row1, row2], [record1, record2])
    coverage = result["token_coverage"]["total_tokens"]
    assert result["total_tokens"]["mean"] is None
    assert coverage["partial_observed_subtotal"] == 22  # shared 10 once + 5 + 7
    assert coverage["observed_attempts"] == 3
    assert coverage["attempted"] is None


def test_cross_record_conflicting_duplicate_excludes_only_conflicted_event():
    row1, record1 = _paired_event_record(1, "shared")
    row2, record2 = _paired_event_record(2, "shared")
    record2["providers"][0]["total_tokens"] = 11
    record1["providers"].append(
        {"run_id": "unique-a", "input_tokens": 2, "output_tokens": 3, "total_tokens": 5}
    )
    record2["providers"].append(
        {"run_id": "unique-b", "input_tokens": 3, "output_tokens": 4, "total_tokens": 7}
    )
    for record in (record1, record2):
        record["usage"]["provider_attempts"] = 2
        for entry in record["usage"]["token_coverage"].values():
            entry.update(attempted=2, observed=2)
    result = _efficiency_metrics([row1, row2], [record1, record2])
    coverage = result["token_coverage"]["total_tokens"]
    assert result["total_tokens"]["mean"] is None
    assert coverage["partial_observed_subtotal"] == 12
    assert coverage["observed_attempts"] == 2
    assert coverage["attempted"] is None


def test_judge_equal_duplicate_request_counts_once_and_reports_known_lower_bound():
    base = {
        "dimension": "graph_usefulness",
        "judge_provider": "fixture",
        "judge_model": "fixture",
        "rubric_version": "v1",
        "status": "success",
        "provider_call_attempted": True,
        "usage": _usage(1, 2, 3, 1, 1, complete=True),
        "observed_request_id": "same",
    }
    result = _semantic_metrics([base, dict(base)])["judge_usage"]
    coverage = result["token_coverage"]["total_tokens"]
    assert result["total_tokens"]["mean"] is None
    assert coverage["partial_observed_subtotal"] == 3
    assert coverage["observed_attempts"] == 1
    assert coverage["known_attempted"] == 1
    assert coverage["attempted"] is None


def test_judge_conflicting_duplicate_request_excludes_ambiguous_usage():
    base = {
        "dimension": "graph_usefulness",
        "judge_provider": "fixture",
        "judge_model": "fixture",
        "rubric_version": "v1",
        "status": "success",
        "provider_call_attempted": True,
        "usage": _usage(1, 2, 3, 1, 1, complete=True),
        "observed_request_id": "same",
    }
    other = base | {"usage": _usage(1, 4, 5, 1, 1, complete=True)}
    result = _semantic_metrics([base, other])["judge_usage"]
    coverage = result["token_coverage"]["total_tokens"]
    assert result["total_tokens"]["mean"] is None
    assert coverage["partial_observed_subtotal"] is None
    assert coverage["observed_attempts"] == 0
    assert coverage["known_attempted"] == 1
    assert coverage["attempted"] is None
