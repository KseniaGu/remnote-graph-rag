from backend.evaluation.runtime_evaluation import (
    ProviderObservation,
    RuntimeUsage,
    _aggregate_event_usage,
    _provider_response_fields,
)


def test_token_fields_distinguish_zero_missing_and_invalid():
    fields = _provider_response_fields({}, {"input_tokens": 0, "output_tokens": "bad"})
    assert fields["input_tokens"] == 0
    assert fields["output_tokens"] is None
    assert fields["total_tokens"] is None
    assert RuntimeUsage().usage_method == "legacy_unknown"


def test_event_totals_distinct_ids_and_missing_metadata():
    providers = [
        ProviderObservation(
            run_id="a",
            node="orchestrator",
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
        ),
        ProviderObservation(
            run_id="b",
            node="researcher",
            input_tokens=3,
            output_tokens=2,
            total_tokens=5,
        ),
        ProviderObservation(
            run_id="c",
            node="researcher",
            input_tokens=3,
            output_tokens=2,
            total_tokens=5,
        ),
    ]
    usage = _aggregate_event_usage(providers, attempted=4)
    assert usage.input_tokens is None
    assert usage.token_coverage["input_tokens"] == {
        "observed": 3,
        "attempted": 4,
        "subtotal": 6,
        "complete": False,
    }
    complete = _aggregate_event_usage(providers, attempted=3)
    assert (complete.input_tokens, complete.output_tokens, complete.total_tokens) == (
        6,
        4,
        10,
    )


def test_duplicate_and_missing_ids_never_invent_completeness():
    event = ProviderObservation(
        run_id="same", input_tokens=1, output_tokens=2, total_tokens=3
    )
    conflict = ProviderObservation(
        run_id="same", input_tokens=7, output_tokens=2, total_tokens=9
    )
    assert _aggregate_event_usage([event, event], attempted=1).total_tokens == 3
    assert _aggregate_event_usage([event, conflict], attempted=1).total_tokens is None
    assert _aggregate_event_usage([event, conflict], attempted=1).output_tokens is None
    assert (
        _aggregate_event_usage(
            [event.model_copy(update={"run_id": ""})], attempted=1
        ).total_tokens
        is None
    )


def test_partial_fields_and_conflicts_remain_field_specific():
    events = [
        ProviderObservation(run_id="a", input_tokens=0, output_tokens=None),
        ProviderObservation(
            run_id="b", input_tokens=2, output_tokens=3, total_tokens=5
        ),
    ]
    usage = _aggregate_event_usage(events, attempted=2)
    assert usage.input_tokens == 2
    assert usage.output_tokens is None
    assert usage.total_tokens is None
    assert usage.token_coverage["output_tokens"]["subtotal"] == 3
    duplicate = events[0].model_copy(update={"input_tokens": 4})
    result = _aggregate_event_usage([*events, duplicate], attempted=2)
    assert result.input_tokens is None
    assert result.token_coverage["input_tokens"]["observed"] == 1


def test_metadata_total_is_preserved_or_derived_only_from_both_components():
    assert (
        _provider_response_fields({}, {"input_tokens": 0, "output_tokens": 0})[
            "total_tokens"
        ]
        == 0
    )
    partial = _provider_response_fields({}, {"input_tokens": 2})
    assert partial["total_tokens"] is None
    result = _provider_response_fields(
        {
            "token_usage": {
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "total_tokens": 9,
            }
        },
        {},
    )
    assert result["total_tokens"] == result["reported_total_tokens"] == 9
    assert result["token_sources"]["total_tokens"] == "response_metadata"


def test_live_uses_all_captured_events_not_partial_turn_counter(monkeypatch):
    import asyncio

    from langchain_core.messages import AIMessage
    from test_runtime_evaluation import make_scenario
    from test_wp41_response_capture import FakeGraph, wrapper

    from backend.evaluation import runtime_live
    from backend.utils.chat_limits import get_current_chat_turn

    class Graph(FakeGraph):
        async def astream_events(self, initial, config, version):
            turn = get_current_chat_turn()
            turn.llm_provider_attempts = 4
            turn.input_tokens, turn.output_tokens, turn.total_tokens = 2, 1, 3
            turn.retries_used = 1
            for rid, node, inp, out in [
                ("o", "orchestrator", 0, 0),
                ("r1", "researcher", 2, 1),
                ("r2", "researcher", 3, 4),
            ]:
                yield {
                    "event": "on_chat_model_end",
                    "run_id": rid,
                    "name": "model",
                    "metadata": {"langgraph_node": node, "ls_provider": "ollama"},
                    "data": {
                        "output": AIMessage(
                            content="",
                            usage_metadata={
                                "input_tokens": inp,
                                "output_tokens": out,
                                "total_tokens": inp + out,
                            },
                        )
                    },
                }
            yield {
                "event": "on_chain_end",
                "name": "LangGraph",
                "data": {"output": self.state(initial)},
            }

    monkeypatch.setattr(runtime_live, "get_workflow", lambda: wrapper(Graph()))
    record = asyncio.run(runtime_live.run_controlled_live_scenario(make_scenario()))
    assert record.usage.provider_attempts == 4
    assert record.usage.total_tokens is None
    assert record.usage.token_coverage["total_tokens"] == {
        "observed": 3,
        "attempted": 4,
        "subtotal": 10,
        "complete": False,
    }
    assert record.usage.turn_telemetry["total_tokens"] == 3


def test_canonical_zero_precedes_legacy_and_legacy_ollama_supported():
    first = _provider_response_fields(
        {"prompt_eval_count": 8, "eval_count": 9},
        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    )
    assert (first["input_tokens"], first["output_tokens"], first["total_tokens"]) == (
        0,
        0,
        0,
    )
    second = _provider_response_fields({"prompt_eval_count": 8, "eval_count": 9}, {})
    assert (
        second["input_tokens"],
        second["output_tokens"],
        second["total_tokens"],
    ) == (8, 9, 17)
    assert second["token_sources"]["total_tokens"] == "derived_input_output"
    for bad in (True, -1, 1.5, "broken"):
        assert (
            _provider_response_fields({}, {"input_tokens": bad})["input_tokens"] is None
        )


def test_unavailable_or_legacy_total_cannot_pass_budget():
    from test_runtime_evaluation import make_scenario, normalize_trace_runs, root_run

    from backend.evaluation.runtime_evaluation import score_runtime_record

    scenario = make_scenario()
    scenario.expectations.budgets.max_total_tokens = 10
    record = normalize_trace_runs(root_run(), scenario)
    record.usage = RuntimeUsage(total_tokens=0)
    result = score_runtime_record(scenario, record)
    check = next(c for c in result.checks if c.name == "total_tokens")
    assert check.status == "not_observed"
    assert result.metrics["total_tokens"] is None
    record.usage = RuntimeUsage(usage_method="event_unique_v1", total_tokens=0)
    assert (
        next(
            c
            for c in score_runtime_record(scenario, record).checks
            if c.name == "total_tokens"
        ).status
        == "not_observed"
    )
    record.usage = _aggregate_event_usage(
        [
            ProviderObservation(
                run_id="z", input_tokens=0, output_tokens=0, total_tokens=0
            )
        ],
        attempted=1,
    )
    assert (
        next(
            c
            for c in score_runtime_record(scenario, record).checks
            if c.name == "total_tokens"
        ).status
        == "pass"
    )


def test_judge_error_and_skip_keep_separate_attempts_and_available_usage():
    from backend.evaluation.runtime_evaluation import _judge_diagnostics

    skipped = _judge_diagnostics(exc=None, raw=None, attempted=False)["usage"]
    assert skipped.provider_attempts == 0
    assert skipped.total_tokens is None
    assert skipped.token_coverage["total_tokens"]["attempted"] == 0
    failed = _judge_diagnostics(
        exc=ValueError("invalid output"),
        raw=None,
        attempted=True,
        transport_metadata={
            "id": "judge-1",
            "http_status_code": 200,
            "token_usage": {
                "prompt_tokens": 4,
                "completion_tokens": 0,
                "total_tokens": 4,
            },
        },
    )["usage"]
    assert failed.provider_attempts == 1
    assert failed.total_tokens == 4
    assert failed.output_tokens == 0
    missing = _judge_diagnostics(exc=TimeoutError(), raw=None, attempted=True)["usage"]
    assert missing.provider_attempts == 1 and missing.total_tokens is None
    assert missing.token_coverage["total_tokens"]["attempted"] == 1


def test_exported_trace_keeps_explicit_attempts_and_never_invents_retries():
    from test_runtime_evaluation import make_scenario, normalize_trace_runs, root_run

    from backend.evaluation.runtime_evaluation import score_runtime_record

    runs = root_run()
    root = next(item for item in runs if not item.get("parent_run_id"))
    root["provider_attempts"] = 4
    root["retries"] = 2
    record = normalize_trace_runs(runs, make_scenario())
    assert record.usage.provider_attempts == 4
    assert record.usage.retries == 2
    assert (
        next(
            c
            for c in score_runtime_record(make_scenario(), record).checks
            if c.name == "provider_attempts"
        ).status
        == "fail"
    )
    assert record.usage.token_coverage["input_tokens"]["attempted"] == 4
    assert record.usage.turn_telemetry["captured_provider_event_observations"] == len(
        record.providers
    )
    del root["provider_attempts"]
    del root["retries"]
    record = normalize_trace_runs(runs, make_scenario())
    assert record.usage.provider_attempts is None
    assert record.usage.retries is None
    assert (
        next(
            c
            for c in score_runtime_record(make_scenario(), record).checks
            if c.name == "provider_attempts"
        ).status
        == "not_observed"
    )
    assert record.usage.token_coverage["input_tokens"]["attempted"] is None


def test_malformed_serialized_provider_usage_never_becomes_observed():
    provider = ProviderObservation.model_validate(
        {
            "run_id": "id",
            "input_tokens": "3",
            "output_tokens": True,
            "total_tokens": "4",
        }
    )
    assert (provider.input_tokens, provider.output_tokens, provider.total_tokens) == (
        None,
        None,
        None,
    )
    usage = _aggregate_event_usage([provider], attempted=1)
    assert usage.input_tokens is usage.output_tokens is usage.total_tokens is None


def test_duplicate_id_with_conflicting_status_or_usage_provenance_is_incomplete():
    event = ProviderObservation(
        run_id="same",
        status="success",
        input_tokens=1,
        output_tokens=2,
        total_tokens=3,
        reported_total_tokens=3,
        token_sources={"total_tokens": "response_metadata"},
    )
    for changed in (
        {"status": "error"},
        {"reported_total_tokens": 4},
        {"token_sources": {"total_tokens": "derived_input_output"}},
    ):
        duplicate = event.model_copy(update=changed)
        result = _aggregate_event_usage([event, duplicate], attempted=1)
        assert (
            result.input_tokens is result.output_tokens is result.total_tokens is None
        )
        assert not result.token_coverage["total_tokens"]["complete"]


def test_conflicting_duplicate_identity_excluded_from_all_observed_subtotals():
    first = ProviderObservation(
        run_id="same",
        input_tokens=1,
        output_tokens=2,
        total_tokens=3,
        configured_provider="provider-a",
        configured_model="model-a",
        stop_reason="stop",
    )
    other = ProviderObservation(
        run_id="other", input_tokens=4, output_tokens=5, total_tokens=9
    )
    for changes in (
        {"configured_provider": "provider-b"},
        {"configured_model": "model-b"},
        {"stop_reason": "length"},
        {"configured_output_token_limit": 128},
    ):
        usage = _aggregate_event_usage(
            [first, first.model_copy(update=changes), other], attempted=2
        )
        assert usage.total_tokens is None
        assert usage.token_coverage["output_tokens"]["observed"] == 1
        assert usage.token_coverage["output_tokens"]["subtotal"] == 5


def test_budget_rejects_serialized_complete_coverage_with_mismatched_attempts():
    from backend.evaluation.runtime_evaluation import (
        RuntimeRecord,
        RuntimeScenario,
        score_runtime_record,
    )

    usage = _aggregate_event_usage(
        [
            ProviderObservation(
                run_id="a", input_tokens=1, output_tokens=2, total_tokens=3
            )
        ],
        attempted=1,
    )
    raw = usage.model_dump()
    raw["provider_attempts"] = 4
    record = RuntimeRecord.model_validate(
        {
            "scenario_id": "s",
            "scenario_version": 1,
            "run_id": "r",
            "source_mode": "controlled_live",
            "input": {"user_message": "hi"},
            "usage": raw,
        }
    )
    scenario = RuntimeScenario.model_validate(
        {
            "id": "s",
            "review_status": "reviewed",
            "description": "fixture",
            "input": {"user_message": "hi"},
            "expectations": {"budgets": {"max_total_tokens": 10}},
        }
    )
    result = score_runtime_record(scenario, record)
    check = next(item for item in result.checks if item.name == "total_tokens")
    assert check.status == "not_observed"


def test_standalone_runtime_summary_reports_field_coverage_and_zero_in_mean():
    from backend.evaluation.runtime_evaluation import (
        RuntimeCaseResult,
        RuntimeRecord,
        summarize_runtime_results,
    )

    def record(key, events, attempted):
        return RuntimeRecord(
            scenario_id="s",
            scenario_version=1,
            run_id=key,
            source_mode="controlled_live",
            input={"user_message": "hi"},
            providers=events,
            usage=_aggregate_event_usage(events, attempted=attempted),
        )

    records = [
        record(
            "zero",
            [
                ProviderObservation(
                    run_id="a", input_tokens=0, output_tokens=0, total_tokens=0
                )
            ],
            1,
        ),
        record(
            "positive",
            [
                ProviderObservation(
                    run_id="b", input_tokens=2, output_tokens=3, total_tokens=5
                )
            ],
            1,
        ),
        record("partial", [ProviderObservation(run_id="c", input_tokens=1)], 2),
    ]
    results = [
        RuntimeCaseResult(
            scenario_id="s",
            scenario_version=1,
            run_id=r.run_id,
            repetition=1,
            source_mode="controlled_live",
            passed=True,
            checks=[],
            failure_types=[],
            metrics={},
        )
        for r in records
    ]
    summary = summarize_runtime_results(records, results)["reliability"]["s"]
    assert summary["tokens"]["mean"] == 2.5
    assert summary["token_coverage"]["total_tokens"]["complete_runs"] == 2
    assert summary["token_coverage"]["total_tokens"]["eligible_runs"] == 3
    assert summary["token_coverage"]["input_tokens"]["observed_attempts"] == 3
    assert summary["token_coverage"]["input_tokens"]["attempted"] == 4
    assert summary["token_coverage"]["input_tokens"]["partial_observed_subtotal"] == 1


def test_standalone_summary_does_not_claim_complete_unmatched_results():
    from backend.evaluation.runtime_evaluation import (
        RuntimeCaseResult,
        RuntimeRecord,
        summarize_runtime_results,
    )

    event = ProviderObservation(
        run_id="evt", input_tokens=1, output_tokens=2, total_tokens=3
    )
    record = RuntimeRecord(
        scenario_id="s",
        scenario_version=1,
        run_id="r",
        source_mode="controlled_live",
        input={"user_message": "hi"},
        providers=[event],
        usage=_aggregate_event_usage([event], attempted=1),
    )
    result = RuntimeCaseResult(
        scenario_id="s",
        scenario_version=1,
        run_id="other",
        repetition=1,
        source_mode="controlled_live",
        passed=True,
        checks=[],
        failure_types=[],
        metrics={},
    )
    summary = summarize_runtime_results([record], [result])["reliability"]["s"]
    assert summary["tokens"]["mean"] is None
    assert summary["token_coverage"]["total_tokens"]["eligible_runs"] is None
    assert summary["token_coverage"]["total_tokens"]["pairing_incomplete"] is True


def test_standalone_summary_keeps_result_without_runtime_record_unavailable():
    from backend.evaluation.runtime_evaluation import (
        RuntimeCaseResult,
        summarize_runtime_results,
    )

    result = RuntimeCaseResult(
        scenario_id="missing",
        scenario_version=1,
        run_id="r",
        repetition=1,
        source_mode="controlled_live",
        passed=False,
        checks=[],
        failure_types=["error"],
        metrics={},
    )
    summary = summarize_runtime_results([], [result])["reliability"]["missing"]
    assert summary["tokens"]["mean"] is None
    assert summary["token_coverage"]["total_tokens"]["eligible_runs"] is None
    assert summary["token_coverage"]["total_tokens"]["pairing_incomplete"] is True
    assert summary["route_consistency"] is None


def test_standalone_summary_rejects_event_ids_shared_across_scenarios():
    from backend.evaluation.runtime_evaluation import (
        RuntimeCaseResult,
        RuntimeRecord,
        summarize_runtime_results,
    )

    def make_pair(scenario_id, event_id):
        event = ProviderObservation(
            run_id=event_id, input_tokens=1, output_tokens=2, total_tokens=3
        )
        record = RuntimeRecord(
            scenario_id=scenario_id,
            scenario_version=1,
            run_id=f"run-{scenario_id}",
            source_mode="controlled_live",
            input={"user_message": "hi"},
            providers=[event],
            usage=_aggregate_event_usage([event], attempted=1),
        )
        result = RuntimeCaseResult(
            scenario_id=scenario_id,
            scenario_version=1,
            run_id=record.run_id,
            repetition=1,
            source_mode="controlled_live",
            passed=True,
            checks=[],
            failure_types=[],
            metrics={},
        )
        return record, result

    first, first_result = make_pair("first", "shared")
    second, second_result = make_pair("second", "shared")
    summary = summarize_runtime_results([first, second], [first_result, second_result])[
        "reliability"
    ]
    for item in summary.values():
        assert item["tokens"]["mean"] is None
        assert item["token_coverage"]["total_tokens"]["identity_incomplete"] is True
        assert item["token_coverage"]["total_tokens"]["attempted"] is None
    unique, unique_result = make_pair("first", "unique")
    unique.run_id = "run-first-unique"
    unique_result.run_id = unique.run_id
    mixed = summarize_runtime_results(
        [first, second, unique], [first_result, second_result, unique_result]
    )["reliability"]
    assert mixed["first"]["tokens"]["count"] == 1
    assert mixed["first"]["tokens"]["mean"] == 3
    assert mixed["first"]["token_coverage"]["total_tokens"]["attempted"] is None
    distinct, distinct_result = make_pair("second", "different")
    summary = summarize_runtime_results(
        [first, distinct], [first_result, distinct_result]
    )["reliability"]
    assert summary["first"]["tokens"]["mean"] == 3
    assert summary["second"]["tokens"]["mean"] == 3
