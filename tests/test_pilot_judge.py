import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from langchain_openrouter import ChatOpenRouter
from openrouter import OpenRouter
from test_runtime_evaluation import make_scenario, normalize_trace_runs, root_run

from backend.evaluation.runtime_evaluation import evaluate_with_openevals_judge
from scripts.evaluate_runtime_workflow import _configure_pilot_judge


def _register_pilot_fixture(cli, args, tmp_path, monkeypatch):
    """Use the protected registration path for isolated existing CLI tests."""
    import hashlib
    from datetime import UTC

    from backend.evaluation import portfolio_budget as pb

    parsed = cli.parse_args(args)
    ledger = cli.campaign_ledger(cli.ROOT, override=parsed.budget_ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(pb, "campaign_ledger", lambda *args, **kwargs: ledger)
    budget = cli._evaluation_session(parsed, parsed.output_dir)
    if not ledger.exists():
        # Initialize the empty synthetic ledger before the separate protected capture.
        budget.require_registered_observation = False
        with budget:
            pass
    state = json.loads(ledger.read_text())
    state.setdefault("policy_amendments", []).append(
        {
            "amendment": "daily-policy-50-20261030",
            "day_change": {
                "source_proof": {
                    "basis": "prospective-authority",
                    "expected_account_id": "dummy-account",
                    "expected_key_id": "dummy-key",
                }
            },
        }
    )
    ledger.write_text(json.dumps(state, sort_keys=True))
    folder = ledger.parent / "registered-fixture"
    folder.mkdir()
    boundary = pb.capture_prospective_boundary(
        checkout=tmp_path,
        ledger=ledger,
        expected_ledger_sha256=hashlib.sha256(ledger.read_bytes()).hexdigest(),
        destination=folder / "capture",
    )
    model = "xingchen-agi/xing4.0-29b-a4b:free"
    now = pb.datetime.now(UTC).isoformat()
    used = state["days"].get(now[:10], {}).get("used", 0)
    q = {
        "observed_at": now,
        "provider": "openrouter",
        "model": model,
        "remaining_requests": 50 - used,
        "demo_reserve": 20,
        "daily_batch_limit": 50,
    }
    account = {
        "openrouter": {
            "method": "GET",
            "url": "https://openrouter.ai/api/v1/key",
            "http_status": 200,
            "status": "success",
            "time": now,
            "quota": {
                "is_free_tier": True,
                "free_model_daily_requests": {
                    "limit": 50,
                    "remaining": 50 - used,
                    "used": used,
                },
            },
        }
    }
    route = {
        "method": "GET",
        "url": f"https://openrouter.ai/api/v1/models/{model}/endpoints",
        "http_status": 200,
        "status": "success",
        "time": now,
        "data": {
            "id": model,
            "endpoints": [
                {
                    "model_id": model,
                    "provider_name": "Phala",
                    "tag": "phala",
                    "max_completion_tokens": 8192,
                    "pricing": {"prompt": "0", "completion": "0"},
                    "supported_parameters": [
                        "response_format",
                        "structured_outputs",
                        "temperature",
                        "reasoning",
                        "max_tokens",
                    ],
                }
            ],
        },
    }
    paths = {
        name: folder / (name + ".json")
        for name in ("account", "route", "quota", "binding")
    }
    for name, value in (("account", account), ("route", route), ("quota", q)):
        paths[name].write_text(json.dumps(value, sort_keys=True))
    binding = {
        "account_id": "dummy-account",
        "key_id": "dummy-key",
        "observed_at": now,
        "provider": "openrouter",
        "model": model,
        "account_capture_sha256": hashlib.sha256(
            paths["account"].read_bytes()
        ).hexdigest(),
        "route_capture_sha256": hashlib.sha256(paths["route"].read_bytes()).hexdigest(),
        "credential_check": "local-equality",
        "checked_at": now,
        "sut_role_credentials_match_queried_key": {"judge": True},
        "external_daily_limit": 50,
    }
    paths["binding"].write_text(json.dumps(binding, sort_keys=True))
    proof = {
        "authority": "daily-50-2026-10-30",
        "capture_started_at": boundary["started_at"],
        "boundary_path": str(folder / "capture/boundary.json"),
        "boundary_sha256": hashlib.sha256(
            (folder / "capture/boundary.json").read_bytes()
        ).hexdigest(),
        "capture_ledger_path": boundary["snapshot_path"],
        "capture_ledger_sha256": boundary["ledger_sha256"],
        "quota": q,
    }
    for name, path in paths.items():
        proof[f"{name}_capture_path"] = str(path)
        proof[f"{name}_capture_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    proof_file = folder / "proof.json"
    proof_file.write_text(json.dumps(proof, sort_keys=True))
    pb.register_prospective_observation(
        checkout=tmp_path,
        ledger=ledger,
        expected_ledger_sha256=hashlib.sha256(ledger.read_bytes()).hexdigest(),
        proof_path=proof_file,
        expected_proof_sha256=hashlib.sha256(proof_file.read_bytes()).hexdigest(),
    )
    parsed.provider_quota.write_text(json.dumps(q))


def test_openevals_pilot_serializes_strict_boolean_and_captures_metadata():
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "offline-id",
                "model": "offline-model",
                "provider": "Phala",
                "object": "chat.completion",
                "created": 0,
                "system_fingerprint": None,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {"score": True, "reasoning": "direct"}
                            ),
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    client = OpenRouter(
        api_key="dummy",
        retry_config=None,
        client=httpx.Client(
            transport=httpx.MockTransport(respond), follow_redirects=False
        ),
    )
    base = ChatOpenRouter(
        model="xingchen-agi/xing4.0-29b-a4b:free",
        api_key="dummy",
        client=client,
        max_retries=0,
    )
    judge = _configure_pilot_judge(base, "graph_usefulness")
    scenario = make_scenario()
    scenario.expectations.judge_dimensions = ["analyst_usefulness"]
    record = normalize_trace_runs(root_run(), scenario)
    with patch(
        "socket.socket.connect", side_effect=AssertionError("network forbidden")
    ):
        result = evaluate_with_openevals_judge(
            scenario,
            record,
            "analyst_usefulness",
            judge=judge,
            judge_provider="openrouter",
            judge_model="xingchen-agi/xing4.0-29b-a4b:free",
            structured_output_method="json_schema",
            strict_structured_output=True,
        )
    assert result.status == "success", result
    assert result.score is True
    assert result.stop_reason == "stop"
    assert result.usage.usage_method == "event_unique_v1"
    assert result.usage.total_tokens == 2
    assert result.usage.token_coverage["total_tokens"] == {
        "observed": 1,
        "attempted": 1,
        "subtotal": 2,
        "complete": True,
    }
    assert len(requests) == 1
    body = requests[0]
    assert body["provider"]["only"] == ["Phala"]
    assert body["provider"]["allow_fallbacks"] is False
    assert body["provider"]["require_parameters"] is True
    assert body["reasoning"]["effort"] == "none"
    assert body["temperature"] == 0
    assert body["max_tokens"] == 512
    assert body["response_format"]["json_schema"]["strict"] is True
    assert set(body["response_format"]["json_schema"]["schema"]["required"]) == {
        "score",
        "reasoning",
    }


def test_pilot_claim_has_8192_cap_and_strict_schema():
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "claim-id",
                "model": "offline-model",
                "provider": "Phala",
                "object": "chat.completion",
                "created": 0,
                "system_fingerprint": None,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "claims": [
                                        {
                                            "claim": "Evidence exists",
                                            "verdict": "supported",
                                            "evidence_ids": ["source-1"],
                                            "reason": "matching text",
                                        }
                                    ]
                                }
                            ),
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    client = OpenRouter(
        api_key="dummy",
        retry_config=None,
        client=httpx.Client(
            transport=httpx.MockTransport(respond), follow_redirects=False
        ),
    )
    judge = _configure_pilot_judge(
        ChatOpenRouter(
            model="xingchen-agi/xing4.0-29b-a4b:free",
            api_key="dummy",
            client=client,
            max_retries=0,
        ),
        "claim_faithfulness",
    )
    scenario = make_scenario()
    scenario.expectations.judge_dimensions = ["claim_faithfulness"]
    record = normalize_trace_runs(root_run(), scenario)
    from backend.evaluation.runtime_evaluation import EvidenceObservation

    record.evidence_capture_status = "captured"
    record.bounded_evidence = [
        EvidenceObservation.model_validate(
            {
                "evidence_id": "source-1",
                "kind": "local_chunk",
                "excerpt": "Evidence exists",
            }
        )
    ]
    with patch(
        "socket.socket.connect", side_effect=AssertionError("network forbidden")
    ):
        result = evaluate_with_openevals_judge(
            scenario,
            record,
            "claim_faithfulness",
            judge=judge,
            judge_provider="openrouter",
            judge_model="xingchen-agi/xing4.0-29b-a4b:free",
            structured_output_method="json_schema",
            strict_structured_output=True,
        )
    assert result.status == "success", result
    assert result.stop_reason == "stop"
    assert result.supported_claims == 1
    assert requests[0]["max_tokens"] == 8192
    assert requests[0]["response_format"]["json_schema"]["strict"] is True


def test_local_prompt_format_error_does_not_reserve_or_dispatch():
    from backend.evaluation import runtime_evaluation as evaluation

    calls = []
    client = OpenRouter(
        api_key="dummy",
        retry_config=None,
        client=httpx.Client(
            transport=httpx.MockTransport(lambda request: calls.append(request))
        ),
    )
    judge = _configure_pilot_judge(
        ChatOpenRouter(
            model="xingchen-agi/xing4.0-29b-a4b:free",
            api_key="dummy",
            client=client,
            max_retries=0,
        ),
        "graph_usefulness",
    )
    scenario = make_scenario()
    scenario.expectations.judge_dimensions = ["analyst_usefulness"]
    record = normalize_trace_runs(root_run(), scenario)
    with patch.dict(
        evaluation.SEMANTIC_JUDGE_RUBRICS,
        {"analyst_usefulness": "Broken {unknown_key}"},
    ):
        result = evaluate_with_openevals_judge(
            scenario,
            record,
            "analyst_usefulness",
            judge=judge,
            judge_provider="openrouter",
            judge_model="xingchen-agi/xing4.0-29b-a4b:free",
            structured_output_method="json_schema",
            strict_structured_output=True,
        )
    assert result.status == "error"
    assert result.error_type == "KeyError"
    assert result.provider_call_attempted is False
    assert result.transport_status == "not_attempted"
    assert result.usage.token_coverage["total_tokens"]["attempted"] == 0
    assert result.usage.total_tokens is None
    assert calls == []


def test_pilot_rejects_unsafe_identity_and_truncation():
    from backend.evaluation.runtime_evaluation import SemanticJudgeResult
    from scripts.evaluate_runtime_workflow import _pilot_result_safe

    result = SemanticJudgeResult(
        scenario_id="x",
        scenario_version=1,
        run_id="r",
        repetition=1,
        dimension="graph_usefulness",
        judge_provider="openrouter",
        judge_model="fixed",
        status="success",
        observed_model="xingchen-agi/xing4.0-29b-a4b-20261015",
        observed_provider="Phala",
    )
    assert _pilot_result_safe(result)
    assert not _pilot_result_safe(result.model_copy(update={"observed_model": "other"}))
    assert not _pilot_result_safe(result.model_copy(update={"observed_provider": ""}))
    assert not _pilot_result_safe(
        result.model_copy(update={"confirmed_truncation": True})
    )
    assert not _pilot_result_safe(result.model_copy(update={"status": "error"}))


def test_real_cli_factory_and_ledger_charge_one_intercepted_request(
    tmp_path, monkeypatch
):
    import openrouter
    from test_portfolio_budget import quota

    from scripts import evaluate_runtime_workflow as cli

    requests = []
    sdk_args = []
    real_sdk = OpenRouter

    def intercepted_sdk(**kwargs):
        sdk_args.append(kwargs)
        kwargs["client"] = httpx.Client(
            transport=httpx.MockTransport(
                lambda request: (
                    requests.append(json.loads(request.content))
                    or httpx.Response(
                        200,
                        json={
                            "id": "offline-id",
                            "model": "xingchen-agi/xing4.0-29b-a4b-20261015",
                            "provider": "Phala",
                            "object": "chat.completion",
                            "created": 0,
                            "system_fingerprint": None,
                            "choices": [
                                {
                                    "index": 0,
                                    "finish_reason": "stop",
                                    "message": {
                                        "role": "assistant",
                                        "content": json.dumps(
                                            {"score": True, "reasoning": "ok"}
                                        ),
                                    },
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 1,
                                "completion_tokens": 1,
                                "total_tokens": 2,
                            },
                        },
                    )
                )
            ),
            follow_redirects=False,
        )
        return real_sdk(**kwargs)

    monkeypatch.setattr(openrouter, "OpenRouter", intercepted_sdk)
    monkeypatch.setattr(
        cli, "campaign_ledger", lambda *args, **kwargs: tmp_path / "ledger.json"
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    scenario = make_scenario()
    scenario.expectations.judge_dimensions = ["graph_usefulness"]
    record = normalize_trace_runs(root_run(), scenario)
    record.worker_sequence = ["visualizer"]
    record.graph.capture_status = "captured"
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(json.dumps([scenario.model_dump(mode="json")]))
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(record.model_dump_json() + "\n")
    quota_path = tmp_path / "quota.json"
    quota_path.write_text(json.dumps(quota(model="xingchen-agi/xing4.0-29b-a4b:free")))
    args = [
        "judge",
        "--confirm-provider-calls",
        "--budget-stage",
        "final",
        "--provider-quota",
        str(quota_path),
        "--evaluation-root",
        str(tmp_path / "evaluation"),
        "--output-dir",
        str(tmp_path / "evaluation" / "runs" / "one"),
        "--scenarios",
        str(scenario_path),
        "--records",
        str(records_path),
        "--case-id",
        scenario.id,
        "--dimension",
        "graph_usefulness",
        "--judge-provider",
        "openrouter",
        "--judge-model",
        "xingchen-agi/xing4.0-29b-a4b:free",
        "--structured-output-method",
        "json_schema",
        "--fixed-judge-pilot",
    ]
    _register_pilot_fixture(cli, args, tmp_path, monkeypatch)
    with patch(
        "socket.socket.connect", side_effect=AssertionError("network forbidden")
    ):
        status = cli.main(args)
    assert status == 0
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert len(ledger["reservations"]) == 1
    assert ledger["reservations"][0]["kind"] == "judge"
    assert len(requests) == 1
    assert requests[0]["model"] == "xingchen-agi/xing4.0-29b-a4b:free"
    assert requests[0]["provider"]["only"] == ["Phala"]
    assert requests[0]["response_format"]["json_schema"]["strict"] is True
    assert sdk_args[0]["retry_config"] is None
    assert sdk_args[0]["client"].follow_redirects is False
    assert len(ledger["jobs"]) == 1
    assert next(iter(ledger["jobs"])).startswith("final:judge:semantic-v2:")
    result = json.loads(
        (tmp_path / "evaluation" / "runs" / "one" / "judge_results.jsonl").read_text()
    )
    assert result["observed_model"] == "xingchen-agi/xing4.0-29b-a4b-20261015"
    assert result["observed_provider"] == "Phala"
    assert result["observed_request_id"] == "offline-id"
    # Byte-different JSON represents the same normalized record, but has a new
    # source-file identity and must not reuse the first assessment.
    records_path.write_text(
        json.dumps(json.loads(record.model_dump_json()), indent=None) + "\n"
    )
    args[args.index("--output-dir") + 1] = str(tmp_path / "evaluation" / "runs" / "two")
    with patch(
        "socket.socket.connect", side_effect=AssertionError("network forbidden")
    ):
        assert cli.main(args) == 0
    assert len(requests) == 2
    assert len(json.loads((tmp_path / "ledger.json").read_text())["jobs"]) == 2
    record.run_id = "different-source"
    records_path.write_text(record.model_dump_json() + "\n")
    quota_path.write_text(json.dumps(quota(model="openrouter/free")))
    args[args.index("--output-dir") + 1] = str(
        tmp_path / "evaluation" / "runs" / "three"
    )
    with patch(
        "socket.socket.connect", side_effect=AssertionError("network forbidden")
    ):
        assert cli.main(args) == 2
    assert len(requests) == 2
    assert len(json.loads((tmp_path / "ledger.json").read_text())["reservations"]) == 2


def test_http_success_then_local_error_does_not_claim_provider_failure():
    from backend.evaluation.runtime_evaluation import _judge_diagnostics

    result = _judge_diagnostics(
        exc=KeyError("private"),
        raw=None,
        attempted=True,
        transport_metadata={"http_status_code": 200},
    )
    assert result["transport_status"] == "completed"
    assert result["http_status_code"] == 200
    assert result["parser_classification"] == "unclassified_failure"


def test_pilot_rejects_different_allowlisted_identity_on_next_job(
    tmp_path, monkeypatch
):
    import openrouter
    from test_portfolio_budget import quota

    from scripts import evaluate_runtime_workflow as cli

    requests = []
    sdk_args = []
    real_sdk = OpenRouter

    def intercepted_sdk(**kwargs):
        sdk_args.append(kwargs)
        kwargs["client"] = httpx.Client(
            transport=httpx.MockTransport(
                lambda request: (
                    requests.append(json.loads(request.content))
                    or httpx.Response(
                        200,
                        json={
                            "id": "offline-id",
                            "model": (
                                "xingchen-agi/xing4.0-29b-a4b-20261015"
                                if len(requests) == 1
                                else "xingchen-agi/xing4.0-29b-a4b:free"
                            ),
                            "provider": "Phala",
                            "object": "chat.completion",
                            "created": 0,
                            "system_fingerprint": None,
                            "choices": [
                                {
                                    "index": 0,
                                    "finish_reason": "stop",
                                    "message": {
                                        "role": "assistant",
                                        "content": json.dumps(
                                            {"score": True, "reasoning": "ok"}
                                        ),
                                    },
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 1,
                                "completion_tokens": 1,
                                "total_tokens": 2,
                            },
                        },
                    )
                )
            ),
            follow_redirects=False,
        )
        return real_sdk(**kwargs)

    monkeypatch.setattr(openrouter, "OpenRouter", intercepted_sdk)
    monkeypatch.setattr(
        cli, "campaign_ledger", lambda *args, **kwargs: tmp_path / "ledger.json"
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    scenario = make_scenario()
    scenario.expectations.judge_dimensions = ["graph_usefulness"]
    record = normalize_trace_runs(root_run(), scenario)
    record.worker_sequence = ["visualizer"]
    record.graph.capture_status = "captured"
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(json.dumps([scenario.model_dump(mode="json")]))
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(record.model_dump_json() + "\n")
    quota_path = tmp_path / "quota.json"
    quota_path.write_text(json.dumps(quota(model="xingchen-agi/xing4.0-29b-a4b:free")))
    args = [
        "judge",
        "--confirm-provider-calls",
        "--budget-stage",
        "final",
        "--provider-quota",
        str(quota_path),
        "--evaluation-root",
        str(tmp_path / "evaluation"),
        "--output-dir",
        str(tmp_path / "evaluation" / "runs" / "one"),
        "--scenarios",
        str(scenario_path),
        "--records",
        str(records_path),
        "--case-id",
        scenario.id,
        "--dimension",
        "graph_usefulness",
        "--judge-provider",
        "openrouter",
        "--judge-model",
        "xingchen-agi/xing4.0-29b-a4b:free",
        "--structured-output-method",
        "json_schema",
        "--fixed-judge-pilot",
    ]
    _register_pilot_fixture(cli, args, tmp_path, monkeypatch)
    with patch(
        "socket.socket.connect", side_effect=AssertionError("network forbidden")
    ):
        status = cli.main(args)
    assert status == 0
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert len(ledger["reservations"]) == 1
    assert ledger["reservations"][0]["kind"] == "judge"
    assert len(requests) == 1
    assert requests[0]["model"] == "xingchen-agi/xing4.0-29b-a4b:free"
    assert requests[0]["provider"]["only"] == ["Phala"]
    assert requests[0]["response_format"]["json_schema"]["strict"] is True
    assert sdk_args[0]["retry_config"] is None
    assert sdk_args[0]["client"].follow_redirects is False
    assert len(ledger["jobs"]) == 1
    assert next(iter(ledger["jobs"])).startswith("final:judge:semantic-v2:")
    result = json.loads(
        (tmp_path / "evaluation" / "runs" / "one" / "judge_results.jsonl").read_text()
    )
    assert result["observed_model"] == "xingchen-agi/xing4.0-29b-a4b-20261015"
    assert result["observed_provider"] == "Phala"
    assert result["observed_request_id"] == "offline-id"
    # Byte-different JSON represents the same normalized record, but has a new
    # source-file identity and must not reuse the first assessment.
    records_path.write_text(
        json.dumps(json.loads(record.model_dump_json()), indent=None) + "\n"
    )
    args[args.index("--output-dir") + 1] = str(tmp_path / "evaluation" / "runs" / "two")
    with patch(
        "socket.socket.connect", side_effect=AssertionError("network forbidden")
    ):
        assert cli.main(args) == 2
    assert len(requests) == 2
    assert len(json.loads((tmp_path / "ledger.json").read_text())["jobs"]) == 2


@pytest.mark.parametrize("failure", ["503", "timeout", "connection"])
def test_pilot_retries_one_confirmed_503_and_counts_both_attempts(
    tmp_path, monkeypatch, failure
):
    import openrouter
    from test_portfolio_budget import quota

    from scripts import evaluate_runtime_workflow as cli

    requests = []
    sdk_args = []
    real_sdk = OpenRouter

    def intercepted_sdk(**kwargs):
        sdk_args.append(kwargs)

        def respond(request):
            if not requests and failure != "503":
                requests.append(json.loads(request.content))
                if failure == "timeout":
                    raise httpx.ReadTimeout("fixture timeout", request=request)
                raise httpx.ConnectError("fixture connection failure", request=request)
            return requests.append(json.loads(request.content)) or (
                httpx.Response(
                    503, json={"error": {"message": "unavailable", "code": 503}}
                )
                if len(requests) != 2
                else httpx.Response(
                    200,
                    json={
                        "id": "offline-id",
                        "model": "xingchen-agi/xing4.0-29b-a4b-20261015",
                        "provider": "Phala",
                        "object": "chat.completion",
                        "created": 0,
                        "system_fingerprint": None,
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {
                                    "role": "assistant",
                                    "content": json.dumps(
                                        {"score": True, "reasoning": "ok"}
                                    ),
                                },
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "total_tokens": 2,
                        },
                    },
                )
            )

        kwargs["client"] = httpx.Client(
            transport=httpx.MockTransport(respond), follow_redirects=False
        )
        return real_sdk(**kwargs)

    monkeypatch.setattr(openrouter, "OpenRouter", intercepted_sdk)
    monkeypatch.setattr(
        cli, "campaign_ledger", lambda *args, **kwargs: tmp_path / "ledger.json"
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    scenario = make_scenario()
    scenario.expectations.judge_dimensions = ["graph_usefulness"]
    record = normalize_trace_runs(root_run(), scenario)
    record.worker_sequence = ["visualizer"]
    record.graph.capture_status = "captured"
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(json.dumps([scenario.model_dump(mode="json")]))
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(record.model_dump_json() + "\n")
    quota_path = tmp_path / "quota.json"
    quota_path.write_text(json.dumps(quota(model="xingchen-agi/xing4.0-29b-a4b:free")))
    args = [
        "judge",
        "--confirm-provider-calls",
        "--budget-stage",
        "final",
        "--provider-quota",
        str(quota_path),
        "--evaluation-root",
        str(tmp_path / "evaluation"),
        "--output-dir",
        str(tmp_path / "evaluation" / "runs" / "one"),
        "--scenarios",
        str(scenario_path),
        "--records",
        str(records_path),
        "--case-id",
        scenario.id,
        "--dimension",
        "graph_usefulness",
        "--judge-provider",
        "openrouter",
        "--judge-model",
        "xingchen-agi/xing4.0-29b-a4b:free",
        "--structured-output-method",
        "json_schema",
        "--fixed-judge-pilot",
    ]
    _register_pilot_fixture(cli, args, tmp_path, monkeypatch)
    with patch(
        "socket.socket.connect", side_effect=AssertionError("network forbidden")
    ):
        status = cli.main(args)
    assert status == 0
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert len(ledger["reservations"]) == 2
    assert ledger["reservations"][0]["kind"] == "judge"
    assert len(requests) == 2
    assert requests[0]["model"] == "xingchen-agi/xing4.0-29b-a4b:free"
    assert requests[0]["provider"]["only"] == ["Phala"]
    assert requests[0]["response_format"]["json_schema"]["strict"] is True
    assert sdk_args[0]["retry_config"] is None
    assert sdk_args[0]["client"].follow_redirects is False
    assert len(ledger["jobs"]) == 1
    assert next(iter(ledger["jobs"])).startswith("final:judge:semantic-v2:")
    result = json.loads(
        (tmp_path / "evaluation" / "runs" / "one" / "judge_results.jsonl").read_text()
    )
    assert result["observed_model"] == "xingchen-agi/xing4.0-29b-a4b-20261015"
    assert result["observed_provider"] == "Phala"
    assert result["observed_request_id"] == "offline-id"
    assert result["usage"]["provider_attempts"] == 2
    assert result["usage"]["token_coverage"]["total_tokens"]["attempted"] == 2
    assert len(result["prior_attempts"]) == 1

    from backend.evaluation.evaluation_reporting import _semantic_metrics

    metrics = _semantic_metrics([result])
    coverage = metrics["judge_usage"]["token_coverage"]["total_tokens"]
    assert coverage["attempted"] == 2
    assert coverage["observed_attempts"] == 1
    assert coverage["partial_observed_subtotal"] == 2
    assert metrics["dimensions"]["graph_usefulness"]["provider_attempted_count"] == 2

    # A restart on another evaluation root cannot create a new shared retry.
    record.run_id = "another-source"
    records_path.write_text(record.model_dump_json() + "\n")
    args[args.index("--evaluation-root") + 1] = str(tmp_path / "other-root")
    args[args.index("--output-dir") + 1] = str(tmp_path / "other-root" / "runs" / "two")
    with patch(
        "socket.socket.connect", side_effect=AssertionError("network forbidden")
    ):
        assert cli.main(args) == 2
    assert len(requests) == 3
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert len(ledger["reservations"]) == 3
    assert ledger["pilot_retry"]["job_id"] == ledger["reservations"][0]["job_id"]


def test_pilot_admits_only_four_dispatched_assessments(tmp_path, monkeypatch):
    import openrouter
    from test_portfolio_budget import quota

    from scripts import evaluate_runtime_workflow as cli

    requests = []
    sdk_args = []
    real_sdk = OpenRouter

    def intercepted_sdk(**kwargs):
        sdk_args.append(kwargs)
        kwargs["client"] = httpx.Client(
            transport=httpx.MockTransport(
                lambda request: (
                    requests.append(json.loads(request.content))
                    or httpx.Response(
                        200,
                        json={
                            "id": "offline-id",
                            "model": "xingchen-agi/xing4.0-29b-a4b-20261015",
                            "provider": "Phala",
                            "object": "chat.completion",
                            "created": 0,
                            "system_fingerprint": None,
                            "choices": [
                                {
                                    "index": 0,
                                    "finish_reason": "stop",
                                    "message": {
                                        "role": "assistant",
                                        "content": json.dumps(
                                            {"score": True, "reasoning": "ok"}
                                        ),
                                    },
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 1,
                                "completion_tokens": 1,
                                "total_tokens": 2,
                            },
                        },
                    )
                )
            ),
            follow_redirects=False,
        )
        return real_sdk(**kwargs)

    monkeypatch.setattr(openrouter, "OpenRouter", intercepted_sdk)
    monkeypatch.setattr(
        cli, "campaign_ledger", lambda *args, **kwargs: tmp_path / "ledger.json"
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    scenario = make_scenario()
    scenario.expectations.judge_dimensions = ["graph_usefulness"]
    record = normalize_trace_runs(root_run(), scenario)
    record.worker_sequence = ["visualizer"]
    record.graph.capture_status = "captured"
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(json.dumps([scenario.model_dump(mode="json")]))
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(record.model_dump_json() + "\n")
    quota_path = tmp_path / "quota.json"
    quota_path.write_text(json.dumps(quota(model="xingchen-agi/xing4.0-29b-a4b:free")))
    args = [
        "judge",
        "--confirm-provider-calls",
        "--budget-stage",
        "final",
        "--provider-quota",
        str(quota_path),
        "--evaluation-root",
        str(tmp_path / "evaluation"),
        "--output-dir",
        str(tmp_path / "evaluation" / "runs" / "one"),
        "--scenarios",
        str(scenario_path),
        "--records",
        str(records_path),
        "--case-id",
        scenario.id,
        "--dimension",
        "graph_usefulness",
        "--judge-provider",
        "openrouter",
        "--judge-model",
        "xingchen-agi/xing4.0-29b-a4b:free",
        "--structured-output-method",
        "json_schema",
        "--fixed-judge-pilot",
    ]
    _register_pilot_fixture(cli, args, tmp_path, monkeypatch)
    with patch(
        "socket.socket.connect", side_effect=AssertionError("network forbidden")
    ):
        status = cli.main(args)
    assert status == 0
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert len(ledger["reservations"]) == 1
    assert ledger["reservations"][0]["kind"] == "judge"
    assert len(requests) == 1
    assert requests[0]["model"] == "xingchen-agi/xing4.0-29b-a4b:free"
    assert requests[0]["provider"]["only"] == ["Phala"]
    assert requests[0]["response_format"]["json_schema"]["strict"] is True
    assert sdk_args[0]["retry_config"] is None
    assert sdk_args[0]["client"].follow_redirects is False
    assert len(ledger["jobs"]) == 1
    assert next(iter(ledger["jobs"])).startswith("final:judge:semantic-v2:")
    result = json.loads(
        (tmp_path / "evaluation" / "runs" / "one" / "judge_results.jsonl").read_text()
    )
    assert result["observed_model"] == "xingchen-agi/xing4.0-29b-a4b-20261015"
    assert result["observed_provider"] == "Phala"
    assert result["observed_request_id"] == "offline-id"
    for index in range(1, 5):
        record.run_id = f"source-{index}"
        records_path.write_text(record.model_dump_json() + "\n")
        args[args.index("--output-dir") + 1] = str(
            tmp_path / "evaluation" / "runs" / str(index)
        )
        with patch(
            "socket.socket.connect", side_effect=AssertionError("network forbidden")
        ):
            assert cli.main(args) == (0 if index < 4 else 2)
    assert len(requests) == 4
    assert len(json.loads((tmp_path / "ledger.json").read_text())["reservations"]) == 4


def test_pilot_non_retryable_outcomes_do_not_spend_retry(tmp_path, monkeypatch):
    import openrouter
    from test_portfolio_budget import quota

    from scripts import evaluate_runtime_workflow as cli

    real_sdk = OpenRouter
    calls = []
    outcome = {"kind": "429"}

    def respond(request):
        calls.append(request)
        kind = outcome["kind"]
        if kind in {"429", "401"}:
            status = int(kind)
            return httpx.Response(
                status, json={"error": {"message": "no", "code": status}}
            )
        content = (
            "" if kind == "empty" else json.dumps({"score": False, "reasoning": "no"})
        )
        return httpx.Response(
            200,
            json={
                "id": "offline-id",
                "model": ""
                if kind == "unknown"
                else "xingchen-agi/xing4.0-29b-a4b-20261015",
                "provider": "Phala",
                "object": "chat.completion",
                "created": 0,
                "system_fingerprint": None,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "length" if kind == "length" else "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    def intercepted_sdk(**kwargs):
        kwargs["client"] = httpx.Client(
            transport=httpx.MockTransport(respond), follow_redirects=False
        )
        return real_sdk(**kwargs)

    monkeypatch.setattr(openrouter, "OpenRouter", intercepted_sdk)
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    scenario = make_scenario()
    scenario.expectations.judge_dimensions = ["graph_usefulness"]
    record = normalize_trace_runs(root_run(), scenario)
    record.worker_sequence = ["visualizer"]
    record.graph.capture_status = "captured"
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(json.dumps([scenario.model_dump(mode="json")]))
    records_path = tmp_path / "records.jsonl"
    quota_path = tmp_path / "quota.json"
    quota_path.write_text(json.dumps(quota(model="xingchen-agi/xing4.0-29b-a4b:free")))
    for kind in ("429", "401", "empty", "length", "unknown", "negative", "skip"):
        outcome["kind"] = kind
        ledger_path = tmp_path / kind / "ledger.json"
        monkeypatch.setattr(
            cli, "campaign_ledger", lambda *args, path=ledger_path, **kwargs: path
        )
        record.worker_sequence = ["analyst"] if kind == "skip" else ["visualizer"]
        records_path.write_text(record.model_dump_json() + "\n")
        args = [
            "judge",
            "--confirm-provider-calls",
            "--budget-stage",
            "final",
            "--provider-quota",
            str(quota_path),
            "--evaluation-root",
            str(tmp_path / kind / "evaluation"),
            "--output-dir",
            str(tmp_path / kind / "evaluation" / "runs" / "one"),
            "--scenarios",
            str(scenario_path),
            "--records",
            str(records_path),
            "--case-id",
            scenario.id,
            "--dimension",
            "graph_usefulness",
            "--judge-provider",
            "openrouter",
            "--judge-model",
            "xingchen-agi/xing4.0-29b-a4b:free",
            "--structured-output-method",
            "json_schema",
            "--fixed-judge-pilot",
        ]
        _register_pilot_fixture(cli, args, tmp_path, monkeypatch)
        before = len(calls)
        with patch(
            "socket.socket.connect", side_effect=AssertionError("network forbidden")
        ):
            status = cli.main(args)
        assert status == (0 if kind in {"negative", "skip"} else 2), kind
        ledger = json.loads(ledger_path.read_text())
        assert len(calls) - before == (0 if kind == "skip" else 1), kind
        assert len(ledger["reservations"]) == (0 if kind == "skip" else 1), kind
        assert "pilot_retry" not in ledger


def test_pilot_retry_rejects_mixed_parser_truncation_and_incidental_timeout():
    from backend.evaluation.runtime_evaluation import SemanticJudgeResult
    from scripts.evaluate_runtime_workflow import _pilot_retryable

    base = SemanticJudgeResult(
        scenario_id="x",
        scenario_version=1,
        run_id="r",
        repetition=1,
        dimension="graph_usefulness",
        judge_provider="openrouter",
        judge_model="fixed",
        status="error",
        provider_call_attempted=True,
        error_type="ValueError",
        error_cause_types=["ReadTimeout"],
        transport_status="unknown",
        parser_classification="unclassified_failure",
    )
    assert not _pilot_retryable(base)
    assert not _pilot_retryable(
        base.model_copy(
            update={
                "http_status_code": 503,
                "parser_classification": "schema_validation_failure",
            }
        )
    )
    assert not _pilot_retryable(
        base.model_copy(update={"http_status_code": 503, "confirmed_truncation": True})
    )


def test_pilot_duplicate_attempt_ids_cannot_claim_complete_usage():
    from backend.evaluation.runtime_evaluation import RuntimeUsage, SemanticJudgeResult
    from scripts.evaluate_runtime_workflow import _combine_pilot_attempts

    usage = RuntimeUsage(
        input_tokens=1,
        output_tokens=1,
        total_tokens=2,
        usage_method="event_unique_v1",
        provider_attempts=1,
        token_coverage={
            field: {"observed": 1, "attempted": 1, "subtotal": value, "complete": True}
            for field, value in (
                ("input_tokens", 1),
                ("output_tokens", 1),
                ("total_tokens", 2),
            )
        },
    )
    failed = SemanticJudgeResult(
        scenario_id="x",
        scenario_version=1,
        run_id="r",
        repetition=1,
        dimension="graph_usefulness",
        judge_provider="openrouter",
        judge_model="fixed",
        status="error",
        provider_call_attempted=True,
        observed_request_id="same",
        usage=usage.model_copy(deep=True),
    )
    result = failed.model_copy(deep=True, update={"status": "success"})
    _combine_pilot_attempts(result, failed)
    assert result.usage.total_tokens is None
    assert result.usage.token_coverage["total_tokens"]["complete"] is False
    assert result.usage.provider_attempts == 2
    assert len(result.prior_attempts) == 1


def test_unrelated_same_model_standard_result_does_not_pin_pilot(tmp_path, monkeypatch):
    import openrouter
    from test_portfolio_budget import quota

    from scripts import evaluate_runtime_workflow as cli

    requests = []
    sdk_args = []
    real_sdk = OpenRouter

    def intercepted_sdk(**kwargs):
        sdk_args.append(kwargs)
        kwargs["client"] = httpx.Client(
            transport=httpx.MockTransport(
                lambda request: (
                    requests.append(json.loads(request.content))
                    or httpx.Response(
                        200,
                        json={
                            "id": "offline-id",
                            "model": (
                                "xingchen-agi/xing4.0-29b-a4b-20261015"
                                if len(requests) == 1
                                else "xingchen-agi/xing4.0-29b-a4b:free"
                            ),
                            "provider": "Phala",
                            "object": "chat.completion",
                            "created": 0,
                            "system_fingerprint": None,
                            "choices": [
                                {
                                    "index": 0,
                                    "finish_reason": "stop",
                                    "message": {
                                        "role": "assistant",
                                        "content": json.dumps(
                                            {"score": True, "reasoning": "ok"}
                                        ),
                                    },
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 1,
                                "completion_tokens": 1,
                                "total_tokens": 2,
                            },
                        },
                    )
                )
            ),
            follow_redirects=False,
        )
        return real_sdk(**kwargs)

    monkeypatch.setattr(openrouter, "OpenRouter", intercepted_sdk)
    monkeypatch.setattr(
        cli, "campaign_ledger", lambda *args, **kwargs: tmp_path / "ledger.json"
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    scenario = make_scenario()
    scenario.expectations.judge_dimensions = ["graph_usefulness"]
    record = normalize_trace_runs(root_run(), scenario)
    record.worker_sequence = ["visualizer"]
    record.graph.capture_status = "captured"
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(json.dumps([scenario.model_dump(mode="json")]))
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(record.model_dump_json() + "\n")
    quota_path = tmp_path / "quota.json"
    quota_path.write_text(json.dumps(quota(model="xingchen-agi/xing4.0-29b-a4b:free")))
    args = [
        "judge",
        "--confirm-provider-calls",
        "--budget-stage",
        "final",
        "--provider-quota",
        str(quota_path),
        "--evaluation-root",
        str(tmp_path / "evaluation"),
        "--output-dir",
        str(tmp_path / "evaluation" / "runs" / "one"),
        "--scenarios",
        str(scenario_path),
        "--records",
        str(records_path),
        "--case-id",
        scenario.id,
        "--dimension",
        "graph_usefulness",
        "--judge-provider",
        "openrouter",
        "--judge-model",
        "xingchen-agi/xing4.0-29b-a4b:free",
        "--structured-output-method",
        "json_schema",
    ]
    with patch(
        "socket.socket.connect", side_effect=AssertionError("network forbidden")
    ):
        status = cli.main(args)
    assert status == 0
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert len(ledger["reservations"]) == 1
    assert ledger["reservations"][0]["kind"] == "judge"
    assert len(requests) == 1
    assert requests[0]["model"] == "xingchen-agi/xing4.0-29b-a4b:free"
    assert sdk_args[0]["retry_config"] is None
    assert sdk_args[0]["client"].follow_redirects is False
    assert len(ledger["jobs"]) == 1
    assert next(iter(ledger["jobs"])).startswith("final:judge:semantic-v2:")
    result = json.loads(
        (tmp_path / "evaluation" / "runs" / "one" / "judge_results.jsonl").read_text()
    )
    assert result["observed_model"] == "xingchen-agi/xing4.0-29b-a4b-20261015"
    assert result["observed_provider"] == "Phala"
    assert result["observed_request_id"] == "offline-id"
    ledger_path = tmp_path / "ledger.json"
    ledger = json.loads(ledger_path.read_text())
    next(iter(ledger["jobs"].values())).pop("fixed_judge_pilot")
    ledger_path.write_text(json.dumps(ledger))
    # Byte-different JSON represents the same normalized record, but has a new
    # source-file identity and must not reuse the first assessment.
    records_path.write_text(
        json.dumps(json.loads(record.model_dump_json()), indent=None) + "\n"
    )
    args[args.index("--output-dir") + 1] = str(tmp_path / "evaluation" / "runs" / "two")
    args.append("--fixed-judge-pilot")
    _register_pilot_fixture(cli, args, tmp_path, monkeypatch)
    with patch(
        "socket.socket.connect", side_effect=AssertionError("network forbidden")
    ):
        assert cli.main(args) == 0
    assert len(requests) == 2
    assert len(json.loads((tmp_path / "ledger.json").read_text())["jobs"]) == 2


def test_unmarked_legacy_pilot_is_reconciled_before_new_dispatch(tmp_path, monkeypatch):
    import openrouter
    from test_portfolio_budget import quota

    from scripts import evaluate_runtime_workflow as cli

    requests = []
    sdk_args = []
    real_sdk = OpenRouter

    def intercepted_sdk(**kwargs):
        sdk_args.append(kwargs)
        kwargs["client"] = httpx.Client(
            transport=httpx.MockTransport(
                lambda request: (
                    requests.append(json.loads(request.content))
                    or httpx.Response(
                        200,
                        json={
                            "id": "offline-id",
                            "model": (
                                "xingchen-agi/xing4.0-29b-a4b-20261015"
                                if len(requests) == 1
                                else "xingchen-agi/xing4.0-29b-a4b:free"
                            ),
                            "provider": "Phala",
                            "object": "chat.completion",
                            "created": 0,
                            "system_fingerprint": None,
                            "choices": [
                                {
                                    "index": 0,
                                    "finish_reason": "stop",
                                    "message": {
                                        "role": "assistant",
                                        "content": json.dumps(
                                            {"score": True, "reasoning": "ok"}
                                        ),
                                    },
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 1,
                                "completion_tokens": 1,
                                "total_tokens": 2,
                            },
                        },
                    )
                )
            ),
            follow_redirects=False,
        )
        return real_sdk(**kwargs)

    monkeypatch.setattr(openrouter, "OpenRouter", intercepted_sdk)
    monkeypatch.setattr(
        cli, "campaign_ledger", lambda *args, **kwargs: tmp_path / "ledger.json"
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    scenario = make_scenario()
    scenario.expectations.judge_dimensions = ["graph_usefulness"]
    record = normalize_trace_runs(root_run(), scenario)
    record.worker_sequence = ["visualizer"]
    record.graph.capture_status = "captured"
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(json.dumps([scenario.model_dump(mode="json")]))
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(record.model_dump_json() + "\n")
    quota_path = tmp_path / "quota.json"
    quota_path.write_text(json.dumps(quota(model="xingchen-agi/xing4.0-29b-a4b:free")))
    args = [
        "judge",
        "--confirm-provider-calls",
        "--budget-stage",
        "final",
        "--provider-quota",
        str(quota_path),
        "--evaluation-root",
        str(tmp_path / "evaluation"),
        "--output-dir",
        str(tmp_path / "evaluation" / "runs" / "one"),
        "--scenarios",
        str(scenario_path),
        "--records",
        str(records_path),
        "--case-id",
        scenario.id,
        "--dimension",
        "graph_usefulness",
        "--judge-provider",
        "openrouter",
        "--judge-model",
        "xingchen-agi/xing4.0-29b-a4b:free",
        "--structured-output-method",
        "json_schema",
        "--fixed-judge-pilot",
    ]
    _register_pilot_fixture(cli, args, tmp_path, monkeypatch)
    with patch(
        "socket.socket.connect", side_effect=AssertionError("network forbidden")
    ):
        status = cli.main(args)
    assert status == 0
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert len(ledger["reservations"]) == 1
    assert ledger["reservations"][0]["kind"] == "judge"
    assert len(requests) == 1
    assert requests[0]["model"] == "xingchen-agi/xing4.0-29b-a4b:free"
    assert requests[0]["provider"]["only"] == ["Phala"]
    assert requests[0]["response_format"]["json_schema"]["strict"] is True
    assert sdk_args[0]["retry_config"] is None
    assert sdk_args[0]["client"].follow_redirects is False
    assert len(ledger["jobs"]) == 1
    assert next(iter(ledger["jobs"])).startswith("final:judge:semantic-v2:")
    result = json.loads(
        (tmp_path / "evaluation" / "runs" / "one" / "judge_results.jsonl").read_text()
    )
    assert result["observed_model"] == "xingchen-agi/xing4.0-29b-a4b-20261015"
    assert result["observed_provider"] == "Phala"
    assert result["observed_request_id"] == "offline-id"
    # Simulate a pilot artifact written before the durable profile marker existed.
    ledger_path = tmp_path / "ledger.json"
    ledger = json.loads(ledger_path.read_text())
    next(iter(ledger["jobs"].values())).pop("fixed_judge_pilot")
    ledger.pop("pilot_serving_identity")
    ledger_path.write_text(json.dumps(ledger))
    # Byte-different JSON represents the same normalized record, but has a new
    # source-file identity and must not reuse the first assessment.
    records_path.write_text(
        json.dumps(json.loads(record.model_dump_json()), indent=None) + "\n"
    )
    args[args.index("--output-dir") + 1] = str(tmp_path / "evaluation" / "runs" / "two")
    with patch(
        "socket.socket.connect", side_effect=AssertionError("network forbidden")
    ):
        assert cli.main(args) == 2
    assert len(requests) == 2
    assert len(json.loads((tmp_path / "ledger.json").read_text())["jobs"]) == 2


def test_ambiguous_legacy_pilot_stops_before_new_request(tmp_path, monkeypatch):
    import openrouter
    from test_portfolio_budget import quota

    from scripts import evaluate_runtime_workflow as cli

    requests = []
    sdk_args = []
    real_sdk = OpenRouter

    def intercepted_sdk(**kwargs):
        sdk_args.append(kwargs)
        kwargs["client"] = httpx.Client(
            transport=httpx.MockTransport(
                lambda request: (
                    requests.append(json.loads(request.content))
                    or httpx.Response(
                        200,
                        json={
                            "id": "offline-id",
                            "model": (
                                "xingchen-agi/xing4.0-29b-a4b-20261015"
                                if len(requests) == 1
                                else "xingchen-agi/xing4.0-29b-a4b:free"
                            ),
                            "provider": "Phala",
                            "object": "chat.completion",
                            "created": 0,
                            "system_fingerprint": None,
                            "choices": [
                                {
                                    "index": 0,
                                    "finish_reason": "stop",
                                    "message": {
                                        "role": "assistant",
                                        "content": json.dumps(
                                            {"score": True, "reasoning": "ok"}
                                        ),
                                    },
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 1,
                                "completion_tokens": 1,
                                "total_tokens": 2,
                            },
                        },
                    )
                )
            ),
            follow_redirects=False,
        )
        return real_sdk(**kwargs)

    monkeypatch.setattr(openrouter, "OpenRouter", intercepted_sdk)
    monkeypatch.setattr(
        cli, "campaign_ledger", lambda *args, **kwargs: tmp_path / "ledger.json"
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    scenario = make_scenario()
    scenario.expectations.judge_dimensions = ["graph_usefulness"]
    record = normalize_trace_runs(root_run(), scenario)
    record.worker_sequence = ["visualizer"]
    record.graph.capture_status = "captured"
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(json.dumps([scenario.model_dump(mode="json")]))
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(record.model_dump_json() + "\n")
    quota_path = tmp_path / "quota.json"
    quota_path.write_text(json.dumps(quota(model="xingchen-agi/xing4.0-29b-a4b:free")))
    args = [
        "judge",
        "--confirm-provider-calls",
        "--budget-stage",
        "final",
        "--provider-quota",
        str(quota_path),
        "--evaluation-root",
        str(tmp_path / "evaluation"),
        "--output-dir",
        str(tmp_path / "evaluation" / "runs" / "one"),
        "--scenarios",
        str(scenario_path),
        "--records",
        str(records_path),
        "--case-id",
        scenario.id,
        "--dimension",
        "graph_usefulness",
        "--judge-provider",
        "openrouter",
        "--judge-model",
        "xingchen-agi/xing4.0-29b-a4b:free",
        "--structured-output-method",
        "json_schema",
        "--fixed-judge-pilot",
    ]
    _register_pilot_fixture(cli, args, tmp_path, monkeypatch)
    with patch(
        "socket.socket.connect", side_effect=AssertionError("network forbidden")
    ):
        status = cli.main(args)
    assert status == 0
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert len(ledger["reservations"]) == 1
    assert ledger["reservations"][0]["kind"] == "judge"
    assert len(requests) == 1
    assert requests[0]["model"] == "xingchen-agi/xing4.0-29b-a4b:free"
    assert requests[0]["provider"]["only"] == ["Phala"]
    assert requests[0]["response_format"]["json_schema"]["strict"] is True
    assert sdk_args[0]["retry_config"] is None
    assert sdk_args[0]["client"].follow_redirects is False
    assert len(ledger["jobs"]) == 1
    assert next(iter(ledger["jobs"])).startswith("final:judge:semantic-v2:")
    result = json.loads(
        (tmp_path / "evaluation" / "runs" / "one" / "judge_results.jsonl").read_text()
    )
    assert result["observed_model"] == "xingchen-agi/xing4.0-29b-a4b-20261015"
    assert result["observed_provider"] == "Phala"
    assert result["observed_request_id"] == "offline-id"
    # Simulate a pilot artifact written before the durable profile marker existed.
    ledger_path = tmp_path / "ledger.json"
    ledger = json.loads(ledger_path.read_text())
    next(iter(ledger["jobs"].values())).pop("fixed_judge_pilot")
    ledger.pop("pilot_serving_identity")
    next(iter(ledger["jobs"].values())).pop("publication_sha256")
    ledger_path.write_text(json.dumps(ledger))
    # Byte-different JSON represents the same normalized record, but has a new
    # source-file identity and must not reuse the first assessment.
    records_path.write_text(
        json.dumps(json.loads(record.model_dump_json()), indent=None) + "\n"
    )
    args[args.index("--output-dir") + 1] = str(tmp_path / "evaluation" / "runs" / "two")
    with patch(
        "socket.socket.connect", side_effect=AssertionError("network forbidden")
    ):
        assert cli.main(args) == 2
    assert len(requests) == 1
    assert len(json.loads((tmp_path / "ledger.json").read_text())["jobs"]) == 1


@pytest.mark.parametrize(
    ("failure_identity", "success_model", "success_provider", "expected"),
    [
        (
            {"model": "xingchen-agi/xing4.0-29b-a4b-20261015", "provider": "Phala"},
            "xingchen-agi/xing4.0-29b-a4b:free",
            "Phala",
            2,
        ),
        (
            {"model": "xingchen-agi/xing4.0-29b-a4b-20261015"},
            "xingchen-agi/xing4.0-29b-a4b:free",
            "Phala",
            2,
        ),
        ({"provider": "phala"}, "xingchen-agi/xing4.0-29b-a4b-20261015", "Phala", 2),
        (
            {"model": "xingchen-agi/xing4.0-29b-a4b-20261015", "provider": "Phala"},
            "xingchen-agi/xing4.0-29b-a4b-20261015",
            "Phala",
            0,
        ),
        ({}, "xingchen-agi/xing4.0-29b-a4b-20261015", "Phala", 0),
    ],
)
def test_pilot_retry_observations_must_match(
    tmp_path, monkeypatch, failure_identity, success_model, success_provider, expected
):
    import openrouter
    from test_portfolio_budget import quota

    from scripts import evaluate_runtime_workflow as cli

    requests = []
    sdk_args = []
    real_sdk = OpenRouter

    def intercepted_sdk(**kwargs):
        sdk_args.append(kwargs)

        def respond(request):
            return requests.append(json.loads(request.content)) or (
                httpx.Response(
                    503,
                    json={
                        "error": {"message": "unavailable", "code": 503},
                        **failure_identity,
                    },
                )
                if len(requests) == 1
                else httpx.Response(
                    200,
                    json={
                        "id": "offline-id",
                        "model": success_model,
                        "provider": success_provider,
                        "object": "chat.completion",
                        "created": 0,
                        "system_fingerprint": None,
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {
                                    "role": "assistant",
                                    "content": json.dumps(
                                        {"score": True, "reasoning": "ok"}
                                    ),
                                },
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "total_tokens": 2,
                        },
                    },
                )
            )

        kwargs["client"] = httpx.Client(
            transport=httpx.MockTransport(respond), follow_redirects=False
        )
        return real_sdk(**kwargs)

    monkeypatch.setattr(openrouter, "OpenRouter", intercepted_sdk)
    monkeypatch.setattr(
        cli, "campaign_ledger", lambda *args, **kwargs: tmp_path / "ledger.json"
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    scenario = make_scenario()
    scenario.expectations.judge_dimensions = ["graph_usefulness"]
    record = normalize_trace_runs(root_run(), scenario)
    record.worker_sequence = ["visualizer"]
    record.graph.capture_status = "captured"
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(json.dumps([scenario.model_dump(mode="json")]))
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(record.model_dump_json() + "\n")
    quota_path = tmp_path / "quota.json"
    quota_path.write_text(json.dumps(quota(model="xingchen-agi/xing4.0-29b-a4b:free")))
    args = [
        "judge",
        "--confirm-provider-calls",
        "--budget-stage",
        "final",
        "--provider-quota",
        str(quota_path),
        "--evaluation-root",
        str(tmp_path / "evaluation"),
        "--output-dir",
        str(tmp_path / "evaluation" / "runs" / "one"),
        "--scenarios",
        str(scenario_path),
        "--records",
        str(records_path),
        "--case-id",
        scenario.id,
        "--dimension",
        "graph_usefulness",
        "--judge-provider",
        "openrouter",
        "--judge-model",
        "xingchen-agi/xing4.0-29b-a4b:free",
        "--structured-output-method",
        "json_schema",
        "--fixed-judge-pilot",
    ]
    _register_pilot_fixture(cli, args, tmp_path, monkeypatch)
    with patch(
        "socket.socket.connect", side_effect=AssertionError("network forbidden")
    ):
        status = cli.main(args)
    assert status == expected
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert len(requests) == 2
    assert len(ledger["reservations"]) == 2
    job = next(iter(ledger["jobs"].values()))
    result = json.loads(Path(job["artifact"]).read_text())
    assert result["usage"]["provider_attempts"] == 2
    assert result["prior_attempts"][0]["http_status_code"] == 503
    assert result["prior_attempts"][0]["observed_model"] == failure_identity.get(
        "model", ""
    )
    assert result["prior_attempts"][0]["observed_provider"] == failure_identity.get(
        "provider", ""
    )
    assert (
        tmp_path / "evaluation" / "runs" / "one" / "judge_results.jsonl"
    ).exists() == (expected == 0)

    if expected == 0 and failure_identity:
        # Reconcile an older saved result whose own attempt history conflicts,
        # with no campaign identity pinned yet.
        import hashlib

        result["prior_attempts"][0]["observed_model"] = (
            "xingchen-agi/xing4.0-29b-a4b:free"
        )
        artifact = Path(job["artifact"])
        artifact.write_text(json.dumps(result))
        job["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        ledger.pop("pilot_serving_identity", None)
        ledger_path = tmp_path / "ledger.json"
        ledger_path.write_text(json.dumps(ledger))
        args[args.index("--output-dir") + 1] = str(
            tmp_path / "evaluation" / "runs" / "restart"
        )
        with patch(
            "socket.socket.connect", side_effect=AssertionError("network forbidden")
        ):
            assert cli.main(args) == 2
        assert len(requests) == 2

        # An interrupted retry remains uncertain, but its durable failed attempt
        # still constrains a different job before any campaign identity is pinned.
        ledger = json.loads(ledger_path.read_text())
        job = next(iter(ledger["jobs"].values()))
        job["status"] = "started"
        ledger.pop("pilot_serving_identity", None)
        ledger_path.write_text(json.dumps(ledger))
        record.run_id = "new-job"
        records_path.write_text(record.model_dump_json() + "\n")
        success_model = "xingchen-agi/xing4.0-29b-a4b:free"
        args[args.index("--output-dir") + 1] = str(
            tmp_path / "evaluation" / "runs" / "new-job"
        )
        with patch(
            "socket.socket.connect", side_effect=AssertionError("network forbidden")
        ):
            assert cli.main(args) == 2
        assert len(requests) == 3


@pytest.mark.parametrize("profile", ["fixed", "standard", "recovered"])
def test_registered_fixed_pilot_dispatches_only_after_durable_reservation(
    tmp_path, monkeypatch, profile
):
    import argparse
    import hashlib
    from datetime import UTC

    import openrouter
    from test_portfolio_budget import registered_quota_fixture

    from backend.evaluation import portfolio_budget as pb
    from scripts import evaluate_runtime_workflow as cli

    arguments, _, old_proof_path, paths = registered_quota_fixture(
        tmp_path, monkeypatch
    )
    ledger = arguments["ledger"]
    monkeypatch.setattr(cli, "campaign_ledger", lambda *args, **kwargs: ledger)
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    scenario = make_scenario()
    scenario.expectations.judge_dimensions = ["graph_usefulness"]
    record = normalize_trace_runs(root_run(), scenario)
    record.worker_sequence = ["visualizer"]
    record.graph.capture_status = "captured"
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(json.dumps([scenario.model_dump(mode="json")]))
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(record.model_dump_json() + "\n")
    quota_path = tmp_path / "quota.json"
    quota_path.write_text("{}")
    output_dir = tmp_path / "evaluation/runs/one"
    settings_args = argparse.Namespace(
        budget_stage="final",
        provider_quota=quota_path,
        budget_ledger=ledger,
        fixed_judge_pilot=True,
        allow_tavily=False,
    )
    configuration = cli._evaluation_session(settings_args, output_dir).configuration
    state = json.loads(ledger.read_text())
    state["stage_configurations"]["final"] = configuration
    for prior_job in state["jobs"].values():
        prior_job.setdefault("run_id", "prior-synthetic")
    ledger.write_text(json.dumps(state, sort_keys=True))
    capture = pb.capture_prospective_boundary(
        checkout=arguments["checkout"],
        ledger=ledger,
        expected_ledger_sha256=hashlib.sha256(ledger.read_bytes()).hexdigest(),
        destination=tmp_path / "judge-boundary",
    )
    proof = json.loads(old_proof_path.read_text())
    proof.update(
        capture_started_at=capture["started_at"],
        boundary_path=str(tmp_path / "judge-boundary/boundary.json"),
        boundary_sha256=hashlib.sha256(
            (tmp_path / "judge-boundary/boundary.json").read_bytes()
        ).hexdigest(),
        capture_ledger_path=capture["snapshot_path"],
        capture_ledger_sha256=capture["ledger_sha256"],
    )
    model = "xingchen-agi/xing4.0-29b-a4b:free"
    q = proof["quota"] | {
        "model": model,
        "observed_at": pb.datetime.now(UTC).isoformat(),
    }
    proof["quota"] = q
    account = json.loads(paths["account"].read_text())
    account["openrouter"]["time"] = q["observed_at"]
    route = {
        "method": "GET",
        "url": f"https://openrouter.ai/api/v1/models/{model}/endpoints",
        "http_status": 200,
        "status": "success",
        "time": q["observed_at"],
        "data": {
            "id": model,
            "endpoints": [
                {
                    "model_id": model,
                    "provider_name": "Phala",
                    "tag": "phala",
                    "max_completion_tokens": 8192,
                    "pricing": {"prompt": "0", "completion": "0"},
                    "supported_parameters": [
                        "response_format",
                        "structured_outputs",
                        "temperature",
                        "reasoning",
                        "max_tokens",
                    ],
                }
            ],
        },
    }
    binding = json.loads(paths["binding"].read_text())
    binding.update(
        model=model,
        observed_at=q["observed_at"],
        checked_at=q["observed_at"],
        sut_role_credentials_match_queried_key={"judge": True},
    )
    paths["account"].write_text(json.dumps(account, sort_keys=True))
    paths["route"].write_text(json.dumps(route, sort_keys=True))
    paths["quota"].write_text(json.dumps(q, sort_keys=True))
    binding["account_capture_sha256"] = hashlib.sha256(
        paths["account"].read_bytes()
    ).hexdigest()
    binding["route_capture_sha256"] = hashlib.sha256(
        paths["route"].read_bytes()
    ).hexdigest()
    paths["binding"].write_text(json.dumps(binding, sort_keys=True))
    for name in paths:
        proof[f"{name}_capture_sha256"] = hashlib.sha256(
            paths[name].read_bytes()
        ).hexdigest()
    old_proof_path.write_text(json.dumps(proof, sort_keys=True))
    quota_path.write_text(json.dumps(q, sort_keys=True))
    before = json.loads(ledger.read_text())
    assert pb.register_prospective_observation(
        checkout=arguments["checkout"],
        ledger=ledger,
        expected_ledger_sha256=hashlib.sha256(ledger.read_bytes()).hexdigest(),
        proof_path=old_proof_path,
        expected_proof_sha256=hashlib.sha256(old_proof_path.read_bytes()).hexdigest(),
    )
    if profile == "recovered":
        # Synthesize the exact failed-setup evidence shape; constants are test-only.
        state = json.loads(ledger.read_text())
        prior = json.loads(json.dumps(state))
        old_config = configuration | {"evaluation_fingerprint": "prior"}
        record_hash = hashlib.sha256(
            json.dumps(
                record.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        input_hash = hashlib.sha256(records_path.read_bytes()).hexdigest()
        old_contract = {
            "scenario": scenario.model_dump(mode="json"),
            "record_sha256": record_hash,
            "source_records_file_sha256": input_hash,
            "dimension": "graph_usefulness",
            "provider": "openrouter",
            "model": model,
            "method": "json_schema",
            "judge_profile": "fixed-pilot-v1",
            "configuration": old_config,
        }
        new_contract = old_contract | {"configuration": configuration}
        old_digest = hashlib.sha256(
            json.dumps(old_contract, sort_keys=True).encode()
        ).hexdigest()
        old_job = "final:judge:semantic-v2:" + old_digest
        monkeypatch.setattr(pb, "_FIXED_SETUP_JOB", old_job)
        monkeypatch.setattr(pb, "_FIXED_INPUT", input_hash)
        original_job = {
            "status": "started",
            "run_id": "01_contextual_mamba_visualization_graph_usefulness",
            "contract_digest": old_digest,
            "fixed_judge_pilot": True,
        }
        prior["stage_configurations"]["final"] = old_config
        stopped = json.loads(json.dumps(prior))
        stopped["jobs"][old_job] = original_job
        evidence = tmp_path / "setup-evidence"
        run = evidence / "runs/01_contextual_mamba_visualization_graph_usefulness"
        run.mkdir(parents=True)
        (evidence / "ledger_after_registration.json").write_text(json.dumps(prior))
        (evidence / "ledger_after_setup_stop.json").write_text(json.dumps(stopped))
        (evidence / "01_invocation.json").write_text(
            json.dumps({"input_sha256": input_hash})
        )
        (evidence / "01_invocation_exit.json").write_text(json.dumps({"exit_code": 2}))
        (evidence / "01_invocation.log").write_text(
            "Runtime evaluation failed: evaluation model differs from quota observation"
        )
        (run / "manifest.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "artifacts": [],
                    "error": "EvaluationBudgetUnavailable: evaluation model differs from quota observation",
                }
            )
        )
        frozen_files = [path for path in evidence.rglob("*") if path.is_file()]
        freeze = evidence / "pilot_setup_stop_freeze.json"
        freeze.write_text(
            json.dumps(
                {
                    "files": {
                        path.relative_to(evidence).as_posix(): hashlib.sha256(
                            path.read_bytes()
                        ).hexdigest()
                        for path in frozen_files
                    }
                }
            )
        )
        monkeypatch.setattr(
            pb, "_FIXED_SETUP_FREEZE", hashlib.sha256(freeze.read_bytes()).hexdigest()
        )
        parent = tmp_path / "parent-source"
        parent.mkdir()
        source = parent / "source.txt"
        source.write_text("synthetic")
        inventory = tmp_path / "parent-inventory.json"
        inventory.write_text(
            json.dumps(
                {
                    "source.txt": {
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "bytes": len(source.read_bytes()),
                    }
                }
            )
        )
        monkeypatch.setattr(
            pb,
            "_FIXED_PARENT_INVENTORY",
            hashlib.sha256(inventory.read_bytes()).hexdigest(),
        )
        state["jobs"][old_job] = original_job
        state.setdefault("final_identity_transitions", []).append(
            {
                "transition": "fixed-judge-quota-recovery",
                "source_proof": {
                    "fingerprint": configuration["evaluation_fingerprint"]
                },
            }
        )
        ledger.write_text(json.dumps(state, sort_keys=True))
        pair = tmp_path / "contracts.json"
        pair.write_text(json.dumps({"old": old_contract, "new": new_contract}))
        pb.recover_fixed_judge_setup(
            checkout=arguments["checkout"],
            ledger=ledger,
            expected_ledger_sha256=hashlib.sha256(ledger.read_bytes()).hexdigest(),
            evidence_dir=evidence,
            parent_source=parent,
            parent_inventory=inventory,
            input_file=records_path,
            contract_pair=pair,
        )
    requests = []
    real_sdk = OpenRouter

    def sdk(**kwargs):
        def respond(request):
            saved = json.loads(ledger.read_text())
            assert len(saved["reservations"]) == len(before["reservations"]) + 1
            assert saved["reservations"][-1]["kind"] == "judge"
            if profile == "recovered":
                assert saved["fixed_judge_recoveries"][0]["consumed"] is True
                assert saved["jobs"][old_job] == original_job
            requests.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "id": "dummy",
                    "model": "xingchen-agi/xing4.0-29b-a4b-20261015",
                    "provider": "Phala",
                    "object": "chat.completion",
                    "created": 0,
                    "system_fingerprint": None,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(
                                    {"score": True, "reasoning": "ok"}
                                ),
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            )

        kwargs["client"] = httpx.Client(
            transport=httpx.MockTransport(respond), follow_redirects=False
        )
        return real_sdk(**kwargs)

    monkeypatch.setattr(openrouter, "OpenRouter", sdk)
    args = [
        "judge",
        "--confirm-provider-calls",
        "--budget-stage",
        "final",
        "--provider-quota",
        str(quota_path),
        "--evaluation-root",
        str(tmp_path / "evaluation"),
        "--output-dir",
        str(output_dir),
        "--scenarios",
        str(scenario_path),
        "--records",
        str(records_path),
        "--case-id",
        scenario.id,
        "--dimension",
        "graph_usefulness",
        "--judge-provider",
        "openrouter",
        "--judge-model",
        model,
        "--structured-output-method",
        "json_schema",
        "--fixed-judge-pilot",
    ]
    registered = json.loads(ledger.read_text())
    if profile == "standard":
        args.remove("--fixed-judge-pilot")
    with patch(
        "socket.socket.connect", side_effect=AssertionError("network forbidden")
    ):
        assert cli.main(args) == (2 if profile == "standard" else 0)
    if profile == "standard":
        after = json.loads(ledger.read_text())
        assert after["reservations"] == registered["reservations"]
        assert after["jobs"] == registered["jobs"]
        assert requests == []
        return
    assert len(requests) == 1
    assert requests[0]["model"] == model
    assert requests[0]["provider"]["only"] == ["Phala"]
    assert requests[0]["response_format"]["json_schema"]["strict"] is True
