import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from test_runtime_evaluation import make_scenario

from backend.evaluation import runtime_live
from backend.evaluation.portfolio_budget import EvaluationBudgetUnavailable
from scripts import evaluate_runtime_workflow as cli


def test_live_repetition_selector_selects_only_requested_repetition():
    args = cli.parse_args(
        [
            "live",
            "--case-id",
            "out_of_scope_biological_cats",
            "--confirm-provider-calls",
            "--repetition",
            "2",
        ]
    )
    assert cli._live_repetitions(args) == [2]
    with pytest.raises(SystemExit):
        cli._live_repetitions(
            cli.parse_args(
                [
                    "live",
                    "--case-id",
                    "out_of_scope_biological_cats",
                    "--confirm-provider-calls",
                    "--repetition",
                    "2",
                    "--repetitions",
                    "2",
                ]
            )
        )


def test_partial_diagnostics_survive_budget_guard_stop(tmp_path):
    class Graph:
        async def astream_events(self, *args, **kwargs):
            yield {
                "event": "on_chain_start",
                "name": "orchestrator",
                "run_id": "node-1",
                "metadata": {"langgraph_node": "orchestrator"},
                "data": {},
            }
            yield {
                "event": "on_chat_model_end",
                "run_id": "model-1",
                "metadata": {
                    "langgraph_node": "orchestrator",
                    "ls_provider": "openrouter",
                    "ls_model_name": "openrouter/free",
                },
                "data": {
                    "output": SimpleNamespace(
                        content="bad route",
                        tool_calls=[],
                        response_metadata={
                            "model": "actual",
                            "finish_reason": "stop",
                            "request_id": "req-1",
                        },
                        usage_metadata={},
                    )
                },
            }
            raise EvaluationBudgetUnavailable("Tavily guard stop")

    wrapper = SimpleNamespace(
        _graph=Graph(),
        _ensure_initialized=lambda: None,
        _convert_messages=lambda messages: [],
    )
    path = tmp_path / "diagnostics.jsonl"
    with patch.object(runtime_live, "get_workflow", return_value=wrapper):
        with pytest.raises(EvaluationBudgetUnavailable):
            asyncio.run(
                runtime_live.run_controlled_live_scenario(
                    make_scenario(), repetition=2, diagnostics_path=path
                )
            )
    events = [json.loads(line) for line in path.read_text().splitlines()]
    model = next(e for e in events if e["event"] == "on_chat_model_end")
    assert model["content"] == "bad route"
    assert model["configured_model"] == "openrouter/free"
    assert model["response_metadata"]["model"] == "actual"
    assert events[-1]["event"] == "exception"
    assert events[-1]["exception_chain"][0]["type"] == "EvaluationBudgetUnavailable"


def test_r2_cli_does_not_replay_uncertain_r1(tmp_path, monkeypatch, capsys):
    from test_portfolio_budget import quota

    from backend.evaluation.runtime_evaluation import RuntimeRecord

    quota_path = tmp_path / "quota.json"
    quota_path.write_text(json.dumps(quota()))
    ledger = tmp_path / "ledger.json"
    monkeypatch.setattr(cli, "campaign_ledger", lambda root, **kwargs: ledger)
    calls = []

    async def fake_run(scenario, *, repetition):
        calls.append(repetition)
        if repetition == 1:
            raise EvaluationBudgetUnavailable("uncertain fixture")
        return RuntimeRecord(
            scenario_id=scenario.id,
            scenario_version=scenario.version,
            run_id="fixture-r2",
            source_mode="controlled_live",
            input=scenario.input,
            repetition=repetition,
        )

    monkeypatch.setattr(cli, "run_controlled_live_scenario", fake_run)
    args = [
        "live",
        "--case-id",
        "out_of_scope_biological_cats",
        "--confirm-provider-calls",
        "--budget-stage",
        "B0",
        "--provider-quota",
        str(quota_path),
        "--evaluation-root",
        str(tmp_path / "eval"),
    ]
    assert cli.main([*args, "--output-dir", str(tmp_path / "r1")]) == 2
    before = json.loads(ledger.read_text())["jobs"][
        "B0:live:out_of_scope_biological_cats:v1:r1"
    ].copy()
    assert before["status"] == "started"
    assert (
        cli.main([*args, "--repetition", "2", "--output-dir", str(tmp_path / "r2")])
        == 0
    )
    assert (
        "provider attempts in published observations: unknown"
        in capsys.readouterr().out
    )
    jobs = json.loads(ledger.read_text())["jobs"]
    assert calls == [1, 2]
    assert jobs["B0:live:out_of_scope_biological_cats:v1:r1"] == before
    assert jobs["B0:live:out_of_scope_biological_cats:v1:r2"]["status"] == "completed"


def test_evaluation_retry_requires_concrete_transport_or_5xx():
    import httpx

    class HttpError(Exception):
        def __init__(self, status):
            self.status_code = status

    assert runtime_live._evaluation_retryable(httpx.ConnectError("fixture"))
    assert runtime_live._evaluation_retryable(TimeoutError("fixture"))
    assert runtime_live._evaluation_retryable(HttpError(503))
    assert not runtime_live._evaluation_retryable(HttpError(429))
    assert not runtime_live._evaluation_retryable(
        ValueError("connection output parsing failed")
    )
    assert not runtime_live._evaluation_retryable(ValueError("timeout schema invalid"))


def test_retry_guard_is_evaluation_only_and_restored(monkeypatch):
    import backend.workflows.learner as learner

    original = learner.is_transient_provider_error
    error = ValueError("connection parsing failed")
    assert original(error)
    monkeypatch.setattr(runtime_live, "current_evaluation_budget", lambda: object())
    with runtime_live._evaluation_retry_guard():
        assert not learner.is_transient_provider_error(error)
        assert learner.is_transient_provider_error(TimeoutError("real timeout"))
    assert learner.is_transient_provider_error is original


def test_diagnostic_redacts_credentials_and_is_valid_bounded_json(
    tmp_path, monkeypatch
):
    from backend.utils.chat_limits import ChatTurnContext

    monkeypatch.setenv("EVALUATION_DUMMY_API_KEY", "supersecretcredential")
    path = tmp_path / "private.jsonl"
    event = {
        "event": "on_chat_model_end",
        "run_id": "event",
        "metadata": {"langgraph_node": "orchestrator"},
        "data": {
            "output": SimpleNamespace(
                content="supersecretcredential",
                tool_calls=[],
                response_metadata={"api_key": "supersecretcredential"},
            )
        },
    }
    runtime_live._write_live_diagnostic(
        path, event, ChatTurnContext(visitor_id="x", session_id="x")
    )
    contents = path.read_text()
    assert "supersecretcredential" not in contents
    saved = json.loads(contents)
    assert saved["content"] == "[REDACTED]"
    assert saved["response_metadata"]["api_key"] == "[REDACTED]"
    assert path.stat().st_mode & 0o777 == 0o600


def test_oversized_diagnostic_preserves_routing_provider_and_guard_prefixes(tmp_path):
    from backend.utils.chat_limits import ChatTurnContext

    path = tmp_path / "large.jsonl"
    context = ChatTurnContext(visitor_id="x", session_id="x")
    runtime_live._write_live_diagnostic(
        path,
        {
            "event": "on_chat_model_end",
            "run_id": "event",
            "metadata": {
                "langgraph_node": "orchestrator",
                "ls_model_name": "openrouter/free",
            },
            "data": {
                "output": SimpleNamespace(
                    content="ROUTE-PREFIX" + "x" * 8000,
                    tool_calls=[{"name": "target", "args": {"topic": "y" * 8000}}],
                    response_metadata={
                        "model": "served-model",
                        "finish_reason": "stop",
                        "request_id": "request-1",
                    },
                )
            },
        },
        context,
    )
    try:
        raise EvaluationBudgetUnavailable("guard-stop-" + "z" * 8000)
    except EvaluationBudgetUnavailable as exc:
        runtime_live._write_live_exception(path, exc, context)
    model, guard = map(json.loads, path.read_text().splitlines())
    assert model["content"].startswith("ROUTE-PREFIX")
    assert model["response_metadata"]["model"] == "served-model"
    assert model["response_metadata"]["request_id"] == "request-1"
    assert guard["exception_chain"][0]["type"] == "EvaluationBudgetUnavailable"
    assert guard["exception_chain"][0]["message"].startswith("guard-stop-")
    assert max(map(len, path.read_text().splitlines())) < 16000


def test_diagnostic_run_does_not_fabricate_record_after_exception(tmp_path):
    class Graph:
        async def astream_events(self, *args, **kwargs):
            yield {
                "event": "on_chain_start",
                "name": "orchestrator",
                "metadata": {"langgraph_node": "orchestrator"},
                "data": {},
            }
            raise ValueError("fixture parser failure")

    wrapper = SimpleNamespace(
        _graph=Graph(),
        _ensure_initialized=lambda: None,
        _convert_messages=lambda messages: [],
    )
    path = tmp_path / "diagnostics.jsonl"
    with patch.object(runtime_live, "get_workflow", return_value=wrapper):
        with pytest.raises(RuntimeError, match="fixture parser failure"):
            asyncio.run(
                runtime_live.run_controlled_live_scenario(
                    make_scenario(), repetition=2, diagnostics_path=path
                )
            )
    assert (
        json.loads(path.read_text().splitlines()[-1])["exception_chain"][0]["type"]
        == "ValueError"
    )


def test_fourth_cats_attempt_is_blocked_before_transport(tmp_path, monkeypatch):
    from datetime import UTC, datetime, timedelta

    from test_portfolio_budget import quota

    from backend.configs.chat_limits import ChatLimitsSettings
    from backend.evaluation import portfolio_budget as budget_module
    from backend.utils.chat_limits import ChatLimitService, get_current_chat_turn

    now = datetime.now(UTC)

    class Clock(datetime):
        moment = None

        @classmethod
        def now(cls, tz=None):
            return cls.moment

    Clock.moment = now - timedelta(days=1)
    monkeypatch.setattr(budget_module, "datetime", Clock)
    ledger = tmp_path / "ledger.json"
    for count in (24, 6):
        with budget_module.EvaluationBudget(
            ledger,
            stage="B0",
            run_id=f"prior-{count}",
            quota=quota(Clock.moment, remaining_requests=100),
        ) as guard:
            for _ in range(count):
                guard.reserve("llm", provider="openrouter", model="openrouter/free")
        Clock.moment = now
    sent = []
    limits = ChatLimitService(
        settings=ChatLimitsSettings(_env_file=None, shared_quotas_enabled=False)
    )

    class Graph:
        async def astream_events(self, *args, **kwargs):
            for number in range(4):
                await limits.reserve_llm_attempt(
                    get_current_chat_turn(), "openrouter", "orchestrator"
                )
                sent.append(number)
                yield {
                    "event": "on_chain_start",
                    "name": "orchestrator",
                    "metadata": {"langgraph_node": "orchestrator"},
                    "data": {},
                }

    wrapper = SimpleNamespace(
        _graph=Graph(),
        _ensure_initialized=lambda: None,
        _convert_messages=lambda messages: [],
    )
    monkeypatch.setattr(runtime_live, "get_workflow", lambda: wrapper)
    monkeypatch.setattr(
        runtime_live, "_disable_tavily_transport_retries", lambda wrapper: None
    )
    path = tmp_path / "guard.jsonl"
    with (
        budget_module.EvaluationBudget(
            ledger,
            stage="B0",
            run_id="cats-r2",
            quota=quota(now, remaining_requests=100),
        ) as guard,
        budget_module.bind_evaluation_budget(guard),
    ):
        guard.begin_job("live:cats:v1:r2")
        with pytest.raises(budget_module.EvaluationBudgetExceeded, match="B0 llm"):
            asyncio.run(
                runtime_live.run_controlled_live_scenario(
                    make_scenario(), repetition=2, diagnostics_path=path
                )
            )
    assert sent == [0, 1, 2]
    assert len(json.loads(ledger.read_text())["reservations"]) == 33
    terminal = json.loads(path.read_text().splitlines()[-1])
    assert terminal["exception_chain"][0]["type"] == "EvaluationBudgetExceeded"
    assert len(terminal["reservation_ids"]) == 3
