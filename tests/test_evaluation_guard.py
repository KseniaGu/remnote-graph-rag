from datetime import UTC, datetime, timedelta

import pytest

from backend.evaluation.request_guard import EvaluationGuard, EvaluationGuardExceeded


def quota(observed_at, remaining=5):
    return {
        "observed_at": observed_at.isoformat(),
        "provider": "openrouter",
        "model": "test/model",
        "remaining_requests": remaining,
        "demo_reserve": 1,
        "daily_batch_limit": 10,
    }


def test_observation_before_reservation_counts_it_once_across_sessions(tmp_path):
    now = datetime.now(UTC)
    ledger = tmp_path / "guard.json"
    observation = quota(now - timedelta(seconds=1), remaining=3)
    with EvaluationGuard(ledger, quota=observation, max_attempts=10) as guard:
        guard.reserve("llm", provider="openrouter", model="test/model")
    with EvaluationGuard(ledger, quota=observation, max_attempts=10) as guard:
        guard.reserve("judge", provider="openrouter", model="test/model")
        with pytest.raises(EvaluationGuardExceeded):
            guard.reserve("llm", provider="openrouter", model="test/model")


def test_new_observation_cannot_increase_same_day_capacity(tmp_path):
    now = datetime.now(UTC)
    ledger = tmp_path / "guard.json"
    with EvaluationGuard(
        ledger, quota=quota(now - timedelta(seconds=2), 3), max_attempts=10
    ) as guard:
        guard.reserve("llm", provider="openrouter", model="test/model")
    with EvaluationGuard(
        ledger, quota=quota(now - timedelta(seconds=1), 100), max_attempts=10
    ) as guard:
        guard.reserve("judge", provider="openrouter", model="test/model")
        with pytest.raises(EvaluationGuardExceeded):
            guard.reserve("judge", provider="openrouter", model="test/model")


def test_new_observation_counts_only_spending_after_its_timestamp(tmp_path):
    ledger = tmp_path / "guard.json"
    first = quota(datetime.now(UTC) - timedelta(seconds=2), 10)
    with EvaluationGuard(ledger, quota=first, max_attempts=10) as guard:
        guard.reserve("llm", provider="openrouter", model="test/model")
        guard.reserve("llm", provider="openrouter", model="test/model")
    second = quota(datetime.now(UTC), 3)
    with EvaluationGuard(ledger, quota=second, max_attempts=10) as guard:
        guard.reserve("judge", provider="openrouter", model="test/model")
        guard.reserve("judge", provider="openrouter", model="test/model")
        with pytest.raises(EvaluationGuardExceeded):
            guard.reserve("judge", provider="openrouter", model="test/model")


def test_guard_blocks_attempts_before_dispatch_and_survives_restart(tmp_path):
    ledger = tmp_path / "guard.json"
    observation = quota(datetime.now(UTC) - timedelta(seconds=1), 10)
    sent = []
    with EvaluationGuard(ledger, quota=observation, max_attempts=1) as guard:
        guard.reserve("llm", provider="openrouter", model="test/model")
        sent.append(1)
        with pytest.raises(EvaluationGuardExceeded):
            guard.reserve("llm", provider="openrouter", model="test/model")
            sent.append(2)
    assert sent == [1]
    with EvaluationGuard(ledger, quota=observation, max_attempts=1) as guard:
        guard.reserve("judge", provider="openrouter", model="test/model")
    assert (
        len(
            __import__("json").loads(ledger.read_text())["days"][
                datetime.now(UTC).date().isoformat()
            ]["reservations"]
        )
        == 2
    )


def test_guard_closes_clients_it_owns_on_exit(tmp_path):
    import asyncio

    class Client:
        closed = False

        def close(self):
            self.closed = True

    class AsyncClient:
        closed = False

        async def aclose(self):
            self.closed = True

    sync, asynchronous = Client(), AsyncClient()

    async def run():
        async with EvaluationGuard(
            tmp_path / "guard.json", quota=quota(datetime.now(UTC)), max_attempts=1
        ) as guard:
            guard.register_clients(sync, asynchronous)
            raise RuntimeError("cancelled work")

    with pytest.raises(RuntimeError):
        asyncio.run(run())
    assert sync.closed
    assert asynchronous.closed


def test_production_quota_works_without_evaluation_context():
    import asyncio

    from backend.configs.chat_limits import ChatLimitsSettings
    from backend.utils.chat_limits import ChatLimitService, ChatTurnContext

    limits = ChatLimitService(settings=ChatLimitsSettings(_env_file=None))
    turn = ChatTurnContext(visitor_id="test", session_id="test")
    asyncio.run(limits.reserve_llm_attempt(turn, "openrouter", "analyst"))
    assert turn.llm_provider_attempts == 1


def test_guarded_turn_counts_retry_and_blocks_missing_context(tmp_path):
    import asyncio

    from backend.configs.chat_limits import ChatLimitsSettings
    from backend.utils.chat_limits import (
        ChatLimitService,
        ChatTurnContext,
        EvaluationBudgetUnavailable,
        bind_evaluation_budget,
    )

    limits = ChatLimitService(settings=ChatLimitsSettings(_env_file=None))
    with (
        EvaluationGuard(
            tmp_path / "guard.json", quota=quota(datetime.now(UTC)), max_attempts=2
        ) as guard,
        bind_evaluation_budget(guard),
    ):
        turn = ChatTurnContext(visitor_id="test", session_id="test")
        asyncio.run(limits.reserve_llm_attempt(turn, "openrouter", "analyst"))
        asyncio.run(limits.reserve_llm_attempt(turn, "openrouter", "analyst"))
        with pytest.raises(EvaluationGuardExceeded):
            asyncio.run(limits.reserve_llm_attempt(turn, "openrouter", "analyst"))
        with pytest.raises(EvaluationBudgetUnavailable):
            asyncio.run(limits.reserve_llm_attempt(None, "openrouter", "analyst"))
        assert guard.attempts == 2


def test_evaluation_web_requires_opt_in_and_respects_attempt_limit(tmp_path):
    import asyncio

    from backend.configs.chat_limits import ChatLimitsSettings
    from backend.utils.chat_limits import (
        ChatLimitService,
        ChatTurnContext,
        bind_evaluation_budget,
    )

    limits = ChatLimitService(settings=ChatLimitsSettings(_env_file=None))
    with (
        EvaluationGuard(
            tmp_path / "guard.json",
            quota=quota(datetime.now(UTC)),
            max_attempts=2,
            max_web_attempts=1,
        ) as guard,
        bind_evaluation_budget(guard),
    ):
        turn = ChatTurnContext(visitor_id="test", session_id="test")
        limits.begin_tavily_search(turn)
        asyncio.run(limits.reserve_tavily_attempt(turn))
        with pytest.raises(EvaluationGuardExceeded):
            asyncio.run(limits.reserve_tavily_attempt(turn))


def test_guarded_openrouter_disables_transport_and_sdk_retries(tmp_path, monkeypatch):
    import httpx
    from openrouter.errors import ServiceUnavailableResponseError

    from backend.configs.enums import ModelRoleType
    from backend.configs.models import OpenRouterSettings
    from backend.utils.chat_limits import bind_evaluation_budget
    from backend.workflows.agents.factory import AgentsFactory

    requests = []

    def fail(request):
        requests.append(request)
        return httpx.Response(
            503, json={"error": {"message": "temporary", "code": 503}}
        )

    monkeypatch.setattr(httpx, "HTTPTransport", lambda **_: httpx.MockTransport(fail))
    monkeypatch.setattr(
        httpx, "AsyncHTTPTransport", lambda **_: httpx.MockTransport(fail)
    )
    settings = OpenRouterSettings(
        _env_file=None,
        role=ModelRoleType.analyst,
        model_name="test/model",
        api_key="fake-offline-key",
    )
    with (
        EvaluationGuard(
            tmp_path / "guard.json", quota=quota(datetime.now(UTC)), max_attempts=2
        ) as guard,
        bind_evaluation_budget(guard),
    ):
        model = AgentsFactory.get_llm_by_role(settings)
        assert model.client.sdk_configuration.retry_config is None
        with pytest.raises(ServiceUnavailableResponseError):
            model.invoke("offline fixture")
    assert len(requests) == 1


def test_stale_quota_and_unsupported_model_fail_before_reservation(tmp_path):
    from backend.utils.chat_limits import EvaluationBudgetUnavailable

    with pytest.raises(EvaluationBudgetUnavailable):
        with EvaluationGuard(
            tmp_path / "guard.json",
            quota=quota(datetime.now(UTC) - timedelta(minutes=16)),
            max_attempts=1,
        ):
            pass
    with EvaluationGuard(
        tmp_path / "guard.json", quota=quota(datetime.now(UTC)), max_attempts=1
    ) as guard:
        with pytest.raises(EvaluationBudgetUnavailable):
            guard.reserve("llm", provider="openrouter", model="different/model")
        assert guard.attempts == 0


def test_web_search_is_blocked_when_not_enabled(tmp_path):
    from backend.utils.chat_limits import EvaluationBudgetUnavailable

    with EvaluationGuard(
        tmp_path / "guard.json", quota=quota(datetime.now(UTC)), max_attempts=1
    ) as guard:
        with pytest.raises(EvaluationBudgetUnavailable):
            guard.reserve("web_search")


def test_provider_429_stops_the_live_batch():
    from backend.evaluation.runtime_live import _raise_capacity_stop
    from backend.utils.chat_limits import EvaluationBudgetUnavailable

    class ResponseError(Exception):
        status_code = 429

    with pytest.raises(EvaluationBudgetUnavailable):
        _raise_capacity_stop(ResponseError())


def test_live_cli_uses_guard_at_workflow_boundary(tmp_path, monkeypatch):
    from backend.configs.chat_limits import ChatLimitsSettings
    from backend.evaluation.runtime_evaluation import RuntimeRecord
    from backend.utils.chat_limits import ChatLimitService, ChatTurnContext
    from scripts import evaluate_runtime_workflow as cli

    ledger = tmp_path / "guard.json"
    observation = quota(datetime.now(UTC), 5)
    monkeypatch.setattr(
        cli,
        "_evaluation_session",
        lambda args, output: EvaluationGuard(ledger, quota=observation, max_attempts=1),
    )
    limits = ChatLimitService(settings=ChatLimitsSettings(_env_file=None))

    async def fake_run(scenario, *, repetition, **kwargs):
        turn = ChatTurnContext(visitor_id="test", session_id="test")
        await limits.reserve_llm_attempt(turn, "openrouter", "analyst")
        with pytest.raises(EvaluationGuardExceeded):
            await limits.reserve_llm_attempt(turn, "openrouter", "analyst")
        return RuntimeRecord(
            scenario_id=scenario.id,
            scenario_version=scenario.version,
            run_id="synthetic-run",
            repetition=repetition,
            source_mode="controlled_live",
            input=scenario.input,
        )

    monkeypatch.setattr(cli, "run_controlled_live_scenario", fake_run)
    assert (
        cli.main(
            [
                "live",
                "--case-id",
                "out_of_scope_biological_cats",
                "--confirm-provider-calls",
                "--output-dir",
                str(tmp_path / "run"),
                "--evaluation-root",
                str(tmp_path / "eval"),
            ]
        )
        == 0
    )
    assert ledger.exists()


def test_judge_429_preserves_private_partial_diagnostics(tmp_path, monkeypatch):
    from synthetic_runtime import make_scenario, root_run

    from backend.evaluation.runtime_evaluation import (
        SemanticJudgeResult,
        normalize_trace_runs,
    )
    from backend.workflows.agents.factory import AgentsFactory
    from scripts import evaluate_runtime_workflow as cli

    scenario = make_scenario()
    scenario.expectations.judge_dimensions = ["analyst_usefulness"]
    record = normalize_trace_runs(root_run(), scenario)
    records = tmp_path / "records.jsonl"
    records.write_text(record.model_dump_json() + "\n")
    monkeypatch.setattr(cli, "load_runtime_scenarios", lambda path: [scenario])
    monkeypatch.setattr(
        cli,
        "_evaluation_session",
        lambda args, output: EvaluationGuard(
            tmp_path / "guard.json", quota=quota(datetime.now(UTC)), max_attempts=1
        ),
    )
    monkeypatch.setattr(AgentsFactory, "get_llm_by_role", lambda settings: object())
    result = SemanticJudgeResult(
        scenario_id=scenario.id,
        scenario_version=scenario.version,
        run_id=record.run_id,
        repetition=1,
        dimension="analyst_usefulness",
        judge_provider="openrouter",
        judge_model="test/model",
        status="error",
        http_status_code=429,
    )
    monkeypatch.setattr(
        cli, "evaluate_with_openevals_judge", lambda *args, **kwargs: result
    )
    output = tmp_path / "run"
    assert (
        cli.main(
            [
                "judge",
                "--records",
                str(records),
                "--case-id",
                scenario.id,
                "--dimension",
                "analyst_usefulness",
                "--judge-provider",
                "openrouter",
                "--judge-model",
                "test/model",
                "--confirm-provider-calls",
                "--output-dir",
                str(output),
                "--evaluation-root",
                str(tmp_path / "eval"),
            ]
        )
        == 2
    )
    assert (
        '"http_status_code":429' in (output / "partial_judge_results.jsonl").read_text()
    )
