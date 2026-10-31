"""Offline evidence for aggregate evaluation attempt and resume boundaries."""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from backend.evaluation import portfolio_budget as budget_module


def quota(day=None, **updates):
    values = {
        "observed_at": (day or budget_module.datetime.now(UTC)).isoformat(),
        "provider": "openrouter",
        "model": "openrouter/free",
        "remaining_requests": 200,
        "demo_reserve": 10,
        "daily_batch_limit": 100,
    }
    return values | updates


def session(tmp_path, stage="B0", **kwargs):
    return budget_module.EvaluationBudget(
        tmp_path / "ledger.json", stage=stage, run_id="run-1", quota=quota(), **kwargs
    )


def test_budget_api_exists():
    assert hasattr(budget_module, "EvaluationBudget")


@pytest.fixture
def campaign_clock(monkeypatch):
    initial = datetime.now(UTC)

    class Clock(datetime):
        moment = initial

        @classmethod
        def now(cls, tz=None):
            return cls.moment

        @classmethod
        def next_day(cls):
            cls.moment += timedelta(days=1)

    monkeypatch.setattr(budget_module, "datetime", Clock)
    return Clock


def spend_across_days(tmp_path, kinds, stage, campaign_clock):
    for offset in range(0, len(kinds), 24):
        if offset:
            campaign_clock.next_day()
        with session(tmp_path, stage=stage) as guard:
            for kind in kinds[offset : offset + 24]:
                guard.reserve(kind, provider="openrouter", model="openrouter/free")


def test_stage_exhaustion_prevents_dispatch_and_survives_restart(
    tmp_path, campaign_clock
):
    spend_across_days(tmp_path, ["llm"] * 33, "B0", campaign_clock)
    for _ in range(2):
        with session(tmp_path) as guard:
            with pytest.raises(budget_module.EvaluationBudgetExceeded, match="B0 llm"):
                guard.reserve("llm", provider="openrouter", model="openrouter/free")
    saved = json.loads((tmp_path / "ledger.json").read_text())
    assert len(saved["reservations"]) == 33
    assert sorted(day["used"] for day in saved["days"].values()) == [9, 24]


def test_global_caps_and_judge_web_sublimits(tmp_path, campaign_clock):
    for stage, runtime_count, search_count, attempt_count in (
        ("B0", 28, 3, 6),
        ("final", 54, 7, 14),
    ):
        if stage == "final":
            campaign_clock.next_day()
        with session(tmp_path, stage=stage) as guard:
            for _ in range(5):
                guard.reserve("judge", provider="openrouter", model="openrouter/free")
            with pytest.raises(budget_module.EvaluationBudgetExceeded, match="judge"):
                guard.reserve("judge", provider="openrouter", model="openrouter/free")
        campaign_clock.next_day()
        spend_across_days(tmp_path, ["llm"] * runtime_count, stage, campaign_clock)
        with session(tmp_path, stage=stage) as guard:
            for _ in range(search_count):
                guard.reserve("web_search")
            for _ in range(attempt_count):
                guard.reserve("web_attempt")
            for kind in ("llm", "judge", "web_search", "web_attempt"):
                with pytest.raises(budget_module.EvaluationBudgetExceeded):
                    guard.reserve(kind, provider="openrouter", model="openrouter/free")
    saved = json.loads((tmp_path / "ledger.json").read_text())
    assert len(saved["reservations"]) == 122  # 92 LLM + 10 logical + 20 web
    assert all(day["used"] <= 24 for day in saved["days"].values())


def test_exclusive_runner_lock(tmp_path):
    with session(tmp_path):
        with pytest.raises(budget_module.EvaluationBudgetUnavailable, match="locked"):
            with session(tmp_path):
                pytest.fail("second runner entered")


def test_daily_batch_includes_judges_and_resumes_without_reset(tmp_path):
    path = tmp_path / "ledger.json"
    observation = quota(remaining_requests=23, demo_reserve=20, daily_batch_limit=2)
    with budget_module.EvaluationBudget(
        path, stage="B0", run_id="a", quota=observation
    ) as guard:
        guard.reserve("llm", provider="openrouter", model="openrouter/free")
        guard.reserve("judge", provider="openrouter", model="openrouter/free")
    with budget_module.EvaluationBudget(
        path, stage="final", run_id="b", quota=observation
    ) as guard:
        with pytest.raises(budget_module.EvaluationBudgetExceeded, match="daily"):
            guard.reserve("llm", provider="openrouter", model="openrouter/free")


def test_demo_reserve_provider_mismatch_and_stale_observation_fail_closed(tmp_path):
    with budget_module.EvaluationBudget(
        tmp_path / "ledger.json",
        stage="B0",
        run_id="a",
        quota=quota(remaining_requests=21),
    ) as guard:
        guard.reserve("llm", provider="openrouter", model="openrouter/free")
        with pytest.raises(budget_module.EvaluationBudgetExceeded, match="reserve"):
            guard.reserve("judge", provider="openrouter", model="openrouter/free")
        with pytest.raises(budget_module.EvaluationBudgetUnavailable):
            guard.reserve("llm", provider="openai", model="different")
    with pytest.raises(budget_module.EvaluationBudgetUnavailable):
        with budget_module.EvaluationBudget(
            tmp_path / "ledger.json",
            stage="B0",
            run_id="a",
            quota=quota(datetime.now(UTC) - timedelta(days=1)),
        ):
            pytest.fail("stale quota accepted")


def test_completed_jobs_resume_and_uncertain_jobs_never_redispatch(tmp_path):
    artifact = tmp_path / "record.json"
    artifact.write_text('{"safe": true}')
    with session(tmp_path) as guard:
        assert guard.begin_job("case-1:r1") is None
        guard.complete_job("case-1:r1", artifact)
        guard.begin_job("uncertain:r1")
    with session(tmp_path) as guard:
        assert guard.begin_job("case-1:r1") == artifact
        with pytest.raises(
            budget_module.EvaluationBudgetUnavailable, match="uncertain"
        ):
            guard.begin_job("uncertain:r1")
    artifact.write_text("changed")
    with session(tmp_path) as guard:
        with pytest.raises(budget_module.EvaluationBudgetUnavailable, match="artifact"):
            guard.begin_job("case-1:r1")


def test_ledger_records_identities_without_request_content(tmp_path):
    with session(tmp_path) as guard:
        guard.begin_job("case-1:r1")
        guard.reserve("llm", provider="openrouter", model="openrouter/free")
    saved = json.loads((tmp_path / "ledger.json").read_text())
    event = saved["reservations"][0]
    assert event["stage"] == "B0"
    assert event["run_id"] == "run-1"
    assert event["job_id"] == "B0:case-1:r1"
    assert set(event) == {
        "kind",
        "stage",
        "run_id",
        "job_id",
        "request_id",
        "reserved_at",
    }


def test_evaluation_hooks_count_failed_retries_and_block_missing_turn(tmp_path):
    import asyncio

    from backend.configs.chat_limits import ChatLimitsSettings
    from backend.utils.chat_limits import ChatLimitService, ChatTurnContext

    limits = ChatLimitService(
        settings=ChatLimitsSettings(_env_file=None, shared_quotas_enabled=False)
    )
    with session(tmp_path) as guard, budget_module.bind_evaluation_budget(guard):
        turn = ChatTurnContext(visitor_id="test", session_id="test")
        asyncio.run(limits.reserve_llm_attempt(turn, "openrouter", "analyst"))
        asyncio.run(limits.reserve_llm_attempt(turn, "openrouter", "analyst"))
        with pytest.raises(budget_module.EvaluationBudgetUnavailable, match="turn"):
            asyncio.run(limits.reserve_llm_attempt(None, "openrouter", "analyst"))
    events = json.loads((tmp_path / "ledger.json").read_text())["reservations"]
    assert [event["kind"] for event in events] == ["llm", "llm"]


def test_web_guard_enforces_opt_in_and_actual_attempt_sublimit(tmp_path):
    import asyncio

    from backend.configs.chat_limits import ChatLimitsSettings
    from backend.utils.chat_limits import ChatLimitService, ChatTurnContext

    limits = ChatLimitService(
        settings=ChatLimitsSettings(_env_file=None, shared_quotas_enabled=False)
    )
    with session(tmp_path) as guard, budget_module.bind_evaluation_budget(guard):
        turn = ChatTurnContext(visitor_id="test", session_id="test")
        with pytest.raises(budget_module.EvaluationBudgetUnavailable, match="Tavily"):
            limits.begin_tavily_search(turn)
        guard.allow_tavily = True
        for _ in range(3):
            turn = ChatTurnContext(visitor_id="test", session_id="test")
            limits.begin_tavily_search(turn)
            asyncio.run(limits.reserve_tavily_attempt(turn))
            asyncio.run(limits.reserve_tavily_attempt(turn))
        with pytest.raises(budget_module.EvaluationBudgetExceeded):
            asyncio.run(limits.reserve_tavily_attempt(turn))
        with pytest.raises(budget_module.EvaluationBudgetExceeded):
            limits.begin_tavily_search(
                ChatTurnContext(visitor_id="test", session_id="test")
            )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_evaluation_openrouter_transport_has_one_request_on_503(
    tmp_path, monkeypatch, asynchronous
):
    import asyncio

    import httpx
    from openrouter.errors import ServiceUnavailableResponseError

    from backend.configs.enums import ModelRoleType
    from backend.configs.models import OpenRouterSettings
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
        model_name="openrouter/free",
        api_key="fake-offline-key",
    )
    with session(tmp_path) as guard, budget_module.bind_evaluation_budget(guard):
        model = AgentsFactory.get_llm_by_role(settings)
        assert model.max_retries == 0
        assert model.client.sdk_configuration.retry_config is None
        with pytest.raises(ServiceUnavailableResponseError):
            if asynchronous:
                asyncio.run(model.ainvoke("offline fixture"))
            else:
                model.invoke("offline fixture")
    assert len(requests) == 1


def test_unknown_evaluation_transport_fails_closed(tmp_path):
    from backend.configs.enums import ModelRoleType
    from backend.configs.models import OllamaSettings
    from backend.workflows.agents.factory import AgentsFactory

    settings = OllamaSettings(
        _env_file=None,
        role=ModelRoleType.analyst,
        model_name="unsupported",
        api_key=None,
    )
    with session(tmp_path) as guard, budget_module.bind_evaluation_budget(guard):
        with pytest.raises(
            budget_module.EvaluationBudgetUnavailable, match="transport"
        ):
            AgentsFactory.get_llm_by_role(settings)


@pytest.mark.parametrize("dimension", ["analyst_usefulness", "claim_faithfulness"])
def test_judge_cannot_bypass_aggregate_sublimit(tmp_path, dimension):
    from test_runtime_evaluation import make_scenario, root_run

    from backend.evaluation.runtime_evaluation import (
        evaluate_with_openevals_judge,
        normalize_trace_runs,
    )

    scenario = make_scenario()
    scenario.expectations.judge_dimensions = [dimension]
    record = normalize_trace_runs(root_run(), scenario)
    if dimension == "claim_faithfulness":
        record.evidence_capture_status = "captured"
        from backend.evaluation.runtime_evaluation import EvidenceObservation

        record.bounded_evidence = [
            EvidenceObservation(
                evidence_id="chunk_mamba", kind="local_chunk", excerpt="Mamba evidence"
            )
        ]
    dispatched = []

    def evaluator_factory(**kwargs):
        def evaluator(**inputs):
            dispatched.append(inputs)
            raise TimeoutError("failed fixture attempt")

        return evaluator

    with session(tmp_path) as guard, budget_module.bind_evaluation_budget(guard):
        for _ in range(5):
            result = evaluate_with_openevals_judge(
                scenario,
                record,
                dimension,
                judge=None,
                judge_provider="openrouter",
                judge_model="openrouter/free",
                evaluator_factory=evaluator_factory,
            )
            assert result.provider_call_attempted
        with pytest.raises(budget_module.EvaluationBudgetExceeded):
            evaluate_with_openevals_judge(
                scenario,
                record,
                dimension,
                judge=None,
                judge_provider="openrouter",
                judge_model="openrouter/free",
                evaluator_factory=evaluator_factory,
            )
    assert len(dispatched) == 5


def test_live_cli_requires_durable_budget_before_dispatch(tmp_path, monkeypatch):
    from scripts import evaluate_runtime_workflow as cli

    dispatched = []

    async def fake_run(*args, **kwargs):
        dispatched.append(args)
        raise AssertionError("unguarded dispatch")

    monkeypatch.setattr(cli, "run_controlled_live_scenario", fake_run)
    status = cli.main(
        [
            "live",
            "--case-id",
            "out_of_scope_biological_cats",
            "--confirm-provider-calls",
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )
    assert status == 2
    assert dispatched == []


def test_campaign_ledger_is_shared_across_worktrees(tmp_path):
    main = tmp_path / "main"
    worktree = tmp_path / "other"
    (main / ".git" / "worktrees" / "other").mkdir(parents=True)
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {main}/.git/worktrees/other\n")
    (main / ".git" / "worktrees" / "other" / "commondir").write_text("../..\n")
    assert (
        budget_module.campaign_ledger(main)
        == main / "data/evaluation/portfolio_budget/ledger.json"
    )
    assert (
        budget_module.campaign_ledger(worktree)
        == main / "data/evaluation/portfolio_budget/ledger.json"
    )


def test_real_headless_workflow_requires_evaluation_guard():
    from backend.evaluation.runtime_live import get_workflow

    with pytest.raises(budget_module.EvaluationBudgetUnavailable):
        get_workflow()


def test_daily_rollover_does_not_reset_stage_spending(tmp_path, monkeypatch):
    today = datetime.now(UTC)
    path = tmp_path / "ledger.json"
    with budget_module.EvaluationBudget(
        path, stage="B0", run_id="day1", quota=quota(daily_batch_limit=1)
    ) as guard:
        guard.reserve("judge", provider="openrouter", model="openrouter/free")
    tomorrow = today + timedelta(days=1)

    class NextDay(datetime):
        @classmethod
        def now(cls, tz=None):
            return tomorrow

    monkeypatch.setattr(budget_module, "datetime", NextDay)
    with budget_module.EvaluationBudget(
        path, stage="B0", run_id="day2", quota=quota(tomorrow, daily_batch_limit=1)
    ) as guard:
        guard.reserve("llm", provider="openrouter", model="openrouter/free")
        with pytest.raises(budget_module.EvaluationBudgetExceeded, match="daily"):
            guard.reserve("llm", provider="openrouter", model="openrouter/free")
    saved = json.loads(path.read_text())
    assert [item["stage"] for item in saved["reservations"]] == ["B0", "B0"]
    assert len(saved["days"]) == 2


def test_corrupted_daily_counters_fail_closed(tmp_path):
    path = tmp_path / "ledger.json"
    with session(tmp_path) as guard:
        guard.reserve("llm", provider="openrouter", model="openrouter/free")
    saved = json.loads(path.read_text())
    saved["days"][datetime.now(UTC).date().isoformat()]["used"] = -20
    path.write_text(json.dumps(saved))
    with pytest.raises(budget_module.EvaluationBudgetUnavailable):
        with session(tmp_path):
            pytest.fail("corrupt counters accepted")


def test_changed_campaign_caps_are_not_a_new_campaign(tmp_path):
    path = tmp_path / "ledger.json"
    with session(tmp_path):
        pass
    saved = json.loads(path.read_text())
    saved["caps"] = {"overall": {"llm": 1000}}
    path.write_text(json.dumps(saved))
    with pytest.raises(budget_module.EvaluationBudgetUnavailable):
        with session(tmp_path):
            pytest.fail("changed ceilings accepted")


def test_job_resume_rejects_changed_contract(tmp_path):
    artifact = tmp_path / "record.json"
    artifact.write_text("{}")
    with session(tmp_path) as guard:
        guard.begin_job("case1:r1", contract_digest="a" * 64)
        guard.complete_job("case1:r1", artifact)
    with session(tmp_path) as guard:
        with pytest.raises(budget_module.EvaluationBudgetUnavailable, match="contract"):
            guard.begin_job("case1:r1", contract_digest="b" * 64)


def test_quota_freshness_expires_before_dispatch(tmp_path, monkeypatch):
    stale = quota(datetime.now(UTC) - timedelta(minutes=16))
    with pytest.raises(budget_module.EvaluationBudgetUnavailable):
        with budget_module.EvaluationBudget(
            tmp_path / "ledger.json", stage="B0", run_id="a", quota=stale
        ):
            pytest.fail("stale observation accepted")
    later = datetime.now(UTC) + timedelta(minutes=16)

    class Later(datetime):
        @classmethod
        def now(cls, tz=None):
            return later

    with session(tmp_path) as guard:
        monkeypatch.setattr(budget_module, "datetime", Later)
        with pytest.raises(budget_module.EvaluationBudgetUnavailable):
            guard.reserve("judge", provider="openrouter", model="openrouter/free")


def test_cli_completed_case_resume_never_dispatches_again(tmp_path, monkeypatch):
    from backend.configs.chat_limits import ChatLimitsSettings
    from backend.evaluation.runtime_evaluation import RuntimeRecord
    from backend.utils.chat_limits import ChatLimitService, ChatTurnContext
    from scripts import evaluate_runtime_workflow as cli

    quota_path = tmp_path / "quota.json"
    quota_path.write_text(json.dumps(quota()))
    ledger = tmp_path / "ledger.json"
    monkeypatch.setattr(cli, "campaign_ledger", lambda root, **kwargs: ledger)
    calls = []

    async def fake_run(scenario, *, repetition):
        calls.append(scenario.id)
        limits = ChatLimitService(
            settings=ChatLimitsSettings(_env_file=None, shared_quotas_enabled=False)
        )
        await limits.reserve_llm_attempt(
            ChatTurnContext(visitor_id="fixture", session_id="fixture"),
            "openrouter",
            "orchestrator",
        )
        return RuntimeRecord(
            scenario_id=scenario.id,
            scenario_version=scenario.version,
            run_id="fixture-runtime",
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
    for name in ("run1", "run2"):
        assert cli.main([*args, "--output-dir", str(tmp_path / name)]) == 0
    assert calls == ["out_of_scope_biological_cats"]
    saved = json.loads(ledger.read_text())
    assert len(saved["reservations"]) == 1
    manifest = json.loads((tmp_path / "run1" / "manifest.json").read_text())
    assert (
        manifest["configuration_snapshot"]["executor_configuration"][
            "evaluation_budget"
        ]["stage"]
        == "B0"
    )


def test_explicit_campaign_ledger_cannot_silently_create_new_budget(
    tmp_path, monkeypatch
):
    import subprocess

    from scripts import evaluate_runtime_workflow as cli

    checkout = tmp_path / "checkout"
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    monkeypatch.setattr(cli, "ROOT", checkout)
    args = cli.parse_args(
        [
            "live",
            "--case-id",
            "out_of_scope_biological_cats",
            "--budget-stage",
            "B0",
            "--provider-quota",
            str(tmp_path / "quota.json"),
            "--budget-ledger",
            str(tmp_path / "missing.json"),
        ]
    )
    with pytest.raises(ValueError, match="existing absolute"):
        cli._evaluation_session(args, tmp_path / "run")
    assert not (tmp_path / "missing.json").exists()
    assert not (checkout / "data/evaluation/portfolio_budget/ledger.json").exists()


def test_quota_metadata_rejects_secret_fields_without_persisting_them(tmp_path):
    path = tmp_path / "ledger.json"
    unsafe = quota(api_key="never-persist-this-key")
    guard = budget_module.EvaluationBudget(
        path, stage="B0", run_id="fixture", quota=unsafe
    )
    with pytest.raises(budget_module.EvaluationBudgetUnavailable):
        with guard:
            pytest.fail("secret-bearing metadata accepted")
    assert not path.exists()
    assert "never-persist-this-key" not in json.dumps(guard.provenance())


def test_tavily_evaluation_adapter_preserves_authenticated_sdk_session(monkeypatch):
    from types import SimpleNamespace

    from requests import HTTPError, Response
    from requests.adapters import HTTPAdapter
    from tavily import TavilyClient

    from backend.evaluation.runtime_live import _disable_tavily_transport_retries

    client = TavilyClient(
        api_key="offline-fixture",
        project_id="fixture-project",
        org_id="fixture-org",
        session_id="fixture-session",
    )
    session = client.session
    session.headers["X-Fixture-Context"] = "retained"
    session.proxies["https"] = "http://proxy.invalid"
    session.verify = "/fixture/ca.pem"
    session.trust_env = False
    session.cookies.set("fixture_cookie", "retained")
    adapter = HTTPAdapter(max_retries=3, pool_maxsize=17)
    session.mount("https://api.tavily.com/", adapter)
    sends = []

    def fail_once(active_adapter, request, **kwargs):
        sends.append(request)
        assert active_adapter is adapter
        assert active_adapter.max_retries.total == 0
        assert request.headers["Authorization"] == "Bearer offline-fixture"
        assert request.headers["Content-Type"] == "application/json"
        assert request.headers["X-Client-Source"] == "tavily-python"
        assert request.headers["X-Project-ID"] == "fixture-project"
        assert request.headers["X-Tavily-Orgid"] == "fixture-org"
        assert request.headers["X-Session-Id"] == "fixture-session"
        assert request.headers["X-Fixture-Context"] == "retained"
        assert "fixture_cookie=retained" in request.headers["Cookie"]
        assert kwargs["proxies"]["https"] == "http://proxy.invalid"
        assert kwargs["verify"] == "/fixture/ca.pem"
        response = Response()
        response.status_code = 503
        response._content = b'{"detail":{"error":"fixture unavailable"}}'
        response.request = request
        response.url = request.url
        return response

    monkeypatch.setattr(HTTPAdapter, "send", fail_once)
    wrapper = SimpleNamespace(_workflow=SimpleNamespace(search_engine=client))
    _disable_tavily_transport_retries(wrapper)
    assert client.session is session
    assert client.session.trust_env is False
    with pytest.raises(HTTPError):
        client.search(
            "offline intercepted fixture",
            search_depth="advanced",
            max_results=5,
            timeout=60,
        )
    assert len(sends) == 1
    client.close()


def test_tavily_transport_does_not_follow_redirects_or_add_retries(monkeypatch):
    from types import SimpleNamespace

    from requests import Response
    from requests.adapters import HTTPAdapter
    from tavily import TavilyClient

    from backend.evaluation.runtime_live import _disable_tavily_transport_retries

    sends = []

    def fake_send(adapter, request, **kwargs):
        sends.append(request)
        assert adapter.max_retries.total == 0
        response = Response()
        response.status_code = 307
        response.headers["Location"] = "https://other.invalid/search"
        response.request = request
        response.url = request.url
        return response

    monkeypatch.setattr(HTTPAdapter, "send", fake_send)
    client = TavilyClient(api_key="offline-redirect-fixture")
    wrapper = SimpleNamespace(_workflow=SimpleNamespace(search_engine=client))
    _disable_tavily_transport_retries(wrapper)
    response = wrapper._workflow.search_engine.session.post(
        "https://fixture.invalid/search", json={"query": "fixture"}
    )
    assert response.status_code == 307
    assert len(sends) == 1
    assert sends[0].headers["Authorization"] == "Bearer offline-redirect-fixture"
    assert sends[0].headers["Content-Type"] == "application/json"
    client.close()


def test_budget_stop_in_live_workflow_stays_an_operational_failure(
    tmp_path, monkeypatch
):
    import asyncio
    from types import SimpleNamespace

    from test_runtime_evaluation import make_scenario

    from backend.evaluation import runtime_live

    class Graph:
        async def astream_events(self, *args, **kwargs):
            raise budget_module.EvaluationBudgetExceeded("fixture ceiling")
            yield  # pragma: no cover

    wrapper = SimpleNamespace(
        _graph=Graph(),
        _ensure_initialized=lambda: None,
        _convert_messages=lambda messages: [],
    )
    monkeypatch.setattr(runtime_live, "get_workflow", lambda: wrapper)
    with pytest.raises(budget_module.EvaluationBudgetExceeded):
        asyncio.run(runtime_live.run_controlled_live_scenario(make_scenario()))
    assert (
        runtime_live._live_error_code(budget_module.EvaluationBudgetExceeded())
        == "evaluation_budget_exhausted"
    )


def test_ledger_alias_cannot_acquire_an_independent_lock(tmp_path):
    ledger = tmp_path / "ledger.json"
    with session(tmp_path):
        alias = tmp_path / "alias.json"
        alias.symlink_to(ledger)
        with pytest.raises(budget_module.EvaluationBudgetUnavailable, match="locked"):
            with budget_module.EvaluationBudget(
                alias, stage="B0", run_id="alias", quota=quota()
            ):
                pytest.fail("alias split the campaign lock")


def test_process_lock_and_restart_share_the_same_ledger(tmp_path):
    import subprocess
    import sys

    script = """
import json, sys
from pathlib import Path
from backend.evaluation.portfolio_budget import EvaluationBudget, EvaluationBudgetUnavailable
try:
    with EvaluationBudget(Path(sys.argv[1]), stage="B0", run_id="child", quota=json.loads(sys.argv[2])) as guard:
        guard.reserve("llm", provider="openrouter", model="openrouter/free")
except EvaluationBudgetUnavailable:
    raise SystemExit(3)
"""
    command = [
        sys.executable,
        "-c",
        script,
        str(tmp_path / "ledger.json"),
        json.dumps(quota()),
    ]
    with session(tmp_path):
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=10, check=False
        )
        assert result.returncode == 3, result.stderr
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=10, check=False
    )
    assert result.returncode == 0, result.stderr
    with session(tmp_path) as guard:
        guard.reserve("judge", provider="openrouter", model="openrouter/free")
    saved = json.loads((tmp_path / "ledger.json").read_text())
    assert [item["kind"] for item in saved["reservations"]] == ["llm", "judge"]


def test_provider_429_stops_the_evaluation_batch(tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from test_runtime_evaluation import make_scenario

    from backend.evaluation import runtime_live
    from backend.utils.chat_errors import AIServiceCapacity

    class QuotaFailure(RuntimeError):
        status_code = 429

    class Graph:
        async def astream_events(self, *args, **kwargs):
            try:
                raise QuotaFailure("safe fixture")
            except QuotaFailure as exc:
                raise AIServiceCapacity() from exc
            yield  # pragma: no cover

    wrapper = SimpleNamespace(
        _graph=Graph(),
        _ensure_initialized=lambda: None,
        _convert_messages=lambda messages: [],
    )
    monkeypatch.setattr(runtime_live, "get_workflow", lambda: wrapper)
    with pytest.raises(budget_module.EvaluationBudgetUnavailable, match="429"):
        asyncio.run(runtime_live.run_controlled_live_scenario(make_scenario()))


def test_same_account_observation_cannot_replenish_quota_for_another_model(tmp_path):
    path = tmp_path / "ledger.json"
    observation = quota(remaining_requests=21, demo_reserve=20)
    with budget_module.EvaluationBudget(
        path, stage="B0", run_id="runtime", quota=observation
    ) as guard:
        guard.reserve("llm", provider="openrouter", model="openrouter/free")
    judge_observation = observation | {"model": "different/judge-model"}
    with budget_module.EvaluationBudget(
        path, stage="B0", run_id="judge", quota=judge_observation
    ) as guard:
        with pytest.raises(budget_module.EvaluationBudgetExceeded, match="reserve"):
            guard.reserve("judge", provider="openrouter", model="different/judge-model")


def test_session_rejects_chat_limit_change_before_new_case(tmp_path, monkeypatch):
    from backend.utils.chat_limits import get_chat_limit_service
    from scripts import evaluate_runtime_workflow as cli

    get_chat_limit_service.cache_clear()
    monkeypatch.setattr(
        cli, "campaign_ledger", lambda root, **kwargs: tmp_path / "ledger.json"
    )
    metadata = tmp_path / "quota.json"
    metadata.write_text(json.dumps(quota()))
    args = cli.parse_args(
        [
            "live",
            "--case-id",
            "out_of_scope_biological_cats",
            "--budget-stage",
            "B0",
            "--provider-quota",
            str(metadata),
        ]
    )
    with cli._evaluation_session(args, tmp_path / "first"):
        pass
    monkeypatch.setenv("CHAT_LIMITS_MAX_LLM_ATTEMPTS_PER_TURN", "7")
    get_chat_limit_service.cache_clear()
    try:
        with pytest.raises(
            budget_module.EvaluationBudgetUnavailable, match="configuration"
        ):
            with cli._evaluation_session(args, tmp_path / "changed"):
                pytest.fail("changed runtime limits mixed into B0")
    finally:
        get_chat_limit_service.cache_clear()


def test_session_rejects_changed_prompt_directory_content(tmp_path, monkeypatch):
    from scripts import evaluate_runtime_workflow as cli

    monkeypatch.setattr(
        cli, "campaign_ledger", lambda root, **kwargs: tmp_path / "ledger.json"
    )
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    prompt = prompt_dir / "fixture.yaml"
    prompt.write_text("instruction: first\n")
    monkeypatch.setenv("PROMPTS_DIR", str(prompt_dir))
    metadata = tmp_path / "quota.json"
    metadata.write_text(json.dumps(quota()))
    args = cli.parse_args(
        [
            "live",
            "--case-id",
            "out_of_scope_biological_cats",
            "--budget-stage",
            "B0",
            "--provider-quota",
            str(metadata),
        ]
    )
    with cli._evaluation_session(args, tmp_path / "first"):
        pass
    prompt.write_text("instruction: changed\n")
    with pytest.raises(
        budget_module.EvaluationBudgetUnavailable, match="configuration"
    ):
        with cli._evaluation_session(args, tmp_path / "changed"):
            pytest.fail("changed external prompts mixed into B0")


@pytest.mark.parametrize("command", ["live", "judge"])
@pytest.mark.parametrize("interrupted_after_manifest", [False, True])
def test_resume_publishes_each_observation_once_in_real_reports(
    tmp_path, monkeypatch, command, interrupted_after_manifest
):
    from test_runtime_evaluation import make_scenario

    from backend.evaluation.evaluation_reporting import build_evaluation_reports
    from backend.evaluation.runtime_evaluation import RuntimeRecord, SemanticJudgeResult
    from backend.workflows.agents.factory import AgentsFactory
    from scripts import evaluate_runtime_workflow as cli

    scenarios = [make_scenario(id=name) for name in ("case_a", "case_b")]
    for scenario in scenarios:
        scenario.expectations.judge_dimensions = ["analyst_usefulness"]
    scenario_file = tmp_path / "scenarios.json"
    scenario_file.write_text(
        json.dumps([item.model_dump(mode="json") for item in scenarios])
    )
    records = [
        RuntimeRecord(
            scenario_id=item.id,
            scenario_version=1,
            run_id=item.id,
            source_mode="controlled_live",
            input=item.input,
            worker_sequence=["analyst"],
            final_response="fixture answer",
        )
        for item in scenarios
    ]
    source = tmp_path / "records.jsonl"
    source.write_text("\n".join(record.model_dump_json() for record in records) + "\n")
    metadata = tmp_path / "quota.json"
    metadata.write_text(json.dumps(quota()))
    monkeypatch.setattr(
        cli, "campaign_ledger", lambda root, **kwargs: tmp_path / "ledger.json"
    )
    monkeypatch.setattr(AgentsFactory, "get_llm_by_role", lambda settings: object())
    dispatched = []

    async def fake_live(scenario, *, repetition):
        dispatched.append(scenario.id)
        budget_module.current_evaluation_budget().reserve(
            "llm", provider="openrouter", model="openrouter/free"
        )
        return records[0 if scenario.id == "case_a" else 1].model_copy(deep=True)

    def fake_judge(scenario, record, dimension, **kwargs):
        dispatched.append(scenario.id)
        budget_module.current_evaluation_budget().reserve(
            "judge", provider="openrouter", model="openrouter/free"
        )
        return SemanticJudgeResult(
            scenario_id=scenario.id,
            scenario_version=1,
            run_id=record.run_id,
            repetition=1,
            dimension=dimension,
            judge_provider="openrouter",
            judge_model="openrouter/free",
            status="success",
            score=True,
            provider_call_attempted=True,
        )

    monkeypatch.setattr(cli, "run_controlled_live_scenario", fake_live)
    monkeypatch.setattr(cli, "evaluate_with_openevals_judge", fake_judge)
    evaluation_root = tmp_path / "evaluation"
    finish = budget_module.EvaluationBudget.finish_publication
    interruptions = []

    def interrupt_once(guard, identities):
        if interrupted_after_manifest and not interruptions:
            interruptions.append(True)
            raise OSError("simulated interruption after successful manifest")
        finish(guard, identities)

    monkeypatch.setattr(
        budget_module.EvaluationBudget, "finish_publication", interrupt_once
    )
    common = [
        command,
        "--confirm-provider-calls",
        "--budget-stage",
        "B0",
        "--provider-quota",
        str(metadata),
        "--evaluation-root",
        str(evaluation_root),
        "--scenarios",
        str(scenario_file),
    ]
    if command == "judge":
        common.extend(
            [
                "--records",
                str(source),
                "--dimension",
                "analyst_usefulness",
                "--judge-provider",
                "openrouter",
                "--judge-model",
                "openrouter/free",
            ]
        )

    for name, cases, expected in (
        ("first", ["case_a"], 1),
        ("full_resume", ["case_a"], 1),
        ("partial_resume", ["case_a", "case_b"], 2),
    ):
        args = [*common, "--output-dir", str(evaluation_root / "runs" / name)]
        for case in cases:
            args.extend(["--case-id", case])
        expected_status = 2 if interrupted_after_manifest and name == "first" else 0
        assert cli.main(args) == expected_status
        reports = build_evaluation_reports(evaluation_root)
        report = next(iter(reports.values()))
        if command == "live":
            assert report["coverage"]["live_sample_count"] == expected
        else:
            assert (
                report["coverage"]["semantic_dimension_coverage"]["analyst_usefulness"][
                    "result_count"
                ]
                == expected
            )
    assert dispatched == ["case_a", "case_b"]


def test_effective_binding_records_supported_local_index_override(
    tmp_path, monkeypatch
):
    from scripts import evaluate_runtime_workflow as cli

    monkeypatch.setattr(
        cli, "campaign_ledger", lambda root, **kwargs: tmp_path / "ledger.json"
    )
    local_index = {
        "storage_type": "local",
        "storage_path": str(tmp_path / "local-index"),
    }
    monkeypatch.setenv("INDEX_STORAGE", json.dumps(local_index))
    metadata = tmp_path / "quota.json"
    metadata.write_text(json.dumps(quota()))
    args = cli.parse_args(
        [
            "live",
            "--case-id",
            "out_of_scope_biological_cats",
            "--budget-stage",
            "B0",
            "--provider-quota",
            str(metadata),
        ]
    )
    with cli._evaluation_session(args, tmp_path / "first") as guard:
        actual = guard.provenance()["effective_configuration"][
            "application_configuration"
        ]["storage"]["index_storage"]
        assert actual == local_index
    monkeypatch.setenv(
        "INDEX_STORAGE",
        json.dumps(local_index | {"storage_path": str(tmp_path / "other-index")}),
    )
    with pytest.raises(
        budget_module.EvaluationBudgetUnavailable, match="configuration"
    ):
        with cli._evaluation_session(args, tmp_path / "changed"):
            pytest.fail("changed local index mixed into B0")


def test_missing_completed_publication_requires_reconciliation(tmp_path):
    artifact = tmp_path / "record.json"
    artifact.write_text("{}")
    manifest = tmp_path / "manifest.json"
    with session(tmp_path) as guard:
        guard.begin_job("case:r1")
        guard.complete_job("case:r1", artifact)
        guard.prepare_publication(["case:r1"], manifest)
        manifest.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "configuration_snapshot": {
                        "executor_configuration": {
                            "evaluation_budget": guard.provenance()
                        }
                    },
                }
            )
        )
        guard.finish_publication(["case:r1"])
    manifest.unlink()
    with session(tmp_path) as guard:
        with pytest.raises(budget_module.EvaluationBudgetUnavailable, match="manifest"):
            guard.is_published("case:r1")


def test_planner_daily_policy_survives_rollover_restart_and_mixed_attempts(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        budget_module,
        "DAILY_POLICY",
        {"llm_attempts_per_utc_day": 24, "demo_headroom_requests": 20},
    )
    import asyncio

    from backend.configs.chat_limits import ChatLimitsSettings
    from backend.utils.chat_limits import ChatLimitService, ChatTurnContext

    initial = datetime.now(UTC)

    class Clock(datetime):
        moment = initial

        @classmethod
        def now(cls, tz=None):
            return cls.moment

    monkeypatch.setattr(budget_module, "datetime", Clock)
    limits = ChatLimitService(
        settings=ChatLimitsSettings(_env_file=None, shared_quotas_enabled=False)
    )
    path = tmp_path / "ledger.json"
    for day in range(2):
        Clock.moment = initial + timedelta(days=day)
        for part in range(4):
            observation = quota(Clock.moment, daily_batch_limit=100, demo_reserve=0)
            with (
                budget_module.EvaluationBudget(
                    path,
                    stage="B0" if day == 0 else "final",
                    run_id=f"d{day}p{part}",
                    quota=observation,
                ) as guard,
                budget_module.bind_evaluation_budget(guard),
            ):
                # Each pair represents a runtime failure and its counted retry.
                for _ in range(2):
                    turn = ChatTurnContext(visitor_id="fixture", session_id="fixture")
                    asyncio.run(
                        limits.reserve_llm_attempt(turn, "openrouter", "analyst")
                    )
                    asyncio.run(
                        limits.reserve_llm_attempt(turn, "openrouter", "analyst")
                    )
                guard.reserve("judge", provider="openrouter", model="openrouter/free")
                guard.reserve("llm", provider="openrouter", model="openrouter/free")
        with budget_module.EvaluationBudget(
            path,
            stage="final",
            run_id="blocked",
            quota=quota(Clock.moment, daily_batch_limit=1000, demo_reserve=0),
        ) as guard:
            with pytest.raises(budget_module.EvaluationBudgetExceeded, match="daily"):
                guard.reserve("judge", provider="openrouter", model="openrouter/free")
    saved = json.loads(path.read_text())
    assert len(saved["reservations"]) == 48
    assert [day["used"] for day in saved["days"].values()] == [24, 24]


def test_planner_minimum_headroom_cannot_be_released_by_fresh_metadata(
    tmp_path, monkeypatch
):
    path = tmp_path / "ledger.json"
    with budget_module.EvaluationBudget(
        path,
        stage="B0",
        run_id="runtime",
        quota=quota(remaining_requests=21, demo_reserve=20),
    ) as guard:
        guard.reserve("llm", provider="openrouter", model="openrouter/free")
    next_day = datetime.now(UTC) + timedelta(days=1)

    class NextDay(datetime):
        @classmethod
        def now(cls, tz=None):
            return next_day

    monkeypatch.setattr(budget_module, "datetime", NextDay)
    with budget_module.EvaluationBudget(
        path,
        stage="final",
        run_id="judge",
        quota=quota(
            next_day, remaining_requests=20, demo_reserve=0, model="another/judge"
        ),
    ) as guard:
        with pytest.raises(budget_module.EvaluationBudgetExceeded, match="reserve"):
            guard.reserve("judge", provider="openrouter", model="another/judge")


def test_planner_alternate_existing_ledger_is_not_an_admission_binding(
    tmp_path, monkeypatch
):
    from scripts import evaluate_runtime_workflow as cli

    checkout = tmp_path / "source"
    (checkout / ".git").mkdir(parents=True)
    monkeypatch.setattr(cli, "ROOT", checkout)
    canonical = budget_module.campaign_ledger(checkout)
    alternate = tmp_path / "other" / "ledger.json"
    with budget_module.EvaluationBudget(
        alternate, stage="B0", run_id="empty", quota=quota()
    ):
        pass
    metadata = tmp_path / "quota.json"
    metadata.write_text(json.dumps(quota()))
    args = cli.parse_args(
        [
            "live",
            "--case-id",
            "case",
            "--budget-stage",
            "B0",
            "--provider-quota",
            str(metadata),
            "--budget-ledger",
            str(alternate),
        ]
    )
    with pytest.raises(budget_module.EvaluationBudgetUnavailable, match="canonical"):
        cli._evaluation_session(args, tmp_path / "run")
    assert not canonical.exists()


def test_planner_copied_or_hardlinked_ledger_cannot_fork_counters(tmp_path):
    import shutil

    canonical = tmp_path / "original" / "ledger.json"
    with budget_module.EvaluationBudget(
        canonical, stage="B0", run_id="runtime", quota=quota()
    ) as guard:
        guard.reserve("llm", provider="openrouter", model="openrouter/free")
    copy = tmp_path / "copied.json"
    shutil.copy2(canonical, copy)
    hardlink = tmp_path / "hardlink.json"
    hardlink.hardlink_to(canonical)
    for alternative in (copy, hardlink):
        with pytest.raises(budget_module.EvaluationBudgetUnavailable, match="binding"):
            with budget_module.EvaluationBudget(
                alternative, stage="B0", run_id="fork", quota=quota()
            ):
                pytest.fail("independent campaign lock accepted")


def test_planner_utc_midnight_requires_fresh_quota_and_preserves_campaign(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        budget_module,
        "DAILY_POLICY",
        {"llm_attempts_per_utc_day": 24, "demo_headroom_requests": 20},
    )
    before_midnight = datetime(2026, 10, 25, 23, 59, 30, tzinfo=UTC)

    class Clock(datetime):
        moment = before_midnight

        @classmethod
        def now(cls, tz=None):
            return cls.moment

    monkeypatch.setattr(budget_module, "datetime", Clock)
    with session(tmp_path) as guard:
        for _ in range(24):
            guard.reserve("llm", provider="openrouter", model="openrouter/free")
        Clock.moment = before_midnight + timedelta(seconds=31)
        with pytest.raises(budget_module.EvaluationBudgetUnavailable, match="expired"):
            guard.reserve("llm", provider="openrouter", model="openrouter/free")
    with session(tmp_path, stage="final") as guard:
        for _ in range(24):
            guard.reserve("llm", provider="openrouter", model="openrouter/free")
        with pytest.raises(budget_module.EvaluationBudgetExceeded, match="daily"):
            guard.reserve("llm", provider="openrouter", model="openrouter/free")
    saved = json.loads((tmp_path / "ledger.json").read_text())
    assert len(saved["reservations"]) == 48
    assert set(saved["days"]) == {"2026-10-25", "2026-10-26"}


@pytest.mark.parametrize("missing", ["ledger_binding", "daily_policy"])
def test_planner_legacy_state_is_rejected_without_migration_or_reset(tmp_path, missing):
    path = tmp_path / "ledger.json"
    with session(tmp_path) as guard:
        guard.reserve("llm", provider="openrouter", model="openrouter/free")
    state = json.loads(path.read_text())
    del state[missing]
    path.write_text(json.dumps(state))
    original = path.read_bytes()
    with pytest.raises(budget_module.EvaluationBudgetUnavailable):
        with session(tmp_path):
            pytest.fail("legacy state silently migrated")
    assert path.read_bytes() == original
    assert len(json.loads(path.read_text())["reservations"]) == 1


def test_planner_original_and_restored_cli_share_state_and_alias_lock(
    tmp_path, monkeypatch
):
    from scripts import evaluate_runtime_workflow as cli

    original = tmp_path / "original"
    restored = tmp_path / "restored"
    (original / ".git").mkdir(parents=True)
    (restored / ".git").mkdir(parents=True)
    canonical = budget_module.campaign_ledger(original)
    (restored / ".git" / "portfolio-restored-source").write_text(
        json.dumps(
            {
                "canonical_ledger": str(canonical),
                "snapshot_manifest_sha256": "fixture-immutable-manifest",
            }
        )
    )
    metadata = tmp_path / "quota.json"
    metadata.write_text(json.dumps(quota()))
    args = cli.parse_args(
        [
            "live",
            "--case-id",
            "case",
            "--budget-stage",
            "B0",
            "--provider-quota",
            str(metadata),
        ]
    )
    monkeypatch.setattr(cli, "ROOT", original)
    with cli._evaluation_session(args, tmp_path / "initial") as guard:
        guard.reserve("llm", provider="openrouter", model="openrouter/free")
        alias = tmp_path / "ledger-alias.json"
        alias.symlink_to(canonical)
        args.budget_ledger = alias
        monkeypatch.setattr(cli, "ROOT", restored)
        with pytest.raises(budget_module.EvaluationBudgetUnavailable, match="locked"):
            with cli._evaluation_session(args, tmp_path / "parallel"):
                pytest.fail("restored alias acquired independent lock")
    with cli._evaluation_session(args, tmp_path / "resume") as guard:
        guard.reserve("judge", provider="openrouter", model="openrouter/free")
    saved = json.loads(canonical.read_text())
    assert [item["kind"] for item in saved["reservations"]] == ["llm", "judge"]
    assert not (restored / "data/evaluation/portfolio_budget/ledger.json").exists()


def test_planner_stricter_quota_policy_remains_in_force(tmp_path):
    observation = quota(remaining_requests=32, demo_reserve=30, daily_batch_limit=2)
    with budget_module.EvaluationBudget(
        tmp_path / "ledger.json", stage="B0", run_id="strict", quota=observation
    ) as guard:
        guard.reserve("llm", provider="openrouter", model="openrouter/free")
        guard.reserve("judge", provider="openrouter", model="openrouter/free")
        with pytest.raises(budget_module.EvaluationBudgetExceeded):
            guard.reserve("llm", provider="openrouter", model="openrouter/free")
    saved = json.loads((tmp_path / "ledger.json").read_text())
    today = next(iter(saved["days"].values()))
    assert today["batch_limit"] == 2
    assert today["demo_reserve"] == 30


def amendment_fixture(
    tmp_path, monkeypatch, *, batch_limit=12, runner_help_amendment=False
):
    """Creates a spent old-policy campaign and two tiny scoped source trees."""
    from backend.evaluation import evaluation_reporting as reporting

    checkout = tmp_path / "campaign"
    previous = tmp_path / "accepted-source"
    (checkout / ".git").mkdir(parents=True)
    for root, content in ((checkout, "approved policy24"), (previous, "policy12")):
        target = root / "backend/evaluation/portfolio_budget.py"
        target.parent.mkdir(parents=True)
        target.write_text(content)
        (root / "backend/unchanged.py").write_text("accepted application")
        if runner_help_amendment:
            runner = root / "scripts/evaluate_runtime_workflow.py"
            runner.parent.mkdir()
            limit = 12 if root == previous else 24
            runner.write_text(
                'parser.add_argument("--provider-quota",\n'
                '        help="Quota JSON no older than 15 minutes on this UTC day; '
                f"supplied limits can only tighten the fixed {limit}-attempt daily "
                'ceiling and 20-request headroom.",\n)\n'
            )
    ledger = budget_module.campaign_ledger(checkout)
    old_policy = {"llm_attempts_per_utc_day": 12, "demo_headroom_requests": 20}
    old_caps = {
        "B0": {"llm": 32, "judge": 5, "web_search": 3, "web_attempt": 6},
        "final": {"llm": 60, "judge": 5, "web_search": 7, "web_attempt": 14},
        "overall": {"llm": 92, "judge": 10, "web_search": 10, "web_attempt": 20},
    }
    with monkeypatch.context() as patch:
        patch.setattr(reporting, "ROOT", previous)
        old_snapshot = reporting.build_evaluation_snapshot({})["fingerprint_input"]
    with monkeypatch.context() as patch:
        patch.setattr(reporting, "ROOT", checkout)
        snapshot = reporting.build_evaluation_snapshot({})["fingerprint_input"]
    old_configuration = {
        "evaluation_fingerprint": reporting._hash_json(old_snapshot),
        "application_configuration": snapshot["application_configuration"],
        "chat_limits": {"max_llm_attempts_per_turn": 6},
        "paths": {"prompts_dir": "/preserved/prompts"},
        "selected_prompt_tree_hash": "unchanged-prompts",
        "selected_storage_manifest_hashes": {"manifest.json": "unchanged-storage"},
    }
    with monkeypatch.context() as patch:
        patch.setattr(budget_module, "DAILY_POLICY", old_policy)
        patch.setattr(budget_module, "CAPS", old_caps)
        # Build actual durable reservations/jobs with the old guard, not invented counters.
        with budget_module.EvaluationBudget(
            ledger,
            stage="B0",
            run_id="old",
            quota=quota(daily_batch_limit=batch_limit),
            configuration=old_configuration,
        ) as guard:
            guard.begin_job("cats:r1", contract_digest="preserve-uncertain")
            for kind in ("llm", "llm", "judge", "web_search", "web_attempt"):
                guard.reserve(kind, provider="openrouter", model="openrouter/free")
    before = json.loads(ledger.read_text())
    new_configuration = old_configuration | {
        "evaluation_fingerprint": reporting._hash_json(snapshot)
    }
    arguments = {
        "checkout": checkout,
        "ledger": ledger,
        "previous_source_root": previous,
        "expected_ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
        "configuration": new_configuration,
    }
    return arguments, before


def test_explicit_policy_amendment_preserves_spend_uncertainty_binding_and_history(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        budget_module,
        "DAILY_POLICY",
        {"llm_attempts_per_utc_day": 24, "demo_headroom_requests": 20},
    )
    arguments, before = amendment_fixture(tmp_path, monkeypatch)
    path = arguments["ledger"]
    # Normal entry must not silently amend the old campaign.
    with pytest.raises(budget_module.EvaluationBudgetUnavailable):
        with budget_module.EvaluationBudget(
            path,
            stage="B0",
            run_id="blocked",
            quota=quota(),
            configuration=arguments["configuration"],
        ):
            pytest.fail("implicit policy migration")
    assert budget_module.amend_b0_completion_budget(**arguments) is True
    after = json.loads(path.read_text())
    for key in ("reservations", "jobs", "ledger_binding", "campaign", "version"):
        assert after[key] == before[key]
    today = datetime.now(UTC).date().isoformat()
    assert after["days"][today] == before["days"][today] | {"batch_limit": 24}
    assert after["stage_configurations"]["B0"] == arguments["configuration"]
    assert after["policy_amendments"][0]["previous_state"] == before
    amended = path.read_bytes()
    assert budget_module.amend_b0_completion_budget(**arguments) is False
    assert path.read_bytes() == amended
    with budget_module.EvaluationBudget(
        path,
        stage="B0",
        run_id="resumed",
        quota=quota(daily_batch_limit=1000),
        configuration=arguments["configuration"],
    ) as guard:
        with pytest.raises(
            budget_module.EvaluationBudgetUnavailable, match="uncertain"
        ):
            guard.begin_job("cats:r1", contract_digest="preserve-uncertain")
        for _ in range(21):
            guard.reserve("llm", provider="openrouter", model="openrouter/free")
        with pytest.raises(budget_module.EvaluationBudgetExceeded, match="daily"):
            guard.reserve("llm", provider="openrouter", model="openrouter/free")


@pytest.mark.parametrize(
    "field",
    [
        "chat_limits",
        "paths",
        "selected_prompt_tree_hash",
        "selected_storage_manifest_hashes",
        "application_configuration",
    ],
)
def test_policy_amendment_rejects_configuration_changes(tmp_path, monkeypatch, field):
    arguments, _ = amendment_fixture(tmp_path, monkeypatch)
    arguments["configuration"] = arguments["configuration"] | {field: "changed"}
    before = arguments["ledger"].read_bytes()
    with pytest.raises(
        budget_module.EvaluationBudgetUnavailable, match="configuration"
    ):
        budget_module.amend_b0_completion_budget(**arguments)
    assert arguments["ledger"].read_bytes() == before


def test_policy_amendment_preserves_independently_stricter_day(tmp_path, monkeypatch):
    monkeypatch.setattr(
        budget_module,
        "DAILY_POLICY",
        {"llm_attempts_per_utc_day": 24, "demo_headroom_requests": 20},
    )
    arguments, before = amendment_fixture(tmp_path, monkeypatch, batch_limit=4)
    budget_module.amend_b0_completion_budget(**arguments)
    assert json.loads(arguments["ledger"].read_text())["days"] == before["days"]
    with budget_module.EvaluationBudget(
        arguments["ledger"],
        stage="B0",
        run_id="strict",
        quota=quota(),
        configuration=arguments["configuration"],
    ) as guard:
        guard.reserve("llm", provider="openrouter", model="openrouter/free")
        with pytest.raises(budget_module.EvaluationBudgetExceeded, match="daily"):
            guard.reserve("llm", provider="openrouter", model="openrouter/free")


@pytest.mark.parametrize(
    "invalid",
    ["sha", "policy", "caps", "copy", "source", "fingerprint", "prior_fingerprint"],
)
def test_policy_amendment_rejects_unproven_or_other_state(
    tmp_path, monkeypatch, invalid
):
    import hashlib
    import shutil

    arguments, state = amendment_fixture(tmp_path, monkeypatch)
    if invalid == "sha":
        arguments["expected_ledger_sha256"] = "0" * 64
    elif invalid in {"policy", "caps", "prior_fingerprint"}:
        if invalid == "policy":
            state["daily_policy"]["llm_attempts_per_utc_day"] = 13
        elif invalid == "caps":
            state["caps"]["B0"]["llm"] = 34
        else:
            state["stage_configurations"]["B0"]["evaluation_fingerprint"] = "0" * 24
        arguments["ledger"].write_text(json.dumps(state))
        arguments["expected_ledger_sha256"] = hashlib.sha256(
            arguments["ledger"].read_bytes()
        ).hexdigest()
    elif invalid == "copy":
        copy = tmp_path / "copied.json"
        shutil.copyfile(arguments["ledger"], copy)
        arguments["ledger"] = copy
    elif invalid == "source":
        (arguments["checkout"] / "backend/unchanged.py").write_text(
            "unapproved application"
        )
    else:
        arguments["configuration"]["evaluation_fingerprint"] = "0" * 24
    before = arguments["ledger"].read_bytes()
    with pytest.raises(budget_module.EvaluationBudgetUnavailable):
        budget_module.amend_b0_completion_budget(**arguments)
    assert arguments["ledger"].read_bytes() == before


def test_amendment_cli_requires_explicit_confirmation_and_resumes_once(
    tmp_path, monkeypatch
):
    from scripts import amend_portfolio_budget as cli

    arguments, _ = amendment_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "ROOT", arguments["checkout"])
    config = tmp_path / "configuration.json"
    config.write_text(json.dumps(arguments["configuration"]))
    command = [
        "--budget-ledger",
        str(arguments["ledger"]),
        "--expected-ledger-sha256",
        arguments["expected_ledger_sha256"],
        "--previous-source-root",
        str(arguments["previous_source_root"]),
        "--configuration",
        str(config),
    ]
    before = arguments["ledger"].read_bytes()
    assert cli.main(command) == 2
    assert arguments["ledger"].read_bytes() == before
    assert cli.main([*command, "--confirm-b0-completion-amendment"]) == 0
    amended = arguments["ledger"].read_bytes()
    assert cli.main([*command, "--confirm-b0-completion-amendment"]) == 0
    assert arguments["ledger"].read_bytes() == amended


def test_policy_amendment_preserves_21_spent_across_days_and_restored_roots(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        budget_module,
        "DAILY_POLICY",
        {"llm_attempts_per_utc_day": 24, "demo_headroom_requests": 20},
    )
    import hashlib

    class Clock(datetime):
        moment = datetime(2026, 10, 25, 12, tzinfo=UTC)

        @classmethod
        def now(cls, tz=None):
            return cls.moment

    monkeypatch.setattr(budget_module, "datetime", Clock)
    arguments, initial = amendment_fixture(tmp_path, monkeypatch)
    old_configuration = initial["stage_configurations"]["B0"]
    with monkeypatch.context() as patch:
        patch.setattr(budget_module, "DAILY_POLICY", initial["daily_policy"])
        patch.setattr(budget_module, "CAPS", initial["caps"])
        with budget_module.EvaluationBudget(
            arguments["ledger"],
            stage="B0",
            run_id="previous-day",
            quota=quota(),
            configuration=old_configuration,
        ) as guard:
            for _ in range(8):
                guard.reserve("llm", provider="openrouter", model="openrouter/free")
        Clock.moment += timedelta(days=1)
        with budget_module.EvaluationBudget(
            arguments["ledger"],
            stage="B0",
            run_id="active-day",
            quota=quota(),
            configuration=old_configuration,
        ) as guard:
            for kind in ["llm"] * 8 + ["judge"] * 2:
                guard.reserve(kind, provider="openrouter", model="openrouter/free")
    before = json.loads(arguments["ledger"].read_text())
    arguments["expected_ledger_sha256"] = hashlib.sha256(
        arguments["ledger"].read_bytes()
    ).hexdigest()
    restored = tmp_path / "restored"
    (restored / ".git").mkdir(parents=True)
    (restored / ".git/portfolio-restored-source").write_text(
        json.dumps({"canonical_ledger": str(arguments["ledger"])})
    )
    import shutil

    shutil.copytree(arguments["checkout"] / "backend", restored / "backend")
    arguments["checkout"] = restored
    assert budget_module.amend_b0_completion_budget(**arguments)
    after = json.loads(arguments["ledger"].read_text())
    assert after["days"]["2026-10-25"] == before["days"]["2026-10-25"]
    assert after["days"]["2026-10-26"] == before["days"]["2026-10-26"] | {
        "batch_limit": 24
    }
    assert after["reservations"] == before["reservations"]
    assert sum(item["kind"] == "llm" for item in after["reservations"]) == 18
    assert sum(item["kind"] == "judge" for item in after["reservations"]) == 3
    assert after["jobs"] == before["jobs"]
    assert not (restored / budget_module.DEFAULT_BUDGET_LEDGER).exists()
    with budget_module.EvaluationBudget(
        arguments["ledger"],
        stage="B0",
        run_id="new",
        quota=quota(),
        configuration=arguments["configuration"],
    ) as guard:
        for _ in range(12):
            guard.reserve("llm", provider="openrouter", model="openrouter/free")
        with pytest.raises(budget_module.EvaluationBudgetExceeded, match="B0 llm"):
            guard.reserve("llm", provider="openrouter", model="openrouter/free")
    spent = arguments["ledger"].read_bytes()
    assert budget_module.amend_b0_completion_budget(**arguments) is False
    assert arguments["ledger"].read_bytes() == spent


def test_policy_amendment_fresh_stricter_metadata_remains_binding(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        budget_module,
        "DAILY_POLICY",
        {"llm_attempts_per_utc_day": 24, "demo_headroom_requests": 20},
    )
    arguments, _ = amendment_fixture(tmp_path, monkeypatch)
    budget_module.amend_b0_completion_budget(**arguments)
    with budget_module.EvaluationBudget(
        arguments["ledger"],
        stage="B0",
        run_id="strict",
        quota=quota(daily_batch_limit=4),
        configuration=arguments["configuration"],
    ) as guard:
        guard.reserve("llm", provider="openrouter", model="openrouter/free")
    with budget_module.EvaluationBudget(
        arguments["ledger"],
        stage="B0",
        run_id="looser",
        quota=quota(daily_batch_limit=1000),
        configuration=arguments["configuration"],
    ) as guard:
        with pytest.raises(budget_module.EvaluationBudgetExceeded, match="daily"):
            guard.reserve("llm", provider="openrouter", model="openrouter/free")


@pytest.mark.parametrize("malformed", [[], {"policy_amendments": [None]}])
def test_policy_amendment_rejects_malformed_ledger_without_writes(
    tmp_path, monkeypatch, malformed
):
    arguments, state = amendment_fixture(tmp_path, monkeypatch)
    broken = state | malformed if isinstance(malformed, dict) else malformed
    arguments["ledger"].write_text(json.dumps(broken))
    arguments["expected_ledger_sha256"] = hashlib.sha256(
        arguments["ledger"].read_bytes()
    ).hexdigest()
    before = arguments["ledger"].read_bytes()
    with pytest.raises(budget_module.EvaluationBudgetUnavailable):
        budget_module.amend_b0_completion_budget(**arguments)
    assert arguments["ledger"].read_bytes() == before


@pytest.mark.parametrize("command", ["live", "judge"])
def test_runtime_cli_quota_help_uses_amended_daily_ceiling(capsys, command):
    from scripts import evaluate_runtime_workflow as cli

    with pytest.raises(SystemExit) as exit_info:
        cli.parse_args([command, "--help"])
    assert exit_info.value.code == 0
    help_output = " ".join(capsys.readouterr().out.split())
    assert "fixed 50-attempt daily ceiling and 20-request headroom" in help_output


def test_policy_amendment_accepts_exact_runner_help_change(tmp_path, monkeypatch):
    arguments, _ = amendment_fixture(tmp_path, monkeypatch, runner_help_amendment=True)
    assert budget_module.amend_b0_completion_budget(**arguments) is True
    proof = json.loads(arguments["ledger"].read_text())["policy_amendments"][0][
        "source_proof"
    ]
    assert set(proof["changed_files"]) == {
        "backend/evaluation/portfolio_budget.py",
        "scripts/evaluate_runtime_workflow.py",
    }


@pytest.mark.parametrize(
    "mutation", ["behavior", "different_help", "duplicate_old_help"]
)
def test_policy_amendment_rejects_other_runner_changes(tmp_path, monkeypatch, mutation):
    arguments, _ = amendment_fixture(tmp_path, monkeypatch, runner_help_amendment=True)
    runner = arguments["checkout"] / "scripts/evaluate_runtime_workflow.py"
    if mutation == "behavior":
        runner.write_text(runner.read_text() + "dispatch_provider_call()\n")
    elif mutation == "different_help":
        runner.write_text(runner.read_text().replace("20-request", "0-request"))
    else:
        previous_runner = (
            arguments["previous_source_root"] / "scripts/evaluate_runtime_workflow.py"
        )
        previous_runner.write_text(previous_runner.read_text() * 2)
        runner.write_text(runner.read_text() * 2)
    before = arguments["ledger"].read_bytes()
    with pytest.raises(
        budget_module.EvaluationBudgetUnavailable, match="source change"
    ):
        budget_module.amend_b0_completion_budget(**arguments)
    assert arguments["ledger"].read_bytes() == before


def test_pilot_retry_claim_and_identity_survive_competing_runner_and_next_day(
    tmp_path, campaign_clock
):
    ledger_path = tmp_path / "ledger.json"
    with session(tmp_path, stage="final") as first:
        first.fixed_judge_pilot = True
        first.begin_job("judge:semantic-v2:first")
        first.reserve("judge", provider="openrouter", model="openrouter/free")
        first.match_pilot_identity("served-v1", "Phala")
        assert first.claim_pilot_retry({"error_type": "ReadTimeout"})
        with pytest.raises(budget_module.EvaluationBudgetUnavailable, match="locked"):
            with session(tmp_path, stage="final"):
                pytest.fail("a competing runner obtained the retry lock")

    campaign_clock.next_day()
    with session(tmp_path, stage="final") as restarted:
        restarted.fixed_judge_pilot = True
        with pytest.raises(
            budget_module.EvaluationBudgetUnavailable, match="uncertain"
        ):
            restarted.begin_job("judge:semantic-v2:first")
        restarted.begin_job("judge:semantic-v2:second")
        assert not restarted.claim_pilot_retry({"error_type": "ConnectError"})
        restarted.match_pilot_identity("served-v1", "Phala")
        with pytest.raises(budget_module.EvaluationBudgetUnavailable, match="identity"):
            restarted.match_pilot_identity("served-v2", "Phala")
    saved = json.loads(ledger_path.read_text())
    assert saved["pilot_retry"]["failed_attempt"]["error_type"] == "ReadTimeout"
    assert len(saved["reservations"]) == 1
    assert len(saved["days"]) == 2


def test_daily_policy_amendment_rejects_unproven_current_day_origin(
    tmp_path, monkeypatch
):
    arguments, state = amendment_fixture(tmp_path, monkeypatch)
    today = datetime.now(UTC)
    previous_day = today - timedelta(days=1)
    older_day = today - timedelta(days=2)
    state["daily_policy"] = {
        "llm_attempts_per_utc_day": 24,
        "demo_headroom_requests": 20,
    }
    state["caps"] = budget_module.CAPS
    state["reservations"] = [
        {
            "kind": "llm",
            "stage": "B0" if i < 33 else "final",
            "run_id": f"run-{i}",
            "job_id": "",
            "request_id": f"request-{i}",
            "reserved_at": (
                older_day if i < 24 else previous_day if i < 33 else today
            ).isoformat(),
        }
        for i in range(53)
    ]
    state["days"] = {
        older_day.date().isoformat(): {
            "used": 24,
            "batch_limit": 24,
            "demo_reserve": 20,
            "observations": {},
        },
        previous_day.date().isoformat(): {
            "used": 9,
            "batch_limit": 24,
            "demo_reserve": 20,
            "observations": {},
        },
        today.date().isoformat(): {
            "used": 20,
            "batch_limit": 24,
            "demo_reserve": 20,
            "observations": {"captured": 0},
        },
    }
    state["jobs"] = {
        f"final:live:case-{i}:v1:r1": {
            "status": "completed",
            "artifact": f"output-{i}",
            "sha256": "a" * 64,
            "contract_digest": f"contract-{i}",
        }
        for i in range(7)
    }
    state["stage_configurations"]["final"] = dict(state["stage_configurations"]["B0"])
    state["policy_amendments"] = [{"amendment": "historical", "previous_state": {}}]
    path = arguments["ledger"]
    path.write_text(json.dumps(state, sort_keys=True))
    arguments["expected_ledger_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    original = path.read_bytes()
    with pytest.raises(budget_module.EvaluationBudgetUnavailable, match="provenance"):
        budget_module.amend_daily_policy_budget(**arguments, quota_provenance=None)
    assert path.read_bytes() == original


def test_final_runtime_admission_uses_locked_observation_baseline(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        budget_module,
        "DAILY_POLICY",
        {"llm_attempts_per_utc_day": 50, "demo_headroom_requests": 20},
    )
    ledger = tmp_path / "ledger.json"
    first = quota(remaining_requests=100, daily_batch_limit=50)
    with budget_module.EvaluationBudget(
        ledger, stage="final", run_id="initial", quota=first
    ) as guard:
        for _ in range(20):
            guard.reserve("llm", provider="openrouter", model="openrouter/free")
    fresh = quota(remaining_requests=31, daily_batch_limit=50)
    with budget_module.EvaluationBudget(
        ledger, stage="final", run_id="next", quota=fresh
    ) as guard:
        guard.require_final_runtime_admission()
        for _ in range(6):
            guard.reserve("llm", provider="openrouter", model="openrouter/free")
        with pytest.raises(budget_module.EvaluationBudgetExceeded):
            guard.require_final_runtime_admission()


def test_final_runtime_admission_rejects_unlocked_and_current_day_exhaustion(tmp_path):
    guard = budget_module.EvaluationBudget(
        tmp_path / "ledger.json",
        stage="final",
        run_id="next",
        quota=quota(daily_batch_limit=24),
    )
    with pytest.raises(budget_module.EvaluationBudgetUnavailable, match="locked"):
        guard.require_final_runtime_admission()
    with guard:
        for _ in range(20):
            guard.reserve("llm", provider="openrouter", model="openrouter/free")
        with pytest.raises(budget_module.EvaluationBudgetExceeded, match="six-attempt"):
            guard.require_final_runtime_admission()


def test_live_cli_admission_blocks_before_dispatch_inside_session(
    tmp_path, monkeypatch
):
    import asyncio
    from types import SimpleNamespace

    from scripts import evaluate_runtime_workflow as cli

    ledger = tmp_path / "ledger.json"
    with budget_module.EvaluationBudget(
        ledger, stage="final", run_id="initial", quota=quota(daily_batch_limit=24)
    ) as guard:
        for _ in range(20):
            guard.reserve("llm", provider="openrouter", model="openrouter/free")
    args = cli.parse_args(
        [
            "live",
            "--case-id",
            "local_transformer_answer_success",
            "--repetition",
            "1",
            "--confirm-provider-calls",
            "--budget-stage",
            "final",
            "--provider-quota",
            str(tmp_path / "quota.json"),
            "--output-dir",
            str(tmp_path / "run"),
            "--require-final-runtime-admission",
        ]
    )
    monkeypatch.setattr(
        cli,
        "KnowledgeGraphSearchSettings",
        lambda: SimpleNamespace(
            analyst_retrieval_mode="legacy_vector_context",
            visualizer_retrieval_mode="optimized",
        ),
    )
    monkeypatch.setattr(
        cli,
        "_evaluation_session",
        lambda args, output: budget_module.EvaluationBudget(
            ledger, stage="final", run_id="next", quota=quota(daily_batch_limit=24)
        ),
    )
    with pytest.raises(budget_module.EvaluationBudgetExceeded, match="six-attempt"):
        asyncio.run(cli.run_live(args))
    assert len(json.loads(ledger.read_text())["reservations"]) == 20


def test_final_runtime_admission_refuses_expired_quota_while_lock_held(
    tmp_path, campaign_clock
):
    guard = budget_module.EvaluationBudget(
        tmp_path / "ledger.json", stage="final", run_id="next", quota=quota()
    )
    with guard:
        campaign_clock.moment += timedelta(minutes=16)
        with pytest.raises(budget_module.EvaluationBudgetUnavailable, match="expired"):
            guard.require_final_runtime_admission()


def test_final_runtime_admission_holds_lock_against_competing_session(tmp_path):
    path = tmp_path / "ledger.json"
    with budget_module.EvaluationBudget(
        path, stage="final", run_id="first", quota=quota()
    ) as guard:
        guard.require_final_runtime_admission()
        with pytest.raises(budget_module.EvaluationBudgetUnavailable, match="locked"):
            with budget_module.EvaluationBudget(
                path, stage="final", run_id="second", quota=quota()
            ):
                pytest.fail("competing session entered")


def test_new_campaign_daily_policy_allows_25_combined_attempts(tmp_path):
    with budget_module.EvaluationBudget(
        tmp_path / "ledger.json",
        stage="B0",
        run_id="new",
        quota=quota(daily_batch_limit=50),
    ) as guard:
        for _ in range(24):
            guard.reserve("llm", provider="openrouter", model="openrouter/free")
        guard.reserve("judge", provider="openrouter", model="openrouter/free")


def daily_amendment_fixture(tmp_path, monkeypatch):
    import difflib

    arguments, state = amendment_fixture(tmp_path, monkeypatch)
    today = datetime.now(UTC)
    state["daily_policy"] = {
        "llm_attempts_per_utc_day": 24,
        "demo_headroom_requests": 20,
    }
    state["caps"] = budget_module.CAPS
    state["reservations"] = [
        {
            "kind": "llm",
            "stage": "B0" if i < 33 else "final",
            "run_id": f"run-{i}",
            "job_id": "",
            "request_id": f"request-{i}",
            "reserved_at": (
                today - timedelta(days=2 if i < 24 else 1 if i < 33 else 0)
            ).isoformat(),
        }
        for i in range(53)
    ]
    state["days"] = {
        (today - timedelta(days=offset)).date().isoformat(): {
            "used": used,
            "batch_limit": 24,
            "demo_reserve": 20,
            "observations": {},
        }
        for offset, used in ((2, 24), (1, 9), (0, 20))
    }
    state["jobs"] = {
        f"final:live:case-{i}:v1:r1": {
            "status": "completed",
            "artifact": f"output-{i}",
            "sha256": "a" * 64,
            "contract_digest": f"contract-{i}",
        }
        for i in range(7)
    }
    state["stage_configurations"]["final"] = dict(state["stage_configurations"]["B0"])
    state["policy_amendments"] = [{"amendment": "historical", "previous_state": {}}]
    path = arguments["ledger"]
    path.write_text(json.dumps(state, sort_keys=True))
    arguments["expected_ledger_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    previous, current = arguments["previous_source_root"], arguments["checkout"]
    inventories = []
    for root in (previous, current):
        files = {}
        for file in (root / "backend").rglob("*.py"):
            data = file.read_bytes()
            files[file.relative_to(root).as_posix()] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
        inventories.append(files)
    for index, name in enumerate(("old", "new")):
        inventory = tmp_path / f"{name}-inventory.json"
        inventory.write_text(json.dumps(inventories[index], sort_keys=True))
        arguments[f"{'previous' if index == 0 else 'current'}_inventory"] = inventory
        arguments[
            f"expected_{'previous' if index == 0 else 'current'}_inventory_sha256"
        ] = hashlib.sha256(inventory.read_bytes()).hexdigest()
    name = "backend/evaluation/portfolio_budget.py"
    patch = "".join(
        difflib.unified_diff(
            (previous / name).read_text().splitlines(keepends=True),
            (current / name).read_text().splitlines(keepends=True),
            fromfile="a/" + name,
            tofile="b/" + name,
        )
    ).encode()
    patch_path = tmp_path / "approved.patch"
    patch_path.write_bytes(patch)
    arguments.update(
        approved_patch=patch_path,
        expected_patch_sha256=hashlib.sha256(patch).hexdigest(),
        preserve_current_day=True,
    )
    return arguments, state


def test_daily_amendment_preserves_spent_history_and_is_idempotent(
    tmp_path, monkeypatch
):
    arguments, state = daily_amendment_fixture(tmp_path, monkeypatch)
    path = arguments["ledger"]
    assert budget_module.amend_daily_policy_budget(**arguments)
    after = json.loads(path.read_text())
    assert after["days"] == state["days"]
    assert after["reservations"] == state["reservations"]
    assert after["jobs"] == state["jobs"]
    assert after["stage_configurations"]["B0"] == state["stage_configurations"]["B0"]
    assert after["policy_amendments"][:-1] == state["policy_amendments"]
    amended = path.read_bytes()
    assert not budget_module.amend_daily_policy_budget(**arguments)
    assert path.read_bytes() == amended


def test_daily_amendment_cli_requires_preserving_ambiguous_day(
    tmp_path, monkeypatch, capsys
):
    from scripts import amend_portfolio_budget as cli

    arguments, _ = amendment_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "ROOT", arguments["checkout"])
    configuration = tmp_path / "configuration.json"
    configuration.write_text(json.dumps(arguments["configuration"]))
    before = arguments["ledger"].read_bytes()
    result = cli.main(
        [
            "--budget-ledger",
            str(arguments["ledger"]),
            "--expected-ledger-sha256",
            arguments["expected_ledger_sha256"],
            "--previous-source-root",
            str(arguments["previous_source_root"]),
            "--configuration",
            str(configuration),
            "--confirm-daily-policy-amendment",
        ]
    )
    assert result == 2
    assert "preserve" in capsys.readouterr().err
    assert arguments["ledger"].read_bytes() == before


@pytest.mark.parametrize("mutation", ["hash", "config", "patch", "source", "lock"])
def test_daily_amendment_rejects_changed_proof_or_competing_lock(
    tmp_path, monkeypatch, mutation
):
    import fcntl

    arguments, _ = daily_amendment_fixture(tmp_path, monkeypatch)
    path = arguments["ledger"]
    original = path.read_bytes()
    lock = None
    if mutation == "hash":
        arguments["expected_ledger_sha256"] = "0" * 64
    elif mutation == "config":
        arguments["configuration"] = arguments["configuration"] | {
            "paths": {"prompts_dir": "/changed"}
        }
    elif mutation == "patch":
        arguments["approved_patch"].write_text("unreviewed")
    elif mutation == "source":
        (arguments["checkout"] / "backend/unchanged.py").write_text("unreviewed")
    else:
        lock = path.with_suffix(".lock").open("a+")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(budget_module.EvaluationBudgetUnavailable):
            budget_module.amend_daily_policy_budget(**arguments)
    finally:
        if lock is not None:
            lock.close()
    assert path.read_bytes() == original


def test_daily_amendment_reapplication_rejects_changed_audit(tmp_path, monkeypatch):
    arguments, _ = daily_amendment_fixture(tmp_path, monkeypatch)
    budget_module.amend_daily_policy_budget(**arguments)
    path = arguments["ledger"]
    changed = json.loads(path.read_text())
    changed["policy_amendments"][-1]["approval"] = "unreviewed"
    path.write_text(json.dumps(changed))
    before = path.read_bytes()
    with pytest.raises(
        budget_module.EvaluationBudgetUnavailable, match="state or proof"
    ):
        budget_module.amend_daily_policy_budget(**arguments)
    assert path.read_bytes() == before


def test_live_cli_writes_pre_dispatch_receipt_when_admission_fails(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from scripts import evaluate_runtime_workflow as cli

    ledger = tmp_path / "ledger.json"
    with budget_module.EvaluationBudget(
        ledger, stage="final", run_id="spent", quota=quota(daily_batch_limit=24)
    ) as guard:
        for _ in range(20):
            guard.reserve("llm", provider="openrouter", model="openrouter/free")
    monkeypatch.setattr(
        cli,
        "KnowledgeGraphSearchSettings",
        lambda: SimpleNamespace(
            analyst_retrieval_mode="legacy_vector_context",
            visualizer_retrieval_mode="optimized",
        ),
    )
    monkeypatch.setattr(
        cli,
        "_evaluation_session",
        lambda args, output: budget_module.EvaluationBudget(
            ledger, stage="final", run_id="next", quota=quota(daily_batch_limit=24)
        ),
    )
    monkeypatch.setattr(cli, "register_failed_run", lambda *args, **kwargs: None)
    receipt = tmp_path / "receipt.json"
    status = cli.main(
        [
            "live",
            "--case-id",
            "local_transformer_answer_success",
            "--repetition",
            "1",
            "--confirm-provider-calls",
            "--budget-stage",
            "final",
            "--provider-quota",
            str(tmp_path / "quota.json"),
            "--output-dir",
            str(tmp_path / "run"),
            "--require-final-runtime-admission",
            "--pre-dispatch-receipt",
            str(receipt),
        ]
    )
    assert status == 2
    assert json.loads(receipt.read_text())["dispatch_started"] is False
    assert len(json.loads(ledger.read_text())["reservations"]) == 20


def test_forward_daily_50_counts_runtime_and_judges_across_rollover(
    tmp_path, campaign_clock
):
    path = tmp_path / "ledger.json"
    with budget_module.EvaluationBudget(
        path, stage="final", run_id="day-one", quota=quota(daily_batch_limit=100)
    ) as guard:
        for _ in range(45):
            guard.reserve("llm", provider="openrouter", model="openrouter/free")
        for _ in range(5):
            guard.reserve("judge", provider="openrouter", model="openrouter/free")
        with pytest.raises(budget_module.EvaluationBudgetExceeded, match="daily"):
            guard.reserve("llm", provider="openrouter", model="openrouter/free")
    campaign_clock.next_day()
    with budget_module.EvaluationBudget(
        path, stage="final", run_id="day-two", quota=quota(daily_batch_limit=100)
    ) as guard:
        guard.reserve("llm", provider="openrouter", model="openrouter/free")
    state = json.loads(path.read_text())
    assert sorted(day["used"] for day in state["days"].values()) == [1, 50]
    assert len(state["reservations"]) == 51


def captured_day_fixture(tmp_path, monkeypatch):
    arguments, before = daily_amendment_fixture(tmp_path, monkeypatch)
    path = arguments["ledger"]
    today = datetime.now(UTC)
    observation = quota(
        today, remaining_requests=50, daily_batch_limit=24, demo_reserve=25
    )
    identity = hashlib.sha256(
        json.dumps(
            {
                key: observation[key]
                for key in ("observed_at", "provider", "remaining_requests")
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    before["days"][today.date().isoformat()]["observations"] = {identity: 0}
    before["days"][today.date().isoformat()]["demo_reserve"] = 25
    path.write_text(json.dumps(before, sort_keys=True))
    arguments["expected_ledger_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    quota_bytes = json.dumps(observation, sort_keys=True).encode()
    quota_path = tmp_path / "captured-quota.json"
    quota_path.write_text(json.dumps(observation, sort_keys=True))
    provider_path = tmp_path / "captured-provider.json"
    provider_path.write_text(
        json.dumps(
            {
                "openrouter": {
                    "status": "success",
                    "http_status": 200,
                    "method": "GET",
                    "time": today.isoformat(),
                    "quota": {
                        "free_model_daily_requests": {"limit": 50, "remaining": 50}
                    },
                }
            }
        )
    )
    evidence = {
        "observation_id": identity,
        "quota_sha256": hashlib.sha256(quota_bytes).hexdigest(),
        "daily_limit_origin": "internal_policy",
        "independent_daily_limit": 35,
        "captured_at": today.isoformat(),
        "quota_capture_path": str(quota_path),
        "quota_capture_sha256": hashlib.sha256(quota_path.read_bytes()).hexdigest(),
        "provider_capture_path": str(provider_path),
        "provider_capture_sha256": hashlib.sha256(
            provider_path.read_bytes()
        ).hexdigest(),
        "no_other_independent_limit": True,
    }
    evidence_path = tmp_path / "captured-provenance.json"
    evidence_path.write_text(json.dumps(evidence, sort_keys=True))
    proof = {
        "day": today.date().isoformat(),
        "observations": [
            {
                "quota": observation,
                "evidence_path": str(evidence_path),
                "evidence_sha256": hashlib.sha256(
                    evidence_path.read_bytes()
                ).hexdigest(),
            }
        ],
    }
    proof_path = tmp_path / "reviewed-day-proof.json"
    proof_path.write_text(json.dumps(proof, sort_keys=True))
    arguments.update(
        preserve_current_day=False,
        quota_provenance=proof_path,
        expected_provenance_sha256=hashlib.sha256(proof_path.read_bytes()).hexdigest(),
    )
    return arguments, before


def test_daily_amendment_expands_only_with_complete_captured_provenance(
    tmp_path, monkeypatch
):
    arguments, before = captured_day_fixture(tmp_path, monkeypatch)
    path = arguments["ledger"]
    today = datetime.now(UTC)
    assert budget_module.amend_daily_policy_budget(**arguments)
    after = json.loads(path.read_text())
    assert after["days"][today.date().isoformat()] == before["days"][
        today.date().isoformat()
    ] | {"batch_limit": 35}
    assert after["reservations"] == before["reservations"]
    assert after["jobs"] == before["jobs"]
    amended = path.read_bytes()
    assert not budget_module.amend_daily_policy_budget(**arguments)
    assert path.read_bytes() == amended


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_observation",
        "stricter",
        "origin",
        "evidence_hash",
        "proof_hash",
        "provider_capture",
        "quota_capture",
        "other_limit",
    ],
)
def test_daily_amendment_rejects_ambiguous_or_stricter_provenance(
    tmp_path, monkeypatch, mutation
):
    arguments, _ = captured_day_fixture(tmp_path, monkeypatch)
    ledger = arguments["ledger"]
    original = ledger.read_bytes()
    proof_path = arguments["quota_provenance"]
    proof = json.loads(proof_path.read_text())
    evidence_path = Path(proof["observations"][0]["evidence_path"])
    evidence = json.loads(evidence_path.read_text())
    if mutation == "missing_observation":
        proof["observations"] = []
    elif mutation == "stricter":
        evidence["independent_daily_limit"] = 24
    elif mutation == "origin":
        evidence["daily_limit_origin"] = "unknown"
    elif mutation == "evidence_hash":
        proof["observations"][0]["evidence_sha256"] = "0" * 64
    elif mutation == "proof_hash":
        arguments["expected_provenance_sha256"] = "0" * 64
    elif mutation == "provider_capture":
        Path(evidence["provider_capture_path"]).write_text("{}")
    elif mutation == "quota_capture":
        Path(evidence["quota_capture_path"]).write_text("{}")
    elif mutation == "other_limit":
        evidence["no_other_independent_limit"] = False
    if mutation in {"stricter", "origin", "other_limit"}:
        evidence_path.write_text(json.dumps(evidence, sort_keys=True))
        proof["observations"][0]["evidence_sha256"] = hashlib.sha256(
            evidence_path.read_bytes()
        ).hexdigest()
    if mutation != "proof_hash":
        proof_path.write_text(json.dumps(proof, sort_keys=True))
        arguments["expected_provenance_sha256"] = hashlib.sha256(
            proof_path.read_bytes()
        ).hexdigest()
    with pytest.raises(budget_module.EvaluationBudgetUnavailable):
        budget_module.amend_daily_policy_budget(**arguments)
    assert ledger.read_bytes() == original


def test_daily_amendment_cli_applies_once_to_real_shape_fixture(tmp_path, monkeypatch):
    from scripts import amend_portfolio_budget as cli

    arguments, before = daily_amendment_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "ROOT", arguments["checkout"])
    config_path = tmp_path / "configuration.json"
    config_path.write_text(json.dumps(arguments["configuration"]))
    command = [
        "--budget-ledger",
        str(arguments["ledger"]),
        "--expected-ledger-sha256",
        arguments["expected_ledger_sha256"],
        "--previous-source-root",
        str(arguments["previous_source_root"]),
        "--configuration",
        str(config_path),
        "--confirm-daily-policy-amendment",
        "--preserve-current-day-limit",
        "--previous-inventory",
        str(arguments["previous_inventory"]),
        "--previous-inventory-sha256",
        arguments["expected_previous_inventory_sha256"],
        "--current-inventory",
        str(arguments["current_inventory"]),
        "--current-inventory-sha256",
        arguments["expected_current_inventory_sha256"],
        "--approved-patch",
        str(arguments["approved_patch"]),
        "--approved-patch-sha256",
        arguments["expected_patch_sha256"],
    ]
    assert cli.main(command) == 0
    path = arguments["ledger"]
    after = json.loads(path.read_text())
    assert after["reservations"] == before["reservations"]
    assert after["jobs"] == before["jobs"]
    assert after["days"] == before["days"]
    assert after["stage_configurations"]["B0"] == before["stage_configurations"]["B0"]
    first = path.read_bytes()
    assert cli.main(command) == 0
    assert path.read_bytes() == first


def test_daily_amendment_refuses_utc_rollover_after_provenance_validation(
    tmp_path, monkeypatch, campaign_clock
):
    arguments, _ = captured_day_fixture(tmp_path, monkeypatch)
    path = arguments["ledger"]
    original = path.read_bytes()
    verify = budget_module._reviewed_day_limit

    def verify_then_rollover(*args, **kwargs):
        result = verify(*args, **kwargs)
        campaign_clock.next_day()
        return result

    monkeypatch.setattr(budget_module, "_reviewed_day_limit", verify_then_rollover)
    with pytest.raises(budget_module.EvaluationBudgetUnavailable, match="day changed"):
        budget_module.amend_daily_policy_budget(**arguments)
    assert path.read_bytes() == original


def test_daily_amendment_refuses_utc_rollover_before_atomic_replace(
    tmp_path, monkeypatch, campaign_clock
):
    arguments, _ = captured_day_fixture(tmp_path, monkeypatch)
    path = arguments["ledger"]
    original = path.read_bytes()
    persist = budget_module.EvaluationBudget._persist

    def rollover_then_persist(guard, *args, **kwargs):
        campaign_clock.next_day()
        return persist(guard, *args, **kwargs)

    monkeypatch.setattr(
        budget_module.EvaluationBudget, "_persist", rollover_then_persist
    )
    with pytest.raises(budget_module.EvaluationBudgetUnavailable, match="day changed"):
        budget_module.amend_daily_policy_budget(**arguments)
    assert path.read_bytes() == original


def prospective_authority_fixture(tmp_path, monkeypatch):
    arguments, before = daily_amendment_fixture(tmp_path, monkeypatch)
    now = budget_module.datetime.now(UTC)
    for event in before["reservations"][33:]:
        event["reserved_at"] = (now - timedelta(minutes=1)).isoformat()
    ledger = arguments["ledger"]
    ledger.write_text(json.dumps(before, sort_keys=True))
    digest = hashlib.sha256(ledger.read_bytes()).hexdigest()
    capture_dir = tmp_path / "capture"
    boundary = budget_module.capture_prospective_boundary(
        checkout=arguments["checkout"],
        ledger=ledger,
        expected_ledger_sha256=digest,
        destination=capture_dir,
    )
    capture_ledger = capture_dir / "ledger_snapshot.json"
    now = budget_module.datetime.now(UTC)
    q = quota(
        now,
        remaining_requests=31,
        demo_reserve=20,
        daily_batch_limit=50,
    )
    account = {
        "openrouter": {
            "method": "GET",
            "url": "https://openrouter.ai/api/v1/key",
            "http_status": 200,
            "status": "success",
            "time": q["observed_at"],
            "quota": {
                "is_free_tier": True,
                "free_model_daily_requests": {"limit": 50, "remaining": 31, "used": 19},
            },
        }
    }
    route = {
        "method": "GET",
        "url": "https://openrouter.ai/api/v1/models",
        "http_status": 200,
        "status": "success",
        "time": q["observed_at"],
        "data": [
            {"id": "openrouter/free", "pricing": {"prompt": "0", "completion": "0"}}
        ],
    }
    for name, value in (
        ("account.json", account),
        ("route.json", route),
        ("quota.json", q),
    ):
        (tmp_path / name).write_text(json.dumps(value, sort_keys=True))
    binding = {
        "account_id": "account-1",
        "key_id": "key-1",
        "observed_at": q["observed_at"],
        "provider": "openrouter",
        "model": "openrouter/free",
        "account_capture_sha256": hashlib.sha256(
            (tmp_path / "account.json").read_bytes()
        ).hexdigest(),
        "route_capture_sha256": hashlib.sha256(
            (tmp_path / "route.json").read_bytes()
        ).hexdigest(),
        "credential_check": "local-equality",
        "checked_at": q["observed_at"],
        "sut_role_credentials_match_queried_key": dict.fromkeys(
            (
                "orchestrator",
                "retriever",
                "analyst",
                "mentor",
                "researcher.with_tools",
                "researcher.structured",
            ),
            True,
        ),
        "external_daily_limit": 50,
    }
    (tmp_path / "binding.json").write_text(json.dumps(binding, sort_keys=True))
    proof = {
        "authority": "daily-50-2026-10-30",
        "capture_started_at": boundary["started_at"],
        "boundary_path": str(capture_dir / "boundary.json"),
        "boundary_sha256": hashlib.sha256(
            (capture_dir / "boundary.json").read_bytes()
        ).hexdigest(),
        "capture_ledger_sha256": digest,
        "capture_ledger_path": str(capture_ledger),
        "quota": q,
        "quota_capture_path": str(tmp_path / "quota.json"),
        "quota_capture_sha256": hashlib.sha256(
            (tmp_path / "quota.json").read_bytes()
        ).hexdigest(),
        "binding_capture_path": str(tmp_path / "binding.json"),
        "binding_capture_sha256": hashlib.sha256(
            (tmp_path / "binding.json").read_bytes()
        ).hexdigest(),
        "account_capture_path": str(tmp_path / "account.json"),
        "account_capture_sha256": hashlib.sha256(
            (tmp_path / "account.json").read_bytes()
        ).hexdigest(),
        "route_capture_path": str(tmp_path / "route.json"),
        "route_capture_sha256": hashlib.sha256(
            (tmp_path / "route.json").read_bytes()
        ).hexdigest(),
    }
    proof_path = tmp_path / "prospective-proof.json"
    proof_path.write_text(json.dumps(proof, sort_keys=True))
    arguments.update(
        expected_ledger_sha256=digest,
        preserve_current_day=False,
        prospective_proof=proof_path,
        expected_prospective_sha256=hashlib.sha256(proof_path.read_bytes()).hexdigest(),
        expected_account_id="account-1",
        expected_key_id="key-1",
    )
    return arguments, before, q


def test_prospective_authority_admits_six_without_historical_origin(
    tmp_path, monkeypatch
):
    arguments, before, observation = prospective_authority_fixture(
        tmp_path, monkeypatch
    )
    path = arguments["ledger"]
    assert budget_module.amend_daily_policy_budget(**arguments)
    amended = json.loads(path.read_text())
    today = datetime.now(UTC).date().isoformat()
    assert amended["days"][today]["batch_limit"] == 50
    assert amended["days"][today]["used"] == 20
    assert amended["reservations"] == before["reservations"]
    assert amended["jobs"] == before["jobs"]
    assert amended["stage_configurations"]["B0"] == before["stage_configurations"]["B0"]
    assert amended["policy_amendments"][:-1] == before["policy_amendments"]
    unchanged = path.read_bytes()
    assert not budget_module.amend_daily_policy_budget(**arguments)
    assert path.read_bytes() == unchanged
    with budget_module.EvaluationBudget(
        path,
        stage="final",
        run_id="pending",
        quota=observation,
        configuration=arguments["configuration"],
    ) as guard:
        guard.require_final_runtime_admission()
        for _ in range(6):
            guard.reserve("llm", provider="openrouter", model="openrouter/free")
        with pytest.raises(budget_module.EvaluationBudgetExceeded):
            guard.require_final_runtime_admission()


def test_prospective_authority_cli_applies_real_shape_once(tmp_path, monkeypatch):
    from scripts import amend_portfolio_budget as cli

    arguments, before, _ = prospective_authority_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "ROOT", arguments["checkout"])
    config = tmp_path / "configuration.json"
    config.write_text(json.dumps(arguments["configuration"]))
    command = [
        "--budget-ledger",
        str(arguments["ledger"]),
        "--expected-ledger-sha256",
        arguments["expected_ledger_sha256"],
        "--previous-source-root",
        str(arguments["previous_source_root"]),
        "--configuration",
        str(config),
        "--confirm-daily-policy-amendment",
        "--prospective-authority-proof",
        str(arguments["prospective_proof"]),
        "--prospective-authority-proof-sha256",
        arguments["expected_prospective_sha256"],
        "--expected-account-id",
        arguments["expected_account_id"],
        "--expected-key-id",
        arguments["expected_key_id"],
        "--previous-inventory",
        str(arguments["previous_inventory"]),
        "--previous-inventory-sha256",
        arguments["expected_previous_inventory_sha256"],
        "--current-inventory",
        str(arguments["current_inventory"]),
        "--current-inventory-sha256",
        arguments["expected_current_inventory_sha256"],
        "--approved-patch",
        str(arguments["approved_patch"]),
        "--approved-patch-sha256",
        arguments["expected_patch_sha256"],
    ]
    assert cli.main(command) == 0
    path = arguments["ledger"]
    after = json.loads(path.read_text())
    assert after["reservations"] == before["reservations"]
    assert after["jobs"] == before["jobs"]
    assert after["policy_amendments"][:-1] == before["policy_amendments"]
    saved = path.read_bytes()
    assert cli.main(command) == 0
    assert path.read_bytes() == saved


@pytest.mark.parametrize(
    "mutation",
    [
        "stale",
        "wrong_account",
        "wrong_key",
        "proof_hash",
        "ledger_hash",
        "capture_hash",
        "route_hash",
        "external_limit",
        "remaining",
        "reused",
        "lock",
        "wrong_model",
        "missing_account",
        "forged_account",
        "binding_role",
        "paid_route",
        "missing_quota",
        "contradictory_quota",
    ],
)
def test_prospective_authority_rejects_unsafe_capacity(tmp_path, monkeypatch, mutation):
    import fcntl

    arguments, state, observation = prospective_authority_fixture(tmp_path, monkeypatch)
    path = arguments["ledger"]
    proof_path = arguments["prospective_proof"]
    proof = json.loads(proof_path.read_text())
    account_path = Path(proof["account_capture_path"])
    account = json.loads(account_path.read_text())
    binding_path = Path(proof["binding_capture_path"])
    binding = json.loads(binding_path.read_text())
    if mutation == "stale":
        proof["capture_started_at"] = (
            datetime.now(UTC) - timedelta(minutes=16)
        ).isoformat()
        boundary_path = Path(proof["boundary_path"])
        boundary = json.loads(boundary_path.read_text())
        boundary["started_at"] = proof["capture_started_at"]
        boundary_path.write_text(json.dumps(boundary, sort_keys=True))
        proof["boundary_sha256"] = hashlib.sha256(
            boundary_path.read_bytes()
        ).hexdigest()
    elif mutation == "wrong_account":
        arguments["expected_account_id"] = "other-account"
    elif mutation == "wrong_key":
        arguments["expected_key_id"] = "other-key"
    elif mutation == "proof_hash":
        arguments["expected_prospective_sha256"] = "0" * 64
    elif mutation == "ledger_hash":
        arguments["expected_ledger_sha256"] = "0" * 64
    elif mutation == "capture_hash":
        proof["capture_ledger_sha256"] = "0" * 64
    elif mutation == "route_hash":
        proof["route_capture_sha256"] = "0" * 64
    elif mutation == "external_limit":
        binding["external_daily_limit"] = 24
    elif mutation == "remaining":
        account["openrouter"]["quota"]["free_model_daily_requests"]["remaining"] = 25
        proof["quota"]["remaining_requests"] = 25
    elif mutation == "wrong_model":
        proof["quota"]["model"] = "paid/model"
    elif mutation == "missing_account":
        account_path.unlink()
    elif mutation == "contradictory_quota":
        account["openrouter"]["quota"]["free_model_daily_requests"]["used"] = 40
    elif mutation == "forged_account":
        account["openrouter"]["status"] = "failed"
    elif mutation == "binding_role":
        binding["sut_role_credentials_match_queried_key"]["mentor"] = False
    elif mutation == "paid_route":
        route_path = Path(proof["route_capture_path"])
        route = json.loads(route_path.read_text())
        route["data"][0]["pricing"]["completion"] = "0.001"
        route_path.write_text(json.dumps(route, sort_keys=True))
        proof["route_capture_sha256"] = hashlib.sha256(
            route_path.read_bytes()
        ).hexdigest()
        binding["route_capture_sha256"] = proof["route_capture_sha256"]
    elif mutation == "missing_quota":
        Path(proof["quota_capture_path"]).unlink()
    elif mutation == "reused":
        identity = hashlib.sha256(
            json.dumps(
                {
                    key: observation[key]
                    for key in ("observed_at", "provider", "remaining_requests")
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        state["days"][datetime.now(UTC).date().isoformat()]["observations"][
            identity
        ] = 0
        path.write_text(json.dumps(state, sort_keys=True))
        arguments["expected_ledger_sha256"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        proof["capture_ledger_sha256"] = arguments["expected_ledger_sha256"]
        Path(proof["capture_ledger_path"]).write_bytes(path.read_bytes())
        boundary_path = Path(proof["boundary_path"])
        boundary = json.loads(boundary_path.read_text())
        boundary["ledger_sha256"] = proof["capture_ledger_sha256"]
        boundary_path.write_text(json.dumps(boundary, sort_keys=True))
        proof["boundary_sha256"] = hashlib.sha256(
            boundary_path.read_bytes()
        ).hexdigest()
    if mutation in {"remaining", "forged_account", "contradictory_quota"}:
        account_path.write_text(json.dumps(account, sort_keys=True))
        proof["account_capture_sha256"] = hashlib.sha256(
            account_path.read_bytes()
        ).hexdigest()
        binding["account_capture_sha256"] = proof["account_capture_sha256"]
        quota_path = Path(proof["quota_capture_path"])
        quota_path.write_text(json.dumps(proof["quota"], sort_keys=True))
        proof["quota_capture_sha256"] = hashlib.sha256(
            quota_path.read_bytes()
        ).hexdigest()
    if mutation in {
        "external_limit",
        "remaining",
        "forged_account",
        "binding_role",
        "paid_route",
    }:
        binding_path.write_text(json.dumps(binding, sort_keys=True))
        proof["binding_capture_sha256"] = hashlib.sha256(
            binding_path.read_bytes()
        ).hexdigest()
    if mutation not in {
        "proof_hash",
        "wrong_account",
        "wrong_key",
        "ledger_hash",
        "lock",
    }:
        proof_path.write_text(json.dumps(proof, sort_keys=True))
        arguments["expected_prospective_sha256"] = hashlib.sha256(
            proof_path.read_bytes()
        ).hexdigest()
    original = path.read_bytes()
    lock = None
    if mutation == "lock":
        lock = path.with_suffix(".lock").open("a+")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(budget_module.EvaluationBudgetUnavailable):
            budget_module.amend_daily_policy_budget(**arguments)
    finally:
        if lock:
            lock.close()
    assert path.read_bytes() == original


def test_prospective_baseline_counts_reservations_after_capture_start(
    tmp_path, monkeypatch
):
    arguments, state, q = prospective_authority_fixture(tmp_path, monkeypatch)
    proof_path = arguments["prospective_proof"]
    proof = json.loads(proof_path.read_text())
    after_start = datetime.fromisoformat(proof["capture_started_at"]) + timedelta(
        microseconds=1
    )
    for index in range(2):
        state["reservations"].append(
            {
                "kind": "llm",
                "stage": "final",
                "run_id": "since-capture",
                "job_id": "",
                "request_id": f"since-{index}",
                "reserved_at": after_start.isoformat(),
            }
        )
    state["days"][after_start.date().isoformat()]["used"] += 2
    path = arguments["ledger"]
    path.write_text(json.dumps(state, sort_keys=True))
    arguments["expected_ledger_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert budget_module.amend_daily_policy_budget(**arguments)
    with budget_module.EvaluationBudget(
        path,
        stage="final",
        run_id="pending",
        quota=q,
        configuration=arguments["configuration"],
    ) as guard:
        guard.require_final_runtime_admission()
        for _ in range(4):
            guard.reserve("llm", provider="openrouter", model="openrouter/free")
        with pytest.raises(budget_module.EvaluationBudgetExceeded):
            guard.require_final_runtime_admission()


def test_prospective_authority_refuses_rollover_before_persistence(
    tmp_path, monkeypatch, campaign_clock
):
    arguments, _, _ = prospective_authority_fixture(tmp_path, monkeypatch)
    campaign_clock.moment = datetime.now(UTC) + timedelta(seconds=1)
    path = arguments["ledger"]
    original = path.read_bytes()
    verify = budget_module._prospective_day_limit

    def verify_then_rollover(*args, **kwargs):
        result = verify(*args, **kwargs)
        campaign_clock.next_day()
        return result

    monkeypatch.setattr(budget_module, "_prospective_day_limit", verify_then_rollover)
    with pytest.raises(budget_module.EvaluationBudgetUnavailable, match="day changed"):
        budget_module.amend_daily_policy_budget(**arguments)
    assert path.read_bytes() == original


def test_prospective_authority_refuses_staleness_before_atomic_replace(
    tmp_path, monkeypatch, campaign_clock
):
    arguments, _, _ = prospective_authority_fixture(tmp_path, monkeypatch)
    campaign_clock.moment = datetime.now(UTC) + timedelta(seconds=1)
    path = arguments["ledger"]
    original = path.read_bytes()
    verify = budget_module._prospective_day_limit

    def verify_then_expire(*args, **kwargs):
        result = verify(*args, **kwargs)
        campaign_clock.moment += timedelta(minutes=16)
        return result

    monkeypatch.setattr(budget_module, "_prospective_day_limit", verify_then_expire)
    with pytest.raises(budget_module.EvaluationBudgetUnavailable, match="stale"):
        budget_module.amend_daily_policy_budget(**arguments)
    assert path.read_bytes() == original


def test_prospective_capture_boundary_is_locked_and_read_only(tmp_path, monkeypatch):
    arguments, _ = daily_amendment_fixture(tmp_path, monkeypatch)
    ledger = arguments["ledger"]
    before = ledger.read_bytes()
    destination = tmp_path / "capture"
    result = budget_module.capture_prospective_boundary(
        checkout=arguments["checkout"],
        ledger=ledger,
        expected_ledger_sha256=arguments["expected_ledger_sha256"],
        destination=destination,
    )
    assert ledger.read_bytes() == before
    assert (destination / "ledger_snapshot.json").read_bytes() == before
    assert result["canonical_ledger"] == str(ledger)
    assert result["ledger_sha256"] == hashlib.sha256(before).hexdigest()
    with pytest.raises(FileExistsError):
        budget_module.capture_prospective_boundary(
            checkout=arguments["checkout"],
            ledger=ledger,
            expected_ledger_sha256=arguments["expected_ledger_sha256"],
            destination=destination,
        )


def test_prospective_capture_cli_uses_canonical_lock_without_session(
    tmp_path, monkeypatch
):
    from scripts import amend_portfolio_budget as cli

    arguments, _ = daily_amendment_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "ROOT", arguments["checkout"])
    path = arguments["ledger"]
    original = path.read_bytes()
    destination = tmp_path / "capture-cli"
    status = cli.main(
        [
            "--budget-ledger",
            str(path),
            "--expected-ledger-sha256",
            arguments["expected_ledger_sha256"],
            "--capture-prospective-boundary",
            str(destination),
        ]
    )
    assert status == 0
    assert path.read_bytes() == original
    assert (destination / "ledger_snapshot.json").read_bytes() == original
    assert json.loads((destination / "boundary.json").read_text())[
        "canonical_ledger"
    ] == str(path)


@pytest.mark.parametrize("mutation", ["lock", "hash", "noncanonical"])
def test_prospective_capture_boundary_rejects_lock_hash_or_path(
    tmp_path, monkeypatch, mutation
):
    import fcntl

    arguments, _ = daily_amendment_fixture(tmp_path, monkeypatch)
    ledger = arguments["ledger"]
    original = ledger.read_bytes()
    expected = arguments["expected_ledger_sha256"]
    selected = ledger
    lock = None
    if mutation == "lock":
        lock = ledger.with_suffix(".lock").open("a+")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    elif mutation == "hash":
        expected = "0" * 64
    else:
        selected = tmp_path / "different-ledger.json"
        selected.write_bytes(original)
    destination = tmp_path / "not-created"
    try:
        with pytest.raises(budget_module.EvaluationBudgetUnavailable):
            budget_module.capture_prospective_boundary(
                checkout=arguments["checkout"],
                ledger=selected,
                expected_ledger_sha256=expected,
                destination=destination,
            )
    finally:
        if lock:
            lock.close()
    assert not destination.exists()
    assert ledger.read_bytes() == original


@pytest.mark.parametrize(
    "mutation",
    ["missing_boundary", "forged_boundary", "changed_snapshot", "nonprefix_snapshot"],
)
def test_prospective_authority_rejects_unverifiable_capture_boundary(
    tmp_path, monkeypatch, mutation
):
    arguments, _, _ = prospective_authority_fixture(tmp_path, monkeypatch)
    path = arguments["ledger"]
    original = path.read_bytes()
    proof_path = arguments["prospective_proof"]
    proof = json.loads(proof_path.read_text())
    boundary_path = Path(proof["boundary_path"])
    snapshot_path = Path(proof["capture_ledger_path"])
    if mutation == "missing_boundary":
        boundary_path.unlink()
    elif mutation == "forged_boundary":
        boundary = json.loads(boundary_path.read_text())
        boundary["canonical_ledger"] = "/different/ledger.json"
        boundary_path.write_text(json.dumps(boundary, sort_keys=True))
        proof["boundary_sha256"] = hashlib.sha256(
            boundary_path.read_bytes()
        ).hexdigest()
    elif mutation == "changed_snapshot":
        snapshot_path.write_text("{}")
    else:
        snapshot = json.loads(snapshot_path.read_text())
        snapshot["reservations"][0]["request_id"] = "forged-event"
        snapshot_path.write_text(json.dumps(snapshot, sort_keys=True))
        proof["capture_ledger_sha256"] = hashlib.sha256(
            snapshot_path.read_bytes()
        ).hexdigest()
        boundary = json.loads(boundary_path.read_text())
        boundary["ledger_sha256"] = proof["capture_ledger_sha256"]
        boundary_path.write_text(json.dumps(boundary, sort_keys=True))
        proof["boundary_sha256"] = hashlib.sha256(
            boundary_path.read_bytes()
        ).hexdigest()
    proof_path.write_text(json.dumps(proof, sort_keys=True))
    arguments["expected_prospective_sha256"] = hashlib.sha256(
        proof_path.read_bytes()
    ).hexdigest()
    with pytest.raises(budget_module.EvaluationBudgetUnavailable):
        budget_module.amend_daily_policy_budget(**arguments)
    assert path.read_bytes() == original


def test_prospective_authority_rejects_quota_observed_before_capture_completed(
    tmp_path, monkeypatch
):
    arguments, _, _ = prospective_authority_fixture(tmp_path, monkeypatch)
    path = arguments["ledger"]
    original = path.read_bytes()
    proof_path = arguments["prospective_proof"]
    proof = json.loads(proof_path.read_text())
    boundary_path = Path(proof["boundary_path"])
    boundary = json.loads(boundary_path.read_text())
    boundary["completed_at"] = (
        datetime.fromisoformat(proof["quota"]["observed_at"]) + timedelta(seconds=1)
    ).isoformat()
    boundary_path.write_text(json.dumps(boundary, sort_keys=True))
    proof["boundary_sha256"] = hashlib.sha256(boundary_path.read_bytes()).hexdigest()
    proof_path.write_text(json.dumps(proof, sort_keys=True))
    arguments["expected_prospective_sha256"] = hashlib.sha256(
        proof_path.read_bytes()
    ).hexdigest()
    with pytest.raises(budget_module.EvaluationBudgetUnavailable):
        budget_module.amend_daily_policy_budget(**arguments)
    assert path.read_bytes() == original


def test_capture_boundary_accepts_existing_daily_50_campaign(tmp_path, monkeypatch):
    arguments, state = daily_amendment_fixture(tmp_path, monkeypatch)
    path = arguments["ledger"]
    state["daily_policy"] = {
        "llm_attempts_per_utc_day": 50,
        "demo_headroom_requests": 20,
    }
    path.write_text(json.dumps(state, sort_keys=True))
    original = path.read_bytes()
    record = budget_module.capture_prospective_boundary(
        checkout=arguments["checkout"],
        ledger=path,
        expected_ledger_sha256=hashlib.sha256(original).hexdigest(),
        destination=tmp_path / "fresh-capture",
    )
    assert path.read_bytes() == original
    assert Path(record["snapshot_path"]).read_bytes() == original


def registered_quota_fixture(tmp_path, monkeypatch, *, capture_next_day=False):
    arguments, _, original_quota = prospective_authority_fixture(tmp_path, monkeypatch)
    assert budget_module.amend_daily_policy_budget(**arguments)
    ledger = arguments["ledger"]
    with budget_module.EvaluationBudget(
        ledger,
        stage="final",
        run_id="spent-after-amendment",
        quota=original_quota,
        configuration=arguments["configuration"],
    ) as guard:
        for _ in range(3):
            guard.reserve("llm", provider="openrouter", model="openrouter/free")
    if capture_next_day:
        budget_module.datetime.next_day()
    before_capture = ledger.read_bytes()
    capture_dir = tmp_path / "registration-capture"
    boundary = budget_module.capture_prospective_boundary(
        checkout=arguments["checkout"],
        ledger=ledger,
        expected_ledger_sha256=hashlib.sha256(before_capture).hexdigest(),
        destination=capture_dir,
    )
    original_proof = json.loads(arguments["prospective_proof"].read_text())
    old_account = json.loads(Path(original_proof["account_capture_path"]).read_text())
    old_route = json.loads(Path(original_proof["route_capture_path"]).read_text())
    old_binding = json.loads(Path(original_proof["binding_capture_path"]).read_text())
    observed = budget_module.datetime.now(UTC).isoformat()
    q = {
        "observed_at": observed,
        "provider": "openrouter",
        "model": "openrouter/free",
        "remaining_requests": 28,
        "demo_reserve": 20,
        "daily_batch_limit": 50,
    }
    old_account["openrouter"]["time"] = observed
    old_account["openrouter"]["quota"]["free_model_daily_requests"].update(
        remaining=28, used=22
    )
    old_route["time"] = observed
    old_binding.update(observed_at=observed, checked_at=observed)
    paths = {
        name: tmp_path / f"registration-{name}.json"
        for name in ("account", "route", "quota", "binding")
    }
    for name, value in (("account", old_account), ("route", old_route), ("quota", q)):
        paths[name].write_text(json.dumps(value, sort_keys=True))
    old_binding["account_capture_sha256"] = hashlib.sha256(
        paths["account"].read_bytes()
    ).hexdigest()
    old_binding["route_capture_sha256"] = hashlib.sha256(
        paths["route"].read_bytes()
    ).hexdigest()
    paths["binding"].write_text(json.dumps(old_binding, sort_keys=True))
    proof = {
        "authority": "daily-50-2026-10-30",
        "capture_started_at": boundary["started_at"],
        "boundary_path": str(capture_dir / "boundary.json"),
        "boundary_sha256": hashlib.sha256(
            (capture_dir / "boundary.json").read_bytes()
        ).hexdigest(),
        "capture_ledger_path": boundary["snapshot_path"],
        "capture_ledger_sha256": boundary["ledger_sha256"],
        "quota": q,
    }
    for name in paths:
        proof[f"{name}_capture_path"] = str(paths[name])
        proof[f"{name}_capture_sha256"] = hashlib.sha256(
            paths[name].read_bytes()
        ).hexdigest()
    proof_path = tmp_path / "registration-proof.json"
    proof_path.write_text(json.dumps(proof, sort_keys=True))
    return arguments, q, proof_path, paths


def test_explicit_registration_preserves_prior_baselines_and_counts_new_reservations(
    tmp_path, monkeypatch
):
    arguments, q, proof, _ = registered_quota_fixture(tmp_path, monkeypatch)
    path = arguments["ledger"]
    before = json.loads(path.read_text())
    with budget_module.EvaluationBudget(
        path,
        stage="final",
        run_id="post-capture",
        quota=json.loads(arguments["prospective_proof"].read_text())["quota"],
        configuration=arguments["configuration"],
    ) as guard:
        for _ in range(2):
            guard.reserve("llm", provider="openrouter", model="openrouter/free")
    current = json.loads(path.read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert budget_module.register_prospective_observation(
        checkout=arguments["checkout"],
        ledger=path,
        expected_ledger_sha256=digest,
        proof_path=proof,
        expected_proof_sha256=hashlib.sha256(proof.read_bytes()).hexdigest(),
    )
    after = json.loads(path.read_text())
    day_key = datetime.now(UTC).date().isoformat()
    assert after["reservations"] == current["reservations"]
    assert after["jobs"] == current["jobs"]
    assert after["stage_configurations"] == current["stage_configurations"]
    assert all(
        after["days"][day_key]["observations"][key] == value
        for key, value in before["days"][day_key]["observations"].items()
    )
    assert after["days"][day_key]["used"] == 25
    with budget_module.EvaluationBudget(
        path,
        stage="final",
        run_id="next",
        quota=q,
        configuration=arguments["configuration"],
    ) as guard:
        guard.require_final_runtime_admission()
        guard.reserve("llm", provider="openrouter", model="openrouter/free")
        with pytest.raises(budget_module.EvaluationBudgetExceeded):
            guard.require_final_runtime_admission()


def test_locked_session_requires_existing_observation_before_configuration(tmp_path):
    path = tmp_path / "ledger.json"
    q = quota(daily_batch_limit=50)
    with budget_module.EvaluationBudget(path, stage="final", run_id="initial", quota=q):
        pass
    before = path.read_bytes()
    fresh = quota(daily_batch_limit=50, remaining_requests=31)
    guard = budget_module.EvaluationBudget(
        path, stage="final", run_id="next", quota=fresh
    )
    guard.require_registered_observation = True
    with pytest.raises(budget_module.EvaluationBudgetUnavailable, match="registered"):
        with guard:
            pytest.fail("unregistered observation entered")
    assert path.read_bytes() == before


def test_live_cli_exposes_registered_quota_guard():
    from scripts import evaluate_runtime_workflow as cli

    args = cli.parse_args(
        [
            "live",
            "--case-id",
            "ambiguous_eagle_without_history",
            "--require-registered-quota",
        ]
    )
    assert args.require_registered_quota is True


def test_registration_rejects_reused_observation_without_reset(tmp_path, monkeypatch):
    arguments, _, proof, _ = registered_quota_fixture(tmp_path, monkeypatch)
    path = arguments["ledger"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    proof_sha = hashlib.sha256(proof.read_bytes()).hexdigest()
    assert budget_module.register_prospective_observation(
        checkout=arguments["checkout"],
        ledger=path,
        expected_ledger_sha256=digest,
        proof_path=proof,
        expected_proof_sha256=proof_sha,
    )
    saved = path.read_bytes()
    with pytest.raises(budget_module.EvaluationBudgetUnavailable, match="already used"):
        budget_module.register_prospective_observation(
            checkout=arguments["checkout"],
            ledger=path,
            expected_ledger_sha256=hashlib.sha256(saved).hexdigest(),
            proof_path=proof,
            expected_proof_sha256=proof_sha,
        )
    assert path.read_bytes() == saved


def test_registration_rejects_wrong_account_capture(tmp_path, monkeypatch):
    arguments, _, proof, paths = registered_quota_fixture(tmp_path, monkeypatch)
    path = arguments["ledger"]
    before = path.read_bytes()
    data = json.loads(proof.read_text())
    binding = json.loads(paths["binding"].read_text())
    binding["account_id"] = "different-account"
    paths["binding"].write_text(json.dumps(binding, sort_keys=True))
    data["binding_capture_sha256"] = hashlib.sha256(
        paths["binding"].read_bytes()
    ).hexdigest()
    proof.write_text(json.dumps(data, sort_keys=True))
    with pytest.raises(
        budget_module.EvaluationBudgetUnavailable, match="capacity evidence"
    ):
        budget_module.register_prospective_observation(
            checkout=arguments["checkout"],
            ledger=path,
            expected_ledger_sha256=hashlib.sha256(before).hexdigest(),
            proof_path=proof,
            expected_proof_sha256=hashlib.sha256(proof.read_bytes()).hexdigest(),
        )
    assert path.read_bytes() == before


def test_registration_rollover_creates_new_day_without_resetting_history(
    tmp_path, monkeypatch, campaign_clock
):
    arguments, q, proof, _ = registered_quota_fixture(
        tmp_path, monkeypatch, capture_next_day=True
    )
    path = arguments["ledger"]
    before = json.loads(path.read_text())
    assert q["observed_at"][:10] not in before["days"]
    assert budget_module.register_prospective_observation(
        checkout=arguments["checkout"],
        ledger=path,
        expected_ledger_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        proof_path=proof,
        expected_proof_sha256=hashlib.sha256(proof.read_bytes()).hexdigest(),
    )
    after = json.loads(path.read_text())
    assert all(after["days"][key] == value for key, value in before["days"].items())
    assert after["days"][q["observed_at"][:10]]["used"] == 0
    assert after["reservations"] == before["reservations"]
    guard = budget_module.EvaluationBudget(
        path,
        stage="final",
        run_id="new-day",
        quota=q,
        configuration=arguments["configuration"],
    )
    guard.require_registered_observation = True
    with guard:
        guard.require_final_runtime_admission()


def test_registration_cli_reuses_reviewed_capture_without_implicit_session(
    tmp_path, monkeypatch
):
    from scripts import amend_portfolio_budget as cli

    arguments, _, proof, _ = registered_quota_fixture(tmp_path, monkeypatch)
    path = arguments["ledger"]
    before = json.loads(path.read_text())
    monkeypatch.setattr(cli, "ROOT", arguments["checkout"])
    result = cli.main(
        [
            "--budget-ledger",
            str(path),
            "--expected-ledger-sha256",
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "--register-prospective-observation",
            str(proof),
            "--prospective-authority-proof-sha256",
            hashlib.sha256(proof.read_bytes()).hexdigest(),
        ]
    )
    assert result == 0
    after = json.loads(path.read_text())
    assert after["reservations"] == before["reservations"]
    assert after["jobs"] == before["jobs"]
    assert after["stage_configurations"] == before["stage_configurations"]


@pytest.mark.parametrize("mutation", ["valid", "missing", "stale", "wrong_account"])
def test_separate_quota_successor_reaches_only_registered_locked_admission(
    tmp_path, monkeypatch, mutation
):
    from types import SimpleNamespace

    from scripts import continue_portfolio_collection as continuation

    arguments, _, proof, paths = registered_quota_fixture(tmp_path, monkeypatch)
    ledger = arguments["ledger"]
    assert budget_module.register_prospective_observation(
        checkout=arguments["checkout"],
        ledger=ledger,
        expected_ledger_sha256=hashlib.sha256(ledger.read_bytes()).hexdigest(),
        proof_path=proof,
        expected_proof_sha256=hashlib.sha256(proof.read_bytes()).hexdigest(),
    )
    if mutation == "wrong_account":
        state = json.loads(ledger.read_text())
        state["quota_observation_registrations"][-1]["source_proof"][
            "expected_account_id"
        ] = "wrong-account"
        ledger.write_text(json.dumps(state, sort_keys=True))
    if mutation == "stale":
        initial = datetime.now(UTC)

        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return initial + timedelta(minutes=16)

        monkeypatch.setattr(budget_module, "datetime", Clock)
    preparation = tmp_path / "entry-preparation"
    preparation.mkdir()
    (preparation / "freeze_manifest.json").write_text(
        json.dumps(
            {
                "ordered_runs": [
                    {"id": f"case-{i}", "repetition": 1, "version": 1}
                    for i in range(1, 16)
                ]
            }
        )
    )
    (preparation / "explicit_environment_v2.json").write_text("{}")
    source = tmp_path / "entry-source"
    contracts = source / "evals/runtime"
    contracts.mkdir(parents=True)
    (contracts / "scenarios.json").write_text(
        json.dumps(
            [
                {
                    "id": "case-8",
                    "version": 1,
                    "expectations": {
                        "required_tools": [],
                        "budgets": {"max_tavily_searches": 0},
                    },
                }
            ]
        )
    )
    output = tmp_path / "entry-output"
    output.mkdir()
    frozen = output / "quota.json"
    frozen.write_text('{"preserved":true}\n')
    frozen_bytes = frozen.read_bytes()
    for name, value in (
        ("BASE", tmp_path),
        ("PREPARATION", preparation),
        ("SOURCE", source),
        ("OUTPUT", output),
        ("LEDGER", ledger),
    ):
        monkeypatch.setattr(continuation, name, value)
    monkeypatch.setattr(continuation, "verify_preflight", lambda digest, env: None)
    admitted = []

    def run(command, **kwargs):
        assert "--require-registered-quota" in command
        loaded = json.loads(
            Path(command[command.index("--provider-quota") + 1]).read_text()
        )
        guard = budget_module.EvaluationBudget(
            ledger,
            stage="final",
            run_id="mocked-admission",
            quota=loaded,
            configuration=arguments["configuration"],
        )
        guard.require_registered_observation = True
        with guard:
            guard.require_final_runtime_admission()
        admitted.append(True)
        run_output = Path(command[command.index("--output-dir") + 1])
        run_output.mkdir(parents=True)
        (run_output / "manifest.json").write_text(json.dumps({"status": "completed"}))
        (run_output / "runtime_records.jsonl").write_text(
            json.dumps(
                {"scenario_id": "case-8", "repetition": 1, "scenario_version": 1}
            )
            + "\n"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(continuation.subprocess, "run", run)
    quota_path = paths["quota"] if mutation != "missing" else tmp_path / "missing.json"
    before = ledger.read_bytes()
    if mutation == "valid":
        assert (
            continuation.main(
                ["8", "case-8", "1", "approved", "--quota-snapshot", str(quota_path)]
            )
            == 0
        )
        assert admitted == [True]
        assert (
            json.loads(ledger.read_text())["reservations"]
            == json.loads(before)["reservations"]
        )
    else:
        with pytest.raises((ValueError, budget_module.EvaluationBudgetUnavailable)):
            continuation.main(
                ["8", "case-8", "1", "approved", "--quota-snapshot", str(quota_path)]
            )
        assert not admitted
        assert ledger.read_bytes() == before
    assert frozen.read_bytes() == frozen_bytes


def final_transition_fixture(tmp_path, monkeypatch):
    import difflib
    import shutil

    from backend.evaluation import evaluation_reporting as reporting

    arguments, _, _ = prospective_authority_fixture(tmp_path, monkeypatch)
    assert budget_module.amend_daily_policy_budget(**arguments)
    ledger = arguments["ledger"]
    before = json.loads(ledger.read_text())
    previous = arguments["checkout"]
    checkout = tmp_path / "successor-source"
    shutil.copytree(previous / "backend", checkout / "backend")
    (checkout / ".git").mkdir()
    (checkout / ".git/portfolio-restored-source").write_text(
        json.dumps({"canonical_ledger": str(ledger)})
    )
    changed = checkout / "backend/evaluation/portfolio_budget.py"
    changed.write_text(changed.read_text() + "\nreviewed-registration\n")
    inventory = {}
    for file in (checkout / "backend").rglob("*.py"):
        data = file.read_bytes()
        inventory[file.relative_to(checkout).as_posix()] = {
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    current_inventory = tmp_path / "successor-inventory.json"
    current_inventory.write_text(json.dumps(inventory, sort_keys=True))
    name = "backend/evaluation/portfolio_budget.py"
    patch = "".join(
        difflib.unified_diff(
            (previous / name).read_text().splitlines(keepends=True),
            (checkout / name).read_text().splitlines(keepends=True),
            fromfile="a/" + name,
            tofile="b/" + name,
        )
    ).encode()
    patch_path = tmp_path / "transition.patch"
    patch_path.write_bytes(patch)
    with monkeypatch.context() as context:
        context.setattr(reporting, "ROOT", checkout)
        fingerprint = reporting._hash_json(
            reporting.build_evaluation_snapshot({})["fingerprint_input"]
        )
    configuration = before["stage_configurations"]["final"] | {
        "evaluation_fingerprint": fingerprint
    }
    transition = {
        "checkout": checkout,
        "ledger": ledger,
        "previous_source_root": previous,
        "expected_ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
        "configuration": configuration,
        "previous_inventory": arguments["current_inventory"],
        "expected_previous_inventory_sha256": arguments[
            "expected_current_inventory_sha256"
        ],
        "current_inventory": current_inventory,
        "expected_current_inventory_sha256": hashlib.sha256(
            current_inventory.read_bytes()
        ).hexdigest(),
        "approved_patch": patch_path,
        "expected_patch_sha256": hashlib.sha256(patch).hexdigest(),
    }
    return transition, before


def test_final_evaluator_transition_preserves_jobs_and_b0_binding(
    tmp_path, monkeypatch
):
    transition, before = final_transition_fixture(tmp_path, monkeypatch)
    ledger = transition["ledger"]
    configuration = transition["configuration"]
    assert budget_module.transition_final_evaluation_identity(**transition)
    after = json.loads(ledger.read_text())
    assert after["reservations"] == before["reservations"]
    assert after["jobs"] == before["jobs"]
    assert after["days"] == before["days"]
    assert after["stage_configurations"]["B0"] == before["stage_configurations"]["B0"]
    assert after["stage_configurations"]["final"] == configuration
    saved = ledger.read_bytes()
    assert not budget_module.transition_final_evaluation_identity(**transition)
    assert ledger.read_bytes() == saved


def test_final_transition_cli_requires_explicit_review_inputs(
    tmp_path, monkeypatch, capsys
):
    from scripts import amend_portfolio_budget as cli

    arguments, _ = daily_amendment_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "ROOT", arguments["checkout"])
    path = arguments["ledger"]
    before = path.read_bytes()
    config = tmp_path / "transition-config.json"
    config.write_text(json.dumps(arguments["configuration"]))
    result = cli.main(
        [
            "--budget-ledger",
            str(path),
            "--expected-ledger-sha256",
            arguments["expected_ledger_sha256"],
            "--previous-source-root",
            str(arguments["previous_source_root"]),
            "--configuration",
            str(config),
            "--confirm-final-evaluator-transition",
        ]
    )
    assert result == 2
    assert path.read_bytes() == before
    assert capsys.readouterr().err


@pytest.mark.parametrize("mutation", ["hash", "configuration", "patch", "lock"])
def test_final_evaluator_transition_rejects_unreviewed_state(
    tmp_path, monkeypatch, mutation
):
    import fcntl

    transition, _ = final_transition_fixture(tmp_path, monkeypatch)
    ledger = transition["ledger"]
    original = ledger.read_bytes()
    lock = None
    if mutation == "hash":
        transition["expected_ledger_sha256"] = "0" * 64
    elif mutation == "configuration":
        transition["configuration"] = transition["configuration"] | {
            "paths": {"prompts_dir": "/wrong"}
        }
    elif mutation == "patch":
        transition["approved_patch"].write_text("unreviewed")
    else:
        lock = ledger.with_suffix(".lock").open("a+")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(budget_module.EvaluationBudgetUnavailable):
            budget_module.transition_final_evaluation_identity(**transition)
    finally:
        if lock:
            lock.close()
    assert ledger.read_bytes() == original


@pytest.mark.parametrize("mutation", ["stale", "ledger_hash", "proof_hash", "lock"])
def test_registration_refuses_stale_hash_or_competing_lock(
    tmp_path, monkeypatch, mutation
):
    import fcntl

    arguments, _, proof, _ = registered_quota_fixture(tmp_path, monkeypatch)
    ledger = arguments["ledger"]
    before = ledger.read_bytes()
    ledger_sha = hashlib.sha256(before).hexdigest()
    proof_sha = hashlib.sha256(proof.read_bytes()).hexdigest()
    lock = None
    if mutation == "stale":
        initial = datetime.now(UTC)

        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return initial + timedelta(minutes=16)

        monkeypatch.setattr(budget_module, "datetime", Clock)
    elif mutation == "ledger_hash":
        ledger_sha = "0" * 64
    elif mutation == "proof_hash":
        proof_sha = "0" * 64
    else:
        lock = ledger.with_suffix(".lock").open("a+")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(budget_module.EvaluationBudgetUnavailable):
            budget_module.register_prospective_observation(
                checkout=arguments["checkout"],
                ledger=ledger,
                expected_ledger_sha256=ledger_sha,
                proof_path=proof,
                expected_proof_sha256=proof_sha,
            )
    finally:
        if lock:
            lock.close()
    assert ledger.read_bytes() == before


def test_final_evaluator_transition_cli_applies_once_to_temporary_ledger(
    tmp_path, monkeypatch
):
    from scripts import amend_portfolio_budget as cli

    transition, before = final_transition_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "ROOT", transition["checkout"])
    config = tmp_path / "final-transition-config.json"
    config.write_text(json.dumps(transition["configuration"]))
    command = [
        "--budget-ledger",
        str(transition["ledger"]),
        "--expected-ledger-sha256",
        transition["expected_ledger_sha256"],
        "--previous-source-root",
        str(transition["previous_source_root"]),
        "--configuration",
        str(config),
        "--confirm-final-evaluator-transition",
        "--previous-inventory",
        str(transition["previous_inventory"]),
        "--previous-inventory-sha256",
        transition["expected_previous_inventory_sha256"],
        "--current-inventory",
        str(transition["current_inventory"]),
        "--current-inventory-sha256",
        transition["expected_current_inventory_sha256"],
        "--approved-patch",
        str(transition["approved_patch"]),
        "--approved-patch-sha256",
        transition["expected_patch_sha256"],
    ]
    assert cli.main(command) == 0
    path = transition["ledger"]
    after = json.loads(path.read_text())
    assert after["reservations"] == before["reservations"]
    assert after["jobs"] == before["jobs"]
    assert after["stage_configurations"]["B0"] == before["stage_configurations"]["B0"]
    saved = path.read_bytes()
    assert cli.main(command) == 0
    assert path.read_bytes() == saved


def test_run10_successor_preserves_no_dispatch_attempt_and_uses_registered_snapshot(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from scripts import continue_portfolio_collection as continuation

    arguments, _, proof, paths = registered_quota_fixture(tmp_path, monkeypatch)
    ledger = arguments["ledger"]
    budget_module.register_prospective_observation(
        checkout=arguments["checkout"],
        ledger=ledger,
        expected_ledger_sha256=hashlib.sha256(ledger.read_bytes()).hexdigest(),
        proof_path=proof,
        expected_proof_sha256=hashlib.sha256(proof.read_bytes()).hexdigest(),
    )
    output = tmp_path / "history"
    output.mkdir()
    frozen_quota = output / "quota.json"
    frozen_quota.write_text("old-frozen-quota")
    for number in (8, 9):
        label = f"{number:02}_attempt_001"
        name = f"{label}_case-{number}"
        run_dir = output / "runs" / name
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(json.dumps({"status": "completed"}))
        (run_dir / "runtime_records.jsonl").write_text(
            json.dumps(
                {
                    "scenario_id": f"case-{number}",
                    "repetition": 1,
                    "scenario_version": 1,
                }
            )
            + "\n"
        )
        (output / f"{label}_invocation.json").write_text(
            json.dumps(
                {
                    "case": f"case-{number}",
                    "version": 1,
                    "repetition": 1,
                    "output_name": name,
                }
            )
        )
        (output / f"{label}_exit.json").write_text(json.dumps({"exit_code": 0}))
    (output / "10_attempt_001_invocation.json").write_text('{"preserve":"original"}\n')
    (output / "10_attempt_001_exit.json").write_text('{"exit_code":2}\n')
    (output / "10_attempt_001_pre_dispatch.json").write_text(
        '{"dispatch_started":false}\n'
    )
    previous = {file.name: file.read_bytes() for file in output.glob("10_attempt_001*")}
    preparation = tmp_path / "run10-preparation"
    preparation.mkdir()
    (preparation / "freeze_manifest.json").write_text(
        json.dumps(
            {
                "ordered_runs": [
                    {"id": f"case-{i}", "repetition": 1, "version": 1}
                    for i in range(1, 16)
                ]
            }
        )
    )
    (preparation / "explicit_environment_v2.json").write_text("{}")
    source = tmp_path / "run10-source"
    contracts = source / "evals/runtime"
    contracts.mkdir(parents=True)
    (contracts / "scenarios.json").write_text(
        json.dumps(
            [
                {
                    "id": "case-10",
                    "version": 1,
                    "expectations": {
                        "required_tools": ["deep_web_research"],
                        "budgets": {"max_tavily_searches": 1},
                    },
                }
            ]
        )
    )
    for name, value in (
        ("BASE", tmp_path),
        ("PREPARATION", preparation),
        ("SOURCE", source),
        ("OUTPUT", output),
        ("LEDGER", ledger),
    ):
        monkeypatch.setattr(continuation, name, value)
    monkeypatch.setattr(continuation, "verify_preflight", lambda digest, env: None)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        quota_path = Path(command[command.index("--provider-quota") + 1])
        guard = budget_module.EvaluationBudget(
            ledger,
            stage="final",
            run_id="mocked-run10",
            quota=json.loads(quota_path.read_text()),
            configuration=arguments["configuration"],
        )
        guard.require_registered_observation = True
        with guard:
            guard.require_final_runtime_admission()
        run_output = Path(command[command.index("--output-dir") + 1])
        run_output.mkdir(parents=True)
        (run_output / "manifest.json").write_text(json.dumps({"status": "completed"}))
        (run_output / "runtime_records.jsonl").write_text(
            json.dumps(
                {"scenario_id": "case-10", "repetition": 1, "scenario_version": 1}
            )
            + "\n"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(continuation.subprocess, "run", run)
    assert (
        continuation.main(
            ["10", "case-10", "1", "reviewed", "--quota-snapshot", str(paths["quota"])]
        )
        == 0
    )
    assert "--allow-tavily" in calls[0] and "--require-registered-quota" in calls[0]
    assert (output / "10_attempt_002_invocation.json").is_file()
    assert {
        file.name: file.read_bytes() for file in output.glob("10_attempt_001*")
    } == previous
    assert frozen_quota.read_text() == "old-frozen-quota"


def test_locked_registered_quota_rejects_malformed_history_without_writing(tmp_path):
    path = tmp_path / "ledger.json"
    q = quota(daily_batch_limit=50)
    with budget_module.EvaluationBudget(path, stage="final", run_id="initial", quota=q):
        pass
    state = json.loads(path.read_text())
    state["quota_observation_registrations"] = [None]
    path.write_text(json.dumps(state, sort_keys=True))
    original = path.read_bytes()
    guard = budget_module.EvaluationBudget(path, stage="final", run_id="next", quota=q)
    guard.require_registered_observation = True
    with pytest.raises(budget_module.EvaluationBudgetUnavailable, match="registered"):
        with guard:
            pytest.fail("malformed registration entered")
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "mutation",
    [
        "valid",
        "nonfree",
        "extra_fee",
        "wrong_provider",
        "missing_schema",
        "wrong_account",
        "wrong_key",
        "wrong_model",
    ],
)
def test_registered_fixed_judge_proof_accepts_exact_free_phala(
    tmp_path, monkeypatch, mutation
):
    arguments, _, proof_path, paths = registered_quota_fixture(tmp_path, monkeypatch)
    proof = json.loads(proof_path.read_text())
    model = "xingchen-agi/xing4.0-29b-a4b:free"
    proof["quota"]["model"] = model
    paths["quota"].write_text(json.dumps(proof["quota"], sort_keys=True))
    route = {
        "method": "GET",
        "url": f"https://openrouter.ai/api/v1/models/{model}/endpoints",
        "http_status": 200,
        "status": "success",
        "time": proof["quota"]["observed_at"],
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
    if mutation == "nonfree":
        route["data"]["endpoints"][0]["pricing"]["completion"] = "1"
    if mutation == "extra_fee":
        route["data"]["endpoints"][0]["pricing"]["request"] = "0.1"
    if mutation == "wrong_provider":
        route["data"]["endpoints"][0]["tag"] = "other"
    if mutation == "missing_schema":
        route["data"]["endpoints"][0]["supported_parameters"].remove(
            "structured_outputs"
        )
    if mutation == "wrong_model":
        proof["quota"]["model"] = "another/model:free"
        paths["quota"].write_text(json.dumps(proof["quota"], sort_keys=True))
    paths["route"].write_text(json.dumps(route, sort_keys=True))
    binding = json.loads(paths["binding"].read_text())
    binding["model"] = model
    if mutation == "wrong_account":
        binding["account_id"] = "other"
    if mutation == "wrong_key":
        binding["key_id"] = "other"
    binding["sut_role_credentials_match_queried_key"] = {"judge": True}
    binding["route_capture_sha256"] = hashlib.sha256(
        paths["route"].read_bytes()
    ).hexdigest()
    paths["binding"].write_text(json.dumps(binding, sort_keys=True))
    for name in ("route", "quota", "binding"):
        proof[f"{name}_capture_sha256"] = hashlib.sha256(
            paths[name].read_bytes()
        ).hexdigest()
    proof_path.write_text(json.dumps(proof, sort_keys=True))
    ledger = arguments["ledger"]
    before = json.loads(ledger.read_text())
    kwargs = {
        "checkout": arguments["checkout"],
        "ledger": ledger,
        "expected_ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
        "proof_path": proof_path,
        "expected_proof_sha256": hashlib.sha256(proof_path.read_bytes()).hexdigest(),
    }
    if mutation != "valid":
        with pytest.raises(budget_module.EvaluationBudgetUnavailable):
            budget_module.register_prospective_observation(**kwargs)
        assert json.loads(ledger.read_text()) == before
        return
    assert budget_module.register_prospective_observation(**kwargs)
    after = json.loads(ledger.read_text())
    assert after["reservations"] == before["reservations"]
    assert after["jobs"] == before["jobs"]


@pytest.mark.parametrize(
    "mutation",
    [
        "valid",
        "tampered_freeze",
        "tampered_source",
        "changed_input",
        "altered_contract",
        "reservation",
        "unrelated_job",
        "wrong_transition",
        "competing_lock",
    ],
)
def test_fixed_setup_recovery_consumed_by_first_claim(tmp_path, monkeypatch, mutation):
    arguments, _, _, _ = registered_quota_fixture(tmp_path, monkeypatch)
    ledger = arguments["ledger"]
    state = json.loads(ledger.read_text())
    old_configuration = state["stage_configurations"]["final"]
    new_configuration = old_configuration | {"evaluation_fingerprint": "successor"}
    input_file = tmp_path / "record.jsonl"
    input_file.write_text("synthetic record")
    input_hash = hashlib.sha256(input_file.read_bytes()).hexdigest()
    monkeypatch.setattr(budget_module, "_FIXED_INPUT", input_hash)
    old_contract = {
        "scenario": {"id": "contextual_mamba_visualization"},
        "source_records_file_sha256": input_hash,
        "dimension": "graph_usefulness",
        "provider": "openrouter",
        "model": "xingchen-agi/xing4.0-29b-a4b:free",
        "method": "json_schema",
        "judge_profile": "fixed-pilot-v1",
        "configuration": old_configuration,
    }
    new_contract = old_contract | {"configuration": new_configuration}
    old_digest = hashlib.sha256(
        json.dumps(old_contract, sort_keys=True).encode()
    ).hexdigest()
    new_digest = hashlib.sha256(
        json.dumps(new_contract, sort_keys=True).encode()
    ).hexdigest()
    old_job = "final:judge:semantic-v2:" + old_digest
    monkeypatch.setattr(budget_module, "_FIXED_SETUP_JOB", old_job)
    original_job = {
        "status": "started",
        "run_id": "01_contextual_mamba_visualization_graph_usefulness",
        "contract_digest": old_digest,
        "fixed_judge_pilot": True,
    }
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    run = evidence / "runs/01_contextual_mamba_visualization_graph_usefulness"
    run.mkdir(parents=True)
    (evidence / "ledger_after_registration.json").write_text(json.dumps(state))
    state["jobs"][old_job] = original_job
    (evidence / "ledger_after_setup_stop.json").write_text(json.dumps(state))
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
    evidence_files = [p for p in evidence.rglob("*") if p.is_file()]
    freeze = {
        "files": {
            p.relative_to(evidence).as_posix(): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in evidence_files
        }
    }
    freeze_path = evidence / "pilot_setup_stop_freeze.json"
    freeze_path.write_text(json.dumps(freeze))
    monkeypatch.setattr(
        budget_module,
        "_FIXED_SETUP_FREEZE",
        hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
    )
    parent = tmp_path / "parent"
    parent.mkdir()
    source = parent / "safe.txt"
    source.write_text("source")
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "safe.txt": {
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "bytes": len(source.read_bytes()),
                }
            }
        )
    )
    monkeypatch.setattr(
        budget_module,
        "_FIXED_PARENT_INVENTORY",
        hashlib.sha256(inventory.read_bytes()).hexdigest(),
    )
    pair = tmp_path / "pair.json"
    pair.write_text(json.dumps({"old": old_contract, "new": new_contract}))
    state["stage_configurations"]["final"] = new_configuration
    state.setdefault("final_identity_transitions", []).append(
        {
            "transition": "fixed-judge-quota-recovery",
            "source_proof": {"fingerprint": "successor"},
        }
    )
    if mutation == "tampered_freeze":
        (evidence / "01_invocation.log").write_text("changed")
    if mutation == "tampered_source":
        source.write_text("changed")
    if mutation == "changed_input":
        input_file.write_text("changed")
    if mutation == "altered_contract":
        pair.write_text(
            json.dumps(
                {"old": old_contract, "new": new_contract | {"dimension": "other"}}
            )
        )
    if mutation == "reservation":
        state["reservations"][0]["job_id"] = old_job
    if mutation == "unrelated_job":
        state["jobs"]["unrelated"] = {"status": "started"}
    if mutation == "wrong_transition":
        state["final_identity_transitions"][-1]["transition"] = "other"
    ledger.write_text(json.dumps(state, sort_keys=True))
    digest = hashlib.sha256(ledger.read_bytes()).hexdigest()
    kwargs = {
        "checkout": arguments["checkout"],
        "ledger": ledger,
        "expected_ledger_sha256": digest,
        "evidence_dir": evidence,
        "parent_source": parent,
        "parent_inventory": inventory,
        "input_file": input_file,
        "contract_pair": pair,
    }
    if mutation != "valid":
        saved = ledger.read_bytes()
        lock = None
        if mutation == "competing_lock":
            import fcntl

            lock = ledger.with_suffix(".lock").open("a+")
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with pytest.raises(budget_module.EvaluationBudgetUnavailable):
                budget_module.recover_fixed_judge_setup(**kwargs)
        finally:
            if lock is not None:
                lock.close()
        assert ledger.read_bytes() == saved
        return
    budget_module.recover_fixed_judge_setup(**kwargs)
    with pytest.raises(budget_module.EvaluationBudgetUnavailable):
        budget_module.recover_fixed_judge_setup(**kwargs)
    guard = budget_module.EvaluationBudget(
        ledger,
        stage="final",
        run_id="one",
        quota=quota(demo_reserve=20, daily_batch_limit=50),
        configuration=new_configuration,
    )
    guard.fixed_judge_pilot = True
    with guard:
        assert (
            guard.begin_job(
                "judge:semantic-v2:" + new_digest,
                contract_digest=new_digest,
                require_setup_recovery=True,
            )
            is None
        )
    with budget_module.EvaluationBudget(
        ledger,
        stage="final",
        run_id="two",
        quota=quota(demo_reserve=20, daily_batch_limit=50),
        configuration=new_configuration,
    ) as again:
        again.fixed_judge_pilot = True
        with pytest.raises(
            budget_module.EvaluationBudgetUnavailable, match="uncertain"
        ):
            again.begin_job(
                "judge:semantic-v2:" + new_digest,
                contract_digest=new_digest,
                require_setup_recovery=True,
            )
    final = json.loads(ledger.read_text())
    assert final["jobs"][old_job] == original_job
    assert final["fixed_judge_recoveries"][0]["consumed"] is True
    assert final["reservations"] == state["reservations"]


def test_fixed_judge_successor_transition_appends_without_rewriting_history(
    tmp_path, monkeypatch
):
    transition, before = final_transition_fixture(tmp_path, monkeypatch)
    path = transition["ledger"]
    before.setdefault("final_identity_transitions", []).append(
        {"transition": "quota-input-lifecycle", "prior": "preserved"}
    )
    path.write_text(json.dumps(before, sort_keys=True))
    transition["expected_ledger_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    transition["fixed_judge_successor"] = True
    monkeypatch.setattr(
        budget_module,
        "_FIXED_PARENT_FINGERPRINT",
        before["stage_configurations"]["final"]["evaluation_fingerprint"],
    )
    monkeypatch.setattr(
        budget_module,
        "_FIXED_PARENT_INVENTORY",
        transition["expected_previous_inventory_sha256"],
    )
    assert budget_module.transition_final_evaluation_identity(**transition)
    after = json.loads(path.read_text())
    assert (
        after["final_identity_transitions"][0]
        == before["final_identity_transitions"][0]
    )
    assert (
        after["final_identity_transitions"][-1]["transition"]
        == "fixed-judge-quota-recovery"
    )
    for key in ("reservations", "jobs", "days", "policy_amendments"):
        assert after[key] == before[key]
    assert after["stage_configurations"]["B0"] == before["stage_configurations"]["B0"]
    saved = path.read_bytes()
    assert not budget_module.transition_final_evaluation_identity(**transition)
    assert path.read_bytes() == saved
