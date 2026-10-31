"""Offline checks for pending-only portfolio continuation."""

import pytest

from scripts import continue_portfolio_collection as continuation


def test_successor_refuses_repeating_completed_run(tmp_path, monkeypatch):
    monkeypatch.setattr(continuation, "OUTPUT", tmp_path)
    with pytest.raises(ValueError, match="pending"):
        continuation.check_pending(7)
    assert not list(tmp_path.iterdir())


def test_successor_requires_previous_pending_run_success(tmp_path, monkeypatch):
    monkeypatch.setattr(continuation, "OUTPUT", tmp_path)
    with pytest.raises(ValueError, match="previous"):
        continuation.check_pending(9)
    assert not list(tmp_path.iterdir())


def test_successor_invokes_single_frozen_run_with_locked_admission(
    tmp_path, monkeypatch
):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    preparation = tmp_path / "preparation"
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
    output = tmp_path / "continuation"
    output.mkdir()
    (output / "quota.json").write_text("{}")
    ledger = tmp_path / "ledger.json"
    ledger.write_text("{}")
    source = tmp_path / "source"
    source.mkdir()
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
    monkeypatch.setattr(continuation, "BASE", tmp_path)
    monkeypatch.setattr(continuation, "PREPARATION", preparation)
    monkeypatch.setattr(continuation, "OUTPUT", output)
    monkeypatch.setattr(continuation, "SOURCE", source)
    monkeypatch.setattr(continuation, "LEDGER", ledger)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
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
    monkeypatch.setattr(continuation, "verify_preflight", lambda digest, env: None)
    assert continuation.main(["8", "case-8", "1", "approved"]) == 0
    command, kwargs = calls[0]
    assert "--require-final-runtime-admission" in command
    assert kwargs["cwd"] == source
    assert command[command.index("--budget-ledger") + 1] == str(ledger)
    assert (
        json.loads((output / "08_attempt_001_exit.json").read_text())["exit_code"] == 0
    )


def test_successor_allows_only_proven_pre_dispatch_retry(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(continuation, "OUTPUT", tmp_path)
    (tmp_path / "08_attempt_001_invocation.json").write_text("{}")
    with pytest.raises(ValueError, match="uncertain"):
        continuation.check_pending(8)
    (tmp_path / "08_attempt_001_exit.json").write_text(json.dumps({"exit_code": 2}))
    (tmp_path / "08_attempt_001_pre_dispatch.json").write_text(
        json.dumps({"dispatch_started": False})
    )
    continuation.check_pending(8)


def test_successor_rejects_unreviewed_freeze_before_any_attempt(tmp_path, monkeypatch):
    freeze = tmp_path / "freeze"
    freeze.mkdir()
    (freeze / "freeze.json").write_text("{}")
    monkeypatch.setattr(continuation, "FREEZE", freeze)
    with pytest.raises(ValueError, match="freeze"):
        continuation.verify_preflight("0" * 64, {})
    assert sorted(path.name for path in freeze.iterdir()) == ["freeze.json"]


def test_successor_does_not_treat_empty_success_as_completed(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(continuation, "OUTPUT", tmp_path)
    (tmp_path / "08_attempt_001_invocation.json").write_text("{}")
    (tmp_path / "08_attempt_001_exit.json").write_text(json.dumps({"exit_code": 0}))
    with pytest.raises(ValueError, match="uncertain"):
        continuation.check_pending(9)


def test_asset_rehash_rejects_changed_model_or_storage(tmp_path):
    import hashlib
    import json

    assets = tmp_path / "assets"
    for name in ("models/example/config.json", "storage/docstore.json"):
        path = assets / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    inventory = {
        name: {
            "bytes": len((assets / name).read_bytes()),
            "sha256": hashlib.sha256((assets / name).read_bytes()).hexdigest(),
        }
        for name in ("models/example/config.json", "storage/docstore.json")
    }
    provenance = tmp_path / "assets.json"
    provenance.write_text(json.dumps({"files": inventory}))
    expected = hashlib.sha256(provenance.read_bytes()).hexdigest()
    continuation.verify_assets(assets, provenance, expected, 2)
    (assets / "storage/docstore.json").write_text("changed")
    with pytest.raises(ValueError, match="asset"):
        continuation.verify_assets(assets, provenance, expected, 2)


@pytest.mark.parametrize(
    ("number", "case", "version", "repetition", "allow_web"),
    [
        (8, "colbert_guidance_mode_continuation", 1, 1, False),
        (10, "ambiguous_eagle_without_history", 2, 1, True),
        (11, "explicit_medusa_web_fallback_success", 1, 1, True),
        (12, "in_scope_retnet_local_miss_fallback", 3, 1, True),
        (13, "researcher_structured_output_truncation_reliability", 2, 1, True),
        (14, "researcher_structured_output_truncation_reliability", 2, 2, True),
        (15, "researcher_structured_output_truncation_reliability", 2, 3, True),
    ],
)
def test_successor_command_uses_frozen_scenario_web_policy(
    tmp_path, monkeypatch, number, case, version, repetition, allow_web
):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    preparation = tmp_path / "preparation"
    preparation.mkdir()
    ordered = [
        {"id": f"unused-{i}", "version": 1, "repetition": 1} for i in range(1, 16)
    ]
    ordered[number - 1] = {"id": case, "version": version, "repetition": repetition}
    (preparation / "freeze_manifest.json").write_text(
        json.dumps({"ordered_runs": ordered})
    )
    (preparation / "explicit_environment_v2.json").write_text("{}")
    source = tmp_path / "source"
    scenarios = source / "evals/runtime/scenarios.json"
    scenarios.parent.mkdir(parents=True)
    tracked_scenarios = (
        Path(__file__).resolve().parents[1] / "evals/runtime/scenarios.json"
    )
    scenarios.write_bytes(tracked_scenarios.read_bytes())
    output = tmp_path / "continuation"
    output.mkdir()
    (output / "quota.json").write_text("frozen-original")
    independent_quota = tmp_path / "separate-quota.json"
    independent_quota.write_text("fresh-separate")
    ledger = tmp_path / "ledger.json"
    ledger.write_text("{}")
    monkeypatch.setattr(continuation, "BASE", tmp_path)
    monkeypatch.setattr(continuation, "PREPARATION", preparation)
    monkeypatch.setattr(continuation, "SOURCE", source)
    monkeypatch.setattr(continuation, "OUTPUT", output)
    monkeypatch.setattr(continuation, "LEDGER", ledger)
    monkeypatch.setattr(continuation, "check_pending", lambda number: None)
    monkeypatch.setattr(continuation, "verify_preflight", lambda digest, env: None)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        run_output = Path(command[command.index("--output-dir") + 1])
        run_output.mkdir(parents=True)
        (run_output / "manifest.json").write_text(json.dumps({"status": "completed"}))
        (run_output / "runtime_records.jsonl").write_text(
            json.dumps(
                {
                    "scenario_id": case,
                    "repetition": repetition,
                    "scenario_version": version,
                }
            )
            + "\n"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(continuation.subprocess, "run", run)
    assert (
        continuation.main(
            [
                str(number),
                case,
                str(repetition),
                "approved",
                "--quota-snapshot",
                str(independent_quota),
            ]
        )
        == 0
    )
    assert ("--allow-tavily" in calls[0]) is allow_web
    assert "--require-registered-quota" in calls[0]
    assert (output / "quota.json").read_text() == "frozen-original"
    assert (
        Path(calls[0][calls[0].index("--provider-quota") + 1]).read_text()
        == "fresh-separate"
    )


def test_successor_uses_separate_quota_snapshot_without_changing_frozen_input(
    tmp_path, monkeypatch
):
    import hashlib
    import json
    from datetime import UTC, datetime
    from pathlib import Path
    from types import SimpleNamespace

    preparation = tmp_path / "preparation"
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
    output = tmp_path / "continuation"
    output.mkdir()
    old = output / "quota.json"
    old.write_text('{"frozen":true}\n')
    old_bytes = old.read_bytes()
    snapshot = tmp_path / "fresh.json"
    q = {
        "observed_at": datetime.now(UTC).isoformat(),
        "provider": "openrouter",
        "model": "openrouter/free",
        "remaining_requests": 31,
        "demo_reserve": 20,
        "daily_batch_limit": 50,
    }
    snapshot.write_text(json.dumps(q))
    ledger = tmp_path / "ledger.json"
    identity = hashlib.sha256(
        json.dumps(
            {key: q[key] for key in ("observed_at", "provider", "remaining_requests")},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    captured = {
        "basis": "prospective-authority",
        "expected_account_id": "account-1",
        "expected_key_id": "key-1",
    }
    registration = captured | {
        "observation_id": identity,
        "baseline": 23,
        "proof": {"quota": q},
    }
    state = {
        "days": {q["observed_at"][:10]: {"used": 23, "observations": {identity: 23}}},
        "policy_amendments": [
            {
                "amendment": "daily-policy-50-20261030",
                "day_change": {"source_proof": captured},
            }
        ],
        "quota_observation_registrations": [
            {"day": q["observed_at"][:10], "source_proof": registration}
        ],
    }
    ledger.write_text(json.dumps(state))
    source = tmp_path / "source"
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
    for key, value in (
        ("BASE", tmp_path),
        ("PREPARATION", preparation),
        ("SOURCE", source),
        ("OUTPUT", output),
        ("LEDGER", ledger),
    ):
        monkeypatch.setattr(continuation, key, value)
    monkeypatch.setattr(continuation, "check_pending", lambda number: None)
    monkeypatch.setattr(continuation, "verify_preflight", lambda digest, env: None)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
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
    assert (
        continuation.main(
            ["8", "case-8", "1", "approved", "--quota-snapshot", str(snapshot)]
        )
        == 0
    )
    assert old.read_bytes() == old_bytes
    passed_path = Path(calls[0][calls[0].index("--provider-quota") + 1])
    assert passed_path.read_bytes() == snapshot.read_bytes()
