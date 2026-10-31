import hashlib
import json
import shutil
from datetime import UTC, datetime

import pytest

from backend.evaluation import evaluation_reporting as reporting
from backend.evaluation import portfolio_budget as budget


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root, target):
    entries = {}
    for path in (root / "backend").rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            entries[path.relative_to(root).as_posix()] = {
                "sha256": digest(path),
                "bytes": path.stat().st_size,
            }
    target.write_text(json.dumps(entries, sort_keys=True))
    return digest(target)


def fingerprint(root, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(reporting, "ROOT", root)
        return reporting.build_evaluation_snapshot({})["fingerprint_input"]


def setup_transition(tmp_path, monkeypatch):
    before = tmp_path / "previous"
    after = tmp_path / "current"
    (before / "backend/evaluation").mkdir(parents=True)
    (before / "backend/evaluation/runtime_evaluation.py").write_text("old evaluator")
    (before / "backend/app.py").write_text("stable app")
    shutil.copytree(before, after)
    (after / ".git").mkdir()
    old_input = fingerprint(before, monkeypatch)
    old_config = {
        "evaluation_fingerprint": reporting._hash_json(old_input),
        "application_configuration": old_input["application_configuration"],
        "selected_prompt_tree_hash": "prompt",
        "selected_storage_manifest_hashes": {"storage": "unchanged"},
        "chat_limits": {"attempts": 6},
    }
    ledger = budget.campaign_ledger(after)
    quota = {
        "observed_at": datetime.now(UTC).isoformat(),
        "provider": "openrouter",
        "model": "openrouter/free",
        "remaining_requests": 100,
        "daily_batch_limit": 24,
        "demo_reserve": 20,
    }
    with budget.EvaluationBudget(
        ledger, stage="B0", run_id="prior", quota=quota, configuration=old_config
    ) as guard:
        guard.begin_job("cats:r1", contract_digest="uncertain")
        for _ in range(3):
            guard.reserve("llm", provider="openrouter", model="openrouter/free")
    (after / "backend/evaluation/runtime_evaluation.py").write_text("new evaluator")
    new_input = fingerprint(after, monkeypatch)
    old_inv, new_inv = tmp_path / "old.json", tmp_path / "new.json"
    args = {
        "checkout": after,
        "ledger": ledger,
        "previous_source_root": before,
        "expected_ledger_sha256": digest(ledger),
        "configuration": old_config
        | {"evaluation_fingerprint": reporting._hash_json(new_input)},
        "previous_inventory": old_inv,
        "expected_previous_inventory_sha256": inventory(before, old_inv),
        "current_inventory": new_inv,
        "expected_current_inventory_sha256": inventory(after, new_inv),
    }
    return args


def test_explicit_identity_transition_keeps_state_and_uncertain_job(
    tmp_path, monkeypatch
):
    args = setup_transition(tmp_path, monkeypatch)
    previous = json.loads(args["ledger"].read_bytes())
    assert budget.transition_b0_evaluation_identity(**args)
    after = json.loads(args["ledger"].read_bytes())
    for key in (
        "reservations",
        "jobs",
        "days",
        "caps",
        "daily_policy",
        "ledger_binding",
    ):
        assert after[key] == previous[key]
    assert after["identity_transitions"][0]["previous_state"] == previous
    assert after["stage_configurations"]["B0"] == args["configuration"]
    with pytest.raises(budget.EvaluationBudgetUnavailable, match="uncertain"):
        with budget.EvaluationBudget(
            args["ledger"],
            stage="B0",
            run_id="resume",
            quota={
                "observed_at": datetime.now(UTC).isoformat(),
                "provider": "openrouter",
                "model": "openrouter/free",
                "remaining_requests": 100,
                "daily_batch_limit": 24,
                "demo_reserve": 20,
            },
            configuration=args["configuration"],
        ) as guard:
            guard.begin_job("cats:r1", contract_digest="uncertain")


@pytest.mark.parametrize("mutation", ["ledger", "app", "inventory", "config", "extra"])
def test_transition_rejects_unreviewed_changes_without_writing(
    tmp_path, monkeypatch, mutation
):
    args = setup_transition(tmp_path, monkeypatch)
    if mutation == "ledger":
        args["expected_ledger_sha256"] = "0" * 64
    if mutation == "app":
        (args["checkout"] / "backend/app.py").write_text("changed app")
        args["expected_current_inventory_sha256"] = inventory(
            args["checkout"], args["current_inventory"]
        )
    if mutation == "inventory":
        args["expected_current_inventory_sha256"] = "0" * 64
    if mutation == "config":
        args["configuration"]["chat_limits"] = {"attempts": 7}
    if mutation == "extra":
        (args["checkout"] / "backend/extra.py").write_text("unreviewed")
    original = args["ledger"].read_bytes()
    with pytest.raises(budget.EvaluationBudgetUnavailable):
        budget.transition_b0_evaluation_identity(**args)
    assert args["ledger"].read_bytes() == original


def test_transition_rejects_copied_ledger_and_repeat(tmp_path, monkeypatch):
    args = setup_transition(tmp_path, monkeypatch)
    copied = tmp_path / "copied-ledger.json"
    copied.write_bytes(args["ledger"].read_bytes())
    with pytest.raises(budget.EvaluationBudgetUnavailable, match="canonical"):
        budget.transition_b0_evaluation_identity(**(args | {"ledger": copied}))
    assert budget.transition_b0_evaluation_identity(**args)
    saved = args["ledger"].read_bytes()
    with pytest.raises(budget.EvaluationBudgetUnavailable, match="ledger SHA256"):
        budget.transition_b0_evaluation_identity(**args)
    assert args["ledger"].read_bytes() == saved


def test_transition_cli_requires_explicit_confirmation(tmp_path, capsys):
    from scripts.transition_b0_evaluation_identity import main

    value = str(tmp_path / "unused")
    assert (
        main(
            [
                "--budget-ledger",
                value,
                "--expected-ledger-sha256",
                "0" * 64,
                "--previous-source-root",
                value,
                "--previous-inventory",
                value,
                "--expected-previous-inventory-sha256",
                "0" * 64,
                "--current-inventory",
                value,
                "--expected-current-inventory-sha256",
                "0" * 64,
                "--configuration",
                value,
            ]
        )
        == 2
    )
    assert "--confirm-b0-evaluator-transition" in capsys.readouterr().err
