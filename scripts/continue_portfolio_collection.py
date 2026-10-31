"""Continue the frozen pending portfolio runs with locked admission."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

BASE = Path("/workspaces/remnote-graph-rag/.worktrees/refactoring-m2")
PREPARATION = BASE / "data/evaluation/portfolio_baseline/final_preparation_20261030"
FREEZE = BASE / "data/evaluation/portfolio_baseline/quota_lifecycle_20261030"
SOURCE = FREEZE / "source"
OUTPUT = (
    BASE / "data/evaluation/portfolio_baseline/daily_policy50_20261030_v4/continuation"
)
LEDGER = Path(
    "/workspaces/remnote-graph-rag/data/evaluation/portfolio_budget/ledger.json"
)


def completed_record(invocation: Path) -> bool:
    """Require one matching record and a completed manifest for a run."""
    try:
        record = json.loads(invocation.read_text())
        output = OUTPUT / "runs" / record["output_name"]
        if output.parent != OUTPUT / "runs" or not output.name:
            return False
        manifest = json.loads((output / "manifest.json").read_text())
        lines = (output / "runtime_records.jsonl").read_text().splitlines()
        records = [json.loads(line) for line in lines if line.strip()]
        return (
            manifest.get("status") == "completed"
            and len(records) == 1
            and records[0].get("scenario_id") == record["case"]
            and records[0].get("repetition") == record["repetition"]
            and records[0].get("scenario_version") == record["version"]
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def check_pending(number: int) -> None:
    """Allow only ordered runs and retries proven to precede dispatch."""
    if not 8 <= number <= 15:
        raise ValueError("run is not a pending observation")
    for current in range(8, number + 1):
        invocations = sorted(OUTPUT.glob(f"{current:02}_attempt_*_invocation.json"))
        if current < number and not invocations:
            raise ValueError("previous pending observation is incomplete or failed")
        completed = False
        for index, invocation in enumerate(invocations, start=1):
            prefix = f"{current:02}_attempt_{index:03}"
            if invocation.name != f"{prefix}_invocation.json":
                raise ValueError("uncertain continuation history")
            exit_path = OUTPUT / f"{prefix}_exit.json"
            receipt_path = OUTPUT / f"{prefix}_pre_dispatch.json"
            if not exit_path.is_file():
                raise ValueError("uncertain continuation history")
            status = json.loads(exit_path.read_text())["exit_code"]
            if (
                status == 0
                and index == len(invocations)
                and not receipt_path.exists()
                and completed_record(invocation)
            ):
                completed = True
                continue
            if (
                status == 0
                or not receipt_path.is_file()
                or json.loads(receipt_path.read_text()).get("dispatch_started")
                is not False
            ):
                raise ValueError("uncertain continuation history")
        if current < number and not completed:
            raise ValueError("previous pending observation is incomplete or failed")
        if current == number and completed:
            raise ValueError("run is not an untouched pending observation")


def verify_assets(
    root: Path, provenance: Path, expected_sha256: str, expected_count: int
) -> None:
    """Rehash the captured model and storage files before a future dispatch."""
    data = provenance.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("asset provenance differs")
    files = json.loads(data)["files"]
    selected = {
        name: entry
        for name, entry in files.items()
        if name.startswith(("models/", "storage/"))
    }
    if len(selected) != expected_count:
        raise ValueError("asset inventory count differs")
    actual = {
        path.relative_to(root).as_posix()
        for name in ("models", "storage")
        for path in (root / name).rglob("*")
        if path.is_file()
    }
    if actual != set(selected):
        raise ValueError("asset inventory membership differs")
    for name, entry in selected.items():
        relative = Path(name)
        path = root / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(root.resolve())
        ):
            raise ValueError("invalid asset path")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if entry != {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}:
            raise ValueError("asset contents differ")


def verify_preflight(expected_freeze_sha256: str, env: dict[str, str]) -> None:
    """Verify a separately reviewed freeze and effective configuration offline."""
    freeze_bytes = (FREEZE / "freeze.json").read_bytes()
    if hashlib.sha256(freeze_bytes).hexdigest() != expected_freeze_sha256:
        raise ValueError("reviewed freeze SHA256 differs")
    freeze = json.loads(freeze_bytes)
    files = {
        FREEZE / "source_manifest.json": freeze["source_inventory_sha256"],
        FREEZE / "previous_source_manifest.json": freeze["previous_inventory_sha256"],
        FREEZE / "policy.patch": freeze["patch_sha256"],
        FREEZE / "effective_configuration.json": freeze["configuration_sha256"],
        PREPARATION / "freeze_manifest.json": freeze["prior_manifest_sha256"],
        PREPARATION / "explicit_environment_v2.json": freeze["environment_sha256"],
        PREPARATION / "effective_config_probe.py": freeze["probe_sha256"],
        SOURCE / ".git/portfolio-restored-source": freeze["restoration_marker_sha256"],
        BASE
        / "data/evaluation/portfolio_baseline/snapshots/B0_CatsEvaluator_7ad22d7_7cbca485/manifest.json": freeze[
            "snapshot_manifest_sha256"
        ],
    }
    for path, expected in files.items():
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError("reviewed freeze file differs")
    preparation = BASE / "data/evaluation/portfolio_baseline/preparation"
    prior = json.loads((PREPARATION / "freeze_manifest.json").read_text())
    provenance = preparation / "asset_provenance.json"
    verify_assets(
        preparation / "assets",
        provenance,
        prior["references"][str(provenance)],
        prior["asset_files_verified"],
    )
    inventory = json.loads((FREEZE / "source_manifest.json").read_text())
    marker = json.loads((SOURCE / ".git/portfolio-restored-source").read_text())
    if (
        marker.get("canonical_ledger") != str(LEDGER)
        or marker.get("snapshot_manifest_sha256") != freeze["snapshot_manifest_sha256"]
    ):
        raise ValueError("restoration ledger binding differs")
    own = Path(__file__).read_bytes()
    if inventory.get("scripts/continue_portfolio_collection.py") != {
        "sha256": hashlib.sha256(own).hexdigest(),
        "bytes": len(own),
    }:
        raise ValueError("successor differs from reviewed frozen source")
    for name, entry in inventory.items():
        relative = Path(name)
        path = SOURCE / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(SOURCE.resolve())
        ):
            raise ValueError("invalid frozen source path")
        data = path.read_bytes()
        if entry != {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}:
            raise ValueError("frozen source differs")
    actual = {
        path.relative_to(SOURCE).as_posix()
        for path in SOURCE.rglob("*")
        if path.is_file()
        and ".git" not in path.relative_to(SOURCE).parts
        and "__pycache__" not in path.relative_to(SOURCE).parts
        and path.suffix not in {".pyc", ".pyo", ".tmp"}
        and path.name != ".DS_Store"
    }
    if actual != set(inventory):
        raise ValueError("unreviewed frozen source files")
    probe = subprocess.run(
        [
            str(BASE / ".venv/bin/python"),
            str(PREPARATION / "effective_config_probe.py"),
            str(SOURCE),
        ],
        cwd=SOURCE,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    actual_configuration = json.loads(probe.stdout)["budget_configuration"]
    expected_configuration = json.loads(
        (FREEZE / "effective_configuration.json").read_text()
    )
    if actual_configuration != expected_configuration:
        raise ValueError("effective configuration differs from reviewed freeze")


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) not in {4, 6} or (
        len(arguments) == 6 and arguments[4] != "--quota-snapshot"
    ):
        raise ValueError(
            "expected run number, scenario ID, repetition, reviewed freeze SHA256 and optional --quota-snapshot"
        )
    number, case, repetition = int(arguments[0]), arguments[1], int(arguments[2])
    check_pending(number)
    manifest = json.loads((PREPARATION / "freeze_manifest.json").read_text())
    item = manifest["ordered_runs"][number - 1]
    if (item["id"], item["repetition"]) != (case, repetition):
        raise ValueError("run differs from the frozen manifest")
    quota_path = Path(arguments[5]) if len(arguments) == 6 else OUTPUT / "quota.json"
    if (
        not quota_path.is_absolute()
        or quota_path.is_symlink()
        or not quota_path.is_file()
    ):
        raise ValueError("fresh quota metadata is required")
    env = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ["HOME"],
        "GIT_OPTIONAL_LOCKS": "0",
        **json.loads((PREPARATION / "explicit_environment_v2.json").read_text()),
    }
    for key in (
        "OPENROUTER_API_KEY",
        "TAVILY_API_KEY",
        "PINECONE_API_KEY",
        "MONGODB_URI",
    ):
        if key in os.environ:
            env[key] = os.environ[key]
    attempt = len(list(OUTPUT.glob(f"{number:02}_attempt_*_invocation.json"))) + 1
    verify_preflight(arguments[3], env)
    scenarios = json.loads((SOURCE / "evals/runtime/scenarios.json").read_text())
    matches = [scenario for scenario in scenarios if scenario.get("id") == case]
    if len(matches) != 1 or matches[0].get("version") != item["version"]:
        raise ValueError("scenario contract differs from the frozen manifest")
    expectations = matches[0]["expectations"]
    needs_web = "deep_web_research" in expectations.get("required_tools", [])
    if needs_web:
        web_limit = expectations.get("budgets", {}).get("max_tavily_searches")
        if type(web_limit) is not int or web_limit < 1:
            raise ValueError("required web tool lacks an approved search budget")
    label = f"{number:02}_attempt_{attempt:03}"
    snapshot = OUTPUT / f"{label}_quota.json"
    quota_bytes = quota_path.read_bytes()
    with snapshot.open("xb") as handle:
        handle.write(quota_bytes)
    receipt = OUTPUT / f"{label}_pre_dispatch.json"
    output_name = f"{label}_{case}"
    command = [
        str(BASE / ".venv/bin/python"),
        "scripts/evaluate_runtime_workflow.py",
        "live",
        "--case-id",
        case,
        "--repetition",
        str(repetition),
        "--capture-diagnostics",
        "--confirm-provider-calls",
        "--budget-stage",
        "final",
        "--budget-ledger",
        str(LEDGER),
        "--provider-quota",
        str(snapshot),
        "--require-final-runtime-admission",
        "--require-registered-quota",
        "--pre-dispatch-receipt",
        str(receipt),
        "--output-dir",
        str(OUTPUT / "runs" / output_name),
        "--evaluation-root",
        str(OUTPUT / "evaluation"),
    ]
    if needs_web:
        command.append("--allow-tavily")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    invocation = OUTPUT / f"{label}_invocation.json"
    with invocation.open("x", encoding="utf-8") as handle:
        json.dump(
            {
                "argv": command,
                "case": case,
                "repetition": repetition,
                "version": item["version"],
                "output_name": output_name,
                "started_at": datetime.now(UTC).isoformat(),
                "quota_sha256": hashlib.sha256(quota_bytes).hexdigest(),
                "quota_source_path": str(quota_path),
                "ledger_sha256_before": hashlib.sha256(LEDGER.read_bytes()).hexdigest(),
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    with (OUTPUT / f"{label}.log").open("x", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=SOURCE,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    (OUTPUT / f"{label}_exit.json").write_text(
        json.dumps(
            {"exit_code": result.returncode, "ended_at": datetime.now(UTC).isoformat()},
            indent=2,
        )
        + "\n"
    )
    if result.returncode == 0 and not completed_record(invocation):
        return 2
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
