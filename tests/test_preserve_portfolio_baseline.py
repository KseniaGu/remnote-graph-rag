import json
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from backend.evaluation.portfolio_budget import (
    EvaluationBudget,
    EvaluationBudgetUnavailable,
    campaign_ledger,
)
from scripts.preserve_portfolio_baseline import safe_path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/preserve_portfolio_baseline.py"


def run(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True
    )


def test_capture_restore_preserves_binary_diff_and_rejects_tampered_assets(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    (source / "payload.bin").write_bytes(b"original\0bytes")
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    (source / "payload.bin").write_bytes(b"changed\0bytes")
    assets = tmp_path / "assets"
    (assets / "storage").mkdir(parents=True)
    (assets / "storage/data.json").write_text('{"private": true}')
    synthetic_secret = tmp_path / "synthetic-secret.fixture"
    synthetic_secret.write_text("SECRET=must-never-copy")
    synthetic_secret.replace(assets / ".env")
    snapshot = tmp_path / "snapshot"
    result = run(
        "capture",
        "--source-root",
        source,
        "--asset-root",
        assets,
        "--asset",
        "storage",
        "--output",
        snapshot,
    )
    assert result.returncode == 0, result.stderr
    destination = tmp_path / "restored"
    result = run("restore", "--snapshot", snapshot, "--destination", destination)
    assert result.returncode == 0, result.stderr
    with pytest.raises(EvaluationBudgetUnavailable, match="--budget-ledger"):
        campaign_ledger(destination)
    assert (destination / "payload.bin").read_bytes() == b"changed\0bytes"
    assert (destination / "storage/data.json").read_text() == '{"private": true}'
    assert not (destination / ".env").exists()
    manifest = json.loads((snapshot / "manifest.json").read_text())
    canonical = Path(manifest["canonical_ledger_binding"])
    assert canonical == source / "data/evaluation/portfolio_budget/ledger.json"
    assert not canonical.exists()
    observation = {
        "observed_at": datetime.now(UTC).isoformat(),
        "provider": "openrouter",
        "model": "openrouter/free",
        "remaining_requests": 50,
        "demo_reserve": 20,
        "daily_batch_limit": 12,
    }
    with EvaluationBudget(canonical, stage="B0", run_id="fixture", quota=observation):
        pass
    assert campaign_ledger(source, override=canonical) == canonical
    assert campaign_ledger(destination, override=canonical) == canonical
    alternative = tmp_path / "other.json"
    alternative.write_bytes(canonical.read_bytes())
    with pytest.raises(EvaluationBudgetUnavailable, match="canonical"):
        campaign_ledger(destination, override=alternative)
    assert (
        run("verify", "--snapshot", snapshot, "--restored-root", destination).returncode
        == 0
    )
    (destination / "payload.bin").write_bytes(b"corruption")
    assert (
        run("verify", "--snapshot", snapshot, "--restored-root", destination).returncode
        != 0
    )
    (snapshot / "assets/storage/data.json").write_text("tampered")
    assert run("verify", "--snapshot", snapshot).returncode != 0


def test_capture_rejects_credential_path_before_archive_creation(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    result = run(
        "capture",
        "--source-root",
        source,
        "--asset-root",
        source,
        "--asset",
        ".env",
        "--output",
        tmp_path / "snapshot",
    )
    assert result.returncode != 0
    assert "credential-bearing snapshot path" in result.stderr
    assert not (tmp_path / "snapshot/assets/.env").exists()


def test_offline_guard_blocks_internet_socket_before_target_import(tmp_path):
    target = tmp_path / "probe.py"
    target.write_text(
        'import socket\ntry:\n socket.socket(socket.AF_INET, socket.SOCK_STREAM)\nexcept PermissionError:\n print("network-blocked")\nelse:\n raise AssertionError("network guard absent")\n'
    )
    result = run("offline-run", str(target))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "network-blocked"


def test_capture_refuses_untracked_source_that_binary_diff_would_omit(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    (source / "tracked.py").write_text("original = True\n")
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    (source / "new.py").write_text("would_be_omitted = True\n")
    (source / "storage").mkdir()
    result = run(
        "capture",
        "--source-root",
        source,
        "--asset-root",
        source,
        "--asset",
        "storage",
        "--output",
        tmp_path / "snapshot",
    )
    assert result.returncode != 0
    assert "Stage untracked source files before preservation" in result.stderr


def test_capture_rejects_git_metadata_inside_asset_directory(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    (source / "tracked.txt").write_text("source")
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    assets = tmp_path / "assets"
    (assets / "storage/.git").mkdir(parents=True)
    (assets / "storage/.git/config").write_text("credential-bearing metadata")
    result = run(
        "capture",
        "--source-root",
        source,
        "--asset-root",
        assets,
        "--asset",
        "storage",
        "--output",
        tmp_path / "snapshot",
    )
    assert result.returncode != 0
    assert "credential-bearing snapshot path" in result.stderr


def test_capture_rejects_credential_file_in_head_even_when_staged_deleted(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    (source / "tracked.txt").write_text("source")
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    synthetic_blob = (
        subprocess.check_output(
            ["git", "-C", str(source), "hash-object", "-w", "--stdin"],
            input=b"SYNTHETIC_SECRET=fixture-only",
        )
        .decode()
        .strip()
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "update-index",
            "--add",
            "--cacheinfo",
            f"100644,{synthetic_blob},.env",
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(source), "rm", "--cached", "-q", ".env"], check=True
    )
    assets = tmp_path / "assets"
    (assets / "storage").mkdir(parents=True)
    snapshot = tmp_path / "snapshot"
    result = run(
        "capture",
        "--source-root",
        source,
        "--asset-root",
        assets,
        "--asset",
        "storage",
        "--output",
        snapshot,
    )
    assert result.returncode != 0
    assert "credential-bearing snapshot path" in result.stderr
    assert not (snapshot / "source.tar").exists()


@pytest.mark.parametrize(
    ("relative", "source_file"),
    [
        (".env.example", False),
        ("nested/.env.example", True),
        (".env", True),
        (".env.production", True),
    ],
)
def test_public_template_exception_keeps_private_and_nested_dotenv_paths_forbidden(
    relative, source_file
):
    with pytest.raises(ValueError, match="credential-bearing snapshot path"):
        safe_path(relative, source_file=source_file)


def test_capture_preserves_only_root_public_source_template_from_git_objects(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    (source / "tracked.txt").write_text("source")
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    template = b"API_KEY=\nCONNECTION=sample-host\n"
    blob = (
        subprocess.check_output(
            ["git", "-C", str(source), "hash-object", "-w", "--stdin"], input=template
        )
        .decode()
        .strip()
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "update-index",
            "--add",
            "--cacheinfo",
            f"100644,{blob},.env.example",
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    assets = tmp_path / "assets"
    (assets / "storage").mkdir(parents=True)
    snapshot = tmp_path / "snapshot"
    result = run(
        "capture",
        "--source-root",
        source,
        "--asset-root",
        assets,
        "--asset",
        "storage",
        "--output",
        snapshot,
    )
    assert result.returncode == 0, result.stderr
    with tarfile.open(snapshot / "source.tar") as archive:
        assert archive.extractfile(".env.example").read() == template
    assert safe_path(".env.example", source_file=True) == Path(".env.example")
