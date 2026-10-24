import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "event", ["socket.connect", "socket.getaddrinfo", "socket.sendto"]
)
def test_startup_guard_records_blocked_network_in_child_python(tmp_path, event):
    log = tmp_path / "network.log"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            # Emit the real audit event without a network operation, even if broken.
            "import sys; "
            f"sys.audit({event!r}, None); "
            "raise AssertionError('network event was allowed')",
        ],
        env={
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(tmp_path),
            "PYTHONPATH": str(ROOT / "tests" / "offline_guard"),
            "OFFLINE_NETWORK_LOG": str(log),
            "OFFLINE_REPOSITORY_ROOT": str(ROOT),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "PermissionError: Offline tests forbid network access" in result.stderr
    assert log.read_text() == f"{event}\n"


def test_startup_guard_denies_dotenv_before_content_is_read(tmp_path):
    dotenv = tmp_path / ".env"
    synthetic = tmp_path / "synthetic"
    synthetic.write_text("SYNTHETIC_OFFLINE_TEST_VALUE=example\n")
    synthetic.rename(dotenv)
    result = subprocess.run(
        [sys.executable, "-c", "import sys; open(sys.argv[1]).read()", str(dotenv)],
        env={
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(tmp_path),
            "PYTHONPATH": str(ROOT / "tests" / "offline_guard"),
            "OFFLINE_NETWORK_LOG": str(tmp_path / "network.log"),
            "OFFLINE_REPOSITORY_ROOT": str(ROOT),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "PermissionError: Offline tests forbid dotenv file access" in result.stderr
    assert "SYNTHETIC_OFFLINE_TEST_VALUE" not in result.stdout + result.stderr


@pytest.mark.parametrize("asset", ["data", "storage", "models"])
@pytest.mark.parametrize("alias_kind", ["traversal", "symlink"])
def test_startup_guard_denies_private_asset_path_aliases(tmp_path, asset, alias_kind):
    root = tmp_path / "repository"
    (root / asset).mkdir(parents=True)
    (root / "tests").mkdir()
    if alias_kind == "traversal":
        probe = root / "tests" / ".." / asset / "synthetic-probe"
    else:
        alias = tmp_path / "alias"
        alias.symlink_to(root / asset, target_is_directory=True)
        probe = alias / "synthetic-probe"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.audit('open', sys.argv[1], 'r', 0)",
            str(probe),
        ],
        env={
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(tmp_path),
            "PYTHONPATH": str(ROOT / "tests" / "offline_guard"),
            "OFFLINE_NETWORK_LOG": str(tmp_path / "network.log"),
            "OFFLINE_REPOSITORY_ROOT": str(root),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "PermissionError: Offline tests forbid private asset access" in result.stderr


@pytest.mark.parametrize("target_name", [".env", ".env.local"])
def test_startup_guard_denies_symlink_to_dotenv(tmp_path, target_name):
    alias = tmp_path / "synthetic-alias"
    alias.symlink_to(tmp_path / target_name)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.audit('open', sys.argv[1], 'r', 0)",
            str(alias),
        ],
        env={
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(tmp_path),
            "PYTHONPATH": str(ROOT / "tests" / "offline_guard"),
            "OFFLINE_NETWORK_LOG": str(tmp_path / "network.log"),
            "OFFLINE_REPOSITORY_ROOT": str(tmp_path / "repository"),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "PermissionError: Offline tests forbid dotenv file access" in result.stderr


def test_runner_isolates_tempfile_and_pytest_directories(tmp_path):
    probe = tmp_path / "test_temporary_isolation.py"
    probe.write_text(
        "import os\n"
        "import tempfile\n"
        "from pathlib import Path\n"
        "def test_temporary_roots(tmp_path_factory):\n"
        "    home = Path(os.environ['HOME'])\n"
        "    assert Path(tempfile.gettempdir()).is_relative_to(home)\n"
        "    assert tmp_path_factory.getbasetemp().is_relative_to(home)\n"
    )
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_offline.py"), str(probe)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout
