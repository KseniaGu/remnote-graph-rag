"""Run the full deterministic suite with empty caches and no network access.

The Python audit guard also covers child Python processes. CI adds an isolated
network namespace so native libraries and non-Python subprocesses cannot connect.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    if (ROOT / ".env").exists():
        raise SystemExit("Offline checks require a checkout without .env")
    with tempfile.TemporaryDirectory(prefix="remnote-offline-") as temporary:
        home = Path(temporary)
        temporary_root = home / "tmp"
        temporary_root.mkdir()
        log = home / "network-attempts.log"
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(home),
            "TMPDIR": str(temporary_root),
            "XDG_CACHE_HOME": str(home / "cache"),
            "HF_HOME": str(home / "huggingface"),
            "NLTK_DATA": str(home / "nltk"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "LANGSMITH_TRACING": "false",
            "LANGCHAIN_TRACING_V2": "false",
            "PYTHON_DOTENV_DISABLED": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONPATH": os.pathsep.join(
                (str(ROOT / "tests" / "offline_guard"), str(ROOT))
            ),
            "OFFLINE_NETWORK_LOG": str(log),
            "OFFLINE_REPOSITORY_ROOT": str(ROOT),
        }
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-ra", *sys.argv[1:]],
            cwd=ROOT,
            env=environment,
            check=False,
        )
        if log.exists():
            print("Offline guard blocked network attempts:", file=sys.stderr)
            print(log.read_text(), file=sys.stderr)
            return 1
        return result.returncode


if __name__ == "__main__":
    sys.exit(main())
