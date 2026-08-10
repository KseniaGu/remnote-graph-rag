#!/usr/bin/env bash
set -euo pipefail

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_PYTHON_DOWNLOADS="${UV_PYTHON_DOWNLOADS:-never}"
export HF_HOME="${HF_HOME:-/home/vscode/.cache/huggingface}"

mkdir -p \
    "/home/vscode/.cache" \
    "${HF_HOME}"

command -v uv >/dev/null 2>&1 || {
    echo "uv is not installed or is unavailable on PATH." >&2
    exit 1
}

uv --version
uv sync --locked

uv run --locked python -c \
    "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/all-MiniLM-L6-v2')"

test -f rxconfig.py || {
    echo "Expected Reflex configuration file rxconfig.py was not found." >&2
    exit 1
}

uv run --locked python --version
