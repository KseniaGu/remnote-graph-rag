#!/usr/bin/env bash
set -euo pipefail

export POETRY_REQUESTS_TIMEOUT="${POETRY_REQUESTS_TIMEOUT:-120}"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
export POETRY_KEYRING_ENABLED=false
export POETRY_CACHE_DIR="${POETRY_CACHE_DIR:-/home/vscode/.cache/pypoetry}"
export POETRY_VIRTUALENVS_PATH="${POETRY_VIRTUALENVS_PATH:-/home/vscode/envs}"

sudo mkdir -p /home/vscode/.cache "$HF_HOME" "$POETRY_CACHE_DIR" "$POETRY_VIRTUALENVS_PATH"
sudo chown -R vscode:vscode /home/vscode/.cache "$POETRY_VIRTUALENVS_PATH"

poetry config virtualenvs.in-project false
poetry config virtualenvs.path "$POETRY_VIRTUALENVS_PATH"
poetry config keyring.enabled false
poetry config installer.parallel false

poetry install --only main --no-root

poetry run python -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/all-MiniLM-L6-v2')"

poetry run reflex init
