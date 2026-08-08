#!/usr/bin/env bash
set -euo pipefail

POETRY_HOME="${POETRY_HOME:-/home/vscode/.local/share/pypoetry}"
POETRY_BIN="${POETRY_HOME}/bin/poetry"

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg

sudo install -d -m 0755 /usr/share/keyrings

if [ ! -f /usr/share/keyrings/cloud.google.gpg ]; then
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | sudo gpg --dearmor \
            -o /usr/share/keyrings/cloud.google.gpg
fi

echo \
    "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list >/dev/null

sudo apt-get update
sudo apt-get install -y --no-install-recommends google-cloud-cli

if [ ! -x "${POETRY_BIN}" ]; then
    curl -sSL https://install.python-poetry.org \
        | POETRY_HOME="${POETRY_HOME}" python3 -
fi

"${POETRY_BIN}" --version
"${POETRY_BIN}" install --no-root