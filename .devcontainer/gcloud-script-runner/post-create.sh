#!/usr/bin/env bash
set -euo pipefail

UV_VERSION="0.12.3"
UV_INSTALL_DIR="${UV_INSTALL_DIR:-/home/vscode/.local/bin}"
UV_BIN="${UV_INSTALL_DIR}/uv"

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

if [ ! -x "${UV_BIN}" ]; then
    curl --fail --show-error --silent --location \
        "https://astral.sh/uv/${UV_VERSION}/install.sh" \
        --output /tmp/install-uv.sh
    UV_INSTALL_DIR="${UV_INSTALL_DIR}" UV_NO_MODIFY_PATH=1 sh /tmp/install-uv.sh
    rm /tmp/install-uv.sh
fi

export PATH="${UV_INSTALL_DIR}:${PATH}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_PYTHON_DOWNLOADS="${UV_PYTHON_DOWNLOADS:-never}"

"${UV_BIN}" --version
"${UV_BIN}" sync --locked --no-dev
"${UV_BIN}" run --locked --no-dev python --version
