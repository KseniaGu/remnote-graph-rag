FROM ghcr.io/astral-sh/uv:0.12.3 AS uv
FROM mcr.microsoft.com/devcontainers/python:3.11

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

COPY --from=uv /uv /uvx /usr/local/bin/

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        libgl1 \
        libglib2.0-0 \
        unzip \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p \
        /home/vscode/.cache/huggingface \
        /home/vscode/.local/share/reflex \
    && chown -R vscode:vscode \
        /home/vscode/.cache \
        /home/vscode/.local

USER vscode

RUN uv --version
