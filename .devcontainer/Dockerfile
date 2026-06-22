FROM mcr.microsoft.com/devcontainers/python:3.11

ENV POETRY_VERSION=1.8.0 \
    POETRY_HOME=/home/vscode/.local/share/pypoetry \
    PATH=/home/vscode/.local/bin:$PATH \
    POETRY_VIRTUALENVS_IN_PROJECT=false \
    POETRY_VIRTUALENVS_PATH=/home/vscode/envs \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    unzip \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

USER vscode
RUN curl -sSL https://install.python-poetry.org | python3 - --version "$POETRY_VERSION"
