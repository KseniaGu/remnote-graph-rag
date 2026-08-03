FROM mcr.microsoft.com/devcontainers/python:3.11

ARG POETRY_VERSION=2.4.1

ENV POETRY_HOME=/home/vscode/.local/share/pypoetry \
    POETRY_VIRTUALENVS_IN_PROJECT=false \
    POETRY_VIRTUALENVS_PATH=/home/vscode/envs \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

ENV PATH="${POETRY_HOME}/bin:/home/vscode/.local/bin:${PATH}"

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
        /home/vscode/.local/bin \
        "${POETRY_HOME}" \
        /home/vscode/.local/share/reflex \
        "${POETRY_VIRTUALENVS_PATH}" \
    && chown -R vscode:vscode \
        /home/vscode/.local \
        "${POETRY_VIRTUALENVS_PATH}"

USER vscode

RUN curl --fail --show-error --silent --location \
        https://install.python-poetry.org \
        --output /tmp/install-poetry.py \
    && python3 /tmp/install-poetry.py --version "${POETRY_VERSION}" \
    && rm /tmp/install-poetry.py \
    && poetry --version