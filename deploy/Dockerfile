# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:0.12.3 AS uv
FROM python:3.11-slim AS builder

ENV HF_HOME=/home/appuser/.cache/huggingface \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=uv /uv /uvx /usr/local/bin/
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

COPY . .

ENV PATH="/app/.venv/bin:${PATH}"

RUN reflex init


FROM python:3.11-slim

ENV HF_HOME=/home/appuser/.cache/huggingface
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV PYTHONUNBUFFERED=1
ENV EMBEDDER_MODEL_PATH=/app/models/all-MiniLM-L6-v2
ENV RERANKER_MODEL_PATH=/app/models/Qwen3-Reranker-0.6B
ENV PATH="/app/.venv/bin:${PATH}"

RUN adduser --disabled-password --no-create-home appuser

WORKDIR /app

COPY --from=model-cache / /app/models/
COPY --from=builder /root/.local/share/reflex /home/appuser/.local/share/reflex
COPY --from=builder /app /app

RUN mkdir -p /app/storage
COPY --from=runtime-storage /docstore.json /app/storage/docstore.json
COPY --from=runtime-storage /property_graph_store.json /app/storage/property_graph_store.json

RUN chown -R appuser:appuser /app /home/appuser

USER appuser

EXPOSE 8000

CMD ["reflex", "run", "--env", "prod", "--backend-only", "--backend-host", "0.0.0.0"]
