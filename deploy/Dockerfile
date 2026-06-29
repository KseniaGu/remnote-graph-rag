# syntax=docker/dockerfile:1.7

FROM python:3.11-slim as builder

ENV HF_HOME=/home/appuser/.cache/huggingface

RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install poetry==1.8.0

COPY pyproject.toml poetry.lock* ./
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-root --no-interaction --no-ansi

COPY . .

RUN reflex init


FROM python:3.11-slim

ENV HF_HOME=/home/appuser/.cache/huggingface
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV PYTHONUNBUFFERED=1
ENV EMBEDDER_MODEL_PATH=/app/models/all-MiniLM-L6-v2
ENV RERANKER_MODEL_PATH=/app/models/Qwen3-Reranker-0.6B

RUN adduser --disabled-password --no-create-home appuser

WORKDIR /app

COPY --from=model-cache / /app/models/
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/.local/share/reflex /home/appuser/.local/share/reflex

COPY --from=builder /app /app

RUN chown -R appuser:appuser /app /home/appuser

USER appuser

EXPOSE 8000

CMD ["reflex", "run", "--env", "prod", "--backend-only", "--backend-host", "0.0.0.0"]
