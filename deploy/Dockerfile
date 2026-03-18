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

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

FROM python:3.11-slim

ENV HF_HOME=/home/appuser/.cache/huggingface
ENV PYTHONUNBUFFERED=1

RUN adduser --disabled-password --no-create-home appuser

WORKDIR /app

COPY --from=builder $HF_HOME $HF_HOME
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/.local/share/reflex /home/appuser/.local/share/reflex
COPY --from=builder /root/.cache/huggingface /home/appuser/.cache/huggingface

COPY --from=builder /app /app

RUN chown -R appuser:appuser /app /home/appuser

USER appuser

EXPOSE 8000

CMD ["reflex", "run", "--env", "prod", "--backend-only", "--backend-host", "0.0.0.0"]
