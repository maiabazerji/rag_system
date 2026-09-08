# syntax=docker/dockerfile:1
#
# Two stages share one base so the runtime image carries no build tooling.
# `--target dev` gives the hot-reload image Compose uses; the default `runtime`
# target is the one to deploy: no dev dependencies, no --reload, non-root.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=180 \
    PIP_RETRIES=10

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first so sentence-transformers does not pull ~2GB of
# CUDA wheels through its default torch dependency.
COPY pyproject.toml ./
RUN mkdir -p app && touch app/__init__.py \
    && pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch


# --- development image: dev dependencies, hot reload, bind-mounted source ----
FROM base AS dev

RUN pip install -e ".[dev,tracing]"

COPY app ./app
COPY tests ./tests

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


# --- production image: runtime dependencies only, non-root, no reload --------
FROM base AS runtime

RUN pip install ".[tracing]"

COPY app ./app

# Run as an unprivileged user, and give it the log directory it writes to.
RUN useradd --create-home --uid 10001 evalrag \
    && mkdir -p /app/logs \
    && chown -R evalrag:evalrag /app
USER evalrag

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
