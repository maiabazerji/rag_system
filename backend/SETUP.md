# Backend setup and troubleshooting

Docker Compose is the fastest path, see the [README quickstart](../README.md#quickstart-docker). This document covers running the backend directly, and what to do when something breaks.

## Running without Docker

You need Python 3.11+ and a running Qdrant. Postgres is only needed if you enable API key auth.

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# CPU-only torch first: the default wheel pulls ~2 GB of CUDA runtime that
# sentence-transformers does not need on a laptop.
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -e ".[dev]"
```

Start the services the backend talks to:

```bash
docker compose -f ../infra/docker-compose.yml up -d qdrant
# only if REQUIRE_API_KEY=true or ADMIN_KEY is set:
docker compose -f ../infra/docker-compose.yml up -d postgres
```

Copy the config and run:

```bash
cp ../.env.example ../.env      # then set ANTHROPIC_API_KEY
uvicorn app.main:app --reload --port 8000
```

`.env` is read relative to the working directory. Running uvicorn from `backend/`, either copy `.env` there or export the variables into your shell.

The first request downloads the embedding model and, on first rerank, the cross-encoder. Both are cached under `~/.cache/huggingface`, which ends up around 470 MB: the cross-encoder is ~90 MB, and the embedding repo ships ONNX and OpenVINO builds alongside the PyTorch weights.

If the cross-encoder cannot be downloaded, reranking silently falls back to BM25. That is by design, but it is worth knowing: `"reranker": "bm25"` in the logs means you are not measuring the cross-encoder.

### Optional extras

```bash
pip install -e ".[openai]"    # GENERATOR_PROVIDER=openai
pip install -e ".[tracing]"   # Langfuse and W&B
```

Both are imported lazily, so the app runs fine without them.

## The checks CI runs

```bash
ruff check app tests   # lint
mypy app               # types
pytest -q              # tests
python -c "import app.main"   # must work with no configuration at all
```

That last one matters more than it looks. Configuration is validated in the FastAPI lifespan, not at import, precisely so tooling works in a bare checkout. If someone moves `validate_startup()` back to import scope, every one of these commands breaks at once.

Coverage: `pytest --cov=app --cov-report=term-missing`.

## Project layout

```
backend/app/
├── main.py              FastAPI app, lifespan, error handlers, /health
├── config.py            All settings, validation, model allowlist
├── auth.py              API keys, rate limiting, usage accounting
├── resilience.py        Retry policy, circuit breaker, timeouts
├── logging_config.py    Structured JSON logging with request IDs
├── api/                 One module per route group
├── rag/
│   ├── generate.py      Orchestrator: strategy dispatch, telemetry, errors
│   ├── embed.py         SentenceTransformer, sync and async
│   ├── retrieve.py      Dense vector search over Qdrant
│   ├── rerank.py        Cross-encoder, with BM25 fallback
│   ├── ingest.py        Load, chunk, embed, index
│   ├── store.py         Qdrant client and collection management
│   ├── providers/       anthropic, openai, local (Ollama)
│   └── strategies/      classic, graph, agentic
├── eval/                judge, metrics, regression, human ratings
└── tracing/             In-memory traces, optional W&B
```

## Common errors

### `ValueError: Configuration validation failed`

`GENERATOR_PROVIDER` names a provider whose key is missing. Either set the key or change the provider. This is raised at server startup, not at import.

### `Qdrant collection 'evalrag' has dim=384 but embedding model produces dim=1024`

You changed `EMBEDDING_MODEL`. Vectors from different models are not comparable, so the collection has to be rebuilt:

```bash
docker compose -f infra/docker-compose.yml down -v   # wipes Qdrant and Postgres
docker compose -f infra/docker-compose.yml up -d
```

Then re-ingest.

### Every answer is "No documents uploaded yet"

Nothing is indexed. Check `GET /ingest/stats`. If it reports chunks but you still see this, Qdrant is probably unreachable, `count()` degrades to `0` rather than erroring. Check `docker compose logs qdrant`.

### Every answer is a refusal, and `/health` shows `"anthropic": false`

`ANTHROPIC_API_KEY` is not reaching the process. Under Compose it comes from `../.env` via `env_file`. Confirm with:

```bash
docker compose -f infra/docker-compose.yml exec backend printenv ANTHROPIC_API_KEY
```

### `503 Authentication service is unavailable`

`REQUIRE_API_KEY=true` but Postgres is unreachable. Check `POSTGRES_URL`, running outside Docker it is host port **5434**, not 5432.

### `429 Rate limit exceeded`

Your API key's per-minute allowance. Create a key with a higher limit:

```bash
python scripts/setup_auth.py --create-key "load-test" --rpm 120
```

### `/admin/*` returns 503

`ADMIN_KEY` is unset. Generate one, put it in `.env`, and restart.

### Answers arrive but citations look wrong

The prompt asks Claude to cite chunk ids like `[9fa3c1b0e2:4]`, and those are preserved in the response. Cross-reference them against the `sources` array. If they name chunks that are not in `sources`, the model is inventing citations, worth an eval run.

### The knowledge graph is empty after a rebuild

`GRAPH_DATA_DIR` must be an absolute path inside Docker. Compose sets `/data/graph`, which is bind-mounted from the host. A relative path resolves against the container's working directory, which is *not* mounted, so the graph is discarded on every rebuild.

### The agentic strategy times out

It makes up to `AGENTIC_MAX_ITERS` model calls in sequence. Raise `PROVIDER_TIMEOUT_SECONDS`, or lower `AGENTIC_MAX_ITERS`.

### `Anthropic service is temporarily unavailable (circuit breaker open)`

Five consecutive provider failures opened the breaker; it retries after 30 seconds. The underlying cause is in the logs, usually an invalid key, an exhausted quota, or no network.

## Reading the logs

Logs are JSON, one object per line, each carrying the `request_id` also returned in the `X-Request-ID` header. To follow one request end to end:

```bash
docker compose -f infra/docker-compose.yml logs backend | grep "request-1234567890-abcd1234"
```

Set `LOG_LEVEL=DEBUG` in `.env` for per-step retrieval and reranking detail. Files rotate under `LOG_DIR` (10 MB, 5 backups); if that directory is not writable the app logs to the console only and says so.

## Resetting

```bash
# Wipe indexed documents and API keys, keep the code
docker compose -f infra/docker-compose.yml down -v

# Wipe evaluation history and the knowledge graph
rm -rf data/eval_runs data/graph data/human_ratings

# Wipe the downloaded models
rm -rf .hf_cache
```
