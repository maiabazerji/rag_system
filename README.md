# EvalRAG

Run the same question through three different RAG retrieval strategies, **Classic**, **Graph**, and **Agentic**, on your own corpus, and measure which one actually answers it better. Built on the Anthropic API (Claude).

## Why

Most RAG projects pick one retrieval strategy and stop there. But a strategy that nails single-fact lookups can flail on multi-hop questions, and the reverse is also true. EvalRAG exists to answer one question: **which strategy actually works better on my data?**

The **Compare view** (`/compare` in the UI, `POST /compare/strategies` on the API) runs one question through every strategy and shows the answers, sources, latency, token cost, and reasoning trace side by side. Paired with an evaluation harness that scores answers with an LLM judge and a regression detector that flags when a change makes things worse on your golden dataset, you can tell whether a retrieval tweak helped or hurt.

For a file-by-file walkthrough, see [LEARN.md](./LEARN.md).

## Three strategies, one corpus

| Strategy | Retrieval logic | Implementation | Best at |
|---|---|---|---|
| **Classic** | embed → vector search → cross-encoder rerank → answer | [`classic.py`](backend/app/rag/strategies/classic.py) | single-fact lookups |
| **Graph** | entity walk over a Claude-extracted knowledge graph | [`graph.py`](backend/app/rag/strategies/graph.py) + [`graph_store.py`](backend/app/rag/graph_store.py) + [`graph_extract.py`](backend/app/rag/graph_extract.py) | "how is X related to Y" multi-hop |
| **Agentic** | Claude drives `search`/`fetch`/`finish` tools in a loop | [`agentic.py`](backend/app/rag/strategies/agentic.py) + [`tool_use_loop`](backend/app/rag/providers/anthropic_provider.py) | ambiguous, multi-step questions |

All three return the same `StrategyResult` shape, which is what makes them comparable 1:1.

**Classic** is fast and cheap but struggles when an answer has to be assembled across documents. **Graph** extracts entity relationships up front (with Claude Haiku) and walks them at query time. **Agentic** hands Claude a toolbox and lets it decide the retrieval path step by step, the most capable and the most expensive.

## Architecture

| Layer | What it does | Where |
|---|---|---|
| Frontend | Ask questions, compare strategies, view eval scores, browse traces | `frontend/src/` |
| Backend API | FastAPI routes for ingest, ask, compare, eval, graph, traces | `backend/app/api/` |
| RAG pipeline | Chunk → embed → retrieve → rerank → generate | `backend/app/rag/` |
| Eval engine | LLM-as-judge scoring, regression tracking | `backend/app/eval/` |

Backing services: **Qdrant** (vectors), **Postgres** (API keys and usage accounting), and optionally **Langfuse** (traces) and **W&B** (eval dashboards).

Everything in this table is implemented. The embedding model is a real SentenceTransformer, the reranker is a real cross-encoder (with a BM25 fallback), and the judge is a real model call.

> **Check the reranker is the one you think it is.** If the cross-encoder cannot be downloaded, reranking degrades to BM25 silently and every request still returns 200. `"reranker": "bm25"` in the logs means the cross-encoder is not running and any comparison you draw is really a comparison of BM25. This bit this project for a long time: the configured model id was a repo that did not exist, so the fallback was the only code path ever taken.

### What happens when you ask a question

1. `POST /ask` with `{"question": "...", "strategy": "classic" | "graph" | "agentic"}` → [`ask.py`](backend/app/api/ask.py)
2. `answer_question` → [`generate.py`](backend/app/rag/generate.py) picks defaults, opens a trace, and calls `get_strategy(strategy).run(...)`
3. The strategy retrieves in its own way but returns the same `StrategyResult` (answer, sources, latency, tokens, trace)
4. All three generate through `AsyncAnthropic` ([`anthropic_provider.py`](backend/app/rag/providers/anthropic_provider.py)); the agentic strategy uses `tool_use_loop`
5. The trace is retrievable from `/traces/{id}` (in memory, so it does not survive a restart)

Retrieval is **dense-only**: the query is embedded and matched against Qdrant. The lexical BM25 signal enters one step later, in [`rerank.py`](backend/app/rag/rerank.py), as the reranker's fallback when the cross-encoder is unavailable. This is a dense-retrieve-then-rerank pipeline, not hybrid retrieval in the fuse-two-retrievers sense.

### What happens when you run an eval

1. `POST /eval/run` → `run_evaluation` in [`metrics.py`](backend/app/eval/metrics.py)
2. Loads a golden dataset (question + ideal answer) from `data/golden/`
3. Answers each question, then scores it with the LLM judge in [`judge.py`](backend/app/eval/judge.py) on faithfulness, answer relevance, context precision and context recall
4. Saves the run to `data/eval_runs/` and compares it against previous runs of **the same configuration** → [`regression.py`](backend/app/eval/regression.py)
5. Optionally streams to W&B via [`wandb_tracer.py`](backend/app/tracing/wandb_tracer.py)

**On unscored examples.** If the judge fails (rate limit, timeout, unparseable response), that example is recorded with `score: null` and left out of the aggregate. The run result reports `n_scored` and `n_unscored` so a partially failed run can never be mistaken for a complete one. A fabricated midpoint score would be worse than a missing one.

---

## Baseline on the bundled corpus

41 documents, 107 chunks, `golden_v1` (34 questions), all 34 scored. Reproduce with
`python scripts/run_eval.py --dataset golden_v1 --all`.

| Metric | Classic | Graph | Agentic |
|---|---|---|---|
| Retrieval precision | 0.185 | 0.185 | **0.250** |
| Retrieval recall | **0.765** | **0.765** | 0.544 |
| Retrieval hit rate | **0.765** | **0.765** | 0.559 |
| Retrieval MRR | **0.554** | **0.554** | 0.356 |
| Faithfulness | 0.507 | **0.519** | 0.249 |
| Answer relevance | **0.862** | 0.854 | 0.521 |
| Context precision | 0.368 | 0.353 | **0.416** |
| Context recall | **0.328** | 0.318 | 0.275 |
| Latency (ms/question) | 21,675 | **15,204** | 17,976 |
| Tokens (total) | **389,251** | 397,431 | 727,101 |
| Refusals | **0** | **0** | 6 |

Three things this says, none of them the marketing answer:

**Classic and Graph score identically on retrieval.** Not a coincidence and not a
bug. The graph strategy walks the entity graph *and* runs the same dense search,
then keeps only the graph hits that dense search missed. With 107 chunks and
`RETRIEVAL_TOP_K=50`, dense search already returns nearly half the corpus, so
there is almost nothing left for the walk to add, and the same reranker picks the
same top-8. Graph RAG needs either a much larger corpus or a much smaller
`RETRIEVAL_TOP_K` before it can differentiate itself. On this corpus it is Classic
with extra steps.

**Agentic is the weakest strategy here, at 1.9x the tokens.** It refused 6 of 34
questions, and a refusal scores near zero on faithfulness and relevance, which is
what drags those columns down. It does win retrieval precision and context
precision: it fetches less, and what it fetches is more on-topic. That is a real
strength, spent badly.

**Faithfulness around 0.5 is the number to attack.** Answer relevance is high, so
the answers address the question; faithfulness says they assert more than the
retrieved context supports. That gap is the interesting bug, and it is the kind of
thing this harness exists to surface.

These are the numbers from one small corpus. The point of the tool is that you run
it on yours.

---

## Quickstart (Docker)

Brings up frontend, backend, Qdrant and Postgres together.

> **Note.** The Compose file lives at `infra/docker-compose.yml`, not at the project root. Every command needs `-f infra/docker-compose.yml`, or `cd infra` first, or `$env:COMPOSE_FILE`. A bare `docker compose down` from the root fails with `no configuration file provided`.

### 1. Configure

```powershell
# Windows PowerShell (from the project root)
copy .env.example .env
```

```bash
# macOS / Linux
cp .env.example .env
```

Open `.env` and set `ANTHROPIC_API_KEY`. Everything else has a working default. Without a key the stack still starts and the UI still loads, but every answer comes back as a refusal.

### 2. Start

```bash
docker compose -f infra/docker-compose.yml up -d --build
```

Add `--profile tracing` if you also want Langfuse. The first build downloads a couple of GB; later starts take seconds.

### 3. Open

| App | URL | Notes |
|---|---|---|
| Frontend | <http://localhost:5173> | Vite dev server with hot reload |
| Backend | <http://localhost:8011/docs> | OpenAPI / Swagger UI |
| Health | <http://localhost:8011/health> | Which providers are configured |
| Langfuse | <http://localhost:3100> | Only with `--profile tracing` |

### 4. Ingest and evaluate

```bash
docker compose -f infra/docker-compose.yml exec backend python /scripts/ingest.py
docker compose -f infra/docker-compose.yml exec backend python /scripts/run_eval.py --dataset golden_v1
```

Or drag files onto the Ingest page. Accepted types: `.pdf`, `.txt`, `.md`, `.markdown`, `.rst`, `.csv`, `.json`, up to `MAX_UPLOAD_MB` (25 MB by default).

### 5. Logs, stop, wipe

```bash
docker compose -f infra/docker-compose.yml logs -f backend      # follow logs
docker compose -f infra/docker-compose.yml ps                   # what's running
docker compose -f infra/docker-compose.yml down                 # stop (keeps data)
docker compose -f infra/docker-compose.yml down -v              # also wipe volumes
```

### Port map (host → container)

| Service | Host | Container | Why this host port |
|---|---|---|---|
| Frontend | `5173` | `5173` | Vite default |
| Backend | `8011` | `8000` | `8001` was taken on the author's machine |
| Langfuse | `3100` | `3000` | `3000` is a common Next.js default |
| Postgres | `5434` | `5432` | `5432`/`5433` taken by other stacks |
| Qdrant | `6333` | `6333` | free |

Containers reach each other by **service name** on the internal network (`qdrant:6333`, `postgres:5432`), never `localhost`. The host ports are only for reaching services from your machine.

`backend/app/` and `frontend/src/` are bind-mounted, so edits appear inside the containers immediately and uvicorn `--reload` / Vite HMR pick them up.

---

## Authentication

Auth is **off by default** so a fresh clone runs with no setup. `/ask`, `/compare`, `/ingest`, `/eval` and `/graph` accept unauthenticated requests, which is fine on localhost and not fine anywhere else.

To turn it on:

```bash
# 1. In .env
REQUIRE_API_KEY=true
ADMIN_KEY=<python -c "import secrets; print(secrets.token_urlsafe(32))">

# 2. Restart, then mint a key
python scripts/setup_auth.py --create-key "my-laptop"
```

The key is shown once, only its SHA-256 hash is stored. Paste it into the field the UI shows (it appears automatically when the backend reports `auth_required`), or send it yourself:

```bash
curl -H "Authorization: Bearer sk_..." http://localhost:8011/ingest/stats
```

Each key carries a per-minute rate limit (`--rpm`, default 10) enforced across every protected route, and its token usage is recorded per request.

```bash
python scripts/setup_auth.py --list-keys           # keys and 24h usage
python scripts/setup_auth.py --deactivate-key 1    # revoke, effective immediately
```

Auth requires Postgres. It is the only thing that does, so with `REQUIRE_API_KEY=false` and no `ADMIN_KEY` the backend never opens a database connection.

---

## Configuration

[`.env.example`](./.env.example) documents every setting. The ones worth knowing:

| Setting | Default | Effect |
|---|---|---|
| `GENERATOR_MODEL` | `claude-sonnet-5` | Model that writes answers |
| `JUDGE_MODEL` | `claude-opus-5` | Model that grades them |
| `RETRIEVAL_TOP_K` | `50` | Candidates fetched before reranking |
| `RERANK_TOP_K` | `8` | Chunks sent to the model, the main cost lever |
| `CHUNK_SIZE_TOKENS` | `600` | Words per chunk (applies to new ingests) |
| `AGENTIC_MAX_ITERS` | `15` | Bounds worst-case cost of one agentic question |
| `MAX_UPLOAD_MB` | `25` | Upload ceiling |
| `REQUIRE_API_KEY` | `false` | Enforce API keys |

Changing `EMBEDDING_MODEL` changes the vector dimension. The backend refuses to start against a collection built with a different model and tells you so; recreate it with `docker compose -f infra/docker-compose.yml down -v`.

---

## Development

```bash
cd backend
python -m venv .venv && . .venv/Scripts/activate    # or bin/activate on macOS/Linux
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -e ".[dev]"

ruff check app tests     # lint
mypy app                 # types
pytest -q                # 258 tests
```

```bash
cd frontend
npm ci
npm run build            # typecheck + production build
npm run dev              # dev server on :5173
```

The app imports without any configuration. `settings.validate_startup()` runs in the FastAPI lifespan rather than at import time, so linters, tests and tooling all work in a bare checkout.

All three gates run in CI on every push and pull request ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

More detail, including troubleshooting: [`backend/SETUP.md`](./backend/SETUP.md).

---

## Troubleshooting

**Every answer is a refusal** → `ANTHROPIC_API_KEY` is unset. Check `GET /health`.

**`Connection refused` from Postgres, Qdrant** → services aren't up. `docker compose -f infra/docker-compose.yml up -d`.

**401 on every request** → `REQUIRE_API_KEY=true` and no key is set. See [Authentication](#authentication).

**429 Rate limit exceeded** → your key's per-minute limit. Raise it with `--rpm` when creating the key.

**Answers are slow** → lower `RETRIEVAL_TOP_K` and `RERANK_TOP_K`; try `claude-haiku-4-5` as `GENERATOR_MODEL`. The agentic strategy is inherently slower, it makes up to `AGENTIC_MAX_ITERS` model calls per question.

**Graph strategy returns nothing** → build the graph first: `POST /graph/build`, then poll `GET /graph/build/{job_id}`. It runs one model call per chunk, so it takes a while on a large corpus.

Fuller guide: [`backend/SETUP.md`](./backend/SETUP.md).

---

## Contributing

1. **New strategy?** Add a `Strategy` subclass in `backend/app/rag/strategies/` and register it in `REGISTRY`.
2. **New metric?** Extend `backend/app/eval/judge.py` and `DIMENSIONS`.
3. **Frontend?** `frontend/src/`.
4. **Bug?** Open an issue with logs (`LOG_LEVEL=DEBUG` in `.env`).

Pull requests should keep `ruff check`, `mypy` and `pytest` green, add tests for new logic, and document any new setting in `.env.example`.

---

## Where to go next

- [`LEARN.md`](./LEARN.md), module-by-module walkthrough
- [`backend/SETUP.md`](./backend/SETUP.md), local install, debugging, common errors
- [`EvalRAG.md`](./EvalRAG.md), theory and extension recipes
- [`.env.example`](./.env.example), every setting, annotated
- <http://localhost:8011/docs>, live API reference

---

## Glossary

**Chunk**, a passage split from a source document (~600 words here, with overlap).
**Embedding**, a fixed-length vector encoding meaning; the same model must embed both documents and queries.
**Vector DB**, an index optimized for approximate nearest-neighbor search. Here: Qdrant.
**Reranking**, a slower cross-encoder rescoring the top-K from retrieval, for higher precision.
**Grounding / citations**, the answer references the chunks it used, which is what makes it auditable.
**Golden dataset**, hand-labeled question/answer pairs used as ground truth.
**LLM-as-judge**, a second model scoring answers against a rubric.
**Faithfulness**, does the answer assert only what the retrieved context supports?
**Regression**, a metric got worse than the previous run *of the same configuration*.
**Circuit breaker**, after repeated provider failures, calls fail fast for 30 seconds instead of piling up.

---

## License

[MIT](./LICENSE)
