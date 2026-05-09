# EvalRAG

> **Status:** work in progress. Core pipeline runs end-to-end; rerank, BM25 hybrid, and LLM-as-judge are scaffolded as next steps.

A platform that pairs a **RAG** (retrieval-augmented generation) pipeline with the **evaluation harness** that keeps it honest — so a prompt or model change can't silently regress quality.

## Architecture

| Layer | What it does | Where it lives |
|---|---|---|
| Frontend | Dashboard: ask, A/B compare, eval scores, traces | `frontend/src/` |
| Backend API | FastAPI — ingest, ask, compare, eval, traces | `backend/app/api/` |
| RAG pipeline | Chunk → embed → retrieve → rerank → generate | `backend/app/rag/` |
| Eval engine | Metrics, LLM-as-judge, regression tracking | `backend/app/eval/` |

**Backing services** (Docker): Qdrant (vectors), Postgres (eval results), Redis (queue), Langfuse (traces).

## Ask flow

1. `POST /ask` → `backend/app/api/ask.py`
2. `answer_question` → `backend/app/rag/generate.py` opens a trace, then:
3. `hybrid_search` → `backend/app/rag/retrieve.py` (embed query, query Qdrant)
4. `rerank` → `backend/app/rag/rerank.py` (cross-encoder, currently pass-through)
5. Prompt assembly → `backend/app/prompts/` (versioned templates)
6. LLM call → pluggable provider (Anthropic / OpenAI / Ollama)
7. Trace closed; answer returned with grounded citations

## Eval flow

1. `POST /eval/run` runs the pipeline against a golden dataset (`data/golden/`).
2. Each example is scored on **faithfulness, relevance, context precision, context recall**.
3. Run snapshots persist to `data/eval_runs/`; regressions are flagged across runs.
4. Optional W&B integration logs per-run tables and aggregates.

## What's wired vs. stubbed

| Module | Status |
|---|---|
| `rag/generate.py` | ✅ Real (Anthropic / OpenAI / Ollama) |
| `rag/embed.py` | ✅ Real (`bge-small-en-v1.5`) |
| `rag/retrieve.py` | ⚠️ Dense-only — BM25 hybrid pending |
| `rag/rerank.py` | ⚠️ Pass-through — cross-encoder pending |
| `eval/judge.py` | ⚠️ Stub — LLM-as-judge wiring pending |
| `eval/metrics.py` | ⚠️ Skeleton — RAGAS integration pending |
| `tracing/wandb_tracer.py` | ✅ Real (no-ops if key missing) |

Missing provider keys are handled gracefully — `/health` reports which are configured.

## Quickstart

```bash
cp .env.example .env
# Fill keys you want (all optional for a smoke test)

docker compose -f infra/docker-compose.yml up -d --build
```

| App      | URL                              |
|----------|----------------------------------|
| Frontend | <http://localhost:5173>          |
| Backend  | <http://localhost:8011/docs>     |
| Health   | <http://localhost:8011/health>   |
| Langfuse | <http://localhost:3100>          |

Ingest sample docs and run an eval:

```bash
docker compose -f infra/docker-compose.yml exec backend python /scripts/ingest.py
docker compose -f infra/docker-compose.yml exec backend python /scripts/run_eval.py --dataset golden_v1
```

Tear down (keep volumes / wipe volumes):

```bash
docker compose -f infra/docker-compose.yml down
docker compose -f infra/docker-compose.yml down -v
```

## Local dev (without full Docker stack)

```bash
# Services only
docker compose -f infra/docker-compose.yml up -d qdrant postgres redis langfuse

# Backend
cd backend
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
uvicorn app.main:app --reload

# Frontend
cd frontend && npm install && npm run dev
```

## Configuration

Every knob lives in `backend/app/config.py` and is overridable via environment variables. See `.env.example` for the full list.

## Tests

```bash
cd backend && pytest
```

## Roadmap

- [ ] Real cross-encoder reranking (`bge-reranker-large`)
- [ ] BM25 sparse retrieval + RRF fusion
- [ ] LLM-as-judge wired to Claude / GPT
- [ ] RAGAS metrics integration
- [ ] Langfuse tracer wiring
- [ ] Async ingest via Redis queue

## License

MIT
