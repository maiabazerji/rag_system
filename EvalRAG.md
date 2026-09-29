# EvalRAG design blueprint

## Table of Contents

- [A. Project Overview](#a-project-overview)
- [B. Learning Goals & Outcomes](#b-learning-goals--outcomes)
- [C. High-Level Architecture](#c-high-level-architecture)
- [D. Tech Stack](#d-tech-stack)
- [E. Repository Structure](#e-repository-structure)
- [F. Environment Setup](#f-environment-setup)
- [G. Data Layer](#g-data-layer)
- [H. Document Ingestion Pipeline](#h-document-ingestion-pipeline)
- [I. Embeddings & Vector Store](#i-embeddings--vector-store)
- [J. Advanced RAG Pipeline](#j-advanced-rag-pipeline)
- [K. Query Processing](#k-query-processing)
- [L. Reranking & Context Compression](#l-reranking--context-compression)
- [M. LLM Inference Layer](#m-llm-inference-layer)
- [N. Structured Output Enforcement](#n-structured-output-enforcement)
- [O. Evaluation Engine](#o-evaluation-engine)
- [P. LLM-as-Judge](#p-llm-as-judge)
- [Q. Golden Dataset & Benchmarks](#q-golden-dataset--benchmarks)
- [R. Regression Testing (CI/CD for Prompts)](#r-regression-testing-cicd-for-prompts)
- [S. Backend API](#s-backend-api)
- [T. Frontend Dashboard](#t-frontend-dashboard)
- [U. Observability & Tracing](#u-observability--tracing)
- [V. Security & Guardrails](#v-security--guardrails)
- [W. Testing Strategy](#w-testing-strategy)
- [X. Deployment](#x-deployment)
- [Y. Roadmap & Milestones](#y-roadmap--milestones)
- [Z. Glossary & References](#z-glossary--references)


## A. Project Overview

**EvalRAG** is a full-stack AI platform demonstrating real-world LLM engineering: retrieval, evaluation, prompt regression testing, structured outputs, reranking, and observability. It models the components companies typically deploy when shipping LLM-based assistants.

> **Blueprint, not a description of what is built.** This is the design the project grew from. Each section starts with a **Status** line saying what is implemented, what differs and what is not implemented, checked against the code. What ships: hybrid dense + BM25 retrieval fused with RRF, cross-encoder reranking, grounded answers with validated citations, the evaluation harness and regression gate, Qdrant and Postgres under Docker Compose (no Redis, no queue, no Kubernetes manifests), and non-streaming JSON responses. The [README](./README.md) and [LEARN.md](./LEARN.md) describe the running system.

**Four layers:**

1. **Frontend**: AI dashboard (ask questions, compare models A/B, view scores, track regressions).
2. **Backend API**: ingestion, embeddings, retrieval, inference, evaluation.
3. **Evaluation Engine**: RAG metrics, LLM-as-judge, regression tests, benchmarks.
4. **Data Layer**: vector DB, document store, golden dataset store.

---

## B. Learning Goals & Outcomes

> **Status:** Implemented: hybrid dense + BM25 retrieval with RRF, cross-encoder reranking, structured answers through tool use, the automated metrics, versioned LLM-as-judge, threshold-based regression checks, tracing with cost and latency. Not implemented: query rewriting, multi-query, HyDE, judge calibration, constrained decoding.

By completion, you will have hands-on experience with:

- Building hybrid dense + BM25 retrieval with RRF fusion and cross-encoder reranking
- Query rewriting, multi-query expansion, and HyDE
- Enforcing structured outputs via Pydantic + constrained decoding
- Automated evaluation: faithfulness, answer relevance, context precision/recall
- LLM-as-judge pipelines with calibration
- Prompt regression testing (CI/CD for prompts)
- Full LLM observability: tracing, cost, latency, failure analysis
- Deploying a FastAPI + React stack with Docker Compose

---

## C. High-Level Architecture

> **Status:** Partly as drawn. The BM25 index exists, in memory per process and per access scope (`backend/app/rag/sparse.py`). Not implemented: SSE (responses are JSON), a document store in Postgres + S3 (chunks live only in the Qdrant payload), a results database (eval runs are JSON files under `DATA_DIR/eval_runs`). Postgres holds API keys, usage and the audit log.

```
┌────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                        │
│   Ask · A/B compare · Eval dashboard · Regression history       │
└────────────────────────────┬────────────────────────────────────┘
                             │ REST / SSE
┌────────────────────────────▼────────────────────────────────────┐
│                      Backend API (FastAPI)                      │
│  Ingest · Retrieve · Generate · Evaluate · Trace                │
└──┬──────────────┬─────────────────┬──────────────┬──────────────┘
   │              │                 │              │
┌──▼───────┐  ┌───▼────────┐  ┌─────▼───────┐  ┌───▼──────────┐
│ Vector   │  │  BM25 /    │  │ LLM Gateway │  │ Evaluation   │
│ Store    │  │  Sparse    │  │ (Claude,    │  │ Engine       │
│(Qdrant)  │  │  Index     │  │  OpenAI)    │  │ (RAGAS+Judge)│
└──────────┘  └────────────┘  └─────────────┘  └──────────────┘
       │            │                                  │
       └─────┬──────┘                                  │
             │                                         │
        ┌────▼──────────────┐                 ┌────────▼────────┐
        │ Document Store    │                 │ Golden Dataset  │
        │ (Postgres + S3)   │                 │ + Results DB    │
        └───────────────────┘                 └─────────────────┘
```

---

## D. Tech Stack

> **Status:** Differs from the table. Embeddings: `intfloat/multilingual-e5-small`; reranker: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`; BM25 is a first-stage retriever fused with RRF, not only a fallback. Generation defaults to `claude-sonnet-5`, the judge to `claude-opus-5-5`. `instructor` and RAGAS are not dependencies (the judge rubric is inspired by RAGAS). No S3/MinIO document storage (MinIO appears only as Langfuse's storage).

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11 + TypeScript | Standard for ML + web |
| Backend | FastAPI | Async, typed, OpenAPI |
| Frontend | React + Vite + Tailwind | Fast iteration |
| Vector DB | Qdrant | Dense search, payload filters |
| Sparse | BM25 via `rank_bm25` | First-stage lexical retriever, fused with dense by RRF; also the reranker fallback |
| Embeddings | `text-embedding-3-large` or `bge-large-en` (planned; see Status) | Retrieval quality |
| Reranker | `bge-reranker-large` / Cohere Rerank | Cross-encoder accuracy |
| LLM | Claude Opus/Sonnet 4.6, GPT-4o (comparison) | Multi-model A/B |
| Structured out | Pydantic + `instructor` / tool-use | Schema safety |
| Eval | RAGAS + custom judges | Industry baseline |
| Tracing | Langfuse or OpenTelemetry | Free, self-host |
| Storage | Postgres + S3/MinIO | Durable |
| Deploy | Docker Compose | Local-first |

---

## E. Repository Structure

> **Status:** Roughly as shown; `rag/` also holds `sparse.py`, `fusion.py`, `grounding.py`, `chunking.py`, `parsers/` and `strategies/`, and `eval/` holds `retrieval.py`, `rubric.py` and `dataset.py`.

```
evalrag/
├── backend/
│   ├── app/
│   │   ├── api/              # FastAPI routers
│   │   ├── rag/              # retrieval + generation
│   │   │   ├── ingest.py
│   │   │   ├── embed.py
│   │   │   ├── retrieve.py   # hybrid search (dense + BM25, RRF)
│   │   │   ├── rerank.py
│   │   │   └── generate.py
│   │   ├── eval/             # evaluation engine
│   │   │   ├── metrics.py
│   │   │   ├── judge.py
│   │   │   └── regression.py
│   │   ├── schemas/          # Pydantic models
│   │   ├── prompts/          # versioned prompts
│   │   ├── tracing/
│   │   └── config.py
│   ├── tests/
│   └── pyproject.toml
├── frontend/
│   ├── src/
│   │   ├── pages/            # Ask, Upload, Compare, Evaluation, Advisor
│   │   ├── components/
│   │   └── api/
│   └── package.json
├── data/
│   ├── docs/                 # source documents
│   └── golden/               # evaluation datasets
├── infra/
│   └── docker-compose.yml
├── scripts/
│   ├── ingest.py
│   └── run_eval.py
└── README.md
```

---

## F. Environment Setup

> **Status:** Implemented; see the README quickstart.

```bash
# Clone + install
git clone <repo> evalrag && cd evalrag
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -e backend

# Frontend
cd frontend && npm ci && cd ..

# Services (see the README quickstart for the .env it needs)
cp .env.example .env
docker compose --env-file .env -f infra/docker-compose.yml up -d   # qdrant, postgres, backend, frontend
# add --profile tracing for langfuse
```

---

## G. Data Layer

> **Status:** Not implemented as described. There is no document store and no eval store in Postgres; `Chunk` has no `embedding` field (`backend/app/schemas/__init__.py`).

Three stores with clear roles:

- **Document Store** (Postgres + object storage): raw docs, chunks, metadata, versions.
- **Vector Store** (Qdrant): dense embeddings + payload for filtering.
- **Eval Store** (Postgres): golden datasets, run results, regression history.

**Chunk schema**

```python
class Chunk(BaseModel):
    id: str
    doc_id: str
    text: str
    tokens: int
    section: str | None
    embedding: list[float]
    metadata: dict  # source, page, created_at, tags
```

---

## H. Document Ingestion Pipeline

> **Status:** Implemented with the project's own parsers (`backend/app/rag/parsers/`, `pypdf` for PDF), structure-aware chunking of at most 300 words by default (capped under the 512-token model window), and content-hash idempotency with stale-revision cleanup. `unstructured` and `trafilatura` are not used.

```
PDF/TXT/MD/RST/CSV/JSON → Loader → Clean → Chunk → Embed → Index (dense)
```

- **Loaders**: `unstructured`, `pypdf`, `trafilatura`.
- **Cleaning**: strip boilerplate, normalize whitespace, dedupe.
- **Chunking**: semantic (sentence-aware) with overlap; target 400-800 tokens.
- **Idempotency**: hash each chunk; skip if unchanged.
- **Sync**: each upload is chunked, embedded and indexed within its request; there is no job queue.

---

## I. Embeddings & Vector Store

> **Status:** Partly. The embedding model is stamped on the Qdrant collection, not on each vector. Payload indexes exist for `tenant` and `acl_groups` only. The BM25 index is separate and in memory.

- Batch embed (64-128) with retries + exponential backoff.
- Store embedding model name + version with each vector (for reindexing).
- Qdrant collection with HNSW + `payload` for filters (`doc_id`, `tags`, `date`).
- Sparse index: BM25 over every chunk's payload text, built in memory per access scope (`backend/app/rag/sparse.py`).

---

## J. Advanced RAG Pipeline

> **Status:** Implemented: query normalisation (step 1, without rewrite/expansion/HyDE), hybrid retrieval with RRF (2), reranking (3), prompt assembly with citation handles (5), structured generation (6), citation validation (7), traces and eval hooks (8). Not implemented: context compression (4) and output guardrails (part of 7).

**Stages:**

1. Query processing (rewrite, expand, HyDE)
2. Hybrid retrieval (dense top-50 from Qdrant + BM25 top-50, fused with RRF)
3. Reranking (cross-encoder)
4. Context compression (extract relevant spans)
5. Prompt assembly
6. Generation (structured output)
7. Post-checks (citations, schema, guardrails)
8. Log trace + eval hooks

---

## K. Query Processing

> **Status:** Not implemented. Queries are only normalised (`backend/app/rag/query.py`).

- **Rewrite**: LLM rewrites ambiguous queries with conversation history.
- **Multi-query**: generate N paraphrases; retrieve for each; union.
- **HyDE**: LLM drafts a hypothetical answer; embed that; retrieve similar real chunks. Helps when queries are short/keyword-like.

Fallback order: rewrite → if low retrieval confidence → multi-query → HyDE.

---

## L. Reranking & Context Compression

> **Status:** Reranking implemented (fused top candidates → cross-encoder → `RERANK_TOP_K`). Compression not implemented.

- **Rerank**: top-50 candidates → cross-encoder → top-k (5-8).
- **Compression**: per-chunk extractive selection (LLM or lightweight extractor) keeps only spans relevant to the query, reduces tokens, raises faithfulness.

---

## M. LLM Inference Layer

> **Status:** Mostly implemented: provider dispatcher (Anthropic, OpenAI, Ollama), timeouts, retries and a circuit breaker, JSON responses without streaming, a generation span per call with tokens and an estimated cost from `config/model_pricing.toml`. Prompt caching is not implemented. Prompt and output text reach traces only when `TELEMETRY_INCLUDE_CONTENT=true`.

Unified gateway with provider adapters (Claude, OpenAI, local).

- Retries, timeouts, rate-limit handling.
- Prompt caching (Anthropic prompt cache for system + retrieved context where stable).
- Responses are returned whole as JSON; there is no SSE streaming.
- Every call emits a trace: inputs, outputs, tokens, cost, latency, model, prompt version.

---

## N. Structured Output Enforcement

> **Status:** Implemented differently. Answers are submitted through a `submit_answer` tool (answer, claims, status, unsupported notes) and validated by `backend/app/rag/grounding.py`; an answer with no valid citation becomes a refusal. There is no retry for answers (unparseable output falls back to JSON-in-text, then plain text); the judge gets one repair retry.

```python
from pydantic import BaseModel, Field

class Source(BaseModel):
    chunk_id: str
    quote: str

class Answer(BaseModel):
    question: str
    answer: str
    sources: list[Source] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    refusal: bool = False
```

- Use tool-use / JSON mode for hard constraints.
- Validate → on failure, retry with error message injected.
- Reject answers with no citations; force `refusal=True` when context is insufficient.

---

## O. Evaluation Engine

> **Status:** Implemented, plus answer correctness, deterministic retrieval metrics at K and refusal metrics. Each run records prompt version, model, judge, rubric version and a retrieval config hash; the commit SHA is not recorded.

Four core metrics (RAGAS-aligned):

| Metric | What it measures |
|---|---|
| **Faithfulness** | Answer claims supported by retrieved context |
| **Answer Relevance** | Answer addresses the actual question |
| **Context Precision** | Retrieved chunks are relevant (ranked) |
| **Context Recall** | All needed info is present in retrieved context |

Run per-example and aggregate. Store every run with: commit SHA, prompt version, model, retriever config.

---

## P. LLM-as-Judge

> **Status:** Partly. A stronger default judge (`claude-opus-5-5`) and a versioned rubric with anchors are implemented. Not implemented: few-shot examples, calibration against human labels (ratings are stored but not compared), order randomisation and pairwise comparison.

- Use a **stronger** model than the generator as judge (e.g., Opus judges Sonnet output).
- Rubrics with explicit criteria + few-shot examples.
- Calibrate: periodically sample judged items for human review; track judge/human agreement (Cohen's κ).
- Mitigate bias: randomize A/B order, strip model names, use pairwise comparisons where possible.

---

## Q. Golden Dataset & Benchmarks

> **Status:** Partly. 175 questions across five JSONL files (the largest has 54), two of them stratified into eight categories with difficulty labels. All were written by the project author. See `data/golden/README.md`.

- 100-500 curated `(question, ideal_answer, expected_sources)` triples.
- Stratified by difficulty, topic, and failure mode (multi-hop, numeric, refusal).
- Versioned in git (JSONL). Changes require PR review.
- Synthetic augmentation via LLM, but **human-reviewed** before entering the set.

---

## R. Regression Testing (CI/CD for Prompts)

> **Status:** Partly. A threshold gate compares a run with a pinned or previous baseline of the same configuration (`eval/regression_thresholds.toml`, `run_eval.py --fail-on-regression`, `scripts/regression_report.py`). Not implemented: a CI job running evals, per-example diffs, PR comments, `--baseline main` (only `--baseline RUN_FILE`), `post_pr_comment.py`.

Every prompt/model/retriever change triggers:

1. Run full golden set.
2. Compute all metrics.
3. Diff vs. previous baseline (per-example + aggregate).
4. Flag regressions (metric drop > threshold or new failures).
5. Post results to PR as a comment + dashboard link.

**GitHub Actions sketch:**

```yaml
on: pull_request
jobs:
  eval:
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e backend
      - run: python scripts/run_eval.py --baseline main --head ${{ github.sha }}
      - run: python scripts/post_pr_comment.py
```

---

## S. Backend API

> **Status:** Implemented, with more routes than listed (graph, advise, privacy, admin, metrics). Per-run regression reports are served at `GET /eval/runs/{id}/regression`.

Core endpoints:

| Method | Path | Purpose |
|---|---|---|
| POST | `/ingest` | Upload/queue documents |
| POST | `/ask` | Run RAG query (JSON, not streamed) |
| POST | `/compare/strategies` | Same question through classic / graph / agentic |
| GET  | `/traces/{id}` | Fetch trace |
| POST | `/eval/run` | Run eval on dataset |
| GET  | `/eval/runs` | List runs + scores |
| GET  | `/eval/runs/{id}/regression` | Regression report of one run against its baseline |

All responses typed via Pydantic; OpenAPI auto-generated.

---

## T. Frontend Dashboard

> **Status:** Partly. Pages: Ask, Upload, Compare (strategies side by side), Evaluation (Overview and Runs tabs, including run history and per-run regression reports) and Advisor. There is no Traces page (traces are served by `GET /traces/{id}`).

Pages:

- **Ask**: query box, answer, retrieved sources, citations.
- **Compare**: side-by-side A/B (two models or two prompt versions).
- **Evaluation**: an Overview tab (quality against cost for the latest run of each configuration, per-question-type breakdown, regression status) and a Runs tab (launch a run, run history, per-run regression report against the baseline).
- **Traces**: per-request drilldown: retrieval → rerank → prompt → output.

Stack: React + Vite + Tailwind + TanStack Query + Recharts.

---

## U. Observability & Tracing

> **Status:** Partly. Langfuse (self-hosted, OTLP) and Prometheus are implemented; per-chunk retrieval and rerank scores are returned in the API's retrieval diagnostics, while spans record counts only. Not implemented: query rewrites (none exist), eval scores on traces, alerts. The Grafana dashboard lacks panels for cost and stage latency.

- **Langfuse** (or OTel + Grafana) for every request.
- Track: query, rewrites, retrieved chunks (with scores), rerank scores, final prompt, model, tokens, cost, latency, eval scores.
- Dashboards: p50/p95 latency, cost/request, faithfulness over time, failure rate.
- Alerts on regression thresholds.

---

## V. Security & Guardrails

> **Status:** Partly. Implemented: PII masking at ingest and redaction in logs and traces, prompts that treat sources as untrusted, per-key rate limits, OIDC (JWT) authentication. Not implemented: stripping instructions from retrieved text, output toxicity/jailbreak filters. OIDC users are not rate-limited.

- **PII redaction** on ingest (optional) and on logs.
- **Prompt injection defenses**: treat retrieved content as untrusted; use role separation; strip instructions from retrieved text; refuse if tool calls requested from context.
- **Output filters**: toxicity/jailbreak classifier (lightweight).
- **Rate limits** per API key; auth via JWT.
- **Secrets**: never in code; `.env` + vault in prod.

---

## W. Testing Strategy

> **Status:** Partly. Unit tests (including RRF) and offline end-to-end integration tests exist. Not implemented: eval thresholds as CI gates, load tests, an OpenAPI-generated client (the frontend client is hand-written and typed).

- **Unit**: chunking, RRF fusion, schema validation.
- **Integration**: ingest → retrieve → generate on a tiny fixture corpus.
- **Eval tests**: golden set thresholds as CI gates.
- **Load**: Locust/k6 on `/ask`; measure p95 under concurrency.
- **Contract**: frontend ↔ backend via OpenAPI-generated client.

---

## X. Deployment

> **Status:** Partly. Docker Compose and a runtime image target exist. Not implemented: Alembic migrations, blue/green rollouts.

- **Local**: `docker compose --env-file .env -f infra/docker-compose.yml up` (API, frontend, qdrant, postgres; langfuse with `--profile tracing`). Ports bind to 127.0.0.1.
- **Staging/Prod**: build `backend/Dockerfile`'s default `runtime` target and run it on any container host; no Kubernetes manifests ship with the repo.
- **Config**: 12-factor; env vars only.
- **Migrations**: Alembic.
- **Rollouts**: blue/green for API; prompt versions are data, not code, so they can roll forward/back without redeploy.

---

## Y. Roadmap & Milestones

> **Status:** Historical plan; not a record of what was delivered.

| Week | Milestone |
|---|---|
| 1 | Repo skeleton, ingestion, dense retrieval, `/ask` MVP |
| 2 | Cross-encoder reranker, structured outputs, tracing |
| 3 | Eval engine (4 metrics) + golden set v1 |
| 4 | LLM-as-judge + regression CI |
| 5 | Frontend dashboard (Ask + Compare + Eval) |
| 6 | Query rewriting, multi-query, HyDE |
| 7 | Guardrails, load testing, docs |
| 8 | Deploy to staging, write-up & demo |

---

## Z. Glossary & References

**Glossary**

- **RAG**: Retrieval-Augmented Generation.
- **BM25**: classical lexical retrieval scoring.
- **RRF**: Reciprocal Rank Fusion; combines ranked lists.
- **HyDE**: Hypothetical Document Embeddings.
- **Cross-encoder**: scores (query, doc) jointly; more accurate, slower.
- **Faithfulness**: answer is grounded in retrieved context.
- **Golden set**: human-curated eval dataset.

**References**

- RAGAS: https://docs.ragas.io
- Langfuse: https://langfuse.com
- Qdrant: https://qdrant.tech
- Anthropic prompt caching: https://docs.anthropic.com
- HyDE paper: Gao et al., 2022- "Precise Zero-Shot Dense Retrieval without Relevance Labels"
- RRF: Cormack et al., 2009

---

**End of blueprint.** Sections whose Status line says "not implemented" have no corresponding module. The README describes the current state; `docs/ARCHITECTURE_AUDIT.md` records the state before the upgrade.
