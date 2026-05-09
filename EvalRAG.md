# 🧠 EvalRAG — Production LLM Evaluation & Advanced RAG Platform

## 📑 Table of Contents

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

**EvalRAG** is a full-stack AI platform demonstrating real-world LLM engineering: retrieval, evaluation, prompt regression testing, structured outputs, reranking, and observability. It simulates the systems production companies deploy when shipping LLM-based assistants.

**Four layers:**

1. **Frontend** — AI dashboard (ask questions, compare models A/B, view scores, track regressions).
2. **Backend API** — ingestion, embeddings, retrieval, inference, evaluation.
3. **Evaluation Engine** — RAG metrics, LLM-as-judge, regression tests, benchmarks.
4. **Data Layer** — vector DB, document store, golden dataset store.

---

## B. Learning Goals & Outcomes

By completion, you will have hands-on experience with:

- Building hybrid retrieval (dense + BM25) with cross-encoder reranking
- Query rewriting, multi-query expansion, and HyDE
- Enforcing structured outputs via Pydantic + constrained decoding
- Automated evaluation: faithfulness, answer relevance, context precision/recall
- LLM-as-judge pipelines with calibration
- Prompt regression testing (CI/CD for prompts)
- Full LLM observability: tracing, cost, latency, failure analysis
- Deploying a production-style FastAPI + React stack

---

## C. High-Level Architecture

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

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11 + TypeScript | Standard for ML + web |
| Backend | FastAPI | Async, typed, OpenAPI |
| Frontend | React + Vite + Tailwind | Fast iteration |
| Vector DB | Qdrant (or pgvector) | Hybrid search, filters |
| Sparse | BM25 via `rank_bm25` | Lexical baseline |
| Embeddings | `text-embedding-3-large` or `bge-large-en` | SOTA quality |
| Reranker | `bge-reranker-large` / Cohere Rerank | Cross-encoder accuracy |
| LLM | Claude Opus/Sonnet 4.6, GPT-4o (comparison) | Multi-model A/B |
| Structured out | Pydantic + `instructor` / tool-use | Schema safety |
| Eval | RAGAS + custom judges | Industry baseline |
| Tracing | Langfuse or OpenTelemetry | Free, self-host |
| Storage | Postgres + S3/MinIO | Durable |
| Queue | Redis + RQ/Celery | Async ingestion |
| Deploy | Docker Compose → k8s | Local-first, scalable |

---

## E. Repository Structure

```
evalrag/
├── backend/
│   ├── app/
│   │   ├── api/              # FastAPI routers
│   │   ├── rag/              # retrieval + generation
│   │   │   ├── ingest.py
│   │   │   ├── embed.py
│   │   │   ├── retrieve.py   # hybrid
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
│   │   ├── pages/            # Ask, Compare, Eval, Regressions
│   │   ├── components/
│   │   └── api/
│   └── package.json
├── data/
│   ├── docs/                 # source documents
│   └── golden/               # evaluation datasets
├── infra/
│   ├── docker-compose.yml
│   └── k8s/
├── scripts/
│   ├── ingest.py
│   └── run_eval.py
└── README.md
```

---

## F. Environment Setup

```bash
# Clone + install
git clone <repo> evalrag && cd evalrag
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -e backend

# Frontend
cd frontend && npm install && cd ..

# Services
docker compose -f infra/docker-compose.yml up -d   # qdrant, postgres, redis, langfuse
```

---

## G. Data Layer

Three stores with clear roles:

- **Document Store** (Postgres + object storage) — raw docs, chunks, metadata, versions.
- **Vector Store** (Qdrant) — dense embeddings + payload for hybrid filtering.
- **Eval Store** (Postgres) — golden datasets, run results, regression history.

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

```
PDF/HTML/MD/DOCX → Loader → Clean → Chunk → Embed → Index (dense+sparse)
```

- **Loaders**: `unstructured`, `pypdf`, `trafilatura`.
- **Cleaning**: strip boilerplate, normalize whitespace, dedupe.
- **Chunking**: semantic (sentence-aware) with overlap; target 400–800 tokens.
- **Idempotency**: hash each chunk; skip if unchanged.
- **Async**: enqueue via Redis; workers process in parallel.

---

## I. Embeddings & Vector Store

- Batch embed (64–128) with retries + exponential backoff.
- Store embedding model name + version with each vector (for reindexing).
- Qdrant collection with HNSW + `payload` for filters (`doc_id`, `tags`, `date`).
- Build parallel BM25 index keyed by `chunk_id` for sparse retrieval.

---

## J. Advanced RAG Pipeline

**Stages:**

1. Query processing (rewrite, expand, HyDE)
2. Hybrid retrieval (dense + BM25, RRF fusion)
3. Reranking (cross-encoder)
4. Context compression (extract relevant spans)
5. Prompt assembly
6. Generation (structured output)
7. Post-checks (citations, schema, guardrails)
8. Log trace + eval hooks

---

## K. Query Processing

- **Rewrite**: LLM rewrites ambiguous queries with conversation history.
- **Multi-query**: generate N paraphrases; retrieve for each; union.
- **HyDE**: LLM drafts a hypothetical answer; embed that; retrieve similar real chunks. Helps when queries are short/keyword-like.

Fallback order: rewrite → if low retrieval confidence → multi-query → HyDE.

---

## L. Reranking & Context Compression

- **Rerank**: top-50 candidates → cross-encoder → top-k (5–8).
- **Compression**: per-chunk extractive selection (LLM or lightweight extractor) keeps only spans relevant to the query — reduces tokens, raises faithfulness.

---

## M. LLM Inference Layer

Unified gateway with provider adapters (Claude, OpenAI, local).

- Retries, timeouts, rate-limit handling.
- Prompt caching (Anthropic prompt cache for system + retrieved context where stable).
- Streaming via SSE to frontend.
- Every call emits a trace: inputs, outputs, tokens, cost, latency, model, prompt version.

---

## N. Structured Output Enforcement

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

- Use a **stronger** model than the generator as judge (e.g., Opus judges Sonnet output).
- Rubrics with explicit criteria + few-shot examples.
- Calibrate: periodically sample judged items for human review; track judge/human agreement (Cohen's κ).
- Mitigate bias: randomize A/B order, strip model names, use pairwise comparisons where possible.

---

## Q. Golden Dataset & Benchmarks

- 100–500 curated `(question, ideal_answer, expected_sources)` triples.
- Stratified by difficulty, topic, and failure mode (multi-hop, numeric, refusal).
- Versioned in git (JSONL). Changes require PR review.
- Synthetic augmentation via LLM, but **human-reviewed** before entering the set.

---

## R. Regression Testing (CI/CD for Prompts)

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

Core endpoints:

| Method | Path | Purpose |
|---|---|---|
| POST | `/ingest` | Upload/queue documents |
| POST | `/ask` | Run RAG query (streams) |
| POST | `/compare` | A/B across models/prompts |
| GET  | `/traces/{id}` | Fetch trace |
| POST | `/eval/run` | Run eval on dataset |
| GET  | `/eval/runs` | List runs + scores |
| GET  | `/eval/regressions` | History + diffs |

All responses typed via Pydantic; OpenAPI auto-generated.

---

## T. Frontend Dashboard

Pages:

- **Ask** — query box, streamed answer, retrieved chunks with highlights, citations.
- **Compare** — side-by-side A/B (two models or two prompt versions).
- **Eval** — run overview, per-metric charts, failure drilldown.
- **Regressions** — timeline of runs, diff viewer for changed examples.
- **Traces** — per-request drilldown: retrieval → rerank → prompt → output.

Stack: React + Vite + Tailwind + TanStack Query + Recharts.

---

## U. Observability & Tracing

- **Langfuse** (or OTel + Grafana) for every request.
- Track: query, rewrites, retrieved chunks (with scores), rerank scores, final prompt, model, tokens, cost, latency, eval scores.
- Dashboards: p50/p95 latency, cost/request, faithfulness over time, failure rate.
- Alerts on regression thresholds.

---

## V. Security & Guardrails

- **PII redaction** on ingest (optional) and on logs.
- **Prompt injection defenses**: treat retrieved content as untrusted; use role separation; strip instructions from retrieved text; refuse if tool calls requested from context.
- **Output filters**: toxicity/jailbreak classifier (lightweight).
- **Rate limits** per API key; auth via JWT.
- **Secrets**: never in code; `.env` + vault in prod.

---

## W. Testing Strategy

- **Unit** — chunking, RRF fusion, schema validation.
- **Integration** — ingest → retrieve → generate on a tiny fixture corpus.
- **Eval tests** — golden set thresholds as CI gates.
- **Load** — Locust/k6 on `/ask`; measure p95 under concurrency.
- **Contract** — frontend ↔ backend via OpenAPI-generated client.

---

## X. Deployment

- **Local**: `docker compose up` (API, frontend, qdrant, postgres, redis, langfuse).
- **Staging/Prod**: container images → k8s (or Fly.io / Render for simplicity).
- **Config**: 12-factor; env vars only.
- **Migrations**: Alembic.
- **Rollouts**: blue/green for API; prompt versions are data, not code, so they can roll forward/back without redeploy.

---

## Y. Roadmap & Milestones

| Week | Milestone |
|---|---|
| 1 | Repo skeleton, ingestion, dense retrieval, `/ask` MVP |
| 2 | Hybrid + reranker, structured outputs, tracing |
| 3 | Eval engine (4 metrics) + golden set v1 |
| 4 | LLM-as-judge + regression CI |
| 5 | Frontend dashboard (Ask + Compare + Eval) |
| 6 | Query rewriting, multi-query, HyDE |
| 7 | Guardrails, load testing, docs |
| 8 | Deploy to staging, write-up & demo |

---

## Z. Glossary & References

**Glossary**

- **RAG** — Retrieval-Augmented Generation.
- **BM25** — classical lexical retrieval scoring.
- **RRF** — Reciprocal Rank Fusion; combines ranked lists.
- **HyDE** — Hypothetical Document Embeddings.
- **Cross-encoder** — scores (query, doc) jointly; more accurate, slower.
- **Faithfulness** — answer is grounded in retrieved context.
- **Golden set** — human-curated eval dataset.

**References**

- RAGAS: https://docs.ragas.io
- Langfuse: https://langfuse.com
- Qdrant: https://qdrant.tech
- Anthropic prompt caching: https://docs.anthropic.com
- HyDE paper: Gao et al., 2022 — "Precise Zero-Shot Dense Retrieval without Relevance Labels"
- RRF: Cormack et al., 2009

---

**End of blueprint.** Start with section F, then build iteratively through the Y roadmap. Every section above maps to a concrete module in `E. Repository Structure`.
