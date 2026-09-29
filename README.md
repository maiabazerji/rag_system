# EvalRAG

Run the same question through three different RAG retrieval strategies, **Classic**, **Graph**, and **Agentic**, on your own corpus, and measure which one actually answers it better. Built on the Anthropic API (Claude).

It is built to be deployable by European and French organisations: multilingual retrieval and a French/English UI, per-document access control with SSO (Entra ID, ProConnect, Keycloak), GDPR tooling (PII masking, erasure, subject-access export, retention), telemetry that is off by default and self-hostable, and parsers for the formats those organisations actually use (`.docx`, `.odt`, `.eml`, scanned PDFs). See [Built for European deployments](#built-for-european-deployments).

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
| Frontend | Ask, compare, advise, ingest, evaluate; French/English UI | `frontend/src/` |
| Backend API | FastAPI routes for ingest, ask, compare, advise, eval, graph, traces, privacy, admin, metrics | `backend/app/api/` |
| Parsing & chunking | One parser per format, OCR, heading- and table-aware chunking | `backend/app/rag/parsers/`, `backend/app/rag/chunking.py` |
| RAG pipeline | Embed → retrieve (ACL-filtered) → rerank → generate | `backend/app/rag/` |
| Access control | Principals, OIDC SSO, tenants and groups, audit log | `backend/app/access.py`, `oidc.py`, `audit.py`, `auth.py` |
| Privacy | PII detection and masking, erasure, export, retention | `backend/app/privacy/` |
| Observability | Telemetry policy, Langfuse (OTLP) exporter, Prometheus metrics | `backend/app/tracing/`, `backend/app/monitoring.py` |
| Eval engine | LLM-as-judge scoring, regression tracking | `backend/app/eval/` |

Backing services: **Qdrant** (vectors), **Postgres** (API keys, usage accounting and the audit log), and optionally a self-hosted **Langfuse** (traces), **Prometheus + Grafana** (metrics) and **W&B** (eval dashboards).

Everything in this table is implemented. The embedding model is a real SentenceTransformer, the reranker is a real cross-encoder (with a BM25 fallback), and the judge is a real model call.

> **Check the reranker is the one you think it is.** If the cross-encoder cannot be downloaded, reranking degrades to BM25 silently and every request still returns 200. `"reranker": "bm25"` in the logs means the cross-encoder is not running and any comparison you draw is really a comparison of BM25. This bit this project for a long time: the configured model id was a repo that did not exist, so the fallback was the only code path ever taken.

### What happens when you ask a question

1. `POST /ask` with `{"question": "...", "strategy": "classic" | "graph" | "agentic"}` → [`ask.py`](backend/app/api/ask.py)
2. `answer_question` → [`generate.py`](backend/app/rag/generate.py) picks defaults, opens a trace, and calls `get_strategy(strategy).run(...)`
3. The strategy retrieves in its own way but returns the same `StrategyResult` (answer, sources, latency, tokens, trace)
4. All three generate through `AsyncAnthropic` ([`anthropic_provider.py`](backend/app/rag/providers/anthropic_provider.py)); the agentic strategy uses `tool_use_loop`
5. The trace is retrievable from `/traces/{id}` (in memory, so it does not survive a restart), and is exported to Langfuse when `TELEMETRY_MODE` allows it ([docs/monitoring.md](docs/monitoring.md))

Retrieval is **dense-only**: the query is embedded and matched against Qdrant. The lexical BM25 signal enters one step later, in [`rerank.py`](backend/app/rag/rerank.py), as the reranker's fallback when the cross-encoder is unavailable. This is a dense-retrieve-then-rerank pipeline, not hybrid retrieval in the fuse-two-retrievers sense.

### What happens when you run an eval

1. `POST /eval/run` → `run_evaluation` in [`metrics.py`](backend/app/eval/metrics.py)
2. Loads a golden dataset (question + ideal answer) from `data/golden/`
3. Answers each question, then scores it with the LLM judge in [`judge.py`](backend/app/eval/judge.py) on faithfulness, answer relevance, context precision, context recall and, when the example has an ideal answer, answer correctness
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

## Advisor: which strategy fits my project?

The **Advisor** page (`/advisor`, API `POST /advise`) takes a plain-language
project description, in English, French or any other language, and ranks the
three strategies for it.

1. **Profile.** Claude reads the description and fills a structured profile:
   corpus size, document types, languages, question mix (single-fact /
   relational / exploratory), latency budget, cost sensitivity, update rate,
   residency and compliance needs, and how entity-rich the documents are.
   Anything you pass in `overrides` wins over what was extracted. With no API
   key, or if the call fails, a keyword heuristic takes over and the response
   says `profile.source: "heuristic"`.
2. **Score.** `app/advisor/scoring.py` is deterministic and has no model calls.
   Each score is a base value plus named contributions, and each contribution
   comes with a sentence explaining it. The weights come from the baseline
   above. Classic is the default for single-fact questions, tight latency and
   cost-sensitive projects. Graph only pulls ahead on large, entity-rich
   corpora with relational questions, and it gets a smaller `RETRIEVAL_TOP_K`.
   Agentic, at ~1.9x tokens and with more refusals, is recommended only for
   ambiguous multi-step research with a generous budget. The response also
   includes a suggested config for each strategy, a hybrid-routing plan when
   the question mix is split, and generic hosting notes: EU residency, GDPR,
   on-prem, and multilingual embeddings such as `BAAI/bge-m3`. These notes are
   guidance, not legal advice.
3. **Validate.** The ranking is a prior, not a measurement. Ingest a sample of
   your documents, then send 10-20 real questions to `POST /advise/validate`.
   A question can include an `ideal_answer`, entered as `question || ideal
   answer` in the UI. By default the endpoint runs the top two strategies. It
   returns refusals, latency, tokens, token-F1 against the ideal answer and a
   judge score for each strategy, and names the `measured_winner`. The limit
   is 20 questions and 3 strategies per call.

```bash
curl -X POST localhost:8011/advise -H 'Content-Type: application/json' -d '{
  "description": "20 000 contrats PDF en français, questions sur les liens entre fournisseurs et filiales, réponse en moins de 5 s, hébergement UE.",
  "overrides": {"cost_sensitivity": "medium", "compliance": ["EU only", "GDPR"]}
}'
```

---

## Built for European deployments

Five features aimed at organisations in France and the EU. Each has its own settings in [`.env.example`](./.env.example); all are on by default except where noted.

### Language: French, English and more

- **Multilingual retrieval.** The default embedder is `intfloat/multilingual-e5-small` (384 dims, CPU-friendly) and the default reranker is the multilingual `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, so a French question finds English passages and vice versa. `BAAI/bge-m3` (1024 dims) is the higher-quality option. E5's `query: ` / `passage: ` prefixes are added automatically; override with `EMBEDDING_QUERY_PREFIX` / `EMBEDDING_PASSAGE_PREFIX`.
- **The model is recorded on the collection.** The embedding model is stamped into the Qdrant collection's metadata. Pointing the backend at a collection built with another model returns a 503 (`embedding_model_mismatch`) with re-ingest instructions instead of silently mixing vectors, even when both models have the same dimension.
- **Answers in the question's language.** Generation, agentic and graph prompts tell Claude to answer in the language of the question, and refusals are localised. Language is detected from the question, falling back to the request's `Accept-Language`. The BM25 fallback folds accents and drops French and English stopwords.
- **French UI.** A FR/EN switch in the navigation; the choice is remembered, and the default follows the browser.
- **French eval set.** `data/golden/golden_fr_v1.jsonl` holds 16 French questions over the bundled (English) corpus, to measure cross-lingual retrieval: `python scripts/run_eval.py --dataset golden_fr_v1`.

> **Upgrading an existing deployment?** The default embedder changed from `BAAI/bge-small-en-v1.5`. Both are 384-dim, so re-ingest: set a new `QDRANT_COLLECTION` and run `scripts/ingest.py`, or wipe with `down -v`. The multilingual reranker is about 2x slower on CPU; `cross-encoder/ms-marco-MiniLM-L-6-v2` remains available for English-only corpora.

### Access control and single sign-on

- **Who can call.** With `REQUIRE_API_KEY=true`, every route needs either an API key or an OIDC bearer token from `OIDC_ISSUER` (Microsoft Entra ID, ProConnect, Keycloak or any OIDC provider). Tokens are verified against the issuer's JWKS (`iss`, `aud`, `exp`, `nbf`). Groups come from `OIDC_GROUPS_CLAIM` (`groups`; dotted paths such as `realm_access.roles` work), admins from `OIDC_ADMIN_GROUP`, the tenant from `OIDC_TENANT_CLAIM` (`tid` on Entra). With auth off, everyone is a local admin in the default tenant, so local use is unchanged.
- **Who can read what.** Every chunk carries a `tenant` and `acl_groups`. A caller sees a chunk only if the tenant matches and they share at least one group (everyone is implicitly in `public`). The filter is applied inside Qdrant on every read path: search, fetch-by-id, the agentic tools, and the graph walk, which drops entities from documents the caller cannot read.
- **Labelling documents.** Uploads default to the uploader's groups (or `public` if they have none, or when `ACL_DEFAULT_PUBLIC=true`). Pass `groups` on `POST /ingest` (the Upload page has a field) to choose; only groups you belong to are accepted, except for admins, who may label with any group but get no extra read access. Chunks indexed before ACLs existed count as public until re-ingested (`ACL_LEGACY_PUBLIC=false` hides them).
- **Managing keys.** `python scripts/setup_auth.py --create-key NAME --groups legal,finance --tenant acme`, or `PUT /admin/keys/{id}/access` with `{groups, tenant, is_admin}`.
- **Audit log.** Every ask, compare, ingest, advise, eval, admin action and privacy erasure is recorded in Postgres: who, when, which documents were returned, status. Question text is stored only as a SHA-256 hash unless `AUDIT_STORE_QUESTIONS=true`. Query it with `GET /admin/audit?principal=&action=&since=`.

Principal ids are `key:<id>`, `oidc:<sub>` or `local`; the privacy endpoints below use the same ids.

### GDPR

- **PII masking at ingest.** Uploads are scanned for emails, French phone numbers, NIR (with its check key, including Corsica), IBAN (mod-97), SIREN/SIRET (Luhn), card numbers and IPv4 addresses. `PII_MODE_INGEST` is `mask` (replace with `[EMAIL]`, `[NIR]`, ...), `reject` (refuse with a 422 listing the types) or `off`. File metadata such as an email's sender is masked too. Only counts are stored, never the values. Person names are not detected; the detector interface is pluggable if you want to add an NER model.
- **Logs and traces.** `PII_REDACT_LOGS` and `PII_REDACT_TRACES` (both on) mask the same patterns in log lines and stored or exported traces. Questions still reach the LLM unmasked.
- **Right to erasure** (admin only): `DELETE /privacy/documents/{doc_id}`, `DELETE /privacy/sources?source_key=...`, `DELETE /privacy/principals/{principal_id}`. Each removes the data from vectors, graph triples, traces, eval runs (answers scrubbed, scores kept), usage rows and the audit log, and returns a count per store. They are idempotent; a partial failure returns 503 with the report, so run it again.
- **Subject access:** `GET /privacy/export/{principal_id}` returns the documents a principal owns, their usage, traces and audit events.
- **Retention.** Traces, eval runs and audit events are purged daily after `RETENTION_TRACES_DAYS` (7), `RETENTION_EVAL_RUNS_DAYS` (365) and `RETENTION_AUDIT_DAYS` (365); `0` keeps forever. `POST /privacy/retention/run` purges now.
- **Paperwork.** [`docs/gdpr/README.md`](docs/gdpr/README.md) covers what is stored where, lawful-basis notes, procedures with `curl` examples and subprocessors (Anthropic, and optional Langfuse/W&B). [`docs/gdpr/ropa_template.md`](docs/gdpr/ropa_template.md) is a CNIL-style *registre des activités de traitement* template in French and English. Neither is legal advice.

### Monitoring that stays in your infrastructure

- **Off by default.** `TELEMETRY_MODE=off` sends nothing anywhere. `self_hosted` exports only to hosts in `TELEMETRY_ALLOWED_HOSTS` (default `localhost,127.0.0.1,langfuse`); a `LANGFUSE_HOST` outside the list is refused at startup. `cloud` also needs `TELEMETRY_CLOUD_OPT_IN=true`.
- **Langfuse, self-hosted.** `--profile tracing` runs Langfuse v4 in the stack (web, worker, ClickHouse, Valkey, MinIO), with its own product telemetry off. Each request becomes a trace with retrieve, rerank, generation and tool-call spans, sent over OTLP from a background queue that never blocks a request. Text is PII-masked first; `TELEMETRY_INCLUDE_CONTENT=false` sends timings and token counts only.
- **W&B is opt-in.** Nothing is sent unless `WANDB_ENABLED=true` and the telemetry mode allows it; `WANDB_MODE=offline` keeps runs on disk.
- **Prometheus.** `METRICS_ENABLED=true` serves `/metrics` to localhost or with the admin key: request latency, per-strategy latency, tokens by model, refusals, circuit-breaker state and dropped telemetry. `--profile monitoring` adds Prometheus and a provisioned Grafana dashboard.

Details, and what leaves the machine in each mode: [`docs/monitoring.md`](docs/monitoring.md).

### Document formats

`.pdf`, `.docx`, `.pptx`, `.odt`, `.ods`, `.eml`, `.html`/`.htm`, `.txt`, `.md`, `.markdown`, `.rst`, `.csv` and `.json`, up to `MAX_UPLOAD_MB` (25 MB). `GET /ingest/formats` returns the live list and whether OCR is available.

- **Structure is kept.** Word, ODF and HTML headings become `#` headings, tables become pipe tables, lists become `-` items. Page headers and footers in `.docx` are skipped. Title, author, dates, page count and language come from the file's own metadata.
- **Chunks follow headings**, including French legal divisions written as plain lines (`Titre I`, `Chapitre 2`, `Section 3`, `Article L. 121-1`, `Art. 12`), then by size. Tables are never cut mid-row; a long table is split into row groups that each repeat the header. Each chunk records a `heading_path` such as `Chapitre 2 > Article 5`.
- **Scanned PDFs are OCR'd** page by page when a page has almost no text layer (`OCR_MIN_CHARS_PER_PAGE`), with tesseract in `OCR_LANGUAGES` (`fra+eng`), up to `OCR_MAX_PAGES`. The Docker image ships tesseract and poppler; without them the backend logs a warning and indexes what text there is.
- **Emails** index their headers and body (plain text preferred, HTML converted without scripts or styles) and parse attachments of a supported type, two levels deep at most.
- **Hostile files get a 422**: password-protected PDFs and Office/ODF files, and zip-based documents that would inflate past `MAX_UNCOMPRESSED_MB` (checked before any parser runs).

---

## Quickstart (Docker)

Brings up frontend, backend, Qdrant and Postgres together.

> **Note.** The Compose file lives at `infra/docker-compose.yml`, not at the project root, and Compose only auto-loads a `.env` sitting next to it. Run every command from the project root as `docker compose --env-file .env -f infra/docker-compose.yml ...`. Without `--env-file`, Compose stops with `required variable POSTGRES_PASSWORD is missing a value`; a bare `docker compose down` fails with `no configuration file provided`.

### 1. Configure

```powershell
# Windows PowerShell (from the project root)
copy .env.example .env
```

```bash
# macOS / Linux
cp .env.example .env
```

Open `.env` and check these:

| Variable | Required? | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes, for answers | Without it the stack starts and the UI loads, but every answer is a refusal |
| `POSTGRES_PASSWORD` | Yes | Compose refuses to start without it. `.env.example` ships a development value; change it |
| `LANGFUSE_*`, `CLICKHOUSE_PASSWORD`, `MINIO_ROOT_PASSWORD` | Only with `--profile tracing` | Blank by default; Langfuse will not start until they are set. [docs/monitoring.md](docs/monitoring.md) lists them |
| `GRAFANA_ADMIN_PASSWORD` | Only with `--profile monitoring` | Blank by default |
| `QDRANT_API_KEY` | No | Blank leaves Qdrant unauthenticated (fine on loopback). Set it to turn Qdrant auth on; Compose passes it to both Qdrant and the backend |
| `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` | No | Default `0`. Add them as `1` only after the models are downloaded (step 2) |

Everything else has a working default.

### 2. Models (optional)

The backend needs two Hugging Face models: the embedder (`EMBEDDING_MODEL`) and the cross-encoder reranker (`RERANKER_MODEL`). By default it downloads them on first use into `.hf_cache/` at the project root, which Compose bind-mounts, so they survive rebuilds. To fetch them up front instead (useful when Docker's DNS is flaky, and required before turning on offline mode):

```bash
pip install huggingface_hub
python scripts/download_models.py --cache-dir .hf_cache
```

Then you may set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` in `.env`. Do not set them on an empty cache: the embedder cannot load, and the reranker silently degrades to BM25.

### 3. Start

```bash
docker compose --env-file .env -f infra/docker-compose.yml up -d --build
```

Add `--profile tracing` for a self-hosted Langfuse (after setting its secrets), or `--profile monitoring` for Prometheus and Grafana. The first build downloads a couple of GB; later starts take seconds.

Every port is published on **127.0.0.1 only**, so nothing is reachable from other machines on your network. That is deliberate: the services run with development credentials and, by default, no API key.

### 4. Open

| App | URL | Notes |
|---|---|---|
| Frontend | <http://localhost:5173> | Vite dev server with hot reload |
| Backend | <http://localhost:8011/docs> | OpenAPI / Swagger UI |
| Health | <http://localhost:8011/health> | Which providers are configured |
| Langfuse | <http://localhost:3100> | Only with `--profile tracing` |
| Prometheus | <http://localhost:9090> | Only with `--profile monitoring` |
| Grafana | <http://localhost:3300> | Only with `--profile monitoring` |

### 5. Ingest and evaluate

```bash
docker compose --env-file .env -f infra/docker-compose.yml exec backend python /scripts/ingest.py
docker compose --env-file .env -f infra/docker-compose.yml exec backend python /scripts/run_eval.py --dataset golden_v1
```

Or drag files onto the Upload page, optionally choosing which groups may read them. Supported formats and how they are parsed: [Document formats](#document-formats).

### 6. Logs, stop, wipe

```bash
docker compose --env-file .env -f infra/docker-compose.yml logs -f backend      # follow logs
docker compose --env-file .env -f infra/docker-compose.yml ps                   # what's running
docker compose --env-file .env -f infra/docker-compose.yml down                 # stop (keeps data)
docker compose --env-file .env -f infra/docker-compose.yml down -v              # also wipe volumes
```

### Port map (host → container)

| Service | Host (127.0.0.1) | Container | Why this host port |
|---|---|---|---|
| Frontend | `5173` | `5173` | Vite default |
| Backend | `8011` | `8000` | `8001` was taken on the author's machine |
| Langfuse | `3100` | `3000` | `3000` is a common Next.js default |
| Postgres | `5434` | `5432` | `5432`/`5433` taken by other stacks |
| Qdrant | `6333` | `6333` | free |
| Prometheus | `9090` | `9090` | `--profile monitoring` |
| Grafana | `3300` | `3000` | `--profile monitoring` |
| MinIO (Langfuse) | `9190` | `9000` | `--profile tracing` |

Containers reach each other by **service name** on the internal network (`qdrant:6333`, `postgres:5432`), never `localhost`. The host ports are only for reaching services from your machine.

`backend/app/` and `frontend/src/` are bind-mounted, so edits appear inside the containers immediately and uvicorn `--reload` / Vite HMR pick them up.

---

## Authentication

Auth is **off by default** so a fresh clone runs with no setup: every caller is a local admin, which is fine on localhost and not fine anywhere else. Turning it on enables both API keys and, if `OIDC_ISSUER` is set, SSO tokens; see [Access control and single sign-on](#access-control-and-single-sign-on) for groups, tenants and the audit log.

To turn it on:

```bash
# 1. In .env
REQUIRE_API_KEY=true
ADMIN_KEY=<python -c "import secrets; print(secrets.token_urlsafe(32))">

# 2. Restart, then mint a key
python scripts/setup_auth.py --create-key "my-laptop" --groups legal --tenant default
```

The key is shown once, only its SHA-256 hash is stored. Paste it into the field the UI shows (it appears automatically when the backend reports `auth_required`), or send it yourself:

```bash
curl -H "Authorization: Bearer sk_..." http://localhost:8011/ingest/stats
```

Each key carries a per-minute rate limit (`--rpm`, default 10) enforced across every protected route and weighted by cost: an agentic question counts 3, and `/compare` and `/eval/run` count per variant or example. Token usage is recorded per request. OIDC users are not rate-limited or metered, since they have no key row. Graph build and reset, `/admin` and `/privacy` also need the `X-Admin-Key` header.

```bash
python scripts/setup_auth.py --list-keys           # keys and 24h usage
python scripts/setup_auth.py --deactivate-key 1    # revoke, effective immediately
```

Auth and the audit log require Postgres. With `REQUIRE_API_KEY=false` and no `ADMIN_KEY` the backend never opens a database connection, and auditing is skipped.

---

## Configuration

[`.env.example`](./.env.example) documents every setting. The ones worth knowing:

| Setting | Default | Effect |
|---|---|---|
| `GENERATOR_MODEL` | `claude-sonnet-5` | Model that writes answers |
| `JUDGE_MODEL` | `claude-opus-5-5` | Model that grades them |
| `RETRIEVAL_MODE` | `hybrid` | `dense`, `sparse` (BM25) or `hybrid` (both, fused with RRF) |
| `DENSE_TOP_K` | `50` | Dense candidates before fusion (`RETRIEVAL_TOP_K` is its deprecated alias) |
| `BM25_TOP_K` | `50` | BM25 candidates before fusion |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant |
| `RERANK_TOP_K` | `8` | Chunks kept after reranking |
| `FINAL_CONTEXT_K` | `RERANK_TOP_K` | Chunks sent to the model, the main cost lever |
| `CHUNK_SIZE_TOKENS` | `300` | Words per chunk (applies to new ingests), capped to fit `CHUNK_MAX_MODEL_TOKENS` |
| `CHUNK_STRATEGY` | `structured` | `structured` (headings, paragraphs, sentences) or `fixed` (legacy word windows) |
| `CHUNK_MAX_MODEL_TOKENS` | `512` | Embedder/reranker input window; chunk size is capped at this / 1.5 words (0 disables) |
| `AGENTIC_MAX_ITERS` | `15` | Bounds worst-case cost of one agentic question |
| `MAX_UPLOAD_MB` | `25` | Upload ceiling |
| `REQUIRE_API_KEY` | `false` | Enforce API keys |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | Changing it requires a re-ingest |
| `RERANKER_MODEL` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Multilingual cross-encoder |
| `OIDC_ISSUER` | empty | Accept SSO tokens from this issuer |
| `DEFAULT_TENANT` | `default` | Tenant for keys and users that carry none |
| `AUDIT_STORE_QUESTIONS` | `false` | Keep question text in the audit log, not just its hash |
| `PII_MODE_INGEST` | `mask` | `off`, `mask` or `reject` personal data found in uploads |
| `RETENTION_TRACES_DAYS` | `7` | Days before request traces are purged (`0` = never) |
| `TELEMETRY_MODE` | `off` | Where traces may go: `off`, `self_hosted`, `cloud` |
| `METRICS_ENABLED` | `false` | Serve Prometheus metrics on `/metrics` |
| `OCR_ENABLED` / `OCR_LANGUAGES` | `true` / `fra+eng` | OCR for scanned PDFs |

The embedding model is stamped on the Qdrant collection; the backend refuses a collection built with a different one (see [Language](#language-french-english-and-more)).

---

## Development

```bash
cd backend
python -m venv .venv && . .venv/Scripts/activate    # or bin/activate on macOS/Linux
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -e ".[dev]"

ruff check app tests     # lint
mypy app                 # types
pytest -q                # ~930 tests
```

```bash
cd frontend
npm ci
npm run lint             # ESLint (typescript-eslint + react-hooks)
npm run typecheck        # tsc, including tests
npm test                 # vitest
npm run build            # typecheck + production build
npm run dev              # dev server on :5173
```

The frontend needs Node 22.12 or newer; `vitest` 5 refuses to start on Node 20.

The app imports without any configuration. `settings.validate_startup()` runs in the FastAPI lifespan rather than at import time, so linters, tests and tooling all work in a bare checkout.

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs on every push to `main` and every pull request: the backend gates above (ruff, mypy, pytest) plus mypy on `scripts/`, the frontend gates (lint, typecheck, tests, build), `docker compose config`, a build of both Dockerfiles, and dependency audits (`npm audit --audit-level=high` blocks; `pip-audit` reports without failing the run).

More detail, including troubleshooting: [`backend/SETUP.md`](./backend/SETUP.md).

---

## Troubleshooting

**Every answer is a refusal** → `ANTHROPIC_API_KEY` is unset. Check `GET /health`.

**`Connection refused` from Postgres, Qdrant** → services aren't up. `docker compose --env-file .env -f infra/docker-compose.yml up -d`.

**401 on every request** → `REQUIRE_API_KEY=true` and no key is set. See [Authentication](#authentication).

**429 Rate limit exceeded** → your key's per-minute limit. Raise it with `--rpm` when creating the key.

**Answers are slow** → lower `RETRIEVAL_TOP_K` and `RERANK_TOP_K`; try `claude-haiku-4-5-20251001` as `GENERATOR_MODEL`. The agentic strategy is inherently slower, it makes up to `AGENTIC_MAX_ITERS` model calls per question.

**503 `embedding_model_mismatch`** → the collection was built with another embedding model. Re-ingest into a new `QDRANT_COLLECTION`, or wipe with `down -v`.

**403 on ingest** → you asked for a group you are not a member of.

**A document is missing from answers for one user** → check its `acl_groups` and tenant against the user's; `GET /admin/audit` shows which documents each request returned.

**422 `pii_rejected` on upload** → `PII_MODE_INGEST=reject` found personal data; the response lists the types.

**Graph strategy returns nothing** → build the graph first: `POST /graph/build` (admin key required), then poll `GET /graph/build/{job_id}`. It runs one model call per chunk, so it takes a while on a large corpus.

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
- [`docs/gdpr/README.md`](docs/gdpr/README.md), personal data, erasure, retention, subprocessors
- [`docs/monitoring.md`](docs/monitoring.md), telemetry modes, Langfuse, Prometheus
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
