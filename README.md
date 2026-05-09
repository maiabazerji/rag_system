# EvalRAG

A hands-on platform for building a **RAG** (retrieval-augmented generation) system *and* the **evaluation harness** that keeps it honest in production. See [EvalRAG.md](./EvalRAG.md) for the full A→Z blueprint.

This README is the **onboarding guide** — it explains what the project is, how the pieces fit together, and where to look in the code.

---

## What you're building, in plain English

A RAG system lets an LLM answer questions grounded in *your* documents (not just its training data). You:

1. **Ingest** docs → split into chunks → embed each chunk → store in a vector DB.
2. **Ask** a question → embed it → retrieve the most similar chunks → stuff them into a prompt → send to the LLM → get an answer.

The hard part isn't the first version — it's knowing whether a change made it *better*. So EvalRAG pairs the RAG pipeline with an **evaluation engine** that scores answers on faithfulness, relevance, and retrieval quality, plus a **regression system** so you catch when a prompt tweak silently breaks things.

## The four layers

| Layer | What it does | Where it lives |
|---|---|---|
| Frontend | Dashboard: ask questions, A/B compare models, view eval scores, browse traces | `frontend/src/` |
| Backend API | FastAPI — routes for ingest, ask, compare, eval, traces | `backend/app/api/` |
| RAG pipeline | Chunk → embed → retrieve → rerank → generate | `backend/app/rag/` |
| Eval engine | Metrics, LLM-as-judge, regression tracking | `backend/app/eval/` |

Backing stores (run via Docker): **Qdrant** (vectors), **Postgres** (docs, eval results), **Redis** (queue), **Langfuse** (traces).

---

## What happens when you ask a question

Follow the trace below to learn the codebase in the order the bytes flow:

1. **`POST /ask`** → [`backend/app/api/ask.py`](backend/app/api/ask.py) — thin FastAPI route, delegates to `answer_question`.
2. **`answer_question`** → [`backend/app/rag/generate.py`](backend/app/rag/generate.py) — the orchestrator. Opens a trace span, then:
3. **`hybrid_search`** → [`backend/app/rag/retrieve.py`](backend/app/rag/retrieve.py) — embeds the query and asks Qdrant for the top-K similar chunks.
4. **`embed_query`** → [`backend/app/rag/embed.py`](backend/app/rag/embed.py) — turns text into a vector.
5. **`rerank`** → [`backend/app/rag/rerank.py`](backend/app/rag/rerank.py) — a cross-encoder re-scores the top-K more precisely. *(Currently a pass-through stub.)*
6. **`compress_context`** → same file — trim each chunk down to spans that actually matter to the query. *(Stub.)*
7. **Prompt assembly** → [`backend/app/prompts/`](backend/app/prompts/) holds versioned prompt templates; `generate.py` fills in `{question}` and `{context}`.
8. **LLM call** → `AsyncAnthropic` sends the filled prompt to Claude, returns the answer.
9. **Trace close** → events recorded under an in-memory trace id you can fetch from `/traces/{id}`.

The shape returned by every endpoint is defined in [`backend/app/schemas/`](backend/app/schemas/) — Pydantic models like `Answer`, `Source`, `AskRequest`. These also enforce structured outputs from the LLM (the "N" section of the blueprint).

### What happens when you run an eval

1. `POST /eval/run` → [`eval_routes.py`](backend/app/api/eval_routes.py) → `run_evaluation` in [`app/eval/metrics.py`](backend/app/eval/metrics.py).
2. It loads a **golden dataset** (question + expected answer pairs) from `data/golden/`.
3. For each question: run the RAG pipeline → score with deterministic metrics *and* an **LLM-as-judge** call ([`app/eval/judge.py`](backend/app/eval/judge.py)).
4. Results saved, compared against prior runs → [`app/eval/regression.py`](backend/app/eval/regression.py) flags regressions.
5. Optionally streamed to **W&B** via [`app/tracing/wandb_tracer.py`](backend/app/tracing/wandb_tracer.py) for dashboards.

---

## What's real vs. stubbed

The scaffold runs end-to-end but many modules are intentionally stubs so you can wire them up as a learning exercise. See `EvalRAG.md` sections I, L, M, P for the recipes.

| Module | Status | To wire |
|---|---|---|
| `rag/generate.py` (Anthropic call) | **Real** — just needs `ANTHROPIC_API_KEY` | — |
| `rag/embed.py` | Stub | Real embedding model (bge-small / OpenAI) |
| `rag/retrieve.py` | Real (talks to Qdrant) | Add BM25 for true hybrid |
| `rag/rerank.py` | Stub (pass-through) | `bge-reranker-large` cross-encoder |
| `eval/judge.py` | Stub (returns zeros) | Real judge LLM call |
| `eval/metrics.py` | Partial | RAGAS metrics |
| `tracing/wandb_tracer.py` | Real (no-ops if key missing) | — |
| Langfuse tracing | Infra only | Wire client into `app/tracing/` |

Missing provider keys are handled gracefully — `generate.py` returns a clear refusal if `ANTHROPIC_API_KEY` is blank; W&B silently no-ops. Check which providers are configured:

```bash
curl http://localhost:8011/health
# {"status":"ok","providers":{"anthropic":true,"openai":false,...}}
```

---

## Suggested learning path

Work outward from the ask flow. Each step is a self-contained concept you can ship:

1. **Get a real answer end-to-end** — add `ANTHROPIC_API_KEY`, ingest a few docs, ask a question. Watch it refuse when context is empty, then succeed after ingest.
2. **Replace the embedding stub** — `rag/embed.py`. Start with a local model (`bge-small-en-v1.5`) — no API key needed.
3. **Add real reranking** — `rag/rerank.py`. Measure retrieval quality before/after (that's what eval is for).
4. **Wire the judge** — `eval/judge.py`. Call Claude with the rubric, parse JSON, return scores.
5. **Build a golden dataset** — 20–50 Q/A pairs in `data/golden/`. Now you can detect regressions.
6. **Add A/B compare** — swap `generator_model` in config, route through `/compare`, compare judge scores.
7. **Turn on tracing** — Langfuse (already in docker-compose) for per-request traces; W&B for eval run dashboards.

Each step is testable in isolation — that's the point of the module layout.

---

## Quickstart (Docker — everything at once)

Spins up frontend + backend + Qdrant + Postgres + Redis + Langfuse together as one **Compose stack**.

> **Golden rule.** The Compose file lives at `infra/docker-compose.yml`, **not** at the project root. Every `docker compose` command needs to point at it — either with `-f infra/docker-compose.yml`, or by `cd`-ing into `infra/` first, or by setting `$env:COMPOSE_FILE`. Plain `docker compose down` from the project root will fail with `no configuration file provided: not found`.

### 1 · Configure (one-time)

```powershell
# Windows PowerShell (cwd = project root)
copy .env.example .env
# Open .env and fill ANTHROPIC_API_KEY (others can stay blank for a smoke test)
```

```bash
# macOS / Linux
cp .env.example .env
```

All provider keys are **optional** for a first smoke test — `generate.py` returns a graceful refusal when `ANTHROPIC_API_KEY` is blank, W&B silently no-ops, etc.

### 2 · Build + start the stack

```bash
docker compose -f infra/docker-compose.yml up -d --build
```

`up` creates the containers, `-d` runs them detached (background), `--build` rebuilds the local `backend` and `frontend` images from their Dockerfiles. First run downloads ~2 GB of base images; subsequent runs are seconds.

### 3 · Open the apps

| App      | URL                              | Notes                              |
|----------|----------------------------------|------------------------------------|
| Frontend | <http://localhost:5173>          | Vite dev server, HMR enabled       |
| Backend  | <http://localhost:8011/docs>     | FastAPI Swagger UI                 |
| Health   | <http://localhost:8011/health>   | Reports which provider keys are set |
| Langfuse | <http://localhost:3100>          | Tracing dashboard (sign up locally) |

### 4 · Ingest some docs and run an eval

```bash
docker compose -f infra/docker-compose.yml exec backend python /scripts/ingest.py
docker compose -f infra/docker-compose.yml exec backend python /scripts/run_eval.py --dataset golden_v1
```

`exec` runs a one-shot command inside an already-running container. The `/scripts` path is bind-mounted from `./scripts/` on your host, so you can edit and re-run without rebuilding the image.

### 5 · Tail logs / stop / wipe

```bash
docker compose -f infra/docker-compose.yml logs -f backend frontend   # follow logs
docker compose -f infra/docker-compose.yml ps                         # what's running
docker compose -f infra/docker-compose.yml down                       # stop + remove containers (keeps data)
docker compose -f infra/docker-compose.yml down -v                    # also delete named volumes (wipes Postgres + Qdrant)
```

### Compose cheat sheet

If you don't want to type `-f infra/docker-compose.yml` every time, pick one:

```powershell
# A. Run commands from inside the infra folder
cd infra
docker compose ps

# B. Sticky for the whole shell session (PowerShell)
$env:COMPOSE_FILE = "infra/docker-compose.yml"
docker compose ps

# C. Sticky for the whole shell session (bash / zsh)
export COMPOSE_FILE=infra/docker-compose.yml
docker compose ps
```

### Service port map (host → container)

| Service   | Host port | Container port | Why this host port? |
|-----------|-----------|----------------|----------------------|
| Frontend  | `5173`    | `5173`         | Vite default         |
| Backend   | `8011`    | `8000`         | `8001` was busy on this machine — bumped by 10 |
| Langfuse  | `3100`    | `3000`         | `3000` was busy — common React/Next.js default |
| Postgres  | `5434`    | `5432`         | `5432`/`5433` already taken by other stacks    |
| Redis     | `6381`    | `6379`         | `6380` already taken by `e-invoice-redis`      |
| Qdrant    | `6333`    | `6333`         | Free                                           |

**`HOST:CONTAINER`** is how `docker compose` publishes ports. Containers always talk to each other over the **container ports** on the internal `infra_default` Docker network (e.g. `redis:6379`, `postgres:5432` — note the service name as DNS, not `localhost`). The **host ports** above are only how *you* reach the services from your laptop. The backend env vars in `infra/docker-compose.yml` use the container ports for exactly this reason.

Code in `backend/app/` and `frontend/src/` is **bind-mounted** — edits on your host instantly appear inside the containers, and uvicorn `--reload` / Vite HMR pick them up automatically. Named volumes (`qdrant_data`, `pg_data`) persist data across `down` but are wiped by `down -v`.

---

## Troubleshooting

### A. `no configuration file provided: not found`

```
$ docker compose down
no configuration file provided: not found
```

You ran `docker compose` from a directory that has no `docker-compose.yml` (probably the project root). The file is at `infra/docker-compose.yml`. Fix any of three ways:

```bash
docker compose -f infra/docker-compose.yml down       # 1. point at it explicitly
cd infra && docker compose down                       # 2. run from the folder it lives in
$env:COMPOSE_FILE = "infra/docker-compose.yml"        # 3. PowerShell — sticky for the session
docker compose down
```

### B. `Bind for 0.0.0.0:<port> failed: port is already allocated`

Another process on your machine is already **listening** on that host port. Docker can't give the same TCP port to two listeners — only one process can hold a `(host, port)` pair at a time. Diagnose:

```powershell
# Windows / PowerShell — find the owner
netstat -ano | findstr :6380           # last column is the PID
docker ps --format "table {{.Names}}\t{{.Ports}}" | findstr 6380
tasklist /FI "PID eq <pid>"            # what process is that PID?
```

```bash
# macOS / Linux
lsof -iTCP:6380 -sTCP:LISTEN
docker ps --format "table {{.Names}}\t{{.Ports}}" | grep 6380
```

Then pick a fix:

```bash
# 1. Stop the other container if you don't need it
docker stop <other-container-name>

# 2. Remap THIS project's host port — edit infra/docker-compose.yml,
#    change "6380:6379" to e.g. "6381:6379", then:
docker compose -f infra/docker-compose.yml up -d --build

# 3. If a non-Docker process holds it, kill it at the OS level
taskkill /PID <pid> /F                 # Windows
kill -9 <pid>                          # macOS / Linux
```

The repo already ships with non-conflicting host ports (`5434`, `6381`, `8011`, `3100`) chosen to coexist with other common local stacks. If you still collide on those, edit the offending `ports:` line in `infra/docker-compose.yml` and re-run `up -d`.

### C. Container exits immediately after `up`

```bash
docker compose -f infra/docker-compose.yml ps         # see exit codes
docker compose -f infra/docker-compose.yml logs backend   # last error before crash
```

Common causes: `.env` missing, a required env var malformed, a Postgres volume from an older schema (`down -v` to reset), or a healthcheck timeout. The `backend` waits for qdrant + postgres + redis to be `healthy` — if one of them is stuck `(starting)`, fix it first.

### D. `Ollama at http://localhost:11434 is unreachable` (from inside the app)

The error message comes from the **backend container**, not your terminal. Inside a container, `localhost` is the container itself — not your laptop. Ollama runs on your laptop, so `localhost:11434` resolves to nothing in the container.

The Compose file already wires the backend to use Docker's special `host.docker.internal` hostname (the host gateway), with a fallback `extra_hosts` entry for Linux portability:

```yaml
backend:
  environment:
    OLLAMA_HOST: ${OLLAMA_HOST:-http://host.docker.internal:11434}
  extra_hosts:
    - "host.docker.internal:host-gateway"
```

If you still see the error after a fresh `up -d --build`:

1. Confirm Ollama is actually running on the host: `curl http://localhost:11434/api/tags` from your terminal.
2. From inside the container, confirm the host is reachable:
   ```bash
   docker compose -f infra/docker-compose.yml exec backend curl -s http://host.docker.internal:11434/api/tags
   ```
3. If your `.env` sets `OLLAMA_HOST=http://localhost:11434` (the old default), either delete that line or set it to `http://host.docker.internal:11434`.
4. On Windows + WSL2 specifically, make sure Ollama is running on the Windows side (not inside WSL) — the host-gateway IP points at the Windows host.

### E. Hot reload not picking up edits

Bind mounts only work if the container actually started with the mount in place. If you edited `docker-compose.yml`, you need `up -d --build` (recreate), not just `restart`. Verify the mount with:

```bash
docker compose -f infra/docker-compose.yml config | grep -A2 volumes
```

### Local (non-Docker) dev

```bash
# Services only
docker compose -f infra/docker-compose.yml up -d qdrant postgres redis langfuse

# Backend — Windows PowerShell
copy .env.example .env       # one-time, from project root
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
uvicorn app.main:app --reload

# Backend — macOS / Linux
cp .env.example .env         # one-time, from project root
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload

# Frontend (new terminal)
cd frontend
npm install
npm run dev                    # http://localhost:5173
```

---

## Where to go next

- **Deep blueprint with theory + code recipes**: [`EvalRAG.md`](./EvalRAG.md) — sections F through P are the meat.
- **API reference**: backend swagger at `http://localhost:8011/docs` once running.
- **Configuration**: [`backend/app/config.py`](backend/app/config.py) — every knob in one file.

---

## Glossary (Docker + RAG keywords used above)

**Docker**

- **Image** — read-only template (e.g. `redis:7`, `postgres:16`); the recipe.
- **Container** — a running instance of an image; isolated process tree, filesystem, **network namespace**.
- **Compose** — declarative orchestrator. `docker-compose.yml` describes a set of services + a shared **bridge network** + named **volumes**, all started together.
- **Service vs. container name** — service is the name in YAML (`redis`); container becomes `infra-redis-1` (project + service + replica index). Project name defaults to the folder of the compose file (here: `infra`).
- **Bridge network** — the virtual switch Compose puts your services on. Containers reach each other by **service name** as DNS (`redis:6379`, `postgres:5432`). Internal traffic does NOT need a published port.
- **Port publishing / mapping** — `HOST:CONTAINER`. Only needed when *your laptop* (outside Docker) wants to reach a service. Two listeners can't share a host port → "already allocated."
- **Bind / listen** — kernel operations. A port is owned by exactly one socket per `(interface, port, protocol)`.
- **Bind mount vs. named volume** — bind mount maps a host path into the container (`../backend/app:/app/app`) — great for hot reload. Named volume is Docker-managed storage (`pg_data`, `qdrant_data`) — survives `down`, wiped by `down -v`.
- **Healthcheck** — a probe inside the container; combined with `depends_on: condition: service_healthy` to delay dependents until upstream is ready.
- **Env file / interpolation** — `${ANTHROPIC_API_KEY:-}` reads from `.env` (loaded via `env_file:`) and defaults to empty if missing.
- **`up`/`down`/`logs`/`exec`/`ps`** — create+start / stop+remove / tail logs / one-shot command in a running container / status. Add `-d` to `up` for detached, `-v` to `down` to wipe volumes.

**RAG**

- **Chunk** — a passage (~300–800 tokens, ~10–15% overlap) split from a source document.
- **Embedding** — a fixed-length vector that encodes semantic meaning; same model must embed both docs and queries.
- **Vector DB** — index optimized for **ANN** (approximate nearest neighbor) search over embeddings. Here: **Qdrant**.
- **Hybrid retrieval** — combine **dense** (embedding similarity) with **sparse** (BM25 / keyword) signals; usually beats either alone.
- **Reranking** — a slower **cross-encoder** that re-scores the top-K from retrieval for higher precision.
- **Context compression** — trim retrieved passages to the spans actually relevant to the question, to fit the context window and reduce noise.
- **Grounding / citations** — the answer references the chunks it used; reduces **hallucination** and gives auditability.
- **Golden dataset** — hand-labeled question/answer pairs used as the ground truth for evaluation.
- **LLM-as-judge** — a second model scores the system's answer against a rubric (faithfulness, relevance, etc.).
- **Faithfulness** — does the answer only assert what's in the retrieved context? (No hallucination.)
- **Regression** — a metric got worse vs. the previous run on the same dataset; the eval engine flags this so a "small" prompt change doesn't silently break things.
- **Tracing** — per-request structured logs of every stage (retrieval, prompts, LLM call). Here: **Langfuse** (web UI) and **W&B** (eval-run dashboards).
