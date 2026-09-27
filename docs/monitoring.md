# Monitoring, tracing and data residency

EvalRAG can report on itself in three ways:

| Signal | What it is | Where it goes | Off by default? |
|---|---|---|---|
| **Traces** | One trace per answered question: retrieve, rerank, each model call, each agent tool call, with timings, token usage and model | Langfuse, via its OpenTelemetry endpoint | Yes (`TELEMETRY_MODE=off`) |
| **Metrics** | Aggregate counters and histograms: requests, latency, tokens, refusals, circuit breakers, dropped traces | `GET /metrics`, scraped by Prometheus | Yes (`METRICS_ENABLED=false`) |
| **Eval dashboards** | Per-run eval scores and tables | Weights & Biases | Yes (`WANDB_ENABLED=false`) |

Separately from all three, **every question is sent to the model provider**
(Anthropic by default) to be answered. That is not telemetry, it is the
product; see [Model provider](#model-provider-anthropic) below.

The policy that decides what may leave the process lives in one module,
[`backend/app/tracing/policy.py`](../backend/app/tracing/policy.py), and is
logged at startup. `GET /health` reports `langfuse` and `wandb` as `true` only
when the policy actually lets them export, not merely when keys are present.

---

## Telemetry modes

Set `TELEMETRY_MODE` in `.env`.

### `off` (default)

- **Langfuse:** nothing is exported, even if keys are set. Traces are still kept
  in memory for `GET /traces/{id}` and are lost on restart.
- **W&B:** only `WANDB_MODE=offline` is allowed (files under `./wandb` on this
  machine). Anything else is forced to `WANDB_MODE=disabled`, so the W&B SDK
  never opens a connection.
- **Leaves the machine:** nothing, apart from model provider calls.

### `self_hosted`

- **Langfuse:** traces are exported only if the host in `LANGFUSE_HOST` is
  listed in `TELEMETRY_ALLOWED_HOSTS` (default `localhost,langfuse,127.0.0.1`).
  Matching is on the exact hostname, so `langfuse.evil.com` does not match
  `langfuse`. Any other host is **refused and logged as an error at startup**;
  the app still starts.
- **W&B:** `offline`, or `online` against a self-hosted W&B server whose
  `WANDB_BASE_URL` host is in the allow-list. The W&B cloud (`api.wandb.ai`) is
  refused.
- **Leaves the machine:** traces to your own Langfuse. With the Compose
  `tracing` profile that Langfuse is on the same host, so nothing leaves it.

### `cloud`

- Requires **both** `TELEMETRY_MODE=cloud` **and** `TELEMETRY_CLOUD_OPT_IN=true`.
  Copying one setting between environments cannot start a cross-border transfer
  on its own; without the opt-in every exporter is refused and logged.
- **Langfuse:** any host, including Langfuse Cloud. Langfuse Cloud offers an EU
  region; choosing it, and having a DPA with the vendor, is your decision.
- **W&B:** `online` to the W&B cloud (needs `WANDB_API_KEY`).
- **Leaves the machine:** traces and eval runs, to the vendors you configured.

### What a trace contains

| Field | Exported? |
|---|---|
| Strategy, model, provider, latency, token counts, refusal flag and reason, step timings, chunk ids, request id | Always (when export is on) |
| Question, answer, prompt with retrieved context, agent tool inputs/outputs | Only if `TELEMETRY_INCLUDE_CONTENT=true` (the default), and always after the redaction hook |
| API keys, user identifiers, IP addresses | Never |

Text is passed through a redaction hook before serialisation
(`app.tracing.langfuse_exporter.set_redactor`); the privacy module installs its
PII redactor there. Long texts are truncated to 20,000 characters. Set
`TELEMETRY_INCLUDE_CONTENT=false` to export timings and counts only.

Export never blocks or fails a request: traces go onto a bounded background
queue (`TELEMETRY_QUEUE_SIZE`, default 1000), are sent in batches, and are
flushed on shutdown. When the queue is full or Langfuse is unreachable, traces
are dropped and counted in `evalrag_telemetry_dropped_total`.

---

## Self-hosting everything in the EU

The Compose stack can run every component on one machine, which you place in
an EU data centre (or on-premises). Nothing in the `tracing` or `monitoring`
profiles calls out to a vendor: Langfuse's own product telemetry
(`TELEMETRY_ENABLED=false`) and Grafana's usage reporting, update checks and
news feed are switched off in the Compose file.

### Traces: Langfuse (profile `tracing`)

Langfuse v4 needs Postgres (reuses the stack's, in its own `langfuse`
database), ClickHouse, Valkey (Redis-compatible) and S3-compatible storage
(MinIO). All of it is in `infra/docker-compose.yml`, image versions pinned,
ports bound to `127.0.0.1`.

```bash
# 1. In .env: pick Langfuse keys and change every secret in the Tracing section
TELEMETRY_MODE=self_hosted
LANGFUSE_PUBLIC_KEY=pk-lf-local
LANGFUSE_SECRET_KEY=sk-lf-<something random>
LANGFUSE_INIT_USER_EMAIL=you@example.eu
LANGFUSE_INIT_USER_PASSWORD=<a password>
LANGFUSE_ENCRYPTION_KEY=<openssl rand -hex 32>
# ...and LANGFUSE_NEXTAUTH_SECRET, LANGFUSE_SALT, CLICKHOUSE_PASSWORD,
#    LANGFUSE_REDIS_PASSWORD, MINIO_ROOT_PASSWORD

# 2. Start it
docker compose --env-file .env -f infra/docker-compose.yml --profile tracing up -d
```

On first start Langfuse creates an organisation, a project with exactly the
keys from `.env`, and the user above, so the backend can export immediately.
Open <http://localhost:3100>. The backend reaches Langfuse at
`http://langfuse:3000`; `langfuse` is in the default allow-list.

The Postgres init script that creates the `langfuse` database only runs on an
empty data volume. On an existing volume, create it once:
`docker compose -f infra/docker-compose.yml exec postgres createdb -U evalrag langfuse`.

To use a Langfuse you run elsewhere (for example on an EU server of your own),
set `LANGFUSE_HOST` to it and add its hostname to `TELEMETRY_ALLOWED_HOSTS`.

### Metrics: Prometheus and Grafana (profile `monitoring`)

```bash
# In .env
METRICS_ENABLED=true
ADMIN_KEY=<python -c "import secrets; print(secrets.token_urlsafe(32))">
GRAFANA_ADMIN_PASSWORD=<a password>

docker compose --env-file .env -f infra/docker-compose.yml --profile monitoring up -d
```

- Prometheus: <http://localhost:9090>, 15 days retention.
- Grafana: <http://localhost:3300> (user `admin`), with the **EvalRAG overview**
  dashboard provisioned from `infra/monitoring/grafana/dashboards/`.

`GET /metrics` answers only when `METRICS_ENABLED=true`, and then only to
loopback callers or to requests carrying `X-Admin-Key: <ADMIN_KEY>`. Prometheus
runs in its own container, so Compose mounts the admin key into it as a file
and it sends the header on every scrape. Behind a reverse proxy on the same
host, run uvicorn with `--proxy-headers` so the proxy's loopback address is not
mistaken for the client's.

| Metric | Labels |
|---|---|
| `evalrag_http_requests_total` | `method`, `route` (template, e.g. `/traces/{trace_id}`), `status` |
| `evalrag_http_request_duration_seconds` | `method`, `route` |
| `evalrag_strategy_duration_seconds` | `strategy`, `outcome` (`ok`, `refusal`, `error`) |
| `evalrag_llm_tokens_total` | `model`, `strategy` (`none` for the judge and graph extraction), `direction` |
| `evalrag_refusals_total` | `strategy`, `reason` (`model`, `no_documents`, `provider_error`, `error`, ...) |
| `evalrag_circuit_breaker_state` / `_failures` | `service` (0 closed, 1 half-open, 2 open) |
| `evalrag_telemetry_exported_total` / `_dropped_total` | `exporter`, `reason` |

Metrics contain no request content and no identifiers.

### Eval dashboards: W&B

Use `WANDB_ENABLED=true` with `WANDB_MODE=offline` to keep runs on disk
(`wandb sync` later, if you choose), or run a self-hosted W&B server, set
`WANDB_BASE_URL` to it and add its host to `TELEMETRY_ALLOWED_HOSTS`. Eval
results are always saved locally under `data/eval_runs/` regardless.

---

## Model provider (Anthropic)

Answer generation, the LLM judge, graph extraction and the agentic strategy
send the question and the retrieved document chunks to the configured model
provider. With the default `GENERATOR_PROVIDER=anthropic`, that is Anthropic's
API. How Anthropic processes and retains API data is governed by its commercial
terms, its data processing addendum, its privacy documentation and any zero
data retention arrangement your organisation has agreed with Anthropic. Read
the current versions at <https://www.anthropic.com/legal> before sending
personal data; this project does not change them.

To keep generation on your own hardware, use `GENERATOR_PROVIDER=local`
(Ollama). The graph and agentic strategies, the judge and graph extraction
still require Anthropic.
