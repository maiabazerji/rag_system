# GDPR guide for EvalRAG deployers

> **This is not legal advice.** It describes what the software does so that you, the deployer and data controller, can do your own assessment with your DPO or counsel. The features below help you meet GDPR obligations. They do not make a deployment compliant on their own.

EvalRAG processes whatever you ingest and whatever your users ask. If either can contain personal data (CVs, support tickets, HR policies with names, questions that mention a colleague), the GDPR applies to your deployment and you are the controller.

- [What personal data the system processes](#what-personal-data-the-system-processes)
- [Where it is stored](#where-it-is-stored)
- [PII detection and masking](#pii-detection-and-masking)
- [Lawful basis](#lawful-basis)
- [Retention](#retention)
- [Right to erasure](#right-to-erasure-art-17)
- [Right of access (export)](#right-of-access-art-15)
- [Subprocessors](#subprocessors)
- [Hosting](#hosting)
- [Checklist](#deployer-checklist)
- [Record of processing template](./ropa_template.md)

---

## What personal data the system processes

| Category | Examples | Source |
|---|---|---|
| Document content | Names, emails, phone numbers, NIR, IBANs in uploaded files | Uploaders via `POST /ingest` or `scripts/ingest.py` |
| Document metadata | Filename, owner, source key, ingestion time | Ingestion |
| Questions and answers | Free text typed by users; answers quoting documents | `/ask`, `/compare`, eval runs |
| Knowledge-graph triples | Entities extracted from chunks ("Alice, works_at, Acme") | `POST /graph/build` |
| API key metadata | Key name (often a person's name), hint, creation and last-use time | `/admin` |
| Usage records | One row per authenticated request: key id, endpoint, timestamp, token counts | `require_api_key` |
| Logs | Request ids, paths, error messages | All components |
| Human ratings | Question, answer, rating, free-text comment | `/eval` rating endpoints |

EvalRAG processes no special-category data (art. 9) on purpose. If your documents contain health, union or other special-category data, you need an art. 9 condition and should seriously consider `PII_MODE_INGEST=reject` for that corpus or leaving it out altogether.

## Where it is stored

| Store | Holds | Location | Erasure | Retention |
|---|---|---|---|---|
| Qdrant | Chunk text, embeddings, metadata (`filename`, `owner`, `source_key`, `ingested_at`, `pii_counts`) | `qdrant` volume | `vectors` target | Until erased |
| Graph store | Triples with `doc_id` and `chunk_id` | `GRAPH_DATA_DIR/triples.jsonl` | `graph` target | Until erased |
| Traces | Question, strategy steps, cited chunk ids, principal | Process memory (max 1000, lost on restart) | `traces` target | `RETENTION_TRACES_DAYS` |
| Eval runs | Golden questions, generated answers, scores, cited docs | `data/eval_runs/*.json` | `eval_runs` target (scrub) | `RETENTION_EVAL_RUNS_DAYS` |
| Postgres `api_keys` | Key name, hash, hint, timestamps | Postgres | Deactivate or delete via `/admin` | Until removed |
| Postgres `api_key_usage` | Per-request accounting | Postgres | `usage` target | Not purged automatically |
| Audit log | Access decisions (added by the access-control module) | Postgres | Registered by that module | `RETENTION_AUDIT_DAYS` |
| Human ratings | Question, answer, comment | `data/human_ratings/ratings.jsonl` | Manual (see below) | Manual |
| Logs | JSON lines | stdout and `LOG_DIR/evalrag.log` (rotated, 5 × 10 MB) | Rotation | Rotation, or your log pipeline's policy |
| Langfuse / W&B (optional) | Traces, eval tables | Those services | In those services | Their settings |

## PII detection and masking

`app/privacy/pii.py` finds personal data with patterns **and validation**, so numbers of the right shape but a wrong check digit are left alone:

| Type | Placeholder | Validation |
|---|---|---|
| Email | `[EMAIL]` | Address shape, sane local part |
| French phone | `[PHONE]` | `0X XX XX XX XX`, `+33 X …`, `0033 …`, `+33 (0)X …` |
| NIR (sécurité sociale) | `[NIR]` | Key = 97 − (first 13 digits mod 97); Corsica `2A`/`2B` read as 19/18 |
| IBAN (any country) | `[IBAN]` | ISO 13616 length per country, mod-97 = 1 |
| SIRET | `[SIRET]` | 14 digits, Luhn (La Poste digit-sum exception) |
| SIREN | `[SIREN]` | 9 digits, Luhn |
| Payment card | `[CARD]` | Network prefix, 13–19 digits, Luhn |
| IPv4 address | `[IPV4]` | Four octets 0–255 |

Where it applies:

| Setting | Default | Effect |
|---|---|---|
| `PII_MODE_INGEST` | `mask` | `off`: index verbatim. `mask`: replace values with placeholders before chunking, embedding or hashing; the per-type counts (never the values) are stored as `pii_counts` on each chunk and returned by `/ingest`. `reject`: refuse the upload with `422 {"code": "pii_rejected", "types": [...]}` |
| `PII_REDACT_LOGS` | `true` | A logging filter masks messages and `extra_fields` on every handler |
| `PII_REDACT_TRACES` | `true` | Trace inputs and events are masked before they are stored. Exporters call `redact_for_telemetry()` so Langfuse and W&B use the same switch |

**Limits you must plan for:**

- **Person names are not detected.** No regular expression can tell a name from a place or a product. `PIIDetector` is a pluggable interface: implement `detect(text) -> Iterable[PIISpan]` (for example with spaCy `fr_core_news_md`) and call `register_detector()` at startup.
- Detection covers the formats above, mostly French ones. Other national identifiers, postal addresses and dates of birth are not detected.
- SIREN is a bare 9-digit number with a 1-in-10 check, so `mask` will sometimes mask an unrelated number. That is the conservative failure.
- **Questions are not masked before they reach the LLM**, only before they are stored in traces and logs. Answering a question needs the question.
- Masking is irreversible. Keep the originals elsewhere if you need them. EvalRAG does not keep them.

## Lawful basis

Choosing a lawful basis (art. 6) is the deployer's job, and it depends on your context. Common patterns:

- **Internal knowledge base for employees:** often legitimate interests (art. 6(1)(f)) with a documented balancing test, or performance of the employment contract for the documents themselves. In France, the works council (CSE) may have to be consulted before deploying a tool that processes employee data.
- **Customer-facing assistant:** usually performance of a contract or legitimate interests. Consent is rarely the right basis for core service processing.
- **Evaluation data (golden sets, eval runs):** usually legitimate interests (quality assurance). Prefer synthetic or anonymised questions in golden datasets.

Also check whether you need a DPIA (art. 35), for example for large-scale processing or systematic monitoring of employees. The CNIL publishes a list of processing types that require one.

## Retention

A background task purges expired records when the app starts and every 24 hours after that. Admins can also trigger it on demand.

| Setting | Default | Store |
|---|---|---|
| `RETENTION_TRACES_DAYS` | `7` | In-memory traces |
| `RETENTION_EVAL_RUNS_DAYS` | `365` | `data/eval_runs/*.json`, by `created_at` (or file time for older files) |
| `RETENTION_AUDIT_DAYS` | `365` | Audit log, when the access-control module is installed |

`0` disables the purge for that store. Documents in Qdrant and the graph are kept until they are erased: their retention follows the retention of the source documents, which you define.

```bash
curl -X POST -H "X-Admin-Key: $ADMIN_KEY" http://localhost:8011/privacy/retention/run
# {"ran_at": "...", "targets": {"traces": 12, "eval_runs": 0},
#  "cutoffs": {"traces": "...", "eval_runs": "..."}, "skipped": [], "errors": {}}
```

## Right to erasure (art. 17)

All `/privacy` endpoints require `X-Admin-Key` (`ADMIN_KEY`). They are operator tools: verify the requester's identity before you act.

```bash
# One document (one revision), by doc_id as returned by /ingest
curl -X DELETE -H "X-Admin-Key: $ADMIN_KEY" http://localhost:8011/privacy/documents/3f2a9c0d1e4b5a67

# Every revision ingested from a source (matches source_key, or the filename)
curl -X DELETE -H "X-Admin-Key: $ADMIN_KEY" \
  "http://localhost:8011/privacy/sources?source_key=alice_cv.pdf"

# A principal: every document they own, their traces and their usage rows
curl -X DELETE -H "X-Admin-Key: $ADMIN_KEY" http://localhost:8011/privacy/principals/42
```

Response (200 when every target succeeded):

```json
{
  "subject": "principal:42",
  "doc_ids": ["3f2a9c0d1e4b5a67"],
  "targets": {"vectors": 14, "graph": 31, "traces": 3, "eval_runs": 2, "usage": 118},
  "errors": {},
  "complete": true
}
```

How it works:

1. **Resolve.** The subject is resolved into document ids and filenames from Qdrant. If Qdrant is unreachable the request fails with `503` and nothing is erased.
2. **Erase.** Every registered target runs with those ids:
   - `vectors`: chunks with those `doc_id`s (and, for a principal, any chunk they own);
   - `graph`: triples from those documents;
   - `traces`: traces that cited those documents or were made by the principal;
   - `eval_runs`: rows citing those documents are **scrubbed, not deleted**. The answer text and the cited filenames become `"[erased]"`, the row is marked `"erased": true`, and the **scores are kept**. We chose this so run aggregates and regression baselines stay comparable over time, and a score is not personal data. If your assessment says rows must go entirely, delete the run files with a retention of `0` days or by hand;
   - `usage`: for a principal, their `api_key_usage` rows. The key itself remains: deactivate it with `POST /admin/keys/{id}/deactivate`.
3. **Report.** If any target fails, the response is `503` with the partial report as `detail`. **Every target is idempotent**, so repeat the request until it returns 200.

A principal is identified by the API key's numeric id or its name until the access-control module introduces principals. Other stores register their own targets with `app.privacy.erasure.register_erasure_target(name, fn)`.

Not covered automatically, so handle these by hand:

- **Human ratings** (`data/human_ratings/ratings.jsonl`): edit the file.
- **Golden datasets** (`data/golden/*.jsonl`): these are yours. Keep personal data out of them.
- **Langfuse / W&B**: delete the traces or runs in those tools.
- **Logs** already written: they age out through rotation. With `PII_REDACT_LOGS=true` they should hold no detected identifiers.
- **Backups** of Qdrant volumes, Postgres or `data/`: follow your backup retention policy.
- **The source file** you ingested, wherever you keep it.

## Right of access (art. 15)

```bash
curl -H "X-Admin-Key: $ADMIN_KEY" http://localhost:8011/privacy/export/42 > export-42.json
```

```json
{
  "principal_id": "42",
  "generated_at": "2026-09-27T10:00:00+00:00",
  "documents": [
    {"doc_id": "3f2a9c0d1e4b5a67", "filename": "alice_cv.pdf", "source_key": null,
     "created": "2026-09-01T10:00:00+00:00", "chunks": 7, "pii_counts": {"EMAIL": 1}}
  ],
  "usage": {"api_keys": [{"id": 42, "name": "alice", "key_hint": "sk_Ab12cd", "created_at": "...", "last_used": "..."}],
            "requests": 118, "tokens_input": 90211, "tokens_output": 20443,
            "first_request": "...", "last_request": "...", "by_endpoint": {"/ask": 110, "/ingest": 8}},
  "traces": {"count": 3, "items": ["..."]},
  "errors": {},
  "complete": true
}
```

`complete: false` with `errors` naming the sections means a store could not be read. Retry before you send the export. Answer within one month (art. 12(3)). The export covers what EvalRAG stores. Add what your other systems hold about the person.

## Subprocessors

Depending on configuration, data leaves your infrastructure to:

| Recipient | When | What is sent |
|---|---|---|
| **Anthropic** (Claude API) | Always for answers with `GENERATOR_PROVIDER=anthropic`, the LLM judge, graph extraction and the agentic strategy | The question, retrieved chunk text (masked at ingest if enabled), eval answers and context |
| OpenAI | Only with `GENERATOR_PROVIDER=openai` | Question and retrieved chunks |
| Langfuse | Only when `LANGFUSE_*` keys are set | Traces (masked when `PII_REDACT_TRACES=true`) |
| Weights & Biases | Only when `WANDB_API_KEY` is set and `WANDB_MODE` is not `disabled` | Eval tables: questions, answers, scores |
| Hugging Face | Model download at first start | Nothing about your data; embeddings and reranking run locally |
| Ollama | `GENERATOR_PROVIDER=local` | Nothing leaves your host |

For Anthropic, review the current terms rather than relying on this page:

- Anthropic's Data Processing Addendum is incorporated into its Commercial Terms of Service: <https://www.anthropic.com/legal/commercial-terms>. Check the current version for subprocessors, international transfer mechanisms and data retention.
- Anthropic's privacy and trust information: <https://privacy.anthropic.com> and <https://trust.anthropic.com>.
- Anthropic offers **zero data retention (ZDR)** arrangements for eligible API customers. Ask your Anthropic account contact whether it fits your use case and which features it covers.

List every subprocessor you enable in your record of processing and in your privacy notice.

## Hosting

- Run Qdrant, Postgres and the backend in the EU/EEA, or document the transfer mechanism you rely on.
- Encrypt volumes at rest. Use TLS between the browser and the backend, and between the backend and Postgres or Qdrant when they are on different hosts.
- Set `REQUIRE_API_KEY=true` and a strong `ADMIN_KEY` before exposing the service. Change the default Postgres password.
- Restrict who can read `data/`, `logs/` and database backups.

## Deployer checklist

- [ ] Decide `PII_MODE_INGEST` per corpus, and add a name detector if names matter.
- [ ] Fill in the [record of processing](./ropa_template.md).
- [ ] Choose and document a lawful basis. Run a DPIA if required.
- [ ] Update your privacy notice (purposes, recipients, retention, rights).
- [ ] Set the retention periods and document why you chose them.
- [ ] Sign or accept the DPAs of the subprocessors you enable. Consider Anthropic ZDR.
- [ ] Write an internal procedure for erasure and access requests (identity checks, one-month deadline, the manual steps above).
- [ ] Plan for breach notification (72 hours to the CNIL, art. 33).
