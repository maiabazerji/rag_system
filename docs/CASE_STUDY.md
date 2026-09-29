# EvalRAG — Measuring and Improving Retrieval-Augmented Generation Systems

This case study describes how EvalRAG was turned from a working RAG comparison bench into a system whose retrieval and answers can be measured. It covers the audit that started the work, the changes to retrieval, grounding and evaluation, and what could and could not be measured. Every claim refers to code in this repository. The before-state is documented in [ARCHITECTURE_AUDIT.md](ARCHITECTURE_AUDIT.md) (commit `054a182`). The current system is described in the [README](../README.md).

---

## 1. Problem

EvalRAG answers questions over a private corpus with three strategies (classic, graph, agentic) and was meant to show which strategy works better on a given corpus. The audit of commit `054a182` found that several of its measurements did not measure what they appeared to:

- **Sources did not match citations.** For the classic and graph strategies, `sources` were the first five reranked chunks, whatever the model cited. Inline `[chunk_id]` citations were never checked, so an invented id went unnoticed (audit §3.9).
- **`confidence` was a constant**: 0.85 by default, 0.9 or 0.6 for the agentic strategy (audit §1).
- **Retrieval was dense-only.** BM25 existed only as the reranker's fallback, computed over the dense candidates, so it could not recover a document that dense retrieval had missed (audit §3.6).
- **Retrieval metrics were coarse.** They were document-level, at a single cutoff that differed per strategy, with no nDCG (audit §3.10).
- **Regression detection ignored configuration.** Runs were grouped without the embedding model, the reranker or the chunking settings, and only judge scores were compared. An embedder swap therefore looked like a regression or an improvement of the same system (audit §3.12).
- **Chunks were truncated by the models.** 600-word chunks exceeded the 512-token input window of both the embedder and the reranker (audit §3.4).
- **The published baseline was stale.** It had been produced by a pipeline in which every major component had since changed (audit §3.15).

The goal of the upgrade was to make each of these measurable and honest before trying to improve any number.

## 2. Architecture

```
question → preprocess → dense (E5 + Qdrant) ─┐
                      → sparse (BM25)       ─┴→ RRF → cross-encoder rerank → context selection
        → grounded generation (submit_answer) → citation validation → response
side channel: spans (/traces, Langfuse), Prometheus metrics, RequestMetrics (tokens, cost, stage latency)
```

- **Backend:** FastAPI, with Qdrant for vectors and Postgres for keys, usage and audit. Eval runs are JSON files under `DATA_DIR/eval_runs`.
- **Frontend:** React, with five pages: Ask, Upload, Compare, Evaluation and Advisor. The Evaluation page has an Overview tab (quality against cost, with the regression status of each configuration) and a Runs tab (launch a run, run history, per-run regression report).
- **Strategies:** all three return one `StrategyResult` contract (`backend/app/rag/strategies/base.py`), which is what makes them comparable.

## 3. Retrieval strategies

| Strategy | Retrieval | Model calls per question |
|---|---|---|
| Classic | Hybrid retrieval, rerank, context | 1 |
| Graph | Question entities (1 call) → 1-hop walk over an LLM-extracted triple graph, merged with fused hybrid candidates, reranked together | 2, plus an offline build with one call per chunk (`POST /graph/build`) |
| Agentic | The model calls `search` (fused, not reranked, k ≤ 12), `fetch_chunk` and `finish` | Up to `AGENTIC_MAX_ITERS` (15) |

The upgrade routed all three strategies through the same retrieval function and the same citation validation (commits `d9b6cce`, `619225c`). Differences in their results can therefore be attributed to the strategy, not to plumbing.

## 4. Hybrid retrieval

`hybrid_search` (`backend/app/rag/retrieve.py`) runs these steps:

1. **Preprocessing.** It normalises the query (NFKC, invisible characters, whitespace, a 1,000-character cap).
2. **Retrieval.** Dense retrieval (`DENSE_TOP_K`) and BM25 (`BM25_TOP_K`) run concurrently.
3. **Fusion.** Reciprocal Rank Fusion with `RRF_K = 60` (`fusion.py`). RRF combines ranks, not scores, because cosine similarities and BM25 scores are on unrelated scales.
4. **Reranking.** The fused pool, capped at `max(DENSE_TOP_K, BM25_TOP_K)` candidates, is reranked with the multilingual cross-encoder, and the top `FINAL_CONTEXT_K` chunks are kept.

Design decisions:

- **Access-scoped BM25.** One BM25 model is built per access scope, over only the chunks that scope may read (`sparse.py`). A shared index would leak another tenant's vocabulary through document frequencies.
- **Visible degradation.** If BM25 fails in hybrid mode, retrieval continues dense-only and reports `degraded: ["sparse"]`. A cross-encoder fallback is reported as `reranker: "bm25-fallback"`. An earlier incident motivated this: at commit `2175ce8` the configured reranker id did not exist, and every query had silently used the BM25 fallback while returning HTTP 200.
- **Per-chunk provenance.** The API returns each context chunk's dense rank, sparse rank, fused score and rerank score, so a retrieval decision can be explained after the fact.

The structured chunker (`chunking.py`) cuts at headings, then packs whole paragraphs, then splits between sentences. Chunk size defaults to 300 words and is capped at 512 / 1.5 = 341 words (commit `602b7ca`). Each chunk carries its heading path, pages and character offsets, which citations reuse.

## 5. Evaluation methodology

- **Datasets** (`data/golden/`): 175 questions over five files. `golden_v3` (54) and `golden_fr_business_v1` (37) are categorised into eight types: single-hop, multi-hop, comparison, aggregation, ambiguous, unanswerable, citation-sensitive and adversarial. Each carries difficulty labels and verbatim evidence quotes. The three older sets are frozen so that historical runs stay comparable.
- **Deterministic retrieval metrics** (`backend/app/eval/retrieval.py`): Recall@K, Precision@K, HitRate@K, MRR and nDCG@K at K = 1, 3, 5, 10. They are scored on every strategy's ranked context at the same K. Examples without relevant documents are excluded, not scored as zero.
- **LLM judge** (`judge.py`, `rubric.py`): five dimensions under a versioned rubric with five anchors each. The reply is validated with Pydantic, gets one repair retry, and out-of-range scores are rejected. Judge reasoning and judge tokens are stored per example.
- **Failure accounting:** each example ends as `scored`, `judge_failed`, `generation_failed` or `not_judged`. Aggregates carry `n` per metric, so a partially failed run cannot pass for a complete one.
- **Refusal scoring:** unanswerable questions are scored by `correct_refusal`, and answerable ones by `false_refusal`, instead of being judged on a rubric that assumes an answer exists.

## 6. Grounding

The model sees each chunk under a handle (`[S1]`…`[Sn]`), never a raw chunk id, and answers through a `submit_answer` tool. The tool call carries the answer with inline markers, per-claim citations with a `supported` flag, a status (`answered` / `partial` / `insufficient_context`) and unsupported notes.

`validate_citations` (`backend/app/rag/grounding.py`) is a pure function:

- It drops unknown handles and lists them in `invalid_citations`.
- It builds `sources` from the cited chunks only.
- It turns an answer with no surviving citation into a localised refusal.
- It sets `grounded` only when every claim is supported by a valid citation.

The constant confidence was replaced by an evidence score: `0.5·coverage + 0.2·status + 0.3·relevance`, reduced by up to half for invalid citations. It is documented as a heuristic for ranking answers, **not a calibrated probability**.

## 7. Regression testing

- **Grouping.** Runs are compared only within a group sharing dataset, strategy, provider, model, prompt version, judge model, rubric version and a `config_hash` of the retrieval configuration (`backend/app/eval/regression.py`, `metrics.py:retrieval_config`).
- **Thresholds.** They are set in `eval/regression_thresholds.toml`:
  - faithfulness, answer relevance, recall@5 and MRR may drop by at most 0.03;
  - p50 latency may rise by at most 20%, and tokens per question by at most 15%.
  - A metric measured on fewer than five examples is `SKIPPED`.
- **Gate.** Each saved run gets a regression report against the pinned baseline for its configuration, or else the previous comparable run. `scripts/run_eval.py --fail-on-regression` and `scripts/regression_report.py --fail-on-regression` exit with code 2 on a `FAIL`.

## 8. Observability

- **Spans in pipeline order:** preprocess, dense, sparse, fusion, rerank, context_selection, generation, citation_validation, response. Graph adds entity_extraction and graph_walk.
- **Where they go:** `GET /traces/{id}` serves them, and they are exported over OTLP to a self-hosted Langfuse when `TELEMETRY_MODE` allows it.
- **Generation spans for every provider.** After the upgrade, OpenAI and Ollama calls produce generation spans as well (commit `b02c1a6`).
- **No content in spans or logs.** Span metadata, trace stages and structured logs contain counts, flags, timings, tokens and costs, never chunk or question text. `backend/tests/test_request_metrics.py` asserts this.

## 9. Cost and latency

- **`RequestMetrics`.** Every `/ask` and `/compare/strategies` response includes it: tokens, estimated USD cost, model-call count, candidate and context chunk counts, and latency split into retrieval, rerank, generation, citation validation and other.
- **Pricing.** Each model call is priced with the model that served it, from `config/model_pricing.toml` (Anthropic list prices dated 2026-09-25). A call to an unlisted model makes the cost `null` rather than a guess.
- **Prometheus.** The same figures are exported as metrics.

---

## 10. Results

Only the following have been measured on the current pipeline.

**Test suite.** 1,495 backend tests pass, with 95% line coverage (CI floor: 90%). The frontend has 102 Vitest tests. ruff and mypy are clean. The audit baseline was 928 tests at 92%. The offline end-to-end tests (commit `7b147c2`) run the real parsers, chunker, PII masking, in-memory Qdrant, BM25 and RRF, citation validation, the eval harness and erasure. The embedder and the model are replaced by documented test doubles.

**BM25 retrieval benchmark** (`scripts/benchmark_retrieval.py`, sparse mode, reranking off, commit `885f3fd`; files in `data/benchmarks/`):

| Dataset | Scored | Recall@5 | HitRate@5 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| golden_v1 | 34/34 | 0.529 | 0.529 | 0.439 | 0.489 |
| golden_v2 | 32/34 | 0.669 | 0.875 | 0.735 | 0.704 |
| golden_fr_v1 | 16/16 | 0.760 | 0.875 | 0.750 | 0.745 |
| golden_v3 | 47/54 | 0.938 | 1.000 | 0.888 | 0.891 |
| golden_fr_business_v1 | 32/37 | 0.867 | 0.906 | 0.717 | 0.766 |

In this script, K counts chunks: the documents behind the top K chunks are deduplicated and scored, and MRR is computed within the top K. These numbers come with three caveats:

- **Dense and hybrid were not measured.** The embedding model could not be downloaded in the benchmark environment, so both modes were skipped.
- **The two newest datasets favour BM25.** `golden_v3` and `golden_fr_business_v1` were written from the documents and share their vocabulary.
- **No generation metrics.** No Anthropic API key was available, so faithfulness, relevance, correctness, refusal accuracy and per-request cost have not been measured after the upgrade. The README lists the exact commands to produce them.

**Defects found and fixed.** Some came from the audit; the rest were found by the new tests.

| Defect | How it was found | Fix |
|---|---|---|
| Classic and graph `sources` were the top-5 reranked chunks regardless of what was cited | Audit | Sources are the validated cited chunks (`619225c`) |
| `confidence` was a hard-coded constant | Audit | Evidence score from claims, status and cited-chunk scores (`399ea04`, `619225c`) |
| BM25 was only a reranker fallback over dense candidates | Audit | Access-scoped BM25 retriever fused with dense retrieval by RRF (`189b7f4`, `6b5fb30`, `1466bd2`) |
| Regression grouping ignored the retrieval configuration | Audit | Config hash and rubric version in the grouping key (`1b20b56`) |
| 600-word chunks silently truncated at 512 tokens | Audit | 300-word default, capped to the model window (`602b7ca`) |
| Eval rows never recorded the retrieval mode: they read a key no strategy set, so every run fell back to the `RETRIEVAL_MODE` setting | Tests | Read it from the retrieval diagnostics (`bfdfb00`) |
| Graph triple parsing crashed on a non-list reply, and a triple with a null field created an entity named `"None"` that linked unrelated chunks | Tests | Accept only well-formed triples (`ae9a578`) |
| A non-JSON reply from Ollama raised `JSONDecodeError` / `AttributeError` instead of `ProviderError`, and was reported as an internal error | Tests | Validate the body and raise `ProviderError` (`2d22cb8`) |
| Human ratings were written to a path derived from the module location, ignoring `DATA_DIR`; `POST /eval/human-rate` failed in the runtime image | Tests | Store under `DATA_DIR` (`819b017`) |

## 11. Failure analysis

What the audit and the datasets revealed about the measurements themselves:

- **`golden_v1` was labelled against a 6-document seed corpus.** Its 34 questions reference only 6 filenames, the documents present at the initial commit, while `data/docs` now holds 41. Later documents on the same topics, such as `reranking_deep_dive.md` and `cross_encoder_reranking.md` next to `reranking.md`, count as misses. This is why `golden_v2` relabels the same questions with every covering document, and why v1 is kept frozen only for comparison with old runs.
- **The corpus contradicts itself.** `dense_embeddings_vector_search.md` gives BGE-Large 768 dimensions, while `embedding_models_guide.md` gives 1024. text-embedding-3-large is priced at $0.02 per million tokens in one document and $0.13 in another. Rather than editing the corpus, `golden_v3` turns these conflicts into `ambiguous` questions whose ideal answer names the conflict (see `data/golden/README.md`).
- **Self-authored datasets have a lexical-overlap bias.** Questions written while reading the source share its terms. BM25 reaches HitRate@5 = 1.000 on `golden_v3` but 0.529 on `golden_v1`. That gap measures how the datasets were written as much as how well retrieval works.
- **The eval corpus includes the documentation.** `scripts/ingest.py` ingests `README.md`, `EvalRAG.md` and `LEARN.md` by default (`--no-meta` skips them). Editing the docs therefore changes the corpus that the `data/docs` datasets are evaluated against. The offline benchmark indexes `data/docs` only.
- **Configuration drift remains possible in two settings.** The config hash covers settings by name prefix. `DENSE_TOP_K` and `FINAL_CONTEXT_K` do not match any prefix, so two runs that differ only in them are grouped together.

## 12. Lessons learned

- **Make degradation visible before optimising.** Before `2175ce8`, a missing reranker and a duplicated index both returned HTTP 200. The reranker actually used, the retrievers that degraded, and the counts per stage are now part of every response.
- **A metric needs its denominator.** Reporting `n` per metric and a status per example removed the possibility of a partially failed run looking complete. Excluding unanswerable questions from retrieval metrics, instead of scoring them zero, did the same for labels.
- **Compare only like with like.** Most apparent regressions in a small project are configuration changes. Putting the rubric version and a retrieval fingerprint in the grouping key is cheaper than explaining false alarms.
- **Low-coverage code hides real bugs.** Four of the defects in §10 were found while raising coverage on paths the audit had listed as weakly tested (§3.14: Ollama provider 31%, human ratings 63%, graph extraction 65%, eval metrics 89%).
- **Numbers that were not measured should not appear.** The benchmark script records skipped modes with the error, instead of leaving a gap that could be filled in by hand.

## 13. Future improvements

- **Calibrate the judge against human labels.** Human ratings are already stored; compute judge-to-human agreement per dimension before relying on judge deltas below the 0.03 threshold.
- **Run the missing benchmarks.** Dense, hybrid and reranked retrieval, and full generation runs, need an environment with Hugging Face access and an Anthropic API key. Then pin baselines per configuration.
- **Use Qdrant native sparse vectors** in place of the in-memory BM25 index, so the index scales and is shared between workers.
- **Incremental BM25 updates** instead of a full rebuild on every collection change.
- **Evaluate query rewriting and HyDE with the harness,** by category, before enabling either by default.
- **Route by question type.** The advisor could choose a strategy per question type from measured per-category results instead of the historical weights it uses today.
- **Add a CI eval gate on a small dataset.** Run `run_eval.py --fail-on-regression` on a subset with a pinned baseline, with the judge enabled only where a key is available.
- **Close the known gaps.** Add `DENSE_TOP_K` and `FINAL_CONTEXT_K` to the config hash, scope `/traces` and `/eval/runs` by principal, and add Grafana panels for cost and stage latency.
