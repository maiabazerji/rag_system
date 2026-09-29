# Retrieval benchmarks

Output of `scripts/benchmark_retrieval.py`: deterministic retrieval metrics
(Recall@K, Precision@K, HitRate@K, MRR, nDCG@K) per golden dataset, computed
at document level (each document ranked by its best chunk, K counting
documents) with the eval harness's `ranked_retrieval_metrics`, against each
question's `relevant_doc_ids` / `expected_sources`. Questions without relevant documents (unanswerable ones)
are excluded from these metrics. No LLM is involved.

## What the committed files contain

The files in this directory were produced on 2026-09-29 from commit `ed63154`
in an environment with no access to Hugging Face. Consequently:

- **Only `sparse` (BM25) mode was measured.** The `dense` and `hybrid` modes
  need the embedding model (`intfloat/multilingual-e5-small`); each file lists
  them under "Skipped modes" with the load error. No dense or hybrid number is
  reported anywhere in this repository until it has actually been measured.
- Reranking is off in the benchmark, so the numbers describe first-stage
  retrieval only.
- Corpus: `data/docs` (41 documents, 467 chunks with the default structured
  chunking, `CHUNK_SIZE_TOKENS=300`), and `data/demo_fr_business/docs`
  (13 documents, 43 chunks) for `golden_fr_business_v1`.

`golden_v3` and `golden_fr_business_v1` were written by reading the source
documents, so their questions share vocabulary with the relevant passages.
BM25 scores on them are therefore optimistic compared with `golden_v1` and
`golden_v2`, and should not be read as evidence that lexical retrieval is
sufficient.

## Reproduce

```bash
# All modes (needs the embedding model: network access or a warm .hf_cache)
PYTHONPATH=backend python scripts/benchmark_retrieval.py --dataset golden_v3
# Offline, BM25 only
PYTHONPATH=backend python scripts/benchmark_retrieval.py --modes sparse --dataset golden_v3
# French business corpus
PYTHONPATH=backend python scripts/benchmark_retrieval.py --modes sparse \
  --dataset golden_fr_business_v1 --docs data/demo_fr_business/docs
```
