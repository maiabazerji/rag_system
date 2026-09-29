# Retrieval benchmark: golden_v2

- Generated: 2026-09-29T11:00:41+00:00
- Corpus: 41 documents, 467 chunks (CHUNK_SIZE_TOKENS=300)
- Questions scored: 32 of 34
- DENSE_TOP_K=50, BM25_TOP_K=50, RRF_K=60, rerank=off
- Embedding model: intfloat/multilingual-e5-small (not loaded)

| Mode | K | Recall@K | Precision@K | HitRate@K | MRR | nDCG@K | mean latency (ms) |
|---|---|---|---|---|---|---|---|
| sparse | 1 | 0.352 | 0.625 | 0.625 | 0.738 | 0.625 | 4.3 |
| sparse | 3 | 0.607 | 0.438 | 0.875 | 0.738 | 0.620 | 4.3 |
| sparse | 5 | 0.701 | 0.312 | 0.906 | 0.738 | 0.651 | 4.3 |
| sparse | 10 | 0.880 | 0.209 | 0.938 | 0.738 | 0.734 | 4.3 |

Skipped modes:
- **dense**: skipped -- embedding model unavailable: RuntimeError: Failed to get embedding dimension: Failed to initialize embedding model 'intfloat/multilingual-e5-small'. Ensure the model name is correct and you have internet access to download it.
- **hybrid**: skipped -- embedding model unavailable: RuntimeError: Failed to get embedding dimension: Failed to initialize embedding model 'intfloat/multilingual-e5-small'. Ensure the model name is correct and you have internet access to download it.
