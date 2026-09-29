# Retrieval benchmark: golden_v3

- Generated: 2026-09-29T10:43:17+00:00
- Corpus: 41 documents, 467 chunks (CHUNK_SIZE_TOKENS=300)
- Questions scored: 47 of 54
- DENSE_TOP_K=50, BM25_TOP_K=50, RRF_K=60, rerank=off
- Embedding model: intfloat/multilingual-e5-small (not loaded)

| Mode | K | Recall@K | Precision@K | HitRate@K | MRR | nDCG@K | mean latency (ms) |
|---|---|---|---|---|---|---|---|
| sparse | 1 | 0.647 | 0.808 | 0.808 | 0.808 | 0.808 | 3.7 |
| sparse | 3 | 0.856 | 0.543 | 0.936 | 0.869 | 0.834 | 3.7 |
| sparse | 5 | 0.938 | 0.402 | 1.000 | 0.888 | 0.877 | 3.7 |
| sparse | 10 | 0.968 | 0.238 | 1.000 | 0.888 | 0.891 | 3.7 |

Skipped modes:
- **dense**: skipped -- embedding model unavailable: RuntimeError: Failed to get embedding dimension: Failed to initialize embedding model 'intfloat/multilingual-e5-small'. Ensure the model name is correct and you have internet access to download it.
- **hybrid**: skipped -- embedding model unavailable: RuntimeError: Failed to get embedding dimension: Failed to initialize embedding model 'intfloat/multilingual-e5-small'. Ensure the model name is correct and you have internet access to download it.
