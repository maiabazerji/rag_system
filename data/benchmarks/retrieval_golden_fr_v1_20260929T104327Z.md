# Retrieval benchmark: golden_fr_v1

- Generated: 2026-09-29T10:43:27+00:00
- Corpus: 41 documents, 467 chunks (CHUNK_SIZE_TOKENS=300)
- Questions scored: 16 of 16
- DENSE_TOP_K=50, BM25_TOP_K=50, RRF_K=60, rerank=off
- Embedding model: intfloat/multilingual-e5-small (not loaded)

| Mode | K | Recall@K | Precision@K | HitRate@K | MRR | nDCG@K | mean latency (ms) |
|---|---|---|---|---|---|---|---|
| sparse | 1 | 0.328 | 0.625 | 0.625 | 0.625 | 0.625 | 5.9 |
| sparse | 3 | 0.661 | 0.521 | 0.875 | 0.750 | 0.657 | 5.9 |
| sparse | 5 | 0.760 | 0.419 | 0.875 | 0.750 | 0.700 | 5.9 |
| sparse | 10 | 0.844 | 0.311 | 0.875 | 0.750 | 0.745 | 5.9 |

Skipped modes:
- **dense**: skipped -- embedding model unavailable: RuntimeError: Failed to get embedding dimension: Failed to initialize embedding model 'intfloat/multilingual-e5-small'. Ensure the model name is correct and you have internet access to download it.
- **hybrid**: skipped -- embedding model unavailable: RuntimeError: Failed to get embedding dimension: Failed to initialize embedding model 'intfloat/multilingual-e5-small'. Ensure the model name is correct and you have internet access to download it.
