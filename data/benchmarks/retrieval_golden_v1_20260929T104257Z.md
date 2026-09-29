# Retrieval benchmark: golden_v1

- Generated: 2026-09-29T10:42:57+00:00
- Corpus: 41 documents, 467 chunks (CHUNK_SIZE_TOKENS=300)
- Questions scored: 34 of 34
- DENSE_TOP_K=50, BM25_TOP_K=50, RRF_K=60, rerank=off
- Embedding model: intfloat/multilingual-e5-small (not loaded)

| Mode | K | Recall@K | Precision@K | HitRate@K | MRR | nDCG@K | mean latency (ms) |
|---|---|---|---|---|---|---|---|
| sparse | 1 | 0.353 | 0.353 | 0.353 | 0.353 | 0.353 | 4.4 |
| sparse | 3 | 0.500 | 0.216 | 0.500 | 0.412 | 0.434 | 4.4 |
| sparse | 5 | 0.529 | 0.132 | 0.529 | 0.422 | 0.449 | 4.4 |
| sparse | 10 | 0.647 | 0.092 | 0.647 | 0.439 | 0.489 | 4.4 |

Skipped modes:
- **dense**: skipped -- embedding model unavailable: RuntimeError: Failed to get embedding dimension: Failed to initialize embedding model 'intfloat/multilingual-e5-small'. Ensure the model name is correct and you have internet access to download it.
- **hybrid**: skipped -- embedding model unavailable: RuntimeError: Failed to get embedding dimension: Failed to initialize embedding model 'intfloat/multilingual-e5-small'. Ensure the model name is correct and you have internet access to download it.
