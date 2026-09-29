# Retrieval benchmark: golden_fr_business_v1

- Generated: 2026-09-29T10:43:31+00:00
- Corpus: 13 documents, 43 chunks (CHUNK_SIZE_TOKENS=300)
- Questions scored: 32 of 37
- DENSE_TOP_K=50, BM25_TOP_K=50, RRF_K=60, rerank=off
- Embedding model: intfloat/multilingual-e5-small (not needed)

| Mode | K | Recall@K | Precision@K | HitRate@K | MRR | nDCG@K | mean latency (ms) |
|---|---|---|---|---|---|---|---|
| sparse | 1 | 0.414 | 0.594 | 0.594 | 0.594 | 0.594 | 1.4 |
| sparse | 3 | 0.812 | 0.469 | 0.906 | 0.708 | 0.712 | 1.4 |
| sparse | 5 | 0.867 | 0.340 | 0.906 | 0.708 | 0.739 | 1.4 |
| sparse | 10 | 0.940 | 0.220 | 0.969 | 0.717 | 0.766 | 1.4 |
