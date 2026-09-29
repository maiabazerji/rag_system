# Retrieval benchmark: golden_fr_business_v1

- Generated: 2026-09-29T11:01:06+00:00
- Corpus: 13 documents, 43 chunks (CHUNK_SIZE_TOKENS=300)
- Questions scored: 32 of 37
- DENSE_TOP_K=50, BM25_TOP_K=50, RRF_K=60, rerank=off
- Embedding model: intfloat/multilingual-e5-small (not needed)

| Mode | K | Recall@K | Precision@K | HitRate@K | MRR | nDCG@K | mean latency (ms) |
|---|---|---|---|---|---|---|---|
| sparse | 1 | 0.414 | 0.594 | 0.594 | 0.720 | 0.594 | 1.9 |
| sparse | 3 | 0.820 | 0.417 | 0.906 | 0.720 | 0.719 | 1.9 |
| sparse | 5 | 0.867 | 0.269 | 0.906 | 0.720 | 0.739 | 1.9 |
| sparse | 10 | 0.958 | 0.153 | 0.969 | 0.720 | 0.775 | 1.9 |
