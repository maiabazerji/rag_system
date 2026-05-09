# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation combines a retriever over an external corpus
with a generative language model. At query time, the retriever fetches
relevant passages, and the generator conditions its output on both the user's
question and the retrieved context. This reduces hallucination and lets the
model answer with information outside its pretraining cutoff.

## Typical Pipeline

1. **Ingestion** — documents are parsed, cleaned, and split into chunks of
   roughly 300–800 tokens with a small overlap (e.g., 10–15%).
2. **Embedding** — each chunk is embedded with a model such as
   `text-embedding-3-large` or `bge-large-en`, and stored in a vector index.
3. **Retrieval** — the query is embedded, and the top-k nearest chunks are
   returned. Hybrid retrieval combines dense vectors with a sparse signal
   like BM25.
4. **Reranking** — a cross-encoder (e.g., `bge-reranker-large`, Cohere Rerank)
   re-scores the top-k candidates for higher precision.
5. **Generation** — the LLM generates an answer grounded in the selected
   passages, typically with citations back to the source chunks.

## When to Use RAG vs Fine-Tuning

RAG is the correct choice when the knowledge is large, frequently updated,
or requires attribution. Fine-tuning is more appropriate for changing the
model's behavior, style, or output format rather than for injecting facts.
