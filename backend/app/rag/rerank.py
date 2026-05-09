from __future__ import annotations

from app.schemas import Chunk


def rerank(query: str, chunks: list[Chunk], top_k: int = 8) -> list[Chunk]:
    """Cross-encoder reranking. Stub preserves order — swap in bge-reranker-large."""
    # from sentence_transformers import CrossEncoder
    # model = CrossEncoder("BAAI/bge-reranker-large")
    # scores = model.predict([(query, c.text) for c in chunks])
    # ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    # return [c for c, _ in ranked[:top_k]]
    return chunks[:top_k]


def compress_context(query: str, chunks: list[Chunk]) -> list[Chunk]:
    """Extractive compression — keep spans relevant to query."""
    return chunks
