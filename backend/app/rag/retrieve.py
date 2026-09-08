"""Retrieval functions for fetching relevant chunks from the vector store.

This module handles semantic search over indexed documents. It provides a single
high-level function (hybrid_search) that orchestrates embedding + vector search
with graceful degradation for service failures.

Retrieval is dense-only: the query is embedded and matched against the vector
store. The lexical (BM25) signal enters later, in `rerank.py`, as the reranker's
fallback scorer -- so this is a dense-retrieve/rerank pipeline, not hybrid
retrieval in the fuse-two-retrievers sense.
"""
from __future__ import annotations

from app.logging_config import get_structured_logger
from app.rag.embed import embed_query_async
from app.rag.store import search
from app.schemas import Chunk

logger = get_structured_logger(__name__)

_RESERVED = {"chunk_id", "doc_id", "text"}


async def dense_search(query: str, top_k: int = 50) -> list[Chunk]:
    """Retrieve relevant chunks from the vector store using dense vector search.

    Encodes the query to a vector embedding, then searches the Qdrant vector store
    for the top-k most similar chunks. Reconstructs Chunk objects from vector store
    payloads.

    This function gracefully degrades: if the vector store is unavailable,
    it returns an empty list instead of raising an exception. This allows the
    answer generation step to proceed with no context (producing a refusal or
    empty answer).

    Args:
        query: The search query string (e.g., a user question or refined search term).
        top_k: Number of chunks to retrieve (default 50). Callers typically retrieve
            more here and rerank to a smaller number later.

    Returns:
        List of Chunk objects sorted by vector similarity (most similar first).
        Chunks include id, doc_id, text, token count, and document metadata.
        Returns empty list on any error (vector store down, embedding error, etc.).

    Raises:
        No exceptions. All errors are caught and logged, with empty list returned.

    Example:
        >>> chunks = await dense_search("What is machine learning?", top_k=10)
        >>> for chunk in chunks:
        ...     print(f"[{chunk.id}] {chunk.text[:100]}...")
    """
    try:
        vec = await embed_query_async(query)
        hits = await search(vec, top_k=top_k)
        chunks = []
        for h in hits:
            chunk_id = h.payload.get("chunk_id")
            doc_id = h.payload.get("doc_id")
            text = h.payload.get("text")
            if chunk_id and doc_id and text:
                chunks.append(Chunk(
                    id=chunk_id,
                    doc_id=doc_id,
                    text=text,
                    tokens=len(text.split()),
                    metadata={k: v for k, v in h.payload.items() if k not in _RESERVED},
                ))

        logger.info(
            "Retrieval completed",
            extra_fields={
                "query_len": len(query),
                "requested_top_k": top_k,
                "retrieved_count": len(hits),
                "chunks_constructed": len(chunks),
            },
        )
        return chunks
    except Exception as e:
        logger.warning(
            f"Retrieval failed: {type(e).__name__}: {e}. Proceeding with empty context.",
            extra_fields={
                "error_type": type(e).__name__,
                "query_len": len(query),
                "requested_top_k": top_k,
            },
        )
        return []  # Graceful degradation: return empty results instead of failing


# Backwards-compatible alias: this function was called `hybrid_search` before the
# name was corrected to match what it actually does.
hybrid_search = dense_search
