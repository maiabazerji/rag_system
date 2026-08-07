"""Reranking functions for sorting chunks by relevance to a query.

Reranking improves retrieval quality by re-scoring candidate chunks after initial
retrieval. This module provides multiple reranking strategies:

1. Cross-encoder: Neural model trained to score (query, chunk) pairs directly.
   Slower but more accurate than embedding-based ranking.
2. BM25 fallback: Sparse lexical ranking (fast, always available).

The approach is to retrieve many candidates via vector search, then rerank
to select the best ones for inclusion in the LLM context window.

Key concepts:
    - Reranking is a second-pass scoring after initial retrieval
    - Cross-encoders score query-chunk pairs directly (semantic + lexical)
    - BM25 provides graceful degradation if cross-encoder unavailable
"""
from __future__ import annotations

import logging
from typing import Optional

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.logging_config import get_structured_logger
from app.schemas import Chunk

logger = get_structured_logger(__name__)

_cross_encoder_model: Optional[CrossEncoder] = None
_cross_encoder_error: Optional[str] = None


def _load_cross_encoder() -> Optional[CrossEncoder]:
    """Load and cache the cross-encoder model.

    Loads the cross-encoder model on first call and caches it globally.
    On failure, caches the error and returns None (fallback to BM25).

    Returns:
        CrossEncoder instance if successful, None if loading failed.
            Subsequent calls return cached result (success or failure).
    """
    global _cross_encoder_model, _cross_encoder_error

    if _cross_encoder_model is not None:
        return _cross_encoder_model

    if _cross_encoder_error is not None:
        return None

    try:
        _cross_encoder_model = CrossEncoder("cross-encoder/mmarco-MiniLMv2-L12-H384-v1")
        logger.info(
            "Cross-encoder model loaded",
            extra_fields={"model": "mmarco-MiniLMv2-L12-H384-v1"},
        )
        return _cross_encoder_model
    except Exception as e:
        _cross_encoder_error = str(e)
        logger.warning(
            f"Failed to load cross-encoder model, will use BM25 fallback: {e}",
            extra_fields={
                "error_type": type(e).__name__,
                "fallback": "bm25",
            },
        )
        return None


def _rerank_with_cross_encoder(
    query: str,
    chunks: list[Chunk],
    top_k: int
) -> list[Chunk]:
    """Rerank chunks using cross-encoder semantic scoring.

    Creates (query, chunk_text) pairs and scores them with a fine-tuned cross-encoder
    model. Chunks are sorted by score and top-k returned.

    If cross-encoder fails, falls back to BM25 ranking.

    Args:
        query: The query string to score pairs against.
        chunks: List of Chunk objects to rerank.
        top_k: Number of top results to return.

    Returns:
        List of top-k chunks sorted by cross-encoder score (highest first).
            Length is min(top_k, len(chunks)).

    Raises:
        No explicit exceptions. Failures fall back to _rerank_with_bm25.
    """
    cross_encoder = _load_cross_encoder()
    if cross_encoder is None:
        return _rerank_with_bm25(query, chunks, top_k)

    try:
        # Prepare pairs for cross-encoder
        pairs = [[query, chunk.text] for chunk in chunks]

        # Score all pairs
        scores = cross_encoder.predict(pairs)

        # Sort chunks by score (descending)
        scored_chunks = list(zip(chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        result = [chunk for chunk, _score in scored_chunks[:top_k]]
        logger.info(
            "Cross-encoder reranking completed",
            extra_fields={
                "input_count": len(chunks),
                "output_count": len(result),
                "requested_top_k": top_k,
                "reranker": "cross-encoder",
            },
        )
        return result
    except Exception as e:
        logger.warning(
            f"Cross-encoder reranking failed, falling back to BM25: {e}",
            extra_fields={
                "error_type": type(e).__name__,
                "input_count": len(chunks),
                "requested_top_k": top_k,
                "fallback": "bm25",
            },
        )
        return _rerank_with_bm25(query, chunks, top_k)


def _rerank_with_bm25(query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
    """Rerank chunks using BM25 lexical scoring.

    Tokenizes query and chunk texts, then applies BM25 algorithm to score
    relevance. Fallback when cross-encoder is unavailable.

    BM25 is fast and reliable but may miss semantic relationships
    compared to neural models.

    Args:
        query: The query string to score against.
        chunks: List of Chunk objects to rerank.
        top_k: Number of top results to return.

    Returns:
        List of top-k chunks sorted by BM25 score (highest first).
            If BM25 fails, returns first top_k chunks in input order.

    Raises:
        No explicit exceptions. On critical error, returns truncated input.
    """
    if not chunks:
        return []

    try:
        # Tokenize texts for BM25
        tokenized_chunks = [chunk.text.lower().split() for chunk in chunks]

        # Create BM25 model
        bm25 = BM25Okapi(tokenized_chunks)

        # Score query
        query_tokens = query.lower().split()
        scores = bm25.get_scores(query_tokens)

        # Sort chunks by score (descending)
        scored_chunks = list(zip(chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        result = [chunk for chunk, _score in scored_chunks[:top_k]]
        logger.info(
            "BM25 reranking completed",
            extra_fields={
                "input_count": len(chunks),
                "output_count": len(result),
                "requested_top_k": top_k,
                "reranker": "bm25",
                "query_tokens": len(query_tokens),
            },
        )
        return result
    except Exception as e:
        logger.error(
            f"BM25 reranking failed: {e}",
            extra_fields={
                "error_type": type(e).__name__,
                "input_count": len(chunks),
                "requested_top_k": top_k,
                "fallback": "truncation",
            },
        )
        # Fallback to just returning top-k by order
        return chunks[:top_k]


def rerank(query: str, chunks: list[Chunk], top_k: int = 8) -> list[Chunk]:
    """Rerank chunks by relevance to query using cross-encoder or BM25.

    Performs reranking to improve retrieval quality. Attempts semantic reranking
    with a cross-encoder model. If that fails, falls back to BM25 sparse ranking.

    This function always returns a result (never raises), implementing graceful
    degradation at each step: cross-encoder -> BM25 -> simple truncation.

    Args:
        query: The query string to rerank by. Typically a user question or
            refined search term.
        chunks: List of candidate chunks to rerank (typically from vector search).
        top_k: Number of top results to return (default 8). For small inputs,
            may return fewer than top_k if len(chunks) < top_k.

    Returns:
        List of chunks sorted by relevance score (best first).
            Length is min(top_k, len(chunks)).
            Returns empty list if input is empty.

    Raises:
        No exceptions. Graceful fallback at each layer ensures a result.

    Example:
        >>> chunks = [Chunk(id="1", text="..."), Chunk(id="2", text="...")]
        >>> reranked = rerank("What is AI?", chunks, top_k=2)
        >>> print(len(reranked))  # 2 or fewer
    """
    if not chunks:
        return []

    if len(chunks) <= top_k:
        # If we have fewer chunks than top_k, try to rerank anyway for scoring
        # but return all of them
        try:
            result = _rerank_with_cross_encoder(query, chunks, len(chunks))
            logger.debug(
                "Reranking small result set",
                extra_fields={
                    "input_count": len(chunks),
                    "output_count": len(result),
                    "requested_top_k": top_k,
                },
            )
            return result
        except Exception as e:
            logger.debug(
                f"Reranking failed for small result set: {e}",
                extra_fields={
                    "error_type": type(e).__name__,
                    "input_count": len(chunks),
                    "requested_top_k": top_k,
                },
            )
            return chunks

    # Normal case: rerank and truncate to top_k
    return _rerank_with_cross_encoder(query, chunks, top_k)
