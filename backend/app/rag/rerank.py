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

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from rank_bm25 import BM25Okapi

from app.config import settings
from app.logging_config import get_structured_logger
from app.rag.query import BM25_STOPWORDS, bm25_tokenize
from app.schemas import Chunk
from app.tracing import instrument

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import CrossEncoder

logger = get_structured_logger(__name__)

# The tokenizer moved to app.rag.query so the sparse retriever shares it; both
# names stay importable from here.
__all__ = ["BM25_STOPWORDS", "RerankOutcome", "bm25_tokenize", "rerank", "rerank_async",
           "rerank_scored", "rerank_scored_async"]

Reranker = Literal["cross-encoder", "bm25-fallback", "none"]


@dataclass(frozen=True)
class RerankOutcome:
    """Reranked chunks with their scores and the scorer that produced them.

    Attributes:
        chunks: The kept chunks, best first.
        scores: One score per kept chunk (``None`` when nothing scored it).
        reranker: ``cross-encoder``, ``bm25-fallback``, or ``none`` when the
            input was passed through unscored.
    """

    chunks: list[Chunk]
    scores: list[float | None] = field(default_factory=list)
    reranker: Reranker = "none"

    @classmethod
    def unscored(cls, chunks: list[Chunk]) -> RerankOutcome:
        return cls(chunks=list(chunks), scores=[None] * len(chunks), reranker="none")


_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rerank")
_cross_encoder_model: CrossEncoder | None = None
_cross_encoder_error: str | None = None
# When the last load failed. A failure is retried after the cooldown rather
# than cached for the life of the process: the usual causes (a model download
# interrupted by a network blip, a cold cache volume) are transient, and a
# permanent fall back to BM25 silently degrades every query until a restart.
_cross_encoder_failed_at: float | None = None
CROSS_ENCODER_RETRY_SECONDS = 300.0


def _load_cross_encoder() -> CrossEncoder | None:
    """Load and cache the cross-encoder model.

    Loads the cross-encoder model on first call and caches it globally.
    On failure, returns None (fallback to BM25) and does not try again until
    ``CROSS_ENCODER_RETRY_SECONDS`` have passed.

    Returns:
        CrossEncoder instance if successful, None if loading failed or a
            recent failure is still cooling down.
    """
    global _cross_encoder_model, _cross_encoder_error, _cross_encoder_failed_at

    if _cross_encoder_model is not None:
        return _cross_encoder_model

    if (
        _cross_encoder_failed_at is not None
        and time.monotonic() - _cross_encoder_failed_at < CROSS_ENCODER_RETRY_SECONDS
    ):
        return None

    try:
        # Imported lazily so this module does not drag in torch at import time.
        from sentence_transformers import CrossEncoder

        _cross_encoder_model = CrossEncoder(settings.reranker_model)
        _cross_encoder_error = None
        _cross_encoder_failed_at = None
        logger.info(
            "Cross-encoder model loaded",
            extra_fields={"model": settings.reranker_model},
        )
        return _cross_encoder_model
    except Exception as e:
        _cross_encoder_error = str(e)
        _cross_encoder_failed_at = time.monotonic()
        logger.warning(
            f"Failed to load cross-encoder model, will use BM25 fallback: {e}",
            extra_fields={
                "error_type": type(e).__name__,
                "fallback": "bm25",
            },
        )
        return None


def _score_with_cross_encoder(query: str, chunks: list[Chunk], top_k: int) -> RerankOutcome:
    """Rerank chunks with the cross-encoder, keeping the scores.

    Creates (query, chunk_text) pairs and scores them with a fine-tuned
    cross-encoder model. Chunks are sorted by score and the top-k returned.
    If the model is unavailable or fails, falls back to BM25 scoring.

    Args:
        query: The query string to score pairs against.
        chunks: Chunks to rerank.
        top_k: Number of top results to return.

    Returns:
        The top ``min(top_k, len(chunks))`` chunks with their scores.
    """
    cross_encoder = _load_cross_encoder()
    if cross_encoder is None:
        return _score_with_bm25(query, chunks, top_k)

    try:
        # `predict` is typed against a wide multimodal union and list is
        # invariant, so a concrete list[list[str]] is rejected even though it
        # is exactly what the method expects.
        pairs: list[Any] = [[query, chunk.text] for chunk in chunks]
        scores = cross_encoder.predict(pairs)

        # Stable sort: equal scores keep their retrieval order.
        scored_chunks = list(zip(chunks, scores, strict=True))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        kept = scored_chunks[:top_k]
        logger.info(
            "Cross-encoder reranking completed",
            extra_fields={
                "input_count": len(chunks),
                "output_count": len(kept),
                "requested_top_k": top_k,
                "reranker": "cross-encoder",
            },
        )
        return RerankOutcome(
            chunks=[c for c, _ in kept],
            scores=[float(sc) for _, sc in kept],
            reranker="cross-encoder",
        )
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
        return _score_with_bm25(query, chunks, top_k)


def _rerank_with_cross_encoder(query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
    """Rerank chunks with the cross-encoder (BM25 fallback); chunks only."""
    return _score_with_cross_encoder(query, chunks, top_k).chunks


def _score_with_bm25(query: str, chunks: list[Chunk], top_k: int) -> RerankOutcome:
    """Rerank chunks by BM25 lexical score, keeping the scores.

    The fallback when the cross-encoder is unavailable: fast and reliable, but
    blind to paraphrase. If BM25 itself fails, the first ``top_k`` chunks are
    returned unscored, in input order.
    """
    if not chunks:
        return RerankOutcome(chunks=[], scores=[], reranker="bm25-fallback")

    try:
        tokenized_chunks = [bm25_tokenize(chunk.text) for chunk in chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        query_tokens = bm25_tokenize(query)
        scores = bm25.get_scores(query_tokens)

        scored_chunks = list(zip(chunks, scores, strict=True))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        kept = scored_chunks[:top_k]
        logger.info(
            "BM25 reranking completed",
            extra_fields={
                "input_count": len(chunks),
                "output_count": len(kept),
                "requested_top_k": top_k,
                "reranker": "bm25",
                "query_tokens": len(query_tokens),
            },
        )
        return RerankOutcome(
            chunks=[c for c, _ in kept],
            scores=[float(sc) for _, sc in kept],
            reranker="bm25-fallback",
        )
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
        return RerankOutcome.unscored(chunks[:top_k])


def _rerank_with_bm25(query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
    """Rerank chunks by BM25 lexical score; chunks only."""
    return _score_with_bm25(query, chunks, top_k).chunks


def rerank_scored(query: str, chunks: list[Chunk], top_k: int = 8) -> RerankOutcome:
    """Rerank chunks by relevance to ``query``, reporting scores and scorer.

    Attempts the cross-encoder first and falls back to BM25, then to plain
    truncation: it always returns a result and never raises.

    Args:
        query: The query to rerank by.
        chunks: Candidate chunks (typically from retrieval).
        top_k: Number of chunks to keep. Fewer come back if fewer went in.

    Returns:
        A :class:`RerankOutcome`, best chunk first.
    """
    if not chunks:
        return RerankOutcome(chunks=[], scores=[], reranker="none")

    # Score even a set smaller than top_k, so it comes back in relevance order.
    keep = min(top_k, len(chunks))
    try:
        return _score_with_cross_encoder(query, chunks, keep)
    except Exception as e:
        logger.debug(
            f"Reranking failed: {e}",
            extra_fields={
                "error_type": type(e).__name__,
                "input_count": len(chunks),
                "requested_top_k": top_k,
            },
        )
        return RerankOutcome.unscored(chunks[:keep])


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

    Example:
        >>> chunks = [Chunk(id="1", text="..."), Chunk(id="2", text="...")]
        >>> reranked = rerank("What is AI?", chunks, top_k=2)
        >>> print(len(reranked))  # 2 or fewer
    """
    return rerank_scored(query, chunks, top_k).chunks


@instrument.rerank
async def rerank_async(query: str, chunks: list[Chunk], top_k: int = 8) -> list[Chunk]:
    """Async wrapper around :func:`rerank`.

    Cross-encoder inference is CPU-bound and would otherwise block the event
    loop for the duration of the scoring pass.

    Args:
        query: The query to rerank against.
        chunks: Candidate chunks from retrieval.
        top_k: Number of chunks to keep.

    Returns:
        The reranked chunks, best first.
    """
    if not chunks:
        return []
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, rerank, query, chunks, top_k)


@instrument.rerank_scored
async def rerank_scored_async(query: str, chunks: list[Chunk], top_k: int = 8) -> RerankOutcome:
    """Async wrapper around :func:`rerank_scored`, off the event loop."""
    if not chunks:
        return RerankOutcome(chunks=[], scores=[], reranker="none")
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, rerank_scored, query, chunks, top_k)
