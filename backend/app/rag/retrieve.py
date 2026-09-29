"""Retrieval: find the chunks that should ground an answer.

The pipeline, run by :func:`hybrid_search`::

    question ─► preprocess ─┬─► dense  (embed + Qdrant, DENSE_TOP_K) ─┐
                            └─► sparse (BM25, BM25_TOP_K)            ─┴─► RRF (RRF_K)
                                                                            │
                     context (FINAL_CONTEXT_K) ◄── rerank (RERANK_TOP_K) ◄──┘

``RETRIEVAL_MODE`` picks the retrievers: ``dense``, ``sparse`` or ``hybrid``
(both, run concurrently and fused with Reciprocal Rank Fusion). The reranker
is the cross-encoder, with BM25 as its fallback when the model is unavailable.

Every stage is a tracing span (``retrieval.preprocess``, ``retrieval.dense``,
``retrieval.sparse``, ``retrieval.fusion``, ``retrieval.rerank``) and is timed
into :class:`~app.schemas.RetrievalDiagnostics`, which strategies return in
``StrategyResult.extra["retrieval"]``. Neither ever records chunk text.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from app.access import AccessScope
from app.config import settings
from app.logging_config import get_structured_logger
from app.rag.embed import embed_query_async
from app.rag.fusion import FusedCandidate, reciprocal_rank_fusion
from app.rag.query import preprocess_query
from app.rag.rerank import RerankOutcome, rerank_scored_async
from app.rag.sparse import sparse_search
from app.rag.store import StoreUnavailable, search
from app.rag.timing import record as record_stage
from app.schemas import (
    Chunk,
    RetrievalDiagnostics,
    RetrievalStageCounts,
    RetrievalStageLatency,
    RetrievedChunkDiagnostics,
)
from app.tracing import instrument, span

logger = get_structured_logger(__name__)

_RESERVED = {"chunk_id", "doc_id", "text"}


def chunk_from_payload(payload: Mapping[str, Any] | None) -> Chunk | None:
    """Rebuild a :class:`Chunk` from a Qdrant payload, or None if it is incomplete."""
    payload = payload or {}
    chunk_id = payload.get("chunk_id")
    doc_id = payload.get("doc_id")
    text = payload.get("text")
    if not (chunk_id and doc_id and text):
        return None
    return Chunk(
        id=chunk_id,
        doc_id=doc_id,
        text=text,
        tokens=len(text.split()),
        metadata={k: v for k, v in payload.items() if k not in _RESERVED},
    )


@instrument.retrieval
async def dense_search(
    query: str, top_k: int = 50, access: AccessScope | None = None
) -> list[Chunk]:
    """Retrieve relevant chunks from the vector store using dense vector search.

    Encodes the query to a vector embedding, then searches the Qdrant vector store
    for the top-k most similar chunks. Reconstructs Chunk objects from vector store
    payloads.

    A vector store outage raises :class:`StoreUnavailable` (served as a 503) so
    it is never mistaken for "no relevant documents". Other failures, such as
    an embedding error, degrade to an empty list.

    Args:
        query: The search query string (e.g., a user question or refined search term).
        top_k: Number of chunks to retrieve (default 50). Callers typically retrieve
            more here and rerank to a smaller number later.
        access: Search only the chunks this scope may read. ``None`` searches
            everything and is for internal jobs only.

    Returns:
        List of Chunk objects sorted by vector similarity (most similar first).
        Returns an empty list on non-store errors (e.g. an embedding failure).

    Raises:
        StoreUnavailable: The vector store is down or its circuit breaker is open.
    """
    try:
        vec = await embed_query_async(query)
        hits = await search(vec, top_k=top_k, access=access)
        chunks = [c for c in (chunk_from_payload(h.payload) for h in hits) if c is not None]

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
    except StoreUnavailable:
        raise
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


@dataclass
class RetrievalResult:
    """What :func:`hybrid_search` found, and how.

    Attributes:
        chunks: The final context, best first.
        diagnostics: Per-stage counts, latencies and per-chunk provenance.
    """

    chunks: list[Chunk]
    diagnostics: RetrievalDiagnostics = field(
        default_factory=lambda: RetrievalDiagnostics(mode="hybrid")
    )

    @property
    def chunk_ids(self) -> list[str]:
        return [c.id for c in self.chunks]

    def diagnostics_for(self, context: list[Chunk]) -> RetrievalDiagnostics:
        """These diagnostics, re-described for a context the caller assembled.

        For strategies that post-process the retrieved chunks (the graph
        strategy adds its own and reranks the lot): per-chunk entries follow
        ``context``, and chunks retrieval never returned get null ranks.
        """
        known = {c.chunk_id: c for c in self.diagnostics.chunks}
        entries = [
            known.get(c.id) or RetrievedChunkDiagnostics(chunk_id=c.id, doc_id=c.doc_id)
            for c in context
        ]
        counts = self.diagnostics.counts.model_copy(update={"final": len(entries)})
        return self.diagnostics.model_copy(update={"chunks": entries, "counts": counts})


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def _set_meta(s: Any, **meta: Any) -> None:
    """Attach counts to a span (``span`` yields None outside a trace)."""
    if s is not None:
        s.metadata.update(meta)


async def hybrid_search(
    query: str,
    access: AccessScope | None = None,
    *,
    mode: str | None = None,
    final_k: int | None = None,
    rerank: bool = True,
    dense_k: int | None = None,
    sparse_k: int | None = None,
    rrf_k: int | None = None,
) -> RetrievalResult:
    """Retrieve the context for ``query``: preprocess, retrieve, fuse, rerank.

    Args:
        query: The user's question. Retrieval uses a normalised copy; callers
            keep the original for generation.
        access: Retrieve only chunks this scope may read. ``None`` is
            unrestricted and is for internal jobs only.
        mode: ``dense``, ``sparse`` or ``hybrid``. Defaults to RETRIEVAL_MODE.
        final_k: Chunks to return. Defaults to FINAL_CONTEXT_K.
        rerank: Rerank the fused candidates. Callers that rerank a larger
            pool themselves (graph) or want raw fused results cheaply (the
            agent's search tool) turn it off.
        dense_k: Dense candidates. Defaults to DENSE_TOP_K.
        sparse_k: BM25 candidates. Defaults to BM25_TOP_K.
        rrf_k: RRF constant. Defaults to RRF_K.

    Returns:
        A :class:`RetrievalResult`.

    Raises:
        StoreUnavailable: The vector store is down (in hybrid mode only if the
            dense retriever needed it; a failed BM25 index is skipped and
            reported in ``diagnostics.degraded``).
        ValueError: If ``mode`` is not one of the three modes.
    """
    requested = (mode or settings.retrieval_mode).lower()
    if requested not in ("dense", "sparse", "hybrid"):
        raise ValueError(f"unknown retrieval mode '{requested}'")
    mode = cast(Literal["dense", "sparse", "hybrid"], requested)
    final_k = settings.final_context_k if final_k is None else final_k
    dense_k = settings.dense_top_k if dense_k is None else dense_k
    sparse_k = settings.bm25_top_k if sparse_k is None else sparse_k
    rrf_k = settings.rrf_k if rrf_k is None else rrf_k

    counts = RetrievalStageCounts()
    latency = RetrievalStageLatency()
    degraded: list[str] = []

    started = time.perf_counter()
    with span("retrieval.preprocess") as s:
        prepared = preprocess_query(query)
        _set_meta(s, chars=len(prepared.text), tokens=len(prepared.tokens),
                  truncated=prepared.truncated)
    latency.preprocess = _ms(started)

    if not prepared.text:
        record_stage("retrieval", latency.preprocess)
        return RetrievalResult(
            chunks=[],
            diagnostics=RetrievalDiagnostics(mode=mode, counts=counts, latency_ms=latency),
        )

    async def run_dense() -> list[Chunk]:
        t0 = time.perf_counter()
        try:
            with span("retrieval.dense", as_type="retriever") as s:
                chunks = await dense_search(prepared.text, top_k=dense_k, access=access)
                _set_meta(s, top_k=dense_k, returned=len(chunks))
            return chunks
        finally:
            latency.dense = _ms(t0)

    async def run_sparse() -> list[Chunk]:
        t0 = time.perf_counter()
        try:
            with span("retrieval.sparse", as_type="retriever") as s:
                try:
                    hits = await sparse_search(prepared.tokens, sparse_k, access)
                except StoreUnavailable:
                    if mode == "sparse":
                        raise
                    degraded.append("sparse")
                    logger.warning("BM25 index unavailable; continuing with dense only")
                    hits = []
                except Exception as e:
                    degraded.append("sparse")
                    logger.warning(
                        f"Sparse retrieval failed: {type(e).__name__}: {e}",
                        extra_fields={"error_type": type(e).__name__},
                    )
                    hits = []
                chunks = [c for c in (chunk_from_payload(h.payload) for h in hits) if c]
                _set_meta(s, top_k=sparse_k, returned=len(chunks))
            return chunks
        finally:
            latency.sparse = _ms(t0)

    dense: list[Chunk] = []
    sparse: list[Chunk] = []
    retrievers_started = time.perf_counter()
    if mode == "hybrid":
        dense, sparse = await asyncio.gather(run_dense(), run_sparse())
    elif mode == "dense":
        dense = await run_dense()
    else:
        sparse = await run_sparse()
    counts.dense, counts.sparse = len(dense), len(sparse)
    retrievers_ms = _ms(retrievers_started)

    t0 = time.perf_counter()
    with span("retrieval.fusion") as s:
        rankings: dict[str, list[Chunk]] = {}
        if mode != "sparse":
            rankings["dense"] = dense
        if mode != "dense":
            rankings["sparse"] = sparse
        fused = reciprocal_rank_fusion(rankings, k=rrf_k)
        _set_meta(s, rrf_k=rrf_k, dense=len(dense), sparse=len(sparse), fused=len(fused))
    latency.fusion = _ms(t0)
    counts.fused = len(fused)
    # The request's stage timer: dense and sparse overlap, so their wall time
    # counts once; the reranker is its own stage.
    record_stage("retrieval", latency.preprocess + retrievers_ms + latency.fusion)

    # The reranker sees as many candidates as the deeper retriever returned,
    # so hybrid costs the cross-encoder no more than dense-only did.
    pool = [c.chunk for c in fused[: max(dense_k, sparse_k)]]
    if rerank and pool:
        t0 = time.perf_counter()
        with span("retrieval.rerank") as s:
            keep = max(settings.rerank_top_k, final_k)
            outcome = await rerank_scored_async(prepared.text, pool, top_k=keep)
            _set_meta(s, candidates=len(pool), kept=len(outcome.chunks),
                      reranker=outcome.reranker)
        latency.rerank = _ms(t0)
        counts.reranked = len(outcome.chunks)
        record_stage("rerank", latency.rerank)
    else:
        outcome = RerankOutcome.unscored(pool)

    final = outcome.chunks[:final_k]
    counts.final = len(final)
    scores = list(outcome.scores[: len(final)])
    scores += [None] * (len(final) - len(scores))
    by_id: dict[str, FusedCandidate] = {c.chunk_id: c for c in fused}
    # Carry the scores on the chunks themselves, so generation can weigh the
    # evidence it cites (see app.rag.grounding) without the diagnostics object.
    for chunk, score in zip(final, scores, strict=True):
        if score is not None:
            chunk.metadata["rerank_score"] = score
        if chunk.id in by_id:
            chunk.metadata["fused_score"] = by_id[chunk.id].fused_score
    diagnostics = RetrievalDiagnostics(
        mode=mode,
        reranker=outcome.reranker,
        query_truncated=prepared.truncated,
        counts=counts,
        latency_ms=latency,
        degraded=degraded,
        chunks=[
            RetrievedChunkDiagnostics(
                chunk_id=chunk.id,
                doc_id=chunk.doc_id,
                dense_rank=by_id[chunk.id].dense_rank if chunk.id in by_id else None,
                sparse_rank=by_id[chunk.id].sparse_rank if chunk.id in by_id else None,
                fused_score=by_id[chunk.id].fused_score if chunk.id in by_id else None,
                rerank_score=score,
            )
            for chunk, score in zip(final, scores, strict=True)
        ],
    )
    logger.info(
        "Hybrid retrieval completed",
        extra_fields={
            "mode": mode,
            "reranker": outcome.reranker,
            "counts": counts.model_dump(),
            "latency_ms": latency.model_dump(),
            "degraded": degraded,
        },
    )
    return RetrievalResult(chunks=final, diagnostics=diagnostics)
