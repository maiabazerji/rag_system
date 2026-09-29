"""Sparse (BM25) retrieval over the indexed chunks.

Dense retrieval matches meaning and misses exact terms: product codes, error
strings, rare names, acronyms. BM25 matches exactly those. This module keeps a
BM25 index over every chunk's payload text, read from Qdrant with
:func:`app.rag.store.scroll_chunks`, so the hybrid pipeline can fuse the two.

Caching and invalidation
    The tokenized corpus is held in memory and rebuilt when either the
    collection's point count changes (any writer, including other processes)
    or this process's corpus generation changes (bumped by every upsert and
    delete in :mod:`app.rag.store`, which also catches an edit that keeps the
    count the same). Checking costs one ``count`` call per query.

Access control
    One BM25 model is built per access scope, over only the chunks that scope
    may read (:meth:`AccessScope.permits`). Unreadable chunks are therefore
    never candidates, and they do not even influence the scores through
    document frequencies or average length -- a shared index would leak
    another tenant's vocabulary statistics. Scoped models are cached (LRU)
    alongside the corpus they were built from.
"""
from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from rank_bm25 import BM25Okapi

from app.access import AccessScope
from app.config import settings
from app.logging_config import get_structured_logger
from app.rag import store
from app.rag.query import bm25_tokenize

logger = get_structured_logger(__name__)

# Scoped BM25 models kept per corpus. A deployment usually has a handful of
# distinct (tenant, groups) combinations; the bound only stops a pathological
# number of them from holding memory.
MAX_CACHED_SCOPES = 32


@dataclass(frozen=True)
class SparseHit:
    """One chunk found by BM25.

    Attributes:
        chunk_id: The chunk's id (payload ``chunk_id``).
        payload: The stored payload: text, doc_id, metadata, ACL fields.
        score: BM25 score. Only comparable within one query.
    """

    chunk_id: str
    payload: dict[str, Any]
    score: float


@dataclass
class _Corpus:
    """Every indexed chunk, tokenized once."""

    generation: int
    point_count: int
    chunk_ids: list[str]
    payloads: list[dict[str, Any]]
    tokens: list[list[str]]
    built_at: float = field(default_factory=time.monotonic)


@dataclass
class _ScopedModel:
    """A BM25 model over the part of the corpus one scope may read."""

    positions: list[int]  # indices into the corpus
    bm25: BM25Okapi | None  # None when the scope can read nothing


def _build_corpus(records: Sequence[Any], generation: int, point_count: int) -> _Corpus:
    chunk_ids: list[str] = []
    payloads: list[dict[str, Any]] = []
    tokens: list[list[str]] = []
    for r in records:
        payload = dict(getattr(r, "payload", None) or {})
        cid, text = payload.get("chunk_id"), payload.get("text")
        if not (cid and payload.get("doc_id") and text):
            continue
        chunk_ids.append(cid)
        payloads.append(payload)
        tokens.append(bm25_tokenize(text))
    return _Corpus(generation, point_count, chunk_ids, payloads, tokens)


def _build_scoped(corpus: _Corpus, access: AccessScope | None) -> _ScopedModel:
    positions = [
        i
        for i, payload in enumerate(corpus.payloads)
        if access is None or access.permits(payload)
    ]
    if not positions:
        return _ScopedModel(positions=[], bm25=None)
    return _ScopedModel(positions=positions, bm25=BM25Okapi([corpus.tokens[i] for i in positions]))


def _rank(
    corpus: _Corpus, model: _ScopedModel, tokens: Sequence[str], top_k: int
) -> list[SparseHit]:
    if model.bm25 is None or not tokens:
        return []
    scores = model.bm25.get_scores(list(tokens))
    # Ties (common: many chunks score 0 on a rare term) break on chunk id so
    # the ranking does not depend on the order Qdrant enumerated the points.
    order = sorted(
        range(len(model.positions)),
        key=lambda j: (-float(scores[j]), corpus.chunk_ids[model.positions[j]]),
    )
    terms = set(tokens)
    hits: list[SparseHit] = []
    for j in order:
        score = float(scores[j])
        # A chunk sharing no term with the query is not a lexical match. The
        # score alone cannot tell: in a tiny corpus Okapi IDF rounds to zero
        # even for terms that do occur.
        if terms.isdisjoint(model.bm25.doc_freqs[j]):
            continue
        pos = model.positions[j]
        hits.append(SparseHit(corpus.chunk_ids[pos], corpus.payloads[pos], score))
        if len(hits) >= top_k:
            break
    return hits


class SparseIndex:
    """A lazily built, self-invalidating BM25 index over the vector store."""

    def __init__(self, max_scopes: int = MAX_CACHED_SCOPES) -> None:
        self._corpus: _Corpus | None = None
        self._scoped: OrderedDict[tuple, _ScopedModel] = OrderedDict()
        self._max_scopes = max_scopes
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None
        self.rebuilds = 0

    def invalidate(self) -> None:
        """Drop everything; the next search rebuilds from the store."""
        self._corpus = None
        self._scoped.clear()

    def _get_lock(self) -> asyncio.Lock:
        # An asyncio.Lock belongs to one event loop; tests and scripts may run
        # several loops over the life of the process.
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock = asyncio.Lock()
            self._lock_loop = loop
        return self._lock

    async def _current_corpus(self) -> _Corpus:
        """The corpus, rebuilt first if the store changed since it was read.

        Raises:
            StoreUnavailable: If Qdrant cannot be counted or enumerated.
        """
        point_count = await store.count()
        generation = store.corpus_generation()
        corpus = self._corpus
        if corpus and corpus.point_count == point_count and corpus.generation == generation:
            return corpus
        async with self._get_lock():
            corpus = self._corpus
            if corpus and corpus.point_count == point_count and corpus.generation == generation:
                return corpus
            started = time.perf_counter()
            records = await store.scroll_chunks(with_payload=True)
            corpus = await asyncio.to_thread(_build_corpus, records, generation, point_count)
            self._corpus = corpus
            self._scoped.clear()
            self.rebuilds += 1
            logger.info(
                "BM25 index rebuilt",
                extra_fields={
                    "chunks": len(corpus.chunk_ids),
                    "point_count": point_count,
                    "generation": generation,
                    "build_ms": round((time.perf_counter() - started) * 1000, 1),
                },
            )
            return corpus

    async def _scoped_model(self, corpus: _Corpus, access: AccessScope | None) -> _ScopedModel:
        # The permission rule also reads these two settings, so they are part
        # of what a cached model was built under.
        key = (access, settings.default_tenant, settings.acl_legacy_public)
        model = self._scoped.get(key)
        if model is not None:
            self._scoped.move_to_end(key)
            return model
        model = await asyncio.to_thread(_build_scoped, corpus, access)
        # The corpus may have been replaced while the model was built.
        if self._corpus is corpus:
            self._scoped[key] = model
            while len(self._scoped) > self._max_scopes:
                self._scoped.popitem(last=False)
        return model

    async def search(
        self,
        tokens: Sequence[str],
        top_k: int,
        access: AccessScope | None = None,
    ) -> list[SparseHit]:
        """The ``top_k`` chunks readable by ``access`` that best match ``tokens``.

        Args:
            tokens: Query terms, from :func:`app.rag.query.bm25_tokenize`.
            top_k: Maximum number of hits.
            access: Rank only chunks this scope may read. ``None`` ranks
                everything and is for internal jobs only.

        Returns:
            Hits with a positive BM25 score, best first; ties broken by chunk id.

        Raises:
            StoreUnavailable: If the index had to be (re)built and Qdrant is down.
        """
        if not tokens or top_k <= 0:
            return []
        corpus = await self._current_corpus()
        model = await self._scoped_model(corpus, access)
        hits = await asyncio.to_thread(_rank, corpus, model, tokens, top_k)
        # Defence in depth, as in store._permitted: never hand back a chunk
        # the scope cannot read, whatever the cache holds.
        if access is not None:
            hits = [h for h in hits if access.permits(h.payload)]
        return hits


_index = SparseIndex()


def get_index() -> SparseIndex:
    """The process-wide sparse index."""
    return _index


async def sparse_search(
    tokens: Sequence[str], top_k: int, access: AccessScope | None = None
) -> list[SparseHit]:
    """BM25 search over the process-wide index. See :meth:`SparseIndex.search`."""
    return await _index.search(tokens, top_k, access)
