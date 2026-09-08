from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

try:
    from qdrant_client.async_client import AsyncQdrantClient
except ImportError:
    from qdrant_client import AsyncQdrantClient

from qdrant_client.http import models as qm

from app.config import settings
from app.rag.embed import embedding_dim
from app.resilience import CircuitBreaker, async_timeout_wrapper

logger = logging.getLogger(__name__)

_client: AsyncQdrantClient | None = None
_qdrant_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=30.0,
    service_name="Qdrant",
)


async def client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        _client = AsyncQdrantClient(url=settings.qdrant_url)
        await _ensure_collection(_client)
    return _client


async def _ensure_collection(c: AsyncQdrantClient) -> None:
    try:
        dim = embedding_dim()
        collections = await async_timeout_wrapper(
            c.get_collections(), timeout=5.0, service_name="Qdrant"
        )
        existing = {col.name for col in collections.collections}
        if settings.qdrant_collection in existing:
            info = await async_timeout_wrapper(
                c.get_collection(settings.qdrant_collection),
                timeout=5.0,
                service_name="Qdrant",
            )
            current = info.config.params.vectors.size
            if current != dim:
                raise RuntimeError(
                    f"Qdrant collection '{settings.qdrant_collection}' has dim={current} "
                    f"but embedding model '{settings.embedding_model}' produces dim={dim}. "
                    "Recreate the collection (delete it via Qdrant API or wipe the volume)."
                )
            return
        await async_timeout_wrapper(
            c.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
            ),
            timeout=5.0,
            service_name="Qdrant",
        )
    except Exception as e:
        logger.error(f"Failed to ensure Qdrant collection: {type(e).__name__}: {e}")
        _qdrant_breaker.record_failure()
        raise


# A single document's superseded revisions, so one page always covers them.
_STALE_SCROLL_LIMIT = 1000


@dataclass(frozen=True)
class StaleRevisions:
    """What a stale-revision cleanup removed."""

    points: int = 0
    doc_ids: frozenset[str] = frozenset()


def _point_id(chunk_id: str) -> int:
    return int(hashlib.sha256(chunk_id.encode()).hexdigest()[:15], 16)


async def upsert(chunks, vectors) -> None:
    if not chunks:
        return
    if not _qdrant_breaker.can_execute():
        logger.error("Qdrant circuit breaker is open; skipping upsert")
        raise RuntimeError("Qdrant service is unavailable (circuit breaker open)")
    try:
        points = [
            qm.PointStruct(
                id=_point_id(chunk.id),
                vector=vec,
                payload={
                    "chunk_id": chunk.id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    **chunk.metadata,
                },
            )
            for chunk, vec in zip(chunks, vectors, strict=True)
        ]
        c = await client()
        await async_timeout_wrapper(
            c.upsert(collection_name=settings.qdrant_collection, points=points),
            timeout=5.0,
            service_name="Qdrant",
        )
        _qdrant_breaker.record_success()
    except Exception as e:
        logger.error(f"Qdrant upsert failed: {type(e).__name__}: {e}")
        _qdrant_breaker.record_failure()
        raise


async def search(vector, top_k: int = 8):
    if not _qdrant_breaker.can_execute():
        logger.warning("Qdrant circuit breaker is open; returning empty context")
        return []  # Graceful degradation: empty results instead of crashing
    try:
        c = await client()
        result = await async_timeout_wrapper(
            c.query_points(
                collection_name=settings.qdrant_collection,
                query=vector,
                limit=top_k,
            ),
            timeout=5.0,
            service_name="Qdrant",
        )
        _qdrant_breaker.record_success()
        return result.points
    except Exception as e:
        logger.error(f"Qdrant search failed: {type(e).__name__}: {e}")
        _qdrant_breaker.record_failure()
        logger.warning("Returning empty context due to Qdrant failure")
        return []  # Graceful degradation


async def count() -> int:
    if not _qdrant_breaker.can_execute():
        logger.warning("Qdrant circuit breaker is open; returning 0 documents")
        return 0  # Graceful degradation
    try:
        c = await client()
        result = await async_timeout_wrapper(
            c.count(collection_name=settings.qdrant_collection, exact=True),
            timeout=5.0,
            service_name="Qdrant",
        )
        _qdrant_breaker.record_success()
        return result.count
    except Exception as e:
        logger.error(f"Qdrant count failed: {type(e).__name__}: {e}")
        _qdrant_breaker.record_failure()
        logger.warning("Returning 0 documents due to Qdrant failure")
        return 0  # Graceful degradation


async def delete_stale_revisions(filename: str, keep_doc_id: str) -> StaleRevisions:
    """Remove chunks left behind by earlier revisions of a document.

    Chunk point ids are derived from ``doc_id``, which is content-derived, so
    re-indexing an edited file writes a *new* set of points and the previous
    revision's points survive untouched. Left alone they accumulate: the index
    ends up holding several near-identical copies of the same document, which
    then crowd each other out of the reranker's top-k and starve genuinely
    relevant documents of a slot.

    Args:
        filename: The document's filename, as recorded in chunk metadata.
        keep_doc_id: The revision being indexed now; its points are preserved.

    Returns:
        The points removed and the document revisions they belonged to. Callers
        need the ids to clear anything keyed off them elsewhere, such as the
        knowledge graph. Empty if the cleanup could not run.
    """
    if not _qdrant_breaker.can_execute():
        logger.error("Qdrant circuit breaker is open; skipping stale-revision cleanup")
        return StaleRevisions()

    selector = qm.Filter(
        must=[qm.FieldCondition(key="filename", match=qm.MatchValue(value=filename))],
        must_not=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=keep_doc_id))],
    )
    try:
        c = await client()
        before = await async_timeout_wrapper(
            c.count(
                collection_name=settings.qdrant_collection,
                count_filter=selector,
                exact=True,
            ),
            timeout=5.0,
            service_name="Qdrant",
        )
        if not before.count:
            _qdrant_breaker.record_success()
            return StaleRevisions()

        # Read the ids before deleting; afterwards there is nothing left to ask.
        # One page is plenty: this matches a single document's own revisions.
        points, _ = await async_timeout_wrapper(
            c.scroll(
                collection_name=settings.qdrant_collection,
                scroll_filter=selector,
                limit=_STALE_SCROLL_LIMIT,
                with_payload=["doc_id"],
                with_vectors=False,
            ),
            timeout=5.0,
            service_name="Qdrant",
        )
        doc_ids = {
            p.payload["doc_id"]
            for p in points
            if p.payload and p.payload.get("doc_id")
        }

        await async_timeout_wrapper(
            c.delete(
                collection_name=settings.qdrant_collection,
                points_selector=qm.FilterSelector(filter=selector),
            ),
            timeout=5.0,
            service_name="Qdrant",
        )
        _qdrant_breaker.record_success()
        return StaleRevisions(points=before.count, doc_ids=frozenset(doc_ids))
    except Exception as e:
        # A failed cleanup leaves duplicates behind but the new revision is
        # already indexed, so the answer path still works. Don't fail ingest.
        logger.error(f"Qdrant stale-revision cleanup failed: {type(e).__name__}: {e}")
        _qdrant_breaker.record_failure()
        return StaleRevisions()
