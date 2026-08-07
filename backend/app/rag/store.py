from __future__ import annotations

import hashlib
import logging

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
    except (TimeoutError, Exception) as e:
        logger.error(f"Failed to ensure Qdrant collection: {type(e).__name__}: {e}")
        _qdrant_breaker.record_failure()
        raise


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
            for chunk, vec in zip(chunks, vectors)
        ]
        c = await client()
        await async_timeout_wrapper(
            c.upsert(collection_name=settings.qdrant_collection, points=points),
            timeout=5.0,
            service_name="Qdrant",
        )
        _qdrant_breaker.record_success()
    except (TimeoutError, Exception) as e:
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
    except (TimeoutError, Exception) as e:
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
    except (TimeoutError, Exception) as e:
        logger.error(f"Qdrant count failed: {type(e).__name__}: {e}")
        _qdrant_breaker.record_failure()
        logger.warning("Returning 0 documents due to Qdrant failure")
        return 0  # Graceful degradation
