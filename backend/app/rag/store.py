from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Iterable
from dataclasses import dataclass

try:
    from qdrant_client.async_client import AsyncQdrantClient
except ImportError:
    from qdrant_client import AsyncQdrantClient

from fastapi import Request
from fastapi.responses import JSONResponse
from qdrant_client.http import models as qm

from app.access import PUBLIC_GROUP, AccessScope
from app.config import settings
from app.middleware.request_id import get_request_id
from app.rag.embed import embedding_dim
from app.resilience import CircuitBreaker, async_timeout_wrapper

logger = logging.getLogger(__name__)

_client: AsyncQdrantClient | None = None
_qdrant_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=30.0,
    service_name="Qdrant",
)


class StoreUnavailable(RuntimeError):
    """The vector store could not be reached, or its circuit breaker is open.

    Raised instead of returning an empty result, so callers cannot mistake an
    outage for an empty index (and, say, prune everything they think is gone).
    """


async def store_unavailable_handler(request: Request, exc: Exception) -> JSONResponse:
    """FastAPI exception handler turning `StoreUnavailable` into a 503."""
    logger.warning(f"Vector store unavailable on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=503,
        content={
            "code": "store_unavailable",
            "message": "Vector store unavailable.",
            "detail": str(exc),
            "request_id": get_request_id(),
        },
    )


async def client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        _client = AsyncQdrantClient(
            url=settings.qdrant_url, api_key=settings.qdrant_api_key or None
        )
        await _ensure_collection(_client)
    return _client


async def close() -> None:
    """Close the shared Qdrant client, if one was opened."""
    global _client
    c, _client = _client, None
    if c is not None:
        await c.close()


async def _ensure_collection(c: AsyncQdrantClient) -> None:
    try:
        # Loading the embedding model is slow, blocking work.
        dim = await asyncio.to_thread(embedding_dim)
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
        else:
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
    await _ensure_payload_indexes(c)


# Payload fields every access-controlled query filters on.
_ACL_INDEXED_FIELDS = ("tenant", "acl_groups")


async def _ensure_payload_indexes(c: AsyncQdrantClient) -> None:
    """Index the access-control fields, so filtered search stays fast.

    Creating an index that exists is a no-op in Qdrant, so this runs on every
    start and also upgrades collections created before ACLs existed. A
    failure is logged, not raised: filtering still works unindexed, only slower.
    """
    for field_name in _ACL_INDEXED_FIELDS:
        try:
            await async_timeout_wrapper(
                c.create_payload_index(
                    collection_name=settings.qdrant_collection,
                    field_name=field_name,
                    field_schema=qm.PayloadSchemaType.KEYWORD,
                ),
                timeout=10.0,
                service_name="Qdrant",
            )
        except Exception as e:
            logger.warning(
                f"Could not create payload index on '{field_name}': {type(e).__name__}: {e}"
            )


def access_filter(access: AccessScope) -> qm.Filter:
    """The Qdrant filter matching exactly the chunks ``access`` may read.

    Mirrors :meth:`AccessScope.permits`: chunks without a ``tenant`` belong to
    the default tenant, and chunks without ``acl_groups`` are public while
    ``ACL_LEGACY_PUBLIC`` is on.
    """
    tenant_match: list[qm.Condition] = [
        qm.FieldCondition(key="tenant", match=qm.MatchValue(value=access.tenant))
    ]
    if access.tenant == settings.default_tenant:
        tenant_match.append(qm.IsEmptyCondition(is_empty=qm.PayloadField(key="tenant")))

    group_match: list[qm.Condition] = [
        qm.FieldCondition(key="acl_groups", match=qm.MatchAny(any=sorted(access.groups)))
    ]
    if settings.acl_legacy_public and PUBLIC_GROUP in access.groups:
        group_match.append(qm.IsEmptyCondition(is_empty=qm.PayloadField(key="acl_groups")))

    return qm.Filter(must=[qm.Filter(should=tenant_match), qm.Filter(should=group_match)])


def _permitted(points: list, access: AccessScope | None) -> list:
    """Re-check each point against ``access``: the filter's second line of defence."""
    if access is None:
        return points
    return [p for p in points if access.permits(p.payload)]


# A single document's superseded revisions, so one page always covers them.
_STALE_SCROLL_LIMIT = 1000
# Points per upsert request, and the timeout each request gets.
_UPSERT_BATCH = 128
_UPSERT_TIMEOUT = 10.0
# Points per page when enumerating the whole collection.
_SCROLL_PAGE = 256


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
        raise StoreUnavailable("Qdrant service is unavailable (circuit breaker open)")
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
        # Batched so a large document is not one request racing one timeout.
        for start in range(0, len(points), _UPSERT_BATCH):
            await async_timeout_wrapper(
                c.upsert(
                    collection_name=settings.qdrant_collection,
                    points=points[start : start + _UPSERT_BATCH],
                ),
                timeout=_UPSERT_TIMEOUT,
                service_name="Qdrant",
            )
        _qdrant_breaker.record_success()
    except Exception as e:
        logger.error(f"Qdrant upsert failed: {type(e).__name__}: {e}")
        _qdrant_breaker.record_failure()
        raise StoreUnavailable(f"Qdrant upsert failed: {type(e).__name__}: {e}") from e


async def search(vector, top_k: int = 8, access: AccessScope | None = None):
    """Nearest chunks to ``vector``.

    Args:
        vector: The query embedding.
        top_k: How many points to return.
        access: Restrict to chunks this scope may read. ``None`` is
            unrestricted and is for internal jobs only; request paths always
            pass a scope.
    """
    if not _qdrant_breaker.can_execute():
        raise StoreUnavailable("Qdrant service is unavailable (circuit breaker open)")
    try:
        c = await client()
        result = await async_timeout_wrapper(
            c.query_points(
                collection_name=settings.qdrant_collection,
                query=vector,
                limit=top_k,
                query_filter=access_filter(access) if access else None,
            ),
            timeout=5.0,
            service_name="Qdrant",
        )
        _qdrant_breaker.record_success()
        return _permitted(result.points, access)
    except Exception as e:
        logger.error(f"Qdrant search failed: {type(e).__name__}: {e}")
        _qdrant_breaker.record_failure()
        raise StoreUnavailable(f"Qdrant search failed: {type(e).__name__}: {e}") from e


async def count(access: AccessScope | None = None) -> int:
    """Number of indexed chunks, or of those ``access`` may read."""
    if not _qdrant_breaker.can_execute():
        raise StoreUnavailable("Qdrant service is unavailable (circuit breaker open)")
    try:
        c = await client()
        result = await async_timeout_wrapper(
            c.count(
                collection_name=settings.qdrant_collection,
                count_filter=access_filter(access) if access else None,
                exact=True,
            ),
            timeout=5.0,
            service_name="Qdrant",
        )
        _qdrant_breaker.record_success()
        return result.count
    except Exception as e:
        logger.error(f"Qdrant count failed: {type(e).__name__}: {e}")
        _qdrant_breaker.record_failure()
        raise StoreUnavailable(f"Qdrant count failed: {type(e).__name__}: {e}") from e


async def fetch_chunks(chunk_ids: Iterable[str], access: AccessScope | None = None) -> list:
    """Look chunks up by id.

    Args:
        chunk_ids: Chunk ids, as recorded in the ``chunk_id`` payload field.
        access: Restrict to chunks this scope may read; the others are
            skipped exactly like unknown ids. ``None`` is unrestricted.

    Returns:
        The stored points (with payload) for the ids that exist and are
        readable; unknown ids are skipped.

    Raises:
        StoreUnavailable: If Qdrant is unreachable or its breaker is open.
    """
    ids = list(dict.fromkeys(chunk_ids))
    if not ids:
        return []
    if not _qdrant_breaker.can_execute():
        raise StoreUnavailable("Qdrant service is unavailable (circuit breaker open)")
    point_ids: list[qm.ExtendedPointId] = [_point_id(cid) for cid in ids]
    try:
        c = await client()
        if access is None:
            points = await async_timeout_wrapper(
                c.retrieve(
                    collection_name=settings.qdrant_collection,
                    ids=point_ids,
                    with_payload=True,
                    with_vectors=False,
                ),
                timeout=5.0,
                service_name="Qdrant",
            )
        else:
            # `retrieve` takes no filter, so look the ids up with a filtered scroll.
            scoped = access_filter(access)
            points, _ = await async_timeout_wrapper(
                c.scroll(
                    collection_name=settings.qdrant_collection,
                    scroll_filter=qm.Filter(
                        must=[qm.HasIdCondition(has_id=point_ids), *(scoped.must or [])]
                    ),
                    limit=len(point_ids),
                    with_payload=True,
                    with_vectors=False,
                ),
                timeout=5.0,
                service_name="Qdrant",
            )
        _qdrant_breaker.record_success()
        return _permitted(list(points), access)
    except Exception as e:
        logger.error(f"Qdrant retrieve failed: {type(e).__name__}: {e}")
        _qdrant_breaker.record_failure()
        raise StoreUnavailable(f"Qdrant retrieve failed: {type(e).__name__}: {e}") from e


async def scroll_chunks(
    limit: int | None = None,
    access: AccessScope | None = None,
    with_payload: bool | list[str] = True,
) -> list:
    """Enumerate stored chunks, paging through the whole collection.

    Unlike a similarity search this sees every point, however many there are,
    so its result can be trusted as the complete set of live chunks.

    Args:
        limit: Stop after this many points. ``None`` reads the whole collection.
        access: Enumerate only the chunks this scope may read.
        with_payload: Payload to return: all of it, or only the named fields.

    Returns:
        The stored points, with payload.

    Raises:
        StoreUnavailable: If any page cannot be read. A partial enumeration is
            never returned, since callers use it to decide what to delete.
    """
    if not _qdrant_breaker.can_execute():
        raise StoreUnavailable("Qdrant service is unavailable (circuit breaker open)")
    records: list = []
    offset = None
    try:
        c = await client()
        while limit is None or len(records) < limit:
            page = _SCROLL_PAGE if limit is None else min(_SCROLL_PAGE, limit - len(records))
            points, offset = await async_timeout_wrapper(
                c.scroll(
                    collection_name=settings.qdrant_collection,
                    limit=page,
                    offset=offset,
                    with_payload=with_payload,
                    with_vectors=False,
                    scroll_filter=access_filter(access) if access else None,
                ),
                timeout=10.0,
                service_name="Qdrant",
            )
            records.extend(_permitted(points, access))
            if offset is None:
                break
        _qdrant_breaker.record_success()
        return records
    except Exception as e:
        logger.error(f"Qdrant scroll failed: {type(e).__name__}: {e}")
        _qdrant_breaker.record_failure()
        raise StoreUnavailable(f"Qdrant scroll failed: {type(e).__name__}: {e}") from e


async def readable_doc_ids(access: AccessScope) -> set[str]:
    """Ids of every document ``access`` may read.

    Used to prune derived data that lives outside Qdrant (the knowledge graph)
    down to what the caller is allowed to see.

    Raises:
        StoreUnavailable: If Qdrant cannot be enumerated.
    """
    records = await scroll_chunks(
        access=access, with_payload=["doc_id", "tenant", "acl_groups"]
    )
    return {r.payload["doc_id"] for r in records if r.payload and r.payload.get("doc_id")}


async def delete_stale_revisions(source_key: str, keep_doc_id: str) -> StaleRevisions:
    """Remove chunks left behind by earlier revisions of a document.

    Chunk point ids are derived from ``doc_id``, which is content-derived, so
    re-indexing an edited file writes a *new* set of points and the previous
    revision's points survive untouched. Left alone they accumulate: the index
    ends up holding several near-identical copies of the same document, which
    then crowd each other out of the reranker's top-k and starve genuinely
    relevant documents of a slot.

    Args:
        source_key: The document's owner-scoped source key, as recorded in
            chunk metadata. Matching on it rather than the bare filename keeps
            one user's upload from deleting another user's (or another
            folder's) document of the same name. Chunks indexed before the
            key existed carry none, so they are never matched.
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
        must=[qm.FieldCondition(key="source_key", match=qm.MatchValue(value=source_key))],
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
