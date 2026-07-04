from __future__ import annotations

import hashlib

try:
    from qdrant_client.async_client import AsyncQdrantClient
except ImportError:
    from qdrant_client import AsyncQdrantClient

from qdrant_client.http import models as qm

from app.config import settings
from app.rag.embed import embedding_dim

_client: AsyncQdrantClient | None = None


async def client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        _client = AsyncQdrantClient(url=settings.qdrant_url)
        await _ensure_collection(_client)
    return _client


async def _ensure_collection(c: AsyncQdrantClient) -> None:
    dim = embedding_dim()
    collections = await c.get_collections()
    existing = {col.name for col in collections.collections}
    if settings.qdrant_collection in existing:
        info = await c.get_collection(settings.qdrant_collection)
        current = info.config.params.vectors.size
        if current != dim:
            raise RuntimeError(
                f"Qdrant collection '{settings.qdrant_collection}' has dim={current} "
                f"but embedding model '{settings.embedding_model}' produces dim={dim}. "
                "Recreate the collection (delete it via Qdrant API or wipe the volume)."
            )
        return
    await c.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    )


def _point_id(chunk_id: str) -> int:
    return int(hashlib.sha256(chunk_id.encode()).hexdigest()[:15], 16)


async def upsert(chunks, vectors) -> None:
    if not chunks:
        return
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
    await c.upsert(collection_name=settings.qdrant_collection, points=points)


async def search(vector, top_k: int = 8):
    c = await client()
    return (await c.query_points(
        collection_name=settings.qdrant_collection,
        query=vector,
        limit=top_k,
    )).points


async def count() -> int:
    c = await client()
    return (await c.count(collection_name=settings.qdrant_collection, exact=True)).count
