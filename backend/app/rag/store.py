from __future__ import annotations

import hashlib
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.config import settings
from app.rag.embed import embedding_dim


@lru_cache(maxsize=1)
def client() -> QdrantClient:
    c = QdrantClient(url=settings.qdrant_url)
    _ensure_collection(c)
    return c


def _ensure_collection(c: QdrantClient) -> None:
    dim = embedding_dim()
    existing = {col.name for col in c.get_collections().collections}
    if settings.qdrant_collection in existing:
        info = c.get_collection(settings.qdrant_collection)
        current = info.config.params.vectors.size
        if current != dim:
            raise RuntimeError(
                f"Qdrant collection '{settings.qdrant_collection}' has dim={current} "
                f"but embedding model '{settings.embedding_model}' produces dim={dim}. "
                "Recreate the collection (delete it via Qdrant API or wipe the volume)."
            )
        return
    c.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    )


def _point_id(chunk_id: str) -> int:
    return int(hashlib.sha256(chunk_id.encode()).hexdigest()[:15], 16)


def upsert(chunks, vectors) -> None:
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
    client().upsert(collection_name=settings.qdrant_collection, points=points)


def search(vector, top_k: int = 8):
    return client().query_points(
        collection_name=settings.qdrant_collection,
        query=vector,
        limit=top_k,
    ).points


def count() -> int:
    return client().count(collection_name=settings.qdrant_collection, exact=True).count
