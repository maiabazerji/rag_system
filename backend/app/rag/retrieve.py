from __future__ import annotations

from app.rag.embed import embed_query
from app.rag.store import search
from app.schemas import Chunk

_RESERVED = {"chunk_id", "doc_id", "text"}


def hybrid_search(query: str, top_k: int = 50) -> list[Chunk]:
    vec = embed_query(query)
    hits = search(vec, top_k=top_k)
    return [
        Chunk(
            id=h.payload["chunk_id"],
            doc_id=h.payload["doc_id"],
            text=h.payload["text"],
            tokens=len(h.payload["text"].split()),
            metadata={k: v for k, v in h.payload.items() if k not in _RESERVED},
        )
        for h in hits
    ]
