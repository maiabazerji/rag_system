from __future__ import annotations

from app.rag.embed import embed_query
from app.rag.store import search
from app.schemas import Chunk

_RESERVED = {"chunk_id", "doc_id", "text"}


async def hybrid_search(query: str, top_k: int = 50) -> list[Chunk]:
    vec = embed_query(query)
    hits = await search(vec, top_k=top_k)
    chunks = []
    for h in hits:
        chunk_id = h.payload.get("chunk_id")
        doc_id = h.payload.get("doc_id")
        text = h.payload.get("text")
        if chunk_id and doc_id and text:
            chunks.append(Chunk(
                id=chunk_id,
                doc_id=doc_id,
                text=text,
                tokens=len(text.split()),
                metadata={k: v for k, v in h.payload.items() if k not in _RESERVED},
            ))
    return chunks
