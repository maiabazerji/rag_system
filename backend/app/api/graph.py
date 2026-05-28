"""HTTP routes for the Graph RAG knowledge graph.

POST /graph/build   — extract triples from already-ingested chunks (idempotent
                      per (chunk_id, triple); call after /ingest).
GET  /graph/stats   — counts of triples, entities, indexed chunks.
GET  /graph/entities?q=foo — search entities by substring.
POST /graph/reset   — wipe the graph (does not touch Qdrant).
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter

from app.rag import graph_store
from app.rag.embed import embed_query
from app.rag.graph_extract import extract_triples
from app.rag.store import search as vector_search

router = APIRouter()


@router.get("/stats")
async def stats() -> dict:
    return graph_store.stats()


@router.get("/entities")
async def entities(q: str | None = None, limit: int = 50) -> dict:
    idx = graph_store.load()
    ents = idx.entities
    if q:
        ql = q.lower()
        ents = [e for e in ents if ql in e]
    return {"entities": ents[:limit], "total": len(ents)}


@router.post("/reset")
async def reset() -> dict:
    graph_store.reset()
    return graph_store.stats()


@router.post("/build")
async def build(limit: int | None = None, concurrency: int = 4) -> dict:
    """Extract triples from every indexed chunk. Re-running is safe but appends —
    call /graph/reset first if you want a clean rebuild."""
    # Pull all chunks back from Qdrant via a wide probe.
    probe = embed_query(" ")
    hits = await vector_search(probe, top_k=limit or 2048)
    chunks = [
        {
            "chunk_id": h.payload["chunk_id"],
            "doc_id": h.payload["doc_id"],
            "text": h.payload["text"],
        }
        for h in hits
        if h.payload.get("text")
    ]

    already_done = {t.chunk_id for t in graph_store.load().triples}
    pending = [c for c in chunks if c["chunk_id"] not in already_done]

    sem = asyncio.Semaphore(concurrency)
    total_added = 0
    total_skipped = 0
    failures: list[str] = []

    async def _work(c: dict) -> None:
        nonlocal total_added, total_skipped
        async with sem:
            try:
                triples = await extract_triples(
                    chunk_id=c["chunk_id"], doc_id=c["doc_id"], text=c["text"]
                )
                result = graph_store.append(triples)
                total_added += result["added"]
                total_skipped += result["skipped"]
            except Exception as e:
                failures.append(f"{c['chunk_id']}: {type(e).__name__}: {e}")

    await asyncio.gather(*[_work(c) for c in pending])

    return {
        "processed_chunks": len(pending),
        "skipped_chunks": len(chunks) - len(pending),
        "new_triples": total_added,
        "duplicate_triples": total_skipped,
        "failures": failures[:10],
        **graph_store.stats(),
    }
