"""HTTP routes for the Graph RAG knowledge graph.

POST /graph/build  - extract triples from already-ingested chunks (idempotent
                      per (chunk_id, triple); call after /ingest).
GET  /graph/stats  - counts of triples, entities, indexed chunks.
GET  /graph/entities?q=foo- search entities by substring.
POST /graph/reset  - wipe the graph (does not touch Qdrant).
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.rag import graph_store
from app.rag.embed import embed_query
from app.rag.graph_extract import extract_triples
from app.rag.store import search as vector_search

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/stats")
async def stats() -> dict:
    try:
        return graph_store.stats()
    except Exception as e:
        logger.exception(f"Graph stats error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to load graph stats."},
        )


@router.get("/entities")
async def entities(q: str | None = None, limit: int = 50) -> dict:
    try:
        if limit < 1 or limit > 500:
            return JSONResponse(
                status_code=400,
                content={"error": "Limit must be between 1 and 500"},
            )

        idx = graph_store.load()
        ents = idx.entities
        if q:
            ql = q.lower().strip()
            ents = [e for e in ents if ql in e.lower()]
        return {"entities": ents[:limit], "total": len(ents)}
    except Exception as e:
        logger.exception(f"Graph entities error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to load entities."},
        )


@router.post("/reset")
async def reset() -> dict:
    try:
        graph_store.reset()
        return graph_store.stats()
    except Exception as e:
        logger.exception(f"Graph reset error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to reset graph."},
        )


@router.post("/build")
async def build(limit: int | None = None, concurrency: int = 4) -> dict:
    try:
        probe = embed_query(" ")
        hits = await vector_search(probe, top_k=limit or 2048)
        chunks = []
        for h in hits:
            chunk_id = h.payload.get("chunk_id")
            doc_id = h.payload.get("doc_id")
            text = h.payload.get("text")
            if chunk_id and doc_id and text:
                chunks.append({"chunk_id": chunk_id, "doc_id": doc_id, "text": text})

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
                    logger.warning(f"Triple extraction failed for {c['chunk_id']}: {e}")
                    failures.append(f"{c['chunk_id']}: {type(e).__name__}")

        await asyncio.gather(*[_work(c) for c in pending], return_exceptions=True)

        return {
            "processed_chunks": len(pending),
            "skipped_chunks": len(chunks) - len(pending),
            "new_triples": total_added,
            "duplicate_triples": total_skipped,
            "failures": failures,
            **graph_store.stats(),
        }
    except Exception as e:
        logger.exception(f"Graph build error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to build knowledge graph. Please try again."},
        )
