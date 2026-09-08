"""HTTP routes for the Graph RAG knowledge graph.

POST /graph/build       - start extracting triples from indexed chunks (async job).
GET  /graph/build/{id}  - poll the status of a build job.
GET  /graph/stats       - counts of triples, entities, indexed chunks.
GET  /graph/entities    - search entities by substring.
POST /graph/reset       - wipe the graph (does not touch Qdrant).
"""
from __future__ import annotations

import asyncio
import uuid
from collections import OrderedDict
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from app.auth import require_api_key
from app.logging_config import get_structured_logger
from app.rag import graph_store
from app.rag.embed import embed_query_async
from app.rag.graph_extract import extract_triples
from app.rag.store import search as vector_search

logger = get_structured_logger(__name__)
router = APIRouter(dependencies=[Depends(require_api_key)])

# Build jobs are in-process and bounded; the graph itself is the durable artifact.
_JOBS: OrderedDict[str, dict[str, Any]] = OrderedDict()
_MAX_JOBS = 50
_build_lock = asyncio.Lock()


def _record_job(job_id: str, **fields: Any) -> None:
    """Create or update a build job record, evicting the oldest when full."""
    job = _JOBS.setdefault(job_id, {"id": job_id})
    job.update(fields)
    _JOBS.move_to_end(job_id)
    while len(_JOBS) > _MAX_JOBS:
        _JOBS.popitem(last=False)


@router.get("/stats", summary="Knowledge graph statistics")
async def stats() -> dict:
    """Return triple, entity and chunk counts for the current graph."""
    return graph_store.stats()


@router.get("/entities", summary="Search graph entities")
async def entities(
    q: str | None = Query(default=None, max_length=200, description="Substring filter"),
    limit: int = Query(default=50, ge=1, le=500, description="Max entities to return"),
) -> dict:
    """List graph entities, optionally filtered by substring.

    Args:
        q: Case-insensitive substring to match.
        limit: Maximum number of entities to return.

    Returns:
        The matching entities (truncated to `limit`) and the total match count.
    """
    ents = graph_store.load().entities
    if q:
        needle = q.lower().strip()
        ents = [e for e in ents if needle in e.lower()]
    return {"entities": ents[:limit], "total": len(ents)}


@router.post("/reset", summary="Wipe the knowledge graph")
async def reset() -> dict:
    """Delete every extracted triple. Does not touch the vector store."""
    graph_store.reset()
    logger.info("Knowledge graph reset")
    return graph_store.stats()


async def _run_build(job_id: str, limit: int | None, concurrency: int) -> None:
    """Extract triples for every not-yet-processed chunk.

    Runs as a background task. Progress and the final summary are written to
    the job record; failures are collected per chunk rather than aborting.
    """
    if _build_lock.locked():
        _record_job(
            job_id,
            status="failed",
            error="Another graph build is already running.",
            finished_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        return

    async with _build_lock:
        try:
            probe = await embed_query_async(" ")
            hits = await vector_search(probe, top_k=limit or 2048)

            chunks = [
                {
                    "chunk_id": h.payload.get("chunk_id"),
                    "doc_id": h.payload.get("doc_id"),
                    "text": h.payload.get("text"),
                }
                for h in hits
            ]
            chunks = [c for c in chunks if c["chunk_id"] and c["doc_id"] and c["text"]]

            # A full pass sees every live chunk, so anything the graph still
            # holds for a document the vector store no longer has is a stranded
            # revision. A limited pass sees only part of the corpus and would
            # mistake the rest for orphans, so it prunes nothing.
            pruned_triples = 0
            if limit is None:
                live_docs = {c["doc_id"] for c in chunks}
                orphaned = {
                    t.doc_id for t in graph_store.load().triples
                } - live_docs
                pruned_triples = graph_store.remove_docs(orphaned)

            already_done = {t.chunk_id for t in graph_store.load().triples}
            pending = [c for c in chunks if c["chunk_id"] not in already_done]

            _record_job(
                job_id,
                status="running",
                total=len(pending),
                completed=0,
                skipped_chunks=len(chunks) - len(pending),
                pruned_triples=pruned_triples,
            )

            sem = asyncio.Semaphore(concurrency)
            added = skipped = completed = 0
            failures: list[str] = []

            async def _work(c: dict) -> None:
                nonlocal added, skipped, completed
                async with sem:
                    try:
                        triples = await extract_triples(
                            chunk_id=c["chunk_id"], doc_id=c["doc_id"], text=c["text"]
                        )
                        result = graph_store.append(triples)
                        added += result["added"]
                        skipped += result["skipped"]
                    except Exception as e:
                        logger.warning(
                            f"Triple extraction failed for {c['chunk_id']}: "
                            f"{type(e).__name__}: {e}",
                            extra_fields={
                                "error_type": type(e).__name__,
                                "chunk_id": c["chunk_id"],
                            },
                        )
                        failures.append(f"{c['chunk_id']}: {type(e).__name__}")
                    finally:
                        completed += 1
                        _record_job(job_id, completed=completed)

            await asyncio.gather(*(_work(c) for c in pending))

            _record_job(
                job_id,
                status="completed",
                processed_chunks=len(pending),
                new_triples=added,
                duplicate_triples=skipped,
                pruned_triples=pruned_triples,
                failures=failures,
                finished_at=datetime.now(UTC).isoformat(timespec="seconds"),
                **graph_store.stats(),
            )
            logger.info(
                "Graph build finished",
                extra_fields={
                    "job_id": job_id,
                    "processed_chunks": len(pending),
                    "new_triples": added,
                    "failures": len(failures),
                },
            )
        except Exception as e:
            logger.exception(
                f"Graph build failed: {type(e).__name__}: {e}",
                extra_fields={"error_type": type(e).__name__, "job_id": job_id},
            )
            _record_job(
                job_id,
                status="failed",
                error=f"{type(e).__name__}: {e}",
                finished_at=datetime.now(UTC).isoformat(timespec="seconds"),
            )


@router.post(
    "/build",
    status_code=202,
    summary="Start a knowledge graph build",
    description=(
        "Starts triple extraction over indexed chunks and returns immediately. "
        "Extraction issues one model call per chunk, so it can run for minutes; "
        "poll GET /graph/build/{job_id} for progress."
    ),
)
async def build(
    background: BackgroundTasks,
    limit: int | None = Query(default=None, ge=1, le=10000, description="Max chunks"),
    concurrency: int = Query(default=4, ge=1, le=16, description="Parallel extractions"),
) -> dict:
    """Queue a graph build and return its job id.

    Args:
        background: FastAPI background task runner.
        limit: Maximum number of chunks to consider.
        concurrency: How many extractions to run at once.

    Returns:
        The job id and a URL to poll for status.

    Raises:
        HTTPException: 409 if a build is already running.
    """
    if _build_lock.locked():
        raise HTTPException(
            status_code=409, detail="A graph build is already running."
        )

    job_id = uuid.uuid4().hex[:12]
    _record_job(
        job_id,
        status="queued",
        started_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    background.add_task(_run_build, job_id, limit, concurrency)
    return {"job_id": job_id, "status": "queued", "poll": f"/graph/build/{job_id}"}


@router.get("/build/{job_id}", summary="Check a graph build job")
async def build_status(job_id: str) -> dict:
    """Return the current state of a build job.

    Args:
        job_id: Id returned by POST /graph/build.

    Returns:
        The job record, including status, progress and final counts.

    Raises:
        HTTPException: 404 if the job is unknown or has been evicted.
    """
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No build job '{job_id}'")
    return job
