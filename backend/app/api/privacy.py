"""GDPR endpoints: erasure, subject access export and retention.

Every route requires the ``X-Admin-Key`` header: these act on other people's
data, so they are operator tools, not self-service. See docs/gdpr/README.md.

DELETE /privacy/documents/{doc_id}       - erase one document everywhere.
DELETE /privacy/sources?source_key=...   - erase every revision of a source.
DELETE /privacy/principals/{principal}   - erase a principal's documents and records.
GET    /privacy/export/{principal}       - export what is held about a principal.
POST   /privacy/retention/run            - purge expired records now.

Erasures are idempotent. A response of 200 means every target succeeded. If
any target failed the response is 503 with the partial report as ``detail``;
repeating the request finishes the job.
"""
from __future__ import annotations

from collections.abc import Awaitable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from app.auth import require_admin_key
from app.logging_config import get_structured_logger
from app.privacy import erasure
from app.privacy.export import export_principal
from app.privacy.retention import run_retention

logger = get_structured_logger(__name__)
router = APIRouter(dependencies=[Depends(require_admin_key)])


async def _run(resolving: Awaitable[erasure.ErasureRequest]) -> dict[str, Any]:
    """Resolve the subject, erase it, and turn the report into a response."""
    try:
        request = await resolving
    except Exception as e:
        logger.error(
            f"Erasure could not resolve its subject: {type(e).__name__}: {e}",
            extra_fields={"error_type": type(e).__name__},
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "The vector store is unavailable, so the documents to erase could "
                "not be determined. Nothing was erased; retry once it is back."
            ),
        ) from e

    report = await erasure.erase(request)
    if not report.complete:
        raise HTTPException(status_code=503, detail=report.as_dict())
    return report.as_dict()


@router.delete("/documents/{doc_id}", summary="Erase a document everywhere")
async def erase_document(
    doc_id: str = Path(..., min_length=1, max_length=256),
) -> dict[str, Any]:
    """Remove a document's chunks, graph triples, traces that used it, and its
    text in saved eval runs.

    Returns:
        ``{subject, doc_ids, targets: {target: count}, errors, complete}``.
    """
    return await _run(erasure.resolve_documents([doc_id]))


@router.delete("/sources", summary="Erase every document from a source")
async def erase_source(
    source_key: str = Query(..., min_length=1, max_length=256),
) -> dict[str, Any]:
    """Erase every document ingested from ``source_key`` (matched against the
    chunk's source key, or its filename for chunks indexed without one)."""
    return await _run(erasure.resolve_source(source_key))


@router.delete("/principals/{principal_id}", summary="Erase a principal")
async def erase_principal(
    principal_id: str = Path(..., min_length=1, max_length=256),
) -> dict[str, Any]:
    """Erase every document the principal owns, plus their traces and usage rows."""
    return await _run(erasure.resolve_principal(principal_id))


@router.get("/export/{principal_id}", summary="Subject access export")
async def export(
    principal_id: str = Path(..., min_length=1, max_length=256),
) -> dict[str, Any]:
    """Return what is held about a principal: owned documents (ids, filenames,
    creation time, chunk counts), API usage totals and their request traces.

    ``complete`` is false, and ``errors`` names the sections, when a store
    could not be read.
    """
    return await export_principal(principal_id)


@router.post("/retention/run", summary="Purge expired records now")
async def retention_run() -> dict[str, Any]:
    """Run the retention purge immediately instead of waiting for the daily run.

    Returns:
        ``{ran_at, targets: {target: deleted}, cutoffs, skipped, errors}``.
    """
    return (await run_retention()).as_dict()
