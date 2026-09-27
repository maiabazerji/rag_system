from typing import Any

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from app.auth import require_api_key
from app.config import settings
from app.logging_config import get_structured_logger
from app.privacy.pii import PIIRejected
from app.rag import parsers
from app.rag.ingest import enqueue_document
from app.rag.store import count as store_count

logger = get_structured_logger(__name__)
router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post(
    "",
    summary="Ingest a document",
    description=(
        "Chunks, embeds and indexes an uploaded document. "
        f"Accepts {', '.join(sorted(parsers.SUPPORTED_EXTENSIONS))}."
    ),
)
async def ingest_document(
    file: UploadFile, auth: dict[str, Any] = Depends(require_api_key)
) -> dict:
    """Index an uploaded document.

    Args:
        file: The uploaded file. Must have a supported extension and be within
            the MAX_UPLOAD_MB limit.
        auth: The caller (resolved once per request with the router's own
            dependency). Scopes which earlier revisions a re-upload replaces.

    Returns:
        The document id, filename, number of chunks written, and the new total
        number of indexed chunks.

    Raises:
        HTTPException: 400 for a missing name, empty file or unsupported type;
            413 if the file exceeds MAX_UPLOAD_MB; 422 if it is encrypted,
            unreadable, a likely zip bomb, holds no text, or holds personal
            data while PII_MODE_INGEST is 'reject'.
            503 (via the StoreUnavailable handler) if Qdrant is unreachable.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a name.")

    if not parsers.is_supported(file.filename, file.content_type):
        raise HTTPException(
            status_code=400, detail=parsers.unsupported_message(file.filename)
        )

    limit = settings.max_upload_bytes
    if file.size is not None and file.size > limit:
        raise HTTPException(
            status_code=413,
            detail=f"File is larger than the {settings.max_upload_mb} MB limit.",
        )

    content = await file.read(limit + 1)
    if len(content) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"File is larger than the {settings.max_upload_mb} MB limit.",
        )
    if not content:
        raise HTTPException(status_code=400, detail="File is empty.")

    try:
        owner = f"key:{auth['id']}" if auth.get("id") is not None else "local"
        result = await enqueue_document(filename=file.filename, content=content, owner=owner)
    except PIIRejected as e:
        raise HTTPException(
            status_code=422,
            detail={"code": "pii_rejected", "message": str(e), "types": e.types},
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    total = await store_count()
    logger.info(
        "Document ingested",
        extra_fields={
            "filename": file.filename,
            "chunks": result["chunks"],
            "indexed_total": total,
        },
    )
    return {**result, "indexed_total": total}


@router.get("/stats", summary="Index statistics")
async def stats() -> dict:
    """Return the number of chunks currently indexed."""
    return {"indexed_chunks": await store_count()}


@router.get("/formats", summary="Supported document formats")
async def formats() -> dict:
    """List the ingestible formats and whether scanned PDFs can be OCR'd.

    The extensions and MIME types come from the parser registry, the same
    list the upload route validates against.
    """
    return {
        "extensions": sorted(parsers.SUPPORTED_EXTENSIONS),
        "formats": parsers.supported_formats(),
        "max_upload_mb": settings.max_upload_mb,
        "ocr": parsers.ocr_status(),
    }
