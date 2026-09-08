from fastapi import APIRouter, Depends, HTTPException, UploadFile

from app.auth import require_api_key
from app.config import settings
from app.logging_config import get_structured_logger
from app.rag.ingest import SUPPORTED_SUFFIXES, enqueue_document
from app.rag.store import count as store_count

logger = get_structured_logger(__name__)
router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post(
    "",
    summary="Ingest a document",
    description=(
        "Chunks, embeds and indexes an uploaded document. "
        f"Accepts {', '.join(sorted(SUPPORTED_SUFFIXES))}."
    ),
)
async def ingest_document(file: UploadFile) -> dict:
    """Index an uploaded document.

    Args:
        file: The uploaded file. Must have a supported extension and be within
            the MAX_UPLOAD_MB limit.

    Returns:
        The document id, filename, number of chunks written, and the new total
        number of indexed chunks.

    Raises:
        HTTPException: 400 for a missing name, empty file or unsupported type;
            413 if the file exceeds MAX_UPLOAD_MB; 422 if it holds no text.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a name.")

    suffix = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{suffix or file.filename}'. "
                f"Supported types: {', '.join(sorted(SUPPORTED_SUFFIXES))}."
            ),
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
        result = await enqueue_document(filename=file.filename, content=content)
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
