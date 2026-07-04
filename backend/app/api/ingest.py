import logging
from fastapi import APIRouter, UploadFile
from fastapi.responses import JSONResponse

from app.rag.ingest import enqueue_document
from app.rag.store import count as store_count

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("")
async def ingest_document(file: UploadFile) -> dict:
    try:
        if not file.filename:
            return JSONResponse(
                status_code=400,
                content={"error": "File must have a name"},
            )

        if file.size == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "File is empty"},
            )

        content = await file.read()
        result = await enqueue_document(filename=file.filename, content=content)
        total = await store_count()
        return {**result, "indexed_total": total}
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)},
        )
    except Exception as e:
        logger.exception(f"Ingest error for {file.filename}: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to ingest file. Check file format (PDF, TXT, or MD)."},
        )


@router.get("/stats")
async def stats() -> dict:
    indexed = await store_count()
    return {"indexed_chunks": indexed}
