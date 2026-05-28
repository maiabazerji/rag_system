from fastapi import APIRouter, UploadFile

from app.rag.ingest import enqueue_document
from app.rag.store import count as store_count

router = APIRouter()


@router.post("")
async def ingest_document(file: UploadFile) -> dict:
    content = await file.read()
    result = await enqueue_document(filename=file.filename or "upload", content=content)
    total = await store_count()
    return {**result, "indexed_total": total}


@router.get("/stats")
async def stats() -> dict:
    indexed = await store_count()
    return {"indexed_chunks": indexed}
