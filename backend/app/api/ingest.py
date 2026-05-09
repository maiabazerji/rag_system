from fastapi import APIRouter, UploadFile

from app.rag.ingest import enqueue_document
from app.rag.store import count as store_count

router = APIRouter()


@router.post("")
async def ingest_document(file: UploadFile) -> dict:
    content = await file.read()
    result = enqueue_document(filename=file.filename or "upload", content=content)
    return {**result, "indexed_total": store_count()}


@router.get("/stats")
async def stats() -> dict:
    return {"indexed_chunks": store_count()}
