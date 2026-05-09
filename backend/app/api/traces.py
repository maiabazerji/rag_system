from fastapi import APIRouter, HTTPException

from app.tracing import get_trace

router = APIRouter()


@router.get("/{trace_id}")
async def trace(trace_id: str) -> dict:
    t = get_trace(trace_id)
    if not t:
        raise HTTPException(404, "trace not found")
    return t
