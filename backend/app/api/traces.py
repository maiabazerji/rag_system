from fastapi import APIRouter, Depends, HTTPException

from app.auth import require_api_key
from app.tracing import get_trace

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.get("/{trace_id}", summary="Fetch a request trace")
async def trace(trace_id: str) -> dict:
    """Return the recorded trace for one request.

    Args:
        trace_id: Trace identifier returned alongside an answer.

    Raises:
        HTTPException: 404 if the trace is unknown or has been evicted. Traces
            are kept in memory and do not survive a restart.
    """
    t = get_trace(trace_id)
    if not t:
        raise HTTPException(
            status_code=404,
            detail="Trace not found. Traces are kept in memory and are lost on restart.",
        )
    return t
