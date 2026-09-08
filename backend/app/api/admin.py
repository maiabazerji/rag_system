"""Admin endpoints for issuing API keys and reviewing usage.

Every route requires the ``X-Admin-Key`` header to match ``ADMIN_KEY``. While
that setting is unset the whole router returns 503.
"""
from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text

from app.auth import (
    create_api_key,
    deactivate_api_key,
    list_api_keys,
    require_admin_key,
    session_scope,
)

router = APIRouter(dependencies=[Depends(require_admin_key)])


class CreateKeyRequest(BaseModel):
    """Request to create a new API key."""

    name: str = Field(..., min_length=1, max_length=256, description="Human-readable name")
    requests_per_minute: int = Field(
        default=10, ge=1, le=1000, description="Rate limit (requests per minute)"
    )


class APIKeyInfo(BaseModel):
    """A newly created API key. The `key` field is shown exactly once."""

    key: str
    name: str
    requests_per_minute: int
    created_at: str


class Usage24h(BaseModel):
    """Request and token totals for the last 24 hours."""

    requests: int
    total_tokens: int


class APIKeyDetail(BaseModel):
    """Stored metadata for an API key. Never includes the key itself."""

    id: int
    name: str
    key_hint: str
    is_active: bool
    requests_per_minute: int
    created_at: str | None
    last_used: str | None
    usage_24h: Usage24h


@router.post(
    "/keys",
    response_model=APIKeyInfo,
    status_code=201,
    summary="Create an API key",
    description=(
        "Mints a new API key. The plaintext key is returned once and never "
        "stored -- only its SHA-256 hash is kept. Save it immediately."
    ),
)
async def create_key(request: CreateKeyRequest) -> APIKeyInfo:
    """Create an API key and return it in plaintext (once).

    Args:
        request: Key name and per-minute rate limit.

    Returns:
        The new key plus its metadata.
    """
    key = create_api_key(
        name=request.name, requests_per_minute=request.requests_per_minute
    )
    return APIKeyInfo(
        key=key,
        name=request.name,
        requests_per_minute=request.requests_per_minute,
        created_at=datetime.now(UTC).isoformat(),
    )


@router.get(
    "/keys",
    response_model=list[APIKeyDetail],
    summary="List API keys",
    description="Lists every API key with its request and token usage over the last 24 hours.",
)
async def list_keys() -> list[APIKeyDetail]:
    """List all API keys with 24-hour usage totals."""
    return [APIKeyDetail(**key) for key in list_api_keys()]


@router.post(
    "/keys/{key_id}/deactivate",
    summary="Deactivate an API key",
    description="Deactivates a key. It stops working on the very next request.",
)
async def deactivate_key(key_id: int) -> dict:
    """Deactivate an API key.

    Args:
        key_id: Row id of the key.

    Returns:
        Confirmation of the deactivation.

    Raises:
        HTTPException: 404 if no key has that id.
    """
    if not deactivate_api_key(key_id):
        raise HTTPException(status_code=404, detail=f"No API key with id {key_id}")
    return {"status": "deactivated", "key_id": key_id}


@router.get(
    "/health",
    summary="Check auth database connectivity",
    description="Verifies that the API key database is reachable.",
)
async def health() -> dict:
    """Check that the auth database answers a trivial query."""
    try:
        with session_scope() as db:
            db.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": type(e).__name__,
        }
