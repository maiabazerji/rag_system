"""Admin endpoints for issuing API keys, scoping their access and reviewing usage and audit.

Every route requires the ``X-Admin-Key`` header to match ``ADMIN_KEY``. While
that setting is unset the whole router returns 503.

Handlers are plain ``def``: they make blocking SQLAlchemy calls, and FastAPI
runs sync handlers in its threadpool instead of on the event loop.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import text

from app.access import normalize_groups
from app.audit import ADMIN_KEY_PRINCIPAL, AuditRecord, emit, list_audit_events
from app.auth import (
    create_api_key,
    deactivate_api_key,
    list_api_keys,
    require_admin_key,
    session_scope,
    set_api_key_access,
)
from app.config import settings

router = APIRouter(dependencies=[Depends(require_admin_key)])

_GROUPS_DESCRIPTION = (
    "Groups the key belongs to; it reads documents labelled with any of them "
    "(and 'public' ones, always)."
)


def _audit_admin(detail: str) -> None:
    """Record an admin action. Admin requests authenticate with X-Admin-Key only."""
    emit(
        AuditRecord(
            principal_id=ADMIN_KEY_PRINCIPAL, tenant=None, action="admin", detail=detail
        )
    )


class CreateKeyRequest(BaseModel):
    """Request to create a new API key."""

    name: str = Field(..., min_length=1, max_length=256, description="Human-readable name")
    requests_per_minute: int = Field(
        default=10, ge=1, le=1000, description="Rate limit (requests per minute)"
    )
    groups: list[str] = Field(default_factory=list, description=_GROUPS_DESCRIPTION)
    tenant: str | None = Field(
        default=None, max_length=128, description="Tenant; defaults to DEFAULT_TENANT."
    )
    is_admin: bool = Field(
        default=False, description="May label uploads with groups it is not a member of."
    )


class APIKeyInfo(BaseModel):
    """A newly created API key. The `key` field is shown exactly once."""

    key: str
    name: str
    requests_per_minute: int
    created_at: str
    tenant: str
    groups: list[str]
    is_admin: bool


class KeyAccessUpdate(BaseModel):
    """New access scope for a key. Omitted fields are left unchanged."""

    groups: list[str] | None = Field(default=None, description=_GROUPS_DESCRIPTION)
    tenant: str | None = Field(
        default=None, max_length=128, description="Tenant; an empty string resets it."
    )
    is_admin: bool | None = None


class KeyAccess(BaseModel):
    """A key's access scope."""

    key_id: int
    tenant: str
    groups: list[str]
    is_admin: bool


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
    tenant: str
    groups: list[str]
    is_admin: bool
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
def create_key(request: CreateKeyRequest) -> APIKeyInfo:
    """Create an API key and return it in plaintext (once).

    Args:
        request: Key name, per-minute rate limit and access scope.

    Returns:
        The new key plus its metadata.

    Raises:
        HTTPException: 422 if the group list is malformed.
    """
    try:
        key = create_api_key(
            name=request.name,
            requests_per_minute=request.requests_per_minute,
            groups=request.groups,
            tenant=request.tenant,
            is_admin=request.is_admin,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    _audit_admin(f"create_key name={request.name}")
    return APIKeyInfo(
        key=key,
        name=request.name,
        requests_per_minute=request.requests_per_minute,
        created_at=datetime.now(UTC).isoformat(),
        tenant=(request.tenant or "").strip() or settings.default_tenant,
        groups=list(normalize_groups(request.groups)),
        is_admin=request.is_admin,
    )


@router.get(
    "/keys",
    response_model=list[APIKeyDetail],
    summary="List API keys",
    description="Lists every API key with its request and token usage over the last 24 hours.",
)
def list_keys() -> list[APIKeyDetail]:
    """List all API keys with 24-hour usage totals."""
    return [APIKeyDetail(**key) for key in list_api_keys()]


@router.post(
    "/keys/{key_id}/deactivate",
    summary="Deactivate an API key",
    description="Deactivates a key. It stops working on the very next request.",
)
def deactivate_key(key_id: int) -> dict:
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
    _audit_admin(f"deactivate_key id={key_id}")
    return {"status": "deactivated", "key_id": key_id}


@router.put(
    "/keys/{key_id}/access",
    response_model=KeyAccess,
    summary="Set an API key's groups, tenant and admin flag",
    description=(
        "Changes what a key can read and how it may label uploads. Takes effect "
        "on the key's next request. Documents it already indexed keep their "
        "labels; re-upload them to relabel."
    ),
)
def set_key_access(key_id: int, update: KeyAccessUpdate) -> KeyAccess:
    """Update a key's access scope.

    Raises:
        HTTPException: 404 if no key has that id, 422 for a malformed group list.
    """
    try:
        access = set_api_key_access(
            key_id, groups=update.groups, tenant=update.tenant, is_admin=update.is_admin
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    if access is None:
        raise HTTPException(status_code=404, detail=f"No API key with id {key_id}")
    _audit_admin(f"set_key_access id={key_id}")
    return KeyAccess(key_id=key_id, **access)


@router.get(
    "/audit",
    summary="Query the audit log",
    description=(
        "Lists audit events, newest first, filtered by principal, action, tenant "
        "and time. Question text is present only with AUDIT_STORE_QUESTIONS=true; "
        "otherwise only its hash is."
    ),
)
def audit_log(
    principal: str | None = Query(
        default=None, max_length=320, description="Principal id, e.g. key:3 or oidc:<sub>"
    ),
    action: str | None = Query(default=None, max_length=32, description="e.g. ask, ingest"),
    tenant: str | None = Query(default=None, max_length=128),
    since: datetime | None = Query(default=None, description="Inclusive lower bound (ISO 8601)"),
    until: datetime | None = Query(default=None, description="Exclusive upper bound (ISO 8601)"),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """Page through the audit log.

    Returns:
        ``{"events": [...], "total", "limit", "offset"}``.
    """
    return list_audit_events(
        principal_id=principal,
        action=action,
        tenant=tenant,
        since=since,
        until=until,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/health",
    summary="Check auth database connectivity",
    description="Verifies that the API key database is reachable.",
)
def health() -> dict:
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
