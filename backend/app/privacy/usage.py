"""Per-principal views of the Postgres usage table, for access and erasure requests.

Usage is recorded per API key. ``principal_id`` is the principal's id as
:class:`app.access.Principal` spells it (``key:<id>``), or for convenience the
key's bare numeric row id or its name. OIDC users (``oidc:<sub>``) have no
usage rows.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.auth import APIKey, APIKeyUsage, session_scope


def _keys(db: Session, principal_id: str) -> list[APIKey]:
    stmt = select(APIKey)
    if principal_id.startswith("oidc:") or principal_id == "local":
        return []
    principal_id = principal_id.removeprefix("key:")
    if principal_id.isdigit():
        stmt = stmt.where(APIKey.id == int(principal_id))
    else:
        stmt = stmt.where(APIKey.name == principal_id)
    return list(db.execute(stmt.order_by(APIKey.id)).scalars().all())


def _iso(value: Any) -> str | None:
    return value.isoformat() if value is not None else None


def usage_summary(principal_id: str) -> dict[str, Any]:
    """Summarise the principal's API keys and recorded usage.

    Returns:
        The keys (metadata only, never key material), total requests and
        tokens, the first and last request time, and requests per endpoint.
    """
    with session_scope() as db:
        keys = _keys(db, principal_id)
        ids = [k.id for k in keys]
        summary: dict[str, Any] = {
            "api_keys": [
                {
                    "id": k.id,
                    "name": k.name,
                    "key_hint": k.key_hint or "",
                    "created_at": _iso(k.created_at),
                    "last_used": _iso(k.last_used),
                }
                for k in keys
            ],
            "requests": 0,
            "tokens_input": 0,
            "tokens_output": 0,
            "first_request": None,
            "last_request": None,
            "by_endpoint": {},
        }
        if not ids:
            return summary

        totals = db.execute(
            select(
                func.count(),
                func.coalesce(func.sum(APIKeyUsage.tokens_input), 0),
                func.coalesce(func.sum(APIKeyUsage.tokens_output), 0),
                func.min(APIKeyUsage.timestamp),
                func.max(APIKeyUsage.timestamp),
            ).where(APIKeyUsage.api_key_id.in_(ids))
        ).one()
        summary.update(
            requests=int(totals[0]),
            tokens_input=int(totals[1]),
            tokens_output=int(totals[2]),
            first_request=_iso(totals[3]),
            last_request=_iso(totals[4]),
        )
        summary["by_endpoint"] = {
            endpoint: int(n)
            for endpoint, n in db.execute(
                select(APIKeyUsage.endpoint, func.count())
                .where(APIKeyUsage.api_key_id.in_(ids))
                .group_by(APIKeyUsage.endpoint)
                .order_by(APIKeyUsage.endpoint)
            ).all()
        }
        return summary


def delete_usage(principal_id: str) -> int:
    """Delete every usage row recorded for the principal's API keys.

    The keys themselves are left alone; deactivate them through ``/admin``.

    Returns:
        The number of usage rows deleted.
    """
    with session_scope() as db:
        ids = [k.id for k in _keys(db, principal_id)]
        if not ids:
            return 0
        result = db.execute(delete(APIKeyUsage).where(APIKeyUsage.api_key_id.in_(ids)))
        return int(getattr(result, "rowcount", 0) or 0)
