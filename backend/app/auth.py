"""API key authentication, rate limiting and usage accounting.

Authentication is opt-in via ``REQUIRE_API_KEY``. When it is off, protected
endpoints run for anyone and no database is touched -- a fresh clone works
without Postgres. When it is on, every protected request must carry
``Authorization: Bearer <key>``.

Keys are stored as SHA-256 hashes. The plaintext key is shown once, at creation.

Rate limiting and usage accounting both hang off a single row written by the
:func:`require_api_key` dependency *before* the handler runs, so every protected
endpoint is counted -- not only the ones whose handler remembers to log.
"""
from __future__ import annotations

import hashlib
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import Header, HTTPException, Request
from sqlalchemy import DateTime, String, create_engine, func, select
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)

from app.config import settings
from app.logging_config import get_structured_logger

logger = get_structured_logger(__name__)

class Base(DeclarativeBase):
    """Declarative base for the auth tables."""

KEY_PREFIX = "sk_"
_ANONYMOUS: dict[str, Any] = {"id": None, "name": "anonymous", "usage_id": None}


class APIKey(Base):
    """An issued API key. Only the hash of the key is stored."""

    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(primary_key=True)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    key_hint: Mapped[str] = mapped_column(String(16), default="")
    name: Mapped[str] = mapped_column(String(256))
    requests_per_minute: Mapped[int] = mapped_column(default=10)
    is_active: Mapped[int] = mapped_column(default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    last_used: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )


class APIKeyUsage(Base):
    """One row per authenticated request, written before the handler runs."""

    __tablename__ = "api_key_usage"

    id: Mapped[int] = mapped_column(primary_key=True)
    api_key_id: Mapped[int] = mapped_column(index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True
    )
    endpoint: Mapped[str] = mapped_column(String(256))
    tokens_input: Mapped[int] = mapped_column(default=0)
    tokens_output: Mapped[int] = mapped_column(default=0)
    model: Mapped[str | None] = mapped_column(String(256), default=None)


_engine = None
_SessionLocal = None


def init_db() -> None:
    """Create the engine and tables. Idempotent; call once at startup.

    Raises:
        Exception: Propagates any SQLAlchemy connection or DDL failure.
    """
    global _engine, _SessionLocal
    if _SessionLocal is not None:
        return

    _engine = create_engine(settings.postgres_url, echo=False, pool_pre_ping=True)
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    Base.metadata.create_all(bind=_engine)
    logger.info("Auth database initialized")


@contextmanager
def session_scope() -> Iterator[Session]:
    """Yield a database session, committing on success and closing always.

    Raises:
        RuntimeError: If the auth database was never initialized.
    """
    if _SessionLocal is None:
        init_db()
    if _SessionLocal is None:  # pragma: no cover - init_db raises first
        raise RuntimeError("Auth database is not initialized")

    db = _SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def hash_key(key: str) -> str:
    """Return the SHA-256 hex digest used as the stored form of an API key."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _parse_bearer(authorization: str | None) -> str:
    """Extract the token from an Authorization header.

    Raises:
        HTTPException: 401 if the header is missing or malformed.
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header. Send 'Authorization: Bearer <api key>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(
            status_code=401,
            detail="Authorization header must be of the form 'Bearer <api key>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token.strip()


def _authenticate(db: Session, key: str) -> APIKey:
    """Look up an active API key by hash.

    Raises:
        HTTPException: 401 if the key is unknown or deactivated.
    """
    stmt = select(APIKey).where(APIKey.key_hash == hash_key(key))
    row = db.execute(stmt).scalar_one_or_none()

    # Constant-time comparison on the hash keeps the failure path uniform even
    # though the lookup itself is indexed.
    if row is None or not secrets.compare_digest(row.key_hash, hash_key(key)):
        raise HTTPException(status_code=401, detail="Invalid API key")
    if not row.is_active:
        raise HTTPException(status_code=401, detail="This API key has been deactivated")
    return row


def _enforce_rate_limit(db: Session, api_key_id: int, rpm_limit: int) -> None:
    """Count this key's requests in the last minute and reject over-limit ones.

    Raises:
        HTTPException: 429 if the key exceeded its per-minute allowance.
    """
    one_minute_ago = datetime.now(UTC) - timedelta(minutes=1)
    recent = db.execute(
        select(func.count())
        .select_from(APIKeyUsage)
        .where(APIKeyUsage.api_key_id == api_key_id)
        .where(APIKeyUsage.timestamp >= one_minute_ago)
    ).scalar_one()

    if recent >= rpm_limit:
        logger.warning(
            "Rate limit exceeded",
            extra_fields={
                "api_key_id": api_key_id,
                "requests_in_window": recent,
                "limit": rpm_limit,
            },
        )
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {rpm_limit} requests per minute",
            headers={"Retry-After": "60"},
        )


async def require_api_key(
    request: Request, authorization: str | None = Header(default=None)
) -> dict[str, Any]:
    """FastAPI dependency enforcing API key auth, rate limits and accounting.

    A no-op returning an anonymous principal when ``REQUIRE_API_KEY`` is off.

    Args:
        request: The incoming request, used to record the endpoint path.
        authorization: The ``Authorization`` header, if present.

    Returns:
        Dict with ``id`` (API key row id, or ``None`` when anonymous), ``name``
        and ``usage_id`` (the usage row to attach token counts to).

    Raises:
        HTTPException: 401 for a missing/invalid key, 429 when rate limited,
            503 if authentication is required but the database is unreachable.
    """
    if not settings.require_api_key:
        return dict(_ANONYMOUS)

    key = _parse_bearer(authorization)
    endpoint = request.url.path

    try:
        with session_scope() as db:
            api_key = _authenticate(db, key)
            _enforce_rate_limit(db, api_key.id, api_key.requests_per_minute)

            api_key.last_used = datetime.now(UTC)
            usage = APIKeyUsage(api_key_id=api_key.id, endpoint=endpoint)
            db.add(usage)
            db.flush()  # assign usage.id before the session closes

            return {"id": api_key.id, "name": api_key.name, "usage_id": usage.id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Auth backend unavailable: {type(e).__name__}: {e}",
            extra_fields={"error_type": type(e).__name__, "endpoint": endpoint},
        )
        raise HTTPException(
            status_code=503, detail="Authentication service is unavailable."
        ) from e


def record_tokens(
    auth: dict[str, Any],
    tokens_input: int = 0,
    tokens_output: int = 0,
    model: str | None = None,
) -> None:
    """Attach token counts to the usage row created for this request.

    Never raises: accounting must not fail a request that already succeeded.

    Args:
        auth: The principal returned by :func:`require_api_key`.
        tokens_input: Input tokens consumed.
        tokens_output: Output tokens generated.
        model: Model that produced them.
    """
    usage_id = auth.get("usage_id")
    if usage_id is None:
        return
    try:
        with session_scope() as db:
            usage = db.get(APIKeyUsage, usage_id)
            if usage is not None:
                usage.tokens_input = tokens_input
                usage.tokens_output = tokens_output
                usage.model = model
    except Exception as e:
        logger.warning(f"Failed to record token usage: {type(e).__name__}: {e}")


def create_api_key(name: str, requests_per_minute: int = 10) -> str:
    """Mint a new API key.

    Args:
        name: Human-readable label for the key.
        requests_per_minute: Per-minute request allowance.

    Returns:
        The plaintext key. It is not recoverable afterwards -- only its hash is
        stored -- so the caller must show it to the user immediately.
    """
    key = KEY_PREFIX + secrets.token_urlsafe(32)
    with session_scope() as db:
        db.add(
            APIKey(
                key_hash=hash_key(key),
                key_hint=key[: len(KEY_PREFIX) + 6],
                name=name,
                requests_per_minute=requests_per_minute,
                is_active=1,
            )
        )
    logger.info(
        "API key created",
        extra_fields={"name": name, "rpm_limit": requests_per_minute},
    )
    return key


def list_api_keys() -> list[dict]:
    """List keys with their 24-hour usage totals.

    Returns:
        One dict per key: metadata plus request and token counts for the last
        24 hours. Never includes key material beyond the short hint.
    """
    one_day_ago = datetime.now(UTC) - timedelta(days=1)

    with session_scope() as db:
        keys = db.execute(select(APIKey).order_by(APIKey.id)).scalars().all()

        totals = {
            row.api_key_id: row
            for row in db.execute(
                select(
                    APIKeyUsage.api_key_id.label("api_key_id"),
                    func.count().label("requests"),
                    func.coalesce(
                        func.sum(APIKeyUsage.tokens_input + APIKeyUsage.tokens_output), 0
                    ).label("total_tokens"),
                )
                .where(APIKeyUsage.timestamp >= one_day_ago)
                .group_by(APIKeyUsage.api_key_id)
            ).all()
        }

        return [
            {
                "id": k.id,
                "name": k.name,
                "key_hint": k.key_hint or "",
                "is_active": bool(k.is_active),
                "requests_per_minute": k.requests_per_minute,
                "created_at": k.created_at.isoformat() if k.created_at else None,
                "last_used": k.last_used.isoformat() if k.last_used else None,
                "usage_24h": {
                    "requests": int(getattr(totals.get(k.id), "requests", 0) or 0),
                    "total_tokens": int(
                        getattr(totals.get(k.id), "total_tokens", 0) or 0
                    ),
                },
            }
            for k in keys
        ]


def deactivate_api_key(key_id: int) -> bool:
    """Deactivate a key. Takes effect on the very next request.

    Args:
        key_id: Row id of the key to deactivate.

    Returns:
        True if a key was deactivated, False if no such key exists.
    """
    with session_scope() as db:
        api_key = db.get(APIKey, key_id)
        if api_key is None:
            return False
        api_key.is_active = 0
    logger.info("API key deactivated", extra_fields={"key_id": key_id})
    return True


def require_admin_key(x_admin_key: str | None = Header(default=None)) -> bool:
    """FastAPI dependency guarding the /admin endpoints.

    Args:
        x_admin_key: The ``X-Admin-Key`` request header.

    Returns:
        True when the header matches the configured admin key.

    Raises:
        HTTPException: 503 if no admin key is configured, 403 if it does not match.
    """
    if not settings.admin_key:
        raise HTTPException(
            status_code=503,
            detail="Admin API is disabled. Set ADMIN_KEY to enable it.",
        )
    if not x_admin_key or not secrets.compare_digest(x_admin_key, settings.admin_key):
        raise HTTPException(status_code=403, detail="Invalid admin key")
    return True


__all__ = [
    "APIKey",
    "APIKeyUsage",
    "create_api_key",
    "deactivate_api_key",
    "hash_key",
    "init_db",
    "list_api_keys",
    "record_tokens",
    "require_admin_key",
    "require_api_key",
    "session_scope",
]
