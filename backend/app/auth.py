"""API key authentication, rate limiting and usage accounting.

Authentication is opt-in via ``REQUIRE_API_KEY``. When it is off, protected
endpoints run for anyone and no database is touched -- a fresh clone works
without Postgres. When it is on, every protected request must carry
``Authorization: Bearer <key>``.

Keys are stored as SHA-256 hashes. The plaintext key is shown once, at creation.

Rate limiting and usage accounting both hang off a single row written by the
:func:`require_api_key` dependency *before* the handler runs, so every protected
endpoint is counted -- not only the ones whose handler remembers to log.

The limit is on *units* per minute, not requests. A plain request costs one
unit; endpoints that fan out to many model calls re-price their row with
:func:`charge` before doing the work (one unit per compare variant, per
strategy -- three for agentic -- and per evaluated example).

Check-and-record is atomic: the key's own row is locked (``SELECT ... FOR
UPDATE``) for the transaction that counts the window and writes the usage row,
so concurrent requests on one key are serialized instead of all passing the
check before any of them is recorded.

Each request resolves to a :class:`~app.access.Principal`. A Bearer token
that looks like a JWT is validated against the OIDC issuer (``app.oidc``) when
one is configured; anything else is looked up as an API key. API keys carry a
tenant and groups (set with the admin API) that scope what they can read.
OIDC users are not rate limited or metered here: they have no key row.

The database calls here are synchronous. The dependencies are plain ``def`` so
FastAPI runs them in its threadpool; async handlers call :func:`charge` and
:func:`record_tokens` through ``run_in_threadpool``.
"""
from __future__ import annotations

import hashlib
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import Header, HTTPException, Request
from sqlalchemy import JSON, DateTime, String, create_engine, func, inspect, select, text
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)

from app.access import Principal, local_principal, normalize_groups
from app.config import settings
from app.logging_config import get_structured_logger
from app.oidc import authenticate_oidc, looks_like_jwt

logger = get_structured_logger(__name__)

class Base(DeclarativeBase):
    """Declarative base for the auth tables."""

KEY_PREFIX = "sk_"

# Rate-limit units per run of a strategy. Agentic makes up to AGENTIC_MAX_ITERS
# model calls per question, so it is priced above the single-call strategies.
STRATEGY_UNITS: dict[str, int] = {"classic": 1, "graph": 1, "agentic": 3}


def strategy_units(strategy: str) -> int:
    """Rate-limit cost of running one question through ``strategy``."""
    return STRATEGY_UNITS.get(strategy, 1)


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
    # Access scope. NULL tenant means DEFAULT_TENANT; NULL groups means none.
    tenant: Mapped[str | None] = mapped_column(String(128), default=None)
    acl_groups: Mapped[list[str] | None] = mapped_column(JSON, default=list)
    is_admin: Mapped[int] = mapped_column(default=0, server_default=text("0"))

    def principal(self, usage_id: int | None = None) -> Principal:
        """The principal this key authenticates as."""
        return Principal(
            id=f"key:{self.id}",
            kind="api_key",
            display_name=self.name,
            tenant=self.tenant or settings.default_tenant,
            groups=normalize_groups(self.acl_groups),
            is_admin=bool(self.is_admin),
            key_id=self.id,
            usage_id=usage_id,
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
    # Rate-limit units this request consumed (see `charge`).
    units: Mapped[int] = mapped_column(default=1, server_default=text("1"))


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

    # Registers the audit table on Base before create_all runs.
    import app.audit  # noqa: F401

    engine = create_engine(settings.postgres_url, echo=False, pool_pre_ping=True)
    Base.metadata.create_all(bind=engine)
    _add_missing_columns(engine)
    _engine = engine
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.info("Auth database initialized")


def db_ready() -> bool:
    """Whether the auth database has been initialized in this process."""
    return _SessionLocal is not None


# (table, column, DDL type) for columns added after a table's first release.
_LATE_COLUMNS: list[tuple[str, str, str]] = [
    ("api_key_usage", "units", "INTEGER NOT NULL DEFAULT 1"),
    ("api_keys", "tenant", "VARCHAR(128)"),
    ("api_keys", "acl_groups", "JSON"),
    ("api_keys", "is_admin", "INTEGER NOT NULL DEFAULT 0"),
]


def _add_missing_columns(engine) -> None:
    """Add columns introduced after a database was first created.

    ``create_all`` creates missing tables but never alters existing ones, so a
    database from an older release lacks the columns in ``_LATE_COLUMNS``.
    """
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    for table, column, ddl in _LATE_COLUMNS:
        if table not in tables:
            continue
        if column in {c["name"] for c in inspector.get_columns(table)}:
            continue
        with engine.begin() as conn:
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))
        logger.info(f"Added {table}.{column} column")


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


def _authenticate(db: Session, key: str, *, lock: bool = False) -> APIKey:
    """Look up an active API key by hash.

    Args:
        db: Open session.
        key: The plaintext key from the request.
        lock: Take a row lock on the key (``FOR UPDATE``) for the rest of the
            transaction. Serializes rate-limit check-and-record per key.

    Raises:
        HTTPException: 401 if the key is unknown or deactivated.
    """
    stmt = select(APIKey).where(APIKey.key_hash == hash_key(key))
    if lock:
        stmt = stmt.with_for_update()
    row = db.execute(stmt).scalar_one_or_none()

    # Constant-time comparison on the hash keeps the failure path uniform even
    # though the lookup itself is indexed.
    if row is None or not secrets.compare_digest(row.key_hash, hash_key(key)):
        raise HTTPException(status_code=401, detail="Invalid API key")
    if not row.is_active:
        raise HTTPException(status_code=401, detail="This API key has been deactivated")
    return row


def _units_in_window(
    db: Session, api_key_id: int, *, exclude_usage_id: int | None = None
) -> int:
    """Rate-limit units this key consumed in the last minute."""
    one_minute_ago = datetime.now(UTC) - timedelta(minutes=1)
    stmt = (
        select(func.coalesce(func.sum(APIKeyUsage.units), 0))
        .where(APIKeyUsage.api_key_id == api_key_id)
        .where(APIKeyUsage.timestamp >= one_minute_ago)
    )
    if exclude_usage_id is not None:
        stmt = stmt.where(APIKeyUsage.id != exclude_usage_id)
    return int(db.execute(stmt).scalar_one())


def _rate_limited(api_key_id: int, used: int, units: int, rpm_limit: int) -> HTTPException:
    logger.warning(
        "Rate limit exceeded",
        extra_fields={
            "api_key_id": api_key_id,
            "units_in_window": used,
            "units_requested": units,
            "limit": rpm_limit,
        },
    )
    return HTTPException(
        status_code=429,
        detail=f"Rate limit exceeded: {rpm_limit} requests per minute",
        headers={"Retry-After": "60"},
    )


def _enforce_rate_limit(
    db: Session, api_key_id: int, rpm_limit: int, units: int = 1
) -> None:
    """Reject the request if ``units`` more would exceed the per-minute limit.

    Call with the key's row locked (``_authenticate(..., lock=True)``) and
    record the usage in the same transaction, or the check can race.

    Raises:
        HTTPException: 429 if the key exceeded its per-minute allowance.
    """
    used = _units_in_window(db, api_key_id)
    if used + units > rpm_limit:
        raise _rate_limited(api_key_id, used, units, rpm_limit)


def _authenticate_api_key(key: str, endpoint: str) -> Principal:
    """Check an API key, enforce its rate limit and open its usage row.

    Raises:
        HTTPException: 401 for an unknown or deactivated key, 429 when rate
            limited, 503 if the database is unreachable.
    """
    try:
        with session_scope() as db:
            api_key = _authenticate(db, key, lock=True)
            _enforce_rate_limit(db, api_key.id, api_key.requests_per_minute)

            api_key.last_used = datetime.now(UTC)
            usage = APIKeyUsage(api_key_id=api_key.id, endpoint=endpoint)
            db.add(usage)
            db.flush()  # assign usage.id before the session closes

            return api_key.principal(usage_id=usage.id)
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


def require_principal(
    request: Request, authorization: str | None = Header(default=None)
) -> Principal:
    """FastAPI dependency resolving the caller to a :class:`Principal`.

    With ``REQUIRE_API_KEY`` off this is the local principal (an admin in the
    default tenant whose documents are public), so local use is unchanged.
    Otherwise the Bearer token is either an OIDC JWT (when ``OIDC_ISSUER`` is
    set) or an API key, which is also rate limited and metered.

    Synchronous on purpose: FastAPI runs it in the threadpool, so the blocking
    database round trips and JWKS fetches do not stall the event loop.

    Raises:
        HTTPException: 401 for a missing/invalid credential, 429 when rate
            limited, 503 if the auth database is unreachable.
    """
    if not settings.require_api_key:
        return local_principal()

    token = _parse_bearer(authorization)
    if settings.oidc_issuer and looks_like_jwt(token):
        return authenticate_oidc(token)
    return _authenticate_api_key(token, request.url.path)


def require_api_key(
    request: Request, authorization: str | None = Header(default=None)
) -> dict[str, Any]:
    """The pre-principal dependency, kept for callers that want a plain dict.

    Routes use :func:`require_principal`. Do not mix the two on one route:
    FastAPI would authenticate (and meter) the request twice.

    Returns:
        :meth:`Principal.as_auth`: ``id`` (API key row id, or ``None``),
        ``name``, ``usage_id``, plus ``principal_id``, ``tenant``, ``groups``
        and ``is_admin``.
    """
    return require_principal(request, authorization).as_auth()


def _account_ids(auth: Principal | dict[str, Any]) -> tuple[int | None, int | None]:
    """``(api key id, usage row id)`` from a principal or a legacy auth dict."""
    if isinstance(auth, Principal):
        return auth.key_id, auth.usage_id
    return auth.get("id"), auth.get("usage_id")


def charge(auth: Principal | dict[str, Any], units: int) -> None:
    """Price this request at ``units`` rate-limit units, before doing the work.

    :func:`require_api_key` records every request as one unit. Endpoints that
    fan out (several variants, strategies or examples per request) call this to
    re-price their usage row, atomically with the limit check.

    A request larger than the whole per-minute allowance is admitted only when
    the key has nothing else in the window, so a big evaluation remains
    possible but then consumes the key's budget for the following minute.

    A no-op for anonymous principals and for ``units <= 1``.

    Args:
        auth: The caller, from :func:`require_principal` (or the legacy dict).
        units: Total cost of this request.

    Raises:
        HTTPException: 429 if the charge would exceed the key's limit, 503 if
            the auth database is unreachable.
    """
    key_id, usage_id = _account_ids(auth)
    if usage_id is None or key_id is None or units <= 1:
        return
    try:
        with session_scope() as db:
            api_key = db.execute(
                select(APIKey).where(APIKey.id == key_id).with_for_update()
            ).scalar_one()
            others = _units_in_window(db, key_id, exclude_usage_id=usage_id)
            if others > 0 and others + units > api_key.requests_per_minute:
                raise _rate_limited(key_id, others, units, api_key.requests_per_minute)
            usage = db.get(APIKeyUsage, usage_id)
            if usage is not None:
                usage.units = units
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Auth backend unavailable while charging: {type(e).__name__}: {e}",
            extra_fields={"error_type": type(e).__name__},
        )
        raise HTTPException(
            status_code=503, detail="Authentication service is unavailable."
        ) from e


def record_tokens(
    auth: Principal | dict[str, Any],
    tokens_input: int = 0,
    tokens_output: int = 0,
    model: str | None = None,
) -> None:
    """Attach token counts to the usage row created for this request.

    Never raises: accounting must not fail a request that already succeeded.

    Args:
        auth: The caller, from :func:`require_principal` (or the legacy dict).
        tokens_input: Input tokens consumed.
        tokens_output: Output tokens generated.
        model: Model that produced them.
    """
    _, usage_id = _account_ids(auth)
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


def create_api_key(
    name: str,
    requests_per_minute: int = 10,
    *,
    groups: list[str] | None = None,
    tenant: str | None = None,
    is_admin: bool = False,
) -> str:
    """Mint a new API key.

    Args:
        name: Human-readable label for the key.
        requests_per_minute: Per-minute request allowance.
        groups: Groups the key belongs to (``public`` is implicit).
        tenant: The key's tenant; ``None`` means ``DEFAULT_TENANT``.
        is_admin: Whether the key may label documents with any group.

    Returns:
        The plaintext key. It is not recoverable afterwards -- only its hash is
        stored -- so the caller must show it to the user immediately.

    Raises:
        ValueError: If the group list is malformed.
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
                tenant=(tenant or "").strip() or None,
                acl_groups=list(normalize_groups(groups)),
                is_admin=int(is_admin),
            )
        )
    logger.info(
        "API key created",
        extra_fields={"name": name, "rpm_limit": requests_per_minute, "tenant": tenant},
    )
    return key


def _key_access(k: APIKey) -> dict[str, Any]:
    return {
        "tenant": k.tenant or settings.default_tenant,
        "groups": list(normalize_groups(k.acl_groups)),
        "is_admin": bool(k.is_admin),
    }


def set_api_key_access(
    key_id: int,
    *,
    groups: list[str] | None = None,
    tenant: str | None = None,
    is_admin: bool | None = None,
) -> dict[str, Any] | None:
    """Change a key's groups, tenant or admin flag. ``None`` leaves a field as is.

    Takes effect on the key's next request. Documents the key already indexed
    keep the tenant and groups they were indexed with.

    Returns:
        The key's resulting ``{"tenant", "groups", "is_admin"}``, or ``None``
        if no key has that id.

    Raises:
        ValueError: If the group list is malformed.
    """
    with session_scope() as db:
        api_key = db.get(APIKey, key_id)
        if api_key is None:
            return None
        if groups is not None:
            api_key.acl_groups = list(normalize_groups(groups))
        if tenant is not None:
            api_key.tenant = tenant.strip() or None
        if is_admin is not None:
            api_key.is_admin = int(is_admin)
        access = _key_access(api_key)
    logger.info("API key access changed", extra_fields={"key_id": key_id, **access})
    return access


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
                **_key_access(k),
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


# Short alias for the admin-only dependency.
require_admin = require_admin_key


__all__ = [
    "APIKey",
    "APIKeyUsage",
    "STRATEGY_UNITS",
    "charge",
    "create_api_key",
    "db_ready",
    "deactivate_api_key",
    "hash_key",
    "init_db",
    "list_api_keys",
    "record_tokens",
    "require_admin",
    "require_admin_key",
    "require_api_key",
    "require_principal",
    "session_scope",
    "set_api_key_access",
    "strategy_units",
]
