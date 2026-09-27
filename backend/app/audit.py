"""Audit log: who asked, ingested or administered what, and which documents it touched.

One row per audited request in ``audit_events``, in the auth database. Rows
are written on a small background thread pool, after the work is done, so
auditing never adds a database round trip to a request's latency and never
fails it: a write that errors is logged and dropped.

Question text is personal data more often than not, so by default only its
SHA-256 hash is stored (enough to correlate repeats of the same question).
``AUDIT_STORE_QUESTIONS=true`` stores the text as well.

Auditing needs the auth database, so it is active exactly when that database
is (``REQUIRE_API_KEY`` on, or ``ADMIN_KEY`` set). In plain local mode events
are discarded.

Retention and erasure hooks: :func:`purge_audit_events` and
:func:`erase_principal_audit`.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import HTTPException
from sqlalchemy import JSON, DateTime, String, Text, delete, func, select
from sqlalchemy.orm import Mapped, mapped_column

from app import auth
from app.access import Principal
from app.config import settings
from app.logging_config import get_structured_logger
from app.schemas import Source

logger = get_structured_logger(__name__)

AuditAction = Literal["ask", "compare", "ingest", "delete", "advise", "eval", "admin"]
# Principal id recorded for requests authenticated with X-Admin-Key.
ADMIN_KEY_PRINCIPAL = "admin-key"
# Bound on stored doc ids per event, so one huge eval cannot bloat a row.
_MAX_DOC_IDS = 500


class AuditEvent(auth.Base):
    """One audited request."""

    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True
    )
    principal_id: Mapped[str] = mapped_column(String(320), index=True)
    tenant: Mapped[str | None] = mapped_column(String(128), default=None, index=True)
    action: Mapped[str] = mapped_column(String(32), index=True)
    strategy: Mapped[str | None] = mapped_column(String(64), default=None)
    doc_ids: Mapped[list[str] | None] = mapped_column(JSON, default=list)
    question_hash: Mapped[str | None] = mapped_column(String(64), default=None)
    question: Mapped[str | None] = mapped_column(Text, default=None)
    status: Mapped[str] = mapped_column(String(32), default="ok")
    detail: Mapped[str | None] = mapped_column(String(512), default=None)


@dataclass
class AuditRecord:
    """An event being assembled by a route. Fields may be filled in as it runs."""

    principal_id: str
    tenant: str | None
    action: str
    strategy: str | None = None
    question: str | None = None
    doc_ids: list[str] = field(default_factory=list)
    status: str = "ok"
    detail: str | None = None

    def add_doc_ids(self, doc_ids: Iterable[str | None]) -> None:
        """Record documents this request returned content from (deduplicated)."""
        for d in doc_ids:
            if d and d not in self.doc_ids:
                self.doc_ids.append(d)

    def add_sources(self, sources: Iterable[Source]) -> None:
        """Record the documents behind a list of cited sources."""
        self.add_doc_ids(doc_id_of(s.chunk_id) for s in sources)


def doc_id_of(chunk_id: str | None) -> str | None:
    """The document id of a chunk id (``<doc_id>:<n>``); ``None`` for placeholders."""
    if not chunk_id or chunk_id == "none":
        return None
    return chunk_id.rsplit(":", 1)[0]


def question_hash(question: str) -> str:
    """SHA-256 of the question, whitespace-normalised so trivial variants match."""
    return hashlib.sha256(" ".join(question.split()).encode("utf-8")).hexdigest()


_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="audit")


def _write(record: AuditRecord) -> None:
    """Persist one event. Never raises."""
    try:
        with auth.session_scope() as db:
            db.add(
                AuditEvent(
                    principal_id=record.principal_id,
                    tenant=record.tenant,
                    action=record.action,
                    strategy=record.strategy,
                    doc_ids=record.doc_ids[:_MAX_DOC_IDS],
                    question_hash=question_hash(record.question) if record.question else None,
                    question=record.question if settings.audit_store_questions else None,
                    status=record.status,
                    detail=(record.detail or None) and record.detail[:512],
                )
            )
    except Exception as e:
        logger.warning(
            f"Failed to write audit event: {type(e).__name__}: {e}",
            extra_fields={"action": record.action, "error_type": type(e).__name__},
        )


def emit(record: AuditRecord) -> Future | None:
    """Queue an event for writing. Never raises and never blocks on the database.

    Returns:
        The write's future (tests wait on it), or ``None`` when auditing is
        inactive because there is no auth database.
    """
    if not auth.db_ready():
        return None
    try:
        return _executor.submit(_write, record)
    except Exception as e:  # e.g. the executor is shutting down
        logger.warning(f"Failed to queue audit event: {type(e).__name__}: {e}")
        return None


@contextmanager
def audited(
    principal: Principal,
    action: AuditAction,
    *,
    strategy: str | None = None,
    question: str | None = None,
) -> Iterator[AuditRecord]:
    """Audit the enclosed block.

    Yields a record the block can enrich (``add_sources``, ``status``); it is
    emitted on exit whatever happens. An ``HTTPException`` is recorded as
    ``rejected`` (4xx) or ``error`` (5xx), any other exception as ``error``,
    and the exception always propagates unchanged.
    """
    record = AuditRecord(
        principal_id=principal.id,
        tenant=principal.tenant,
        action=action,
        strategy=strategy,
        question=question,
    )
    try:
        yield record
    except HTTPException as e:
        record.status = "rejected" if e.status_code < 500 else "error"
        record.detail = f"{e.status_code}: {e.detail}"
        raise
    except Exception as e:
        record.status = "error"
        record.detail = type(e).__name__
        raise
    finally:
        emit(record)


def _event_dict(e: AuditEvent) -> dict[str, Any]:
    return {
        "id": e.id,
        "ts": e.ts.isoformat() if e.ts else None,
        "principal_id": e.principal_id,
        "tenant": e.tenant,
        "action": e.action,
        "strategy": e.strategy,
        "doc_ids": list(e.doc_ids or []),
        "question_hash": e.question_hash,
        "question": e.question,
        "status": e.status,
        "detail": e.detail,
    }


def list_audit_events(
    *,
    principal_id: str | None = None,
    action: str | None = None,
    tenant: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    """Query the audit log, newest first.

    Returns:
        ``{"events": [...], "total": <matching rows>, "limit", "offset"}``.
    """
    conditions = []
    if principal_id:
        conditions.append(AuditEvent.principal_id == principal_id)
    if action:
        conditions.append(AuditEvent.action == action)
    if tenant:
        conditions.append(AuditEvent.tenant == tenant)
    if since is not None:
        conditions.append(AuditEvent.ts >= since)
    if until is not None:
        conditions.append(AuditEvent.ts < until)

    with auth.session_scope() as db:
        total = db.execute(
            select(func.count()).select_from(AuditEvent).where(*conditions)
        ).scalar_one()
        rows = db.execute(
            select(AuditEvent)
            .where(*conditions)
            .order_by(AuditEvent.ts.desc(), AuditEvent.id.desc())
            .limit(limit)
            .offset(offset)
        ).scalars()
        events = [_event_dict(e) for e in rows]
    return {"events": events, "total": int(total), "limit": limit, "offset": offset}


def purge_audit_events(before: datetime) -> int:
    """Delete every audit event older than ``before``. For retention policies.

    Returns:
        The number of events deleted.
    """
    with auth.session_scope() as db:
        result = db.execute(delete(AuditEvent).where(AuditEvent.ts < before))
        removed = int(getattr(result, "rowcount", 0) or 0)
    logger.info("Audit events purged", extra_fields={"before": before.isoformat(), "removed": removed})
    return removed


def erase_principal_audit(principal_id: str) -> int:
    """Delete every audit event recorded for one principal. For erasure requests.

    Returns:
        The number of events deleted.
    """
    with auth.session_scope() as db:
        result = db.execute(delete(AuditEvent).where(AuditEvent.principal_id == principal_id))
        removed = int(getattr(result, "rowcount", 0) or 0)
    logger.info("Audit events erased for principal", extra_fields={"removed": removed})
    return removed


__all__ = [
    "ADMIN_KEY_PRINCIPAL",
    "AuditEvent",
    "AuditRecord",
    "audited",
    "doc_id_of",
    "emit",
    "erase_principal_audit",
    "list_audit_events",
    "purge_audit_events",
    "question_hash",
]
