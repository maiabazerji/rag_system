"""Right of access (GDPR art. 15): gather what the system holds about a principal.

Each store contributes a section through :func:`register_export_source`; the
built-ins cover the documents the principal owns, their API usage and their
request traces. A section that cannot be read is reported under ``errors``
rather than silently left out, so an incomplete export is never mistaken for a
complete one.
"""
from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from qdrant_client.http import models as qm

from app.logging_config import get_structured_logger
from app.privacy.erasure import OWNER_PAYLOAD_KEY, SOURCE_PAYLOAD_KEY
from app.rag.store import scroll_payloads
from app.tracing import traces_for_principal

logger = get_structured_logger(__name__)

ExportSource = Callable[[str], Any | Awaitable[Any]]

_sources: dict[str, ExportSource] = {}


def register_export_source(name: str, fn: ExportSource) -> None:
    """Add (or replace) a section of the export.

    Args:
        name: Key of the section in the export.
        fn: Called with the principal id; returns (or resolves to) any
            JSON-serialisable value.
    """
    _sources[name] = fn


def unregister_export_source(name: str) -> None:
    """Remove a section. A no-op if it is not registered."""
    _sources.pop(name, None)


async def export_principal(principal_id: str) -> dict[str, Any]:
    """Assemble every registered section for ``principal_id``."""
    export: dict[str, Any] = {
        "principal_id": principal_id,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    errors: dict[str, str] = {}
    for name, fn in list(_sources.items()):
        try:
            value = fn(principal_id)
            if inspect.isawaitable(value):
                value = await value
            export[name] = value
        except Exception as e:
            errors[name] = f"{type(e).__name__}: {e}"
            logger.error(
                f"Export section '{name}' failed: {type(e).__name__}: {e}",
                extra_fields={"section": name},
            )
    export["errors"] = errors
    export["complete"] = not errors
    return export


async def owned_documents(principal_id: str) -> list[dict[str, Any]]:
    """Documents whose chunks carry the principal as owner, one entry per document."""
    payloads = await scroll_payloads(
        qm.Filter(
            must=[
                qm.FieldCondition(
                    key=OWNER_PAYLOAD_KEY, match=qm.MatchValue(value=principal_id)
                )
            ]
        ),
        fields=["doc_id", "filename", "ingested_at", "pii_counts", SOURCE_PAYLOAD_KEY],
    )
    docs: dict[str, dict[str, Any]] = {}
    for p in payloads:
        doc_id = p.get("doc_id")
        if not doc_id:
            continue
        doc = docs.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "filename": p.get("filename"),
                "source_key": p.get(SOURCE_PAYLOAD_KEY),
                "created": p.get("ingested_at"),
                "chunks": 0,
                "pii_counts": p.get("pii_counts") or {},
            },
        )
        doc["chunks"] += 1
    return sorted(docs.values(), key=lambda d: (d["created"] or "", d["doc_id"]))


def usage(principal_id: str) -> dict[str, Any]:
    from app.privacy.usage import usage_summary

    return usage_summary(principal_id)


def traces(principal_id: str) -> dict[str, Any]:
    items = traces_for_principal(principal_id)
    return {"count": len(items), "items": items}


def audit_events(principal_id: str) -> dict[str, Any]:
    """The principal's audit events, newest first (up to 1000)."""
    from app import audit, auth

    if not auth.db_ready():
        return {"events": [], "total": 0}
    return audit.list_audit_events(principal_id=principal_id, limit=1000)


def register_builtin_sources() -> None:
    """(Re)register the built-in sections. Called at import."""
    register_export_source("documents", owned_documents)
    register_export_source("usage", usage)
    register_export_source("traces", traces)
    register_export_source("audit", audit_events)


register_builtin_sources()
