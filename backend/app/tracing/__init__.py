"""Request tracing.

Every traced request is kept in a bounded in-memory store (served by
``/traces/{id}``), counted in the Prometheus metrics, and, when the telemetry
policy allows it, handed to the Langfuse exporter. See docs/monitoring.md.

Traces hold the question and whatever the strategy logged along the way, so
they are personal data whenever a question is. Three things keep that in check:

- with ``PII_REDACT_TRACES`` on, every string is masked before the trace is
  stored (see :func:`app.privacy.pii.mask_value`);
- each record carries ``created_at`` so :func:`purge_traces` can apply
  ``RETENTION_TRACES_DAYS``;
- each record carries the principal that asked and the document ids it
  touched, so :func:`erase_traces` can honour an erasure request.
"""
from __future__ import annotations

import logging
import time
from collections import OrderedDict
from collections.abc import Collection, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from app.config import settings
from app.privacy.pii import mask_value
from app.tracing.spans import Span, Trace, activate, current_trace, span, traced
from app.tracing.stages import pipeline_stages

logger = logging.getLogger(__name__)

_TRACES: OrderedDict[str, dict] = OrderedDict()
_MAX_TRACES = 1000

# Keys whose values name a document, directly or through a chunk id
# ("<doc_id>:<n>"). Used to find the traces an erasure has to remove.
_DOC_KEYS = {"doc_id", "doc_ids"}
_CHUNK_KEYS = {"chunk_id", "chunk_ids", "context_chunks"}


def referenced_doc_ids(value: Any) -> set[str]:
    """Collect document ids named anywhere in a trace's inputs or events."""
    found: set[str] = set()

    def ids(v: Any) -> list[str]:
        if isinstance(v, str):
            return [v]
        if isinstance(v, list | tuple | set | frozenset):
            return [x for x in v if isinstance(x, str)]
        return []

    def walk(v: Any) -> None:
        if isinstance(v, dict):
            for k, child in v.items():
                if k in _DOC_KEYS:
                    found.update(ids(child))
                elif k in _CHUNK_KEYS:
                    found.update(c.split(":", 1)[0] for c in ids(child) if ":" in c)
                walk(child)
        elif isinstance(v, list | tuple | set | frozenset):
            for child in v:
                walk(child)

    walk(value)
    found.discard("")
    return found


def _on_trace_end(t: Trace) -> None:
    """Publish a finished trace to metrics and the exporter. Never raises."""
    # Imported here: both modules import app.tracing.spans, and keeping this
    # package's import light lets them do so without a cycle.
    from app import monitoring
    from app.tracing import langfuse_exporter

    try:
        monitoring.observe_trace(t)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Recording trace metrics failed: %s", e)
    try:
        langfuse_exporter.export(t)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Queueing trace for export failed: %s", e)


@contextmanager
def start_trace(
    name: str, inputs: dict, principal_id: str | None = None
) -> Iterator[Trace]:
    """Open a trace for one request; spans recorded inside attach to it.

    Args:
        name: Trace name, e.g. ``ask:classic``.
        inputs: The request inputs.
        principal_id: Who made the request, so the trace can be found when that
            principal exercises their right to erasure or access.
    """
    t = Trace.new(name=name, inputs=inputs, principal_id=principal_id)
    try:
        with activate(t):
            yield t
    except BaseException as e:
        if t.output is None:
            t.fail(type(e).__name__)
        raise
    finally:
        t.end_ns = time.time_ns()
        _store(t)
        _on_trace_end(t)


def _store(t: Trace) -> None:
    # Document ids are read before masking, which must not be able to hide them.
    doc_ids = referenced_doc_ids([t.inputs, t.events])
    inputs, events, output = t.inputs, t.events, t.output
    # The pipeline stages in order, with latencies, for a waterfall view. Span
    # metadata only: counts, flags, timings, tokens and cost, never content.
    stages = pipeline_stages(t)
    if settings.pii_redact_traces:
        inputs, events, output = mask_value(inputs), mask_value(events), mask_value(output)
        stages = mask_value(stages)
    _TRACES[t.id] = {
        "id": t.id,
        "name": t.name,
        "inputs": inputs,
        "events": events,
        "output": output,
        "stages": stages,
        "duration_ms": round(t.duration_seconds * 1000, 3),
        "created_at": datetime.now(UTC).isoformat(),
        "principal_id": t.principal_id,
        "doc_ids": sorted(doc_ids),
    }
    if len(_TRACES) > _MAX_TRACES:
        _TRACES.popitem(last=False)


def get_trace(trace_id: str) -> dict | None:
    return _TRACES.get(trace_id)


def traces_for_principal(principal_id: str) -> list[dict]:
    """Every stored trace recorded for ``principal_id``, oldest first."""
    return [t for t in _TRACES.values() if t.get("principal_id") == principal_id]


def erase_traces(
    doc_ids: Collection[str] = (), principal_id: str | None = None
) -> int:
    """Delete traces that touched any of ``doc_ids`` or were made by ``principal_id``.

    Returns:
        The number of traces deleted.
    """
    wanted = set(doc_ids)
    doomed = [
        tid
        for tid, t in _TRACES.items()
        if (principal_id is not None and t.get("principal_id") == principal_id)
        or wanted.intersection(t.get("doc_ids", ()))
    ]
    for tid in doomed:
        del _TRACES[tid]
    return len(doomed)


def purge_traces(before: datetime) -> int:
    """Delete traces recorded before ``before``. Returns how many were deleted."""
    doomed = [
        tid
        for tid, t in _TRACES.items()
        if datetime.fromisoformat(t["created_at"]) < before
    ]
    for tid in doomed:
        del _TRACES[tid]
    return len(doomed)


__all__ = [
    "Span",
    "Trace",
    "current_trace",
    "erase_traces",
    "get_trace",
    "purge_traces",
    "span",
    "start_trace",
    "traced",
    "traces_for_principal",
]
