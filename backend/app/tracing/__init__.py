"""Request tracing.

Every traced request is kept in a bounded in-memory store (served by
``/traces/{id}``), counted in the Prometheus metrics, and, when the telemetry
policy allows it, handed to the Langfuse exporter. See docs/monitoring.md.
"""
from __future__ import annotations

import logging
import time
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager

from app.tracing.spans import Span, Trace, activate, current_trace, span, traced

logger = logging.getLogger(__name__)

_TRACES: OrderedDict[str, dict] = OrderedDict()
_MAX_TRACES = 1000


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
def start_trace(name: str, inputs: dict) -> Iterator[Trace]:
    """Open a trace for one request; spans recorded inside attach to it."""
    t = Trace.new(name=name, inputs=inputs)
    try:
        with activate(t):
            yield t
    except BaseException as e:
        if t.output is None:
            t.fail(type(e).__name__)
        raise
    finally:
        t.end_ns = time.time_ns()
        _TRACES[t.id] = {
            "id": t.id,
            "name": t.name,
            "inputs": t.inputs,
            "events": t.events,
            "output": t.output,
        }
        if len(_TRACES) > _MAX_TRACES:
            _TRACES.popitem(last=False)
        _on_trace_end(t)


def get_trace(trace_id: str) -> dict | None:
    return _TRACES.get(trace_id)


__all__ = ["Span", "Trace", "current_trace", "get_trace", "span", "start_trace", "traced"]
