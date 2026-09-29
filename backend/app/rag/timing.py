"""Per-request stage timing.

One :class:`StageTimer` is active per request (a context variable, so
concurrent requests never mix). Pipeline code times its work with
:func:`stage`, or reports a measured duration with :func:`record`; both are
no-ops when no timer is active, so strategies and retrieval cost nothing when
called outside a request (scripts, the eval harness's own helpers).

Timing is *exclusive*: when stages nest, the inner stage's time counts only
toward the inner stage. Graph RAG's retrieval stage contains a hybrid search
that records its own retrieval and the agent's generation stage contains the
searches its tools run, and neither is counted twice, so the stages of a
request never add up to more than its wall time.

:func:`stage` can also open the tracing span for the step and emits one
structured log line per stage (request id included by the log formatter).
Neither carries question or chunk text: only the stage name, its latency and
whatever counts the caller attaches to the span.
"""
from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from app.logging_config import get_structured_logger
from app.tracing import span as trace_span
from app.tracing.spans import Span, current_strategy

logger = get_structured_logger(__name__)


@dataclass
class StageTimer:
    """Accumulates the exclusive time spent in each named stage of one request."""

    durations_ms: dict[str, float] = field(default_factory=dict)
    started: float = field(default_factory=time.perf_counter)

    def add(self, stage: str, ms: float) -> None:
        self.durations_ms[stage] = self.durations_ms.get(stage, 0.0) + max(0.0, ms)

    def get(self, stage: str) -> float:
        return round(self.durations_ms.get(stage, 0.0), 3)

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started) * 1000


@dataclass
class _Frame:
    """An open stage; ``child_ms`` is the time its nested stages took."""

    name: str
    child_ms: float = 0.0


_timer: ContextVar[StageTimer | None] = ContextVar("evalrag_stage_timer", default=None)
_frame: ContextVar[_Frame | None] = ContextVar("evalrag_stage_frame", default=None)


def current_timer() -> StageTimer | None:
    return _timer.get()


@contextmanager
def activate(timer: StageTimer | None = None) -> Iterator[StageTimer]:
    """Make ``timer`` (or a new one) the current request's timer."""
    timer = timer or StageTimer()
    timer_token = _timer.set(timer)
    frame_token = _frame.set(None)
    try:
        yield timer
    finally:
        _frame.reset(frame_token)
        _timer.reset(timer_token)


def record(stage: str, ms: float) -> None:
    """Attribute ``ms`` of already-measured time to ``stage``.

    For code that times its own sub-steps (hybrid retrieval): the time is also
    deducted from the enclosing stage, exactly as a nested :func:`stage` is.
    """
    timer = _timer.get()
    if timer is None:
        return
    timer.add(stage, ms)
    parent = _frame.get()
    if parent is not None:
        parent.child_ms += max(0.0, ms)


@contextmanager
def stage(
    name: str,
    *,
    span_name: str | None = None,
    as_type: str = "span",
    log: bool = True,
) -> Iterator[Span | None]:
    """Time one stage of the request, optionally as a tracing span.

    Args:
        name: Timer bucket, e.g. ``generation`` or ``citation_validation``.
        span_name: Also open a span with this name on the current trace. The
            span is yielded (None outside a trace) so the caller can attach
            counts; its ``latency_ms`` is set on exit.
        as_type: Langfuse observation type of the span.
        log: Emit a structured "Pipeline stage completed" line on exit.
    """
    parent = _frame.get()
    frame = _Frame(name)
    token = _frame.set(frame)
    started = time.perf_counter()
    s: Span | None = None
    try:
        if span_name is None:
            yield None
        else:
            with trace_span(span_name, as_type=as_type) as s:
                yield s
                if s is not None:
                    s.metadata["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    finally:
        _frame.reset(token)
        elapsed = (time.perf_counter() - started) * 1000
        timer = _timer.get()
        if timer is not None:
            timer.add(name, elapsed - frame.child_ms)
        if parent is not None:
            parent.child_ms += elapsed
        if log:
            logger.info(
                "Pipeline stage completed",
                extra_fields={
                    "stage": name,
                    "strategy": current_strategy(),
                    "latency_ms": round(elapsed, 3),
                    **(_safe_attributes(s) if s is not None else {}),
                },
            )


def _safe_attributes(s: Span) -> dict:
    """Scalar span attributes for the log line (counts and flags only)."""
    return {
        k: v
        for k, v in s.metadata.items()
        if k != "latency_ms" and isinstance(v, bool | int | float)
    }
