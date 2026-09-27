"""Request traces and the timed spans inside them.

A :class:`Trace` covers one question answered by one strategy. While it is open
it is the *current* trace (a context variable, so concurrent requests and
``asyncio.gather`` fan-out never see each other's spans), and anything that runs
inside it can record a :class:`Span`: retrieval, reranking, each model call and
each agent tool call.

Recording is always on and costs a few object allocations per step. Whether a
finished trace leaves the process is decided elsewhere, by the telemetry policy
and the Langfuse exporter. Outside a trace, :func:`span` is a no-op, so
instrumented helpers (the judge, graph extraction) cost nothing when called on
their own.
"""
from __future__ import annotations

import logging
import secrets
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, ParamSpec, TypeVar

from app.middleware.request_id import get_request_id

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def _span_id() -> str:
    """A random 64-bit OpenTelemetry span id, as 16 hex characters."""
    return secrets.token_hex(8)


@dataclass
class Span:
    """One timed step inside a trace.

    Attributes:
        name: Step name, e.g. ``retrieve`` or ``tool:search``.
        as_type: Langfuse observation type: ``span``, ``generation``,
            ``retriever`` or ``tool``.
        parent_id: Enclosing span, or None for a top-level step.
        input: What went in. Exported only after redaction, and only when
            TELEMETRY_INCLUDE_CONTENT is on.
        output: What came out, under the same rules as ``input``.
        model: Model ID, for generations.
        usage: Token counts, for generations: ``{"input": n, "output": m}``.
        metadata: Small non-content facts (counts, stop reasons).
        error: Exception type name when the step raised.
    """

    name: str
    as_type: str = "span"
    parent_id: str | None = None
    id: str = field(default_factory=_span_id)
    start_ns: int = field(default_factory=time.time_ns)
    end_ns: int | None = None
    input: Any = None
    output: Any = None
    model: str | None = None
    usage: dict[str, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def duration_ms(self) -> int:
        end = self.end_ns if self.end_ns is not None else time.time_ns()
        return (end - self.start_ns) // 1_000_000


@dataclass
class Trace:
    """One traced request: its inputs, outcome, events and spans.

    ``events`` is the free-form log served by ``/traces/{id}``; ``spans`` is the
    timed structure the exporter sends to Langfuse.
    """

    id: str
    name: str
    inputs: dict
    events: list[dict] = field(default_factory=list)
    spans: list[Span] = field(default_factory=list)
    root_span_id: str = field(default_factory=_span_id)
    start_ns: int = field(default_factory=time.time_ns)
    end_ns: int | None = None
    output: dict[str, Any] | None = None
    outcome: str = "ok"  # ok | refusal | error
    refusal_reason: str | None = None
    request_id: str | None = None

    @classmethod
    def new(cls, name: str, inputs: dict) -> Trace:
        return cls(
            id=str(uuid.uuid4()), name=name, inputs=inputs, request_id=get_request_id() or None
        )

    @property
    def otel_trace_id(self) -> str:
        """The trace id as 32 hex characters, as OpenTelemetry expects."""
        return self.id.replace("-", "")

    @property
    def strategy(self) -> str:
        return str(self.inputs.get("strategy") or "unknown")

    @property
    def duration_seconds(self) -> float:
        end = self.end_ns if self.end_ns is not None else time.time_ns()
        return (end - self.start_ns) / 1e9

    def log(self, step: str, data: Any) -> None:
        self.events.append({"step": step, "data": data})

    def finish(self, result: R, *, reason: str | None = None) -> R:
        """Record the outcome of the request and return ``result`` unchanged.

        Accepts anything shaped like an ``Answer`` or a ``StrategyResult``
        (``answer``, ``refusal``, token counts), so call sites can write
        ``return trace.finish(answer)``.

        Args:
            result: The answer being returned.
            reason: Why it is a refusal, when the caller knows (for example
                ``provider_error``). A refusal without a reason is the
                strategy's own decision and is recorded as ``model``.
        """
        refusal = bool(getattr(result, "refusal", False))
        self.outcome = "refusal" if refusal else "ok"
        self.refusal_reason = (reason or "model") if refusal else None
        self.output = {
            "answer": getattr(result, "answer", None),
            "refusal": refusal,
            "model": getattr(result, "model", None),
            "input_tokens": getattr(result, "input_tokens", 0),
            "output_tokens": getattr(result, "output_tokens", 0),
            "latency_ms": getattr(result, "latency_ms", None),
            "confidence": getattr(result, "confidence", None),
        }
        return result

    def fail(self, message: str, *, reason: str = "error") -> None:
        """Record that the request produced no answer at all."""
        self.outcome = "error"
        self.refusal_reason = reason
        self.output = {"error": message}


_current_trace: ContextVar[Trace | None] = ContextVar("evalrag_trace", default=None)
_current_span: ContextVar[str | None] = ContextVar("evalrag_span", default=None)


def current_trace() -> Trace | None:
    """The trace open in this context, if any."""
    return _current_trace.get()


def current_strategy() -> str:
    """Strategy of the open trace, or ``none`` outside one (e.g. the judge)."""
    trace = _current_trace.get()
    return trace.strategy if trace is not None else "none"


@contextmanager
def activate(trace: Trace) -> Iterator[Trace]:
    """Make ``trace`` the current trace for the duration of the block."""
    trace_token = _current_trace.set(trace)
    span_token = _current_span.set(None)
    try:
        yield trace
    finally:
        _current_span.reset(span_token)
        _current_trace.reset(trace_token)


@contextmanager
def span(name: str, *, as_type: str = "span", input: Any = None) -> Iterator[Span | None]:
    """Time a step inside the current trace.

    Yields None outside a trace so callers can guard enrichment with
    ``if s is not None``. Exceptions are recorded on the span and re-raised;
    tracing never changes control flow.
    """
    trace = _current_trace.get()
    if trace is None:
        yield None
        return
    s = Span(name=name, as_type=as_type, parent_id=_current_span.get(), input=input)
    trace.spans.append(s)
    token = _current_span.set(s.id)
    try:
        yield s
    except BaseException as e:
        s.error = type(e).__name__
        raise
    finally:
        s.end_ns = time.time_ns()
        _current_span.reset(token)


Describe = Callable[[Span, tuple, dict, Any], None]


def traced(
    name: str, *, as_type: str = "span", describe: Describe | None = None
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Decorate an async function so each call becomes a span.

    Args:
        name: Span name.
        as_type: Langfuse observation type.
        describe: Optional ``(span, args, kwargs, result)`` callback that fills
            in model, usage, input/output or metadata once the call returns.
            Only runs inside a trace, and its own failures are logged, never
            raised.
    """

    def decorator(fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with span(name, as_type=as_type) as s:
                result = await fn(*args, **kwargs)
                if s is not None and describe is not None:
                    try:
                        describe(s, args, kwargs, result)
                    except Exception as e:  # pragma: no cover - defensive
                        logger.debug("Span describe for %s failed: %s", name, e)
                return result

        return wrapper

    return decorator
