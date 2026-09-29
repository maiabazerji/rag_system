"""Langfuse trace exporter: OTLP/HTTP JSON, off the request path.

Why not the ``langfuse`` SDK: its public API was rewritten between v2 and v3
(and v3 is itself an OpenTelemetry wrapper), and pulling in the OpenTelemetry
SDK for one exporter is a lot of dependency for little gain. Langfuse's stable
ingestion surface is its OpenTelemetry endpoint, ``POST
/api/public/otel/v1/traces``, which accepts OTLP JSON with Basic auth
(public key : secret key). That is what this module speaks, over httpx.

Guarantees:

- **Never blocks or fails a request.** :func:`export` only puts the trace on a
  bounded queue. A background thread builds payloads, redacts them and sends
  them in batches.
- **Bounded memory.** When the queue is full the trace is dropped and counted
  (``evalrag_telemetry_dropped_total{reason="queue_full"}``).
- **Flushed on shutdown.** The app lifespan calls :func:`shutdown`, which
  drains what it can within a timeout.
- **Policy-gated.** Nothing is created unless :mod:`app.tracing.policy` allows
  Langfuse for the current settings.
- **Redacted.** Every text field passes through the redactor hook
  (:func:`set_redactor`) before it is serialised. With
  TELEMETRY_INCLUDE_CONTENT=false no question, answer, prompt or context text
  is sent at all.
"""
from __future__ import annotations

import atexit
import base64
import importlib.metadata
import json
import logging
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

import httpx

from app import monitoring
from app.tracing import policy
from app.tracing.spans import Span, Trace
from app.tracing.stages import stage_of

logger = logging.getLogger(__name__)

EXPORTER_NAME = "langfuse"
OTLP_PATH = "/api/public/otel/v1/traces"
SCOPE_NAME = "evalrag"
# Long strings are truncated before export: a 50-chunk context can be hundreds
# of KB, and Langfuse rejects bodies over a few MB.
MAX_TEXT_CHARS = 20_000

# ----------------------------------------------------------------------------
# Redaction hook
# ----------------------------------------------------------------------------

Redactor = Callable[[str], str]


def _identity(text: str) -> str:
    return text


_redactor: Redactor = _identity


def set_redactor(fn: Redactor | None) -> None:
    """Install the function applied to every exported text field.

    The privacy module wires ``app.privacy.pii.redact_for_telemetry`` here.
    Passing None restores the identity function.
    """
    global _redactor
    _redactor = fn or _identity


def _redact(value: Any) -> Any:
    """Redact and truncate strings, recursively through lists and dicts."""
    if isinstance(value, str):
        text = _redactor(value)
        if len(text) > MAX_TEXT_CHARS:
            text = text[:MAX_TEXT_CHARS] + f"... [truncated {len(text) - MAX_TEXT_CHARS} chars]"
        return text
    if isinstance(value, dict):
        return {k: _redact(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_redact(v) for v in value]
    return value


# ----------------------------------------------------------------------------
# OTLP JSON payload
# ----------------------------------------------------------------------------


def _attr(key: str, value: Any) -> dict:
    """One OTLP KeyValue. Non-scalar values are sent as JSON strings."""
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    if isinstance(value, str):
        return {"key": key, "value": {"stringValue": value}}
    return {"key": key, "value": {"stringValue": json.dumps(value, default=str)}}


def _attrs(pairs: dict[str, Any]) -> list[dict]:
    return [_attr(k, v) for k, v in pairs.items() if v is not None]


def _content(value: Any, include_content: bool) -> Any:
    """Content fields are redacted, or omitted entirely when capture is off."""
    if not include_content or value is None:
        return None
    return _redact(value)


def _status(error: str | None) -> dict:
    # OTLP status codes: 1 OK, 2 ERROR.
    return {"code": 2, "message": error} if error else {"code": 1}


def _span_to_otlp(s: Span, trace: Trace, include_content: bool) -> dict:
    pairs: dict[str, Any] = {
        "langfuse.observation.type": s.as_type,
        "langfuse.observation.input": _content(s.input, include_content),
        "langfuse.observation.output": _content(s.output, include_content),
        "langfuse.observation.metadata": s.metadata or None,
    }
    if s.model:
        pairs["langfuse.observation.model.name"] = s.model
        pairs["gen_ai.request.model"] = s.model
    if s.usage:
        pairs["langfuse.observation.usage_details"] = s.usage
    if s.cost:
        # Langfuse takes the cost as given rather than re-deriving it from its
        # own model price list, so the UI shows what the app estimated.
        pairs["langfuse.observation.cost_details"] = s.cost
    stage = stage_of(s)
    if stage:
        pairs["evalrag.stage"] = stage
    if s.error:
        pairs["langfuse.observation.level"] = "ERROR"
        pairs["langfuse.observation.status_message"] = s.error
    return {
        "traceId": trace.otel_trace_id,
        "spanId": s.id,
        "parentSpanId": s.parent_id or trace.root_span_id,
        "name": s.name,
        "kind": 1,
        "startTimeUnixNano": str(s.start_ns),
        "endTimeUnixNano": str(s.end_ns or s.start_ns),
        "attributes": _attrs(pairs),
        "status": _status(s.error),
    }


def _root_span(trace: Trace, include_content: bool) -> dict:
    """The trace itself, as the root span carrying the trace-level attributes."""
    inputs = dict(trace.inputs)
    question = inputs.pop("question", None)
    output = dict(trace.output or {})
    answer = output.pop("answer", None) if "answer" in output else output.pop("error", None)
    usage = {
        "input": output.get("input_tokens") or 0,
        "output": output.get("output_tokens") or 0,
    }
    metadata = {
        **inputs,
        **{k: v for k, v in output.items() if v is not None},
        "outcome": trace.outcome,
        "refusal_reason": trace.refusal_reason,
        "evalrag_trace_id": trace.id,
        "request_id": trace.request_id,
    }
    error = trace.refusal_reason if trace.outcome == "error" else None
    pairs: dict[str, Any] = {
        "langfuse.trace.name": trace.name,
        "langfuse.trace.tags": [t for t in (trace.strategy, trace.outcome) if t],
        "langfuse.trace.input": _content(question, include_content),
        "langfuse.trace.output": _content(answer, include_content),
        "langfuse.trace.metadata": {k: v for k, v in metadata.items() if v is not None},
        "langfuse.observation.type": "span",
        "langfuse.observation.input": _content(question, include_content),
        "langfuse.observation.output": _content(answer, include_content),
        "langfuse.observation.metadata": {"total_usage": usage},
    }
    if error:
        pairs["langfuse.observation.level"] = "ERROR"
        pairs["langfuse.observation.status_message"] = error
    return {
        "traceId": trace.otel_trace_id,
        "spanId": trace.root_span_id,
        "name": trace.name,
        "kind": 2,  # SERVER
        "startTimeUnixNano": str(trace.start_ns),
        "endTimeUnixNano": str(trace.end_ns or time.time_ns()),
        "attributes": _attrs(pairs),
        "status": _status(error),
    }


def build_payload(
    traces: list[Trace], *, include_content: bool = True, service_version: str = ""
) -> dict:
    """Build one OTLP ExportTraceServiceRequest (JSON form) for a batch."""
    spans: list[dict] = []
    for t in traces:
        spans.append(_root_span(t, include_content))
        spans.extend(_span_to_otlp(s, t, include_content) for s in t.spans)
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": _attrs(
                        {"service.name": "evalrag", "service.version": service_version or None}
                    )
                },
                "scopeSpans": [{"scope": {"name": SCOPE_NAME}, "spans": spans}],
            }
        ]
    }


# ----------------------------------------------------------------------------
# Exporter
# ----------------------------------------------------------------------------


class LangfuseExporter:
    """Background, bounded, batching exporter.

    Args:
        host: Langfuse base URL.
        public_key: Project public key.
        secret_key: Project secret key.
        max_queue: Traces buffered before new ones are dropped.
        batch_size: Traces per HTTP request.
        flush_interval: Seconds a partial batch waits before it is sent.
        timeout: HTTP timeout per request, in seconds.
        include_content: Export text fields (after redaction).
        transport: Optional httpx transport, for tests.
    """

    def __init__(
        self,
        *,
        host: str,
        public_key: str,
        secret_key: str,
        max_queue: int = 1000,
        batch_size: int = 20,
        flush_interval: float = 2.0,
        timeout: float = 5.0,
        include_content: bool = True,
        service_version: str = "",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.endpoint = host.rstrip("/") + OTLP_PATH
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.include_content = include_content
        self.service_version = service_version
        self.dropped = 0
        self.exported = 0
        self._queue: queue.Queue[Trace] = queue.Queue(maxsize=max_queue)
        token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        self._client = httpx.Client(
            timeout=timeout,
            transport=transport,
            headers={"Authorization": f"Basic {token}", "Content-Type": "application/json"},
        )
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._failures_logged = 0

    # -- producer side (request path) ---------------------------------------

    def submit(self, trace: Trace) -> bool:
        """Queue a finished trace. Returns False, and counts a drop, if full."""
        if self._stop.is_set():
            return False
        self._ensure_worker()
        try:
            self._queue.put_nowait(trace)
            return True
        except queue.Full:
            self._drop("queue_full")
            return False

    # -- worker side --------------------------------------------------------

    def _ensure_worker(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run, name="langfuse-exporter", daemon=True
                )
                self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            batch = self._next_batch(self.flush_interval)
            if batch:
                self._send(batch)

    def _next_batch(self, wait: float) -> list[Trace]:
        """Block up to ``wait`` for a first item, then take what is ready."""
        try:
            batch = [self._queue.get(timeout=wait)]
        except queue.Empty:
            return []
        while len(batch) < self.batch_size:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return batch

    def _send(self, batch: list[Trace]) -> None:
        try:
            payload = build_payload(
                batch,
                include_content=self.include_content,
                service_version=self.service_version,
            )
            resp = self._client.post(self.endpoint, content=json.dumps(payload, default=str))
            if resp.status_code >= 300:
                raise httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}: {resp.text[:200]}",
                    request=resp.request,
                    response=resp,
                )
        except Exception as e:
            self._drop("export_failed", len(batch))
            # A down Langfuse would otherwise log once per batch forever.
            if self._failures_logged < 5 or self._failures_logged % 100 == 0:
                logger.warning(
                    "Langfuse export failed (%s traces dropped): %s: %s",
                    len(batch),
                    type(e).__name__,
                    e,
                )
            self._failures_logged += 1
            return
        self.exported += len(batch)
        monitoring.observe_exported(EXPORTER_NAME, len(batch))

    def _drop(self, reason: str, n: int = 1) -> None:
        self.dropped += n
        monitoring.observe_dropped(EXPORTER_NAME, reason, n)

    # -- lifecycle ----------------------------------------------------------

    def flush(self, timeout: float = 5.0) -> None:
        """Send everything queued, from the calling thread, within ``timeout``.

        Used on shutdown, after the worker has stopped: whatever is still
        queued past the deadline is counted as dropped.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            batch = self._next_batch(0)
            if not batch:
                return
            self._send(batch)
        remaining = self._queue.qsize()
        if remaining:
            self._drop("export_failed", remaining)

    def shutdown(self, timeout: float = 5.0) -> None:
        """Stop the worker, flush what is left, and close the HTTP client."""
        if self._stop.is_set():
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.flush_interval + 1)
        self.flush(timeout)
        self._client.close()


# ----------------------------------------------------------------------------
# Process-wide singleton
# ----------------------------------------------------------------------------

def _package_version() -> str:
    try:
        return importlib.metadata.version("evalrag")
    except importlib.metadata.PackageNotFoundError:
        return ""


_exporter: LangfuseExporter | None = None
_configured = False
_config_lock = threading.Lock()


def configure(
    s: Any | None = None, *, transport: httpx.BaseTransport | None = None
) -> LangfuseExporter | None:
    """(Re)build the exporter from settings and the telemetry policy.

    Called from the app lifespan. Also called lazily by :func:`export` so that
    scripts and tests that never run the lifespan still follow the policy.
    """
    global _exporter, _configured
    s = s if s is not None else policy._app_settings()
    with _config_lock:
        if _exporter is not None:
            _exporter.shutdown(timeout=1.0)
            _exporter = None
        decision = policy.evaluate(s).langfuse
        if decision.enabled:
            _exporter = LangfuseExporter(
                host=s.langfuse_host,
                public_key=s.langfuse_public_key,
                secret_key=s.langfuse_secret_key,
                max_queue=s.telemetry_queue_size,
                include_content=s.telemetry_include_content,
                service_version=_package_version(),
                transport=transport,
            )
        _configured = True
        return _exporter


def get_exporter() -> LangfuseExporter | None:
    """The configured exporter, or None when Langfuse is not allowed."""
    if not _configured:
        configure()
    return _exporter


def export(trace: Trace) -> bool:
    """Queue a trace for export if Langfuse is enabled. Never raises."""
    exporter = get_exporter()
    return exporter.submit(trace) if exporter is not None else False


def shutdown(timeout: float = 5.0) -> None:
    """Flush and stop the exporter. Safe to call more than once."""
    global _exporter, _configured
    with _config_lock:
        if _exporter is not None:
            _exporter.shutdown(timeout)
        _exporter = None
        _configured = False


atexit.register(shutdown, 2.0)
