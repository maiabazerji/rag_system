"""Prometheus metrics.

Everything here is aggregate and content-free: counts, durations, token totals
and states, labelled by route, strategy and model. No question, answer, key or
user identifier is ever a label. Served by ``GET /metrics`` (see
app/api/metrics.py) when METRICS_ENABLED is on.

The metrics live in a dedicated registry rather than prometheus_client's global
one, so importing a library that registers its own collectors cannot change
what this endpoint exposes.
"""
from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    ProcessCollector,
)
from prometheus_client.core import GaugeMetricFamily, Metric
from prometheus_client.registry import Collector

from app.resilience import registered_breakers
from app.tracing.spans import Trace

REGISTRY = CollectorRegistry(auto_describe=True)
ProcessCollector(registry=REGISTRY)

HTTP_REQUESTS = Counter(
    "evalrag_http_requests_total",
    "HTTP requests handled, by route template and status code.",
    ["method", "route", "status"],
    registry=REGISTRY,
)
HTTP_LATENCY = Histogram(
    "evalrag_http_request_duration_seconds",
    "HTTP request latency, by route template.",
    ["method", "route"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60, 120),
    registry=REGISTRY,
)
STRATEGY_LATENCY = Histogram(
    "evalrag_strategy_duration_seconds",
    "Wall time of one strategy run, by strategy and outcome (ok, refusal, error).",
    ["strategy", "outcome"],
    buckets=(0.5, 1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180),
    registry=REGISTRY,
)
LLM_TOKENS = Counter(
    "evalrag_llm_tokens_total",
    "Tokens consumed by model calls, by model, strategy and direction (input/output).",
    ["model", "strategy", "direction"],
    registry=REGISTRY,
)
LLM_COST = Counter(
    "evalrag_llm_cost_usd_total",
    "Estimated USD cost of model calls, by model and strategy. Each call is priced "
    "with the model that served it; calls to unpriced models are not included.",
    ["model", "strategy"],
    registry=REGISTRY,
)
LLM_UNPRICED_CALLS = Counter(
    "evalrag_llm_unpriced_calls_total",
    "Model calls whose model has no listed price, so no cost was estimated.",
    ["model", "strategy"],
    registry=REGISTRY,
)
STAGE_LATENCY = Histogram(
    "evalrag_stage_duration_seconds",
    "Exclusive wall time of one pipeline stage in a request, by strategy and stage "
    "(retrieval, rerank, generation, citation_validation).",
    ["strategy", "stage"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120),
    registry=REGISTRY,
)
CONTEXT_CHUNKS = Histogram(
    "evalrag_context_chunks",
    "Chunks in the context an answer was generated from, by strategy.",
    ["strategy"],
    buckets=(0, 1, 2, 4, 6, 8, 10, 12, 16, 24, 32, 50),
    registry=REGISTRY,
)
REQUEST_COST = Histogram(
    "evalrag_request_cost_usd",
    "Estimated USD cost of one strategy run, by strategy (priced runs only).",
    ["strategy"],
    buckets=(0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1),
    registry=REGISTRY,
)
REFUSALS = Counter(
    "evalrag_refusals_total",
    "Answers returned as refusals, by strategy and reason.",
    ["strategy", "reason"],
    registry=REGISTRY,
)
TELEMETRY_DROPPED = Counter(
    "evalrag_telemetry_dropped_total",
    "Traces that were not exported, by exporter and reason (queue_full, export_failed).",
    ["exporter", "reason"],
    registry=REGISTRY,
)
TELEMETRY_EXPORTED = Counter(
    "evalrag_telemetry_exported_total",
    "Traces delivered to an exporter backend.",
    ["exporter"],
    registry=REGISTRY,
)

# 0 closed, 1 half-open, 2 open: higher is worse, so one alert threshold works.
_BREAKER_STATE_VALUES = {"closed": 0, "half_open": 1, "open": 2}


class CircuitBreakerCollector(Collector):
    """Reads every live circuit breaker at scrape time."""

    def collect(self) -> Iterable[Metric]:
        state = GaugeMetricFamily(
            "evalrag_circuit_breaker_state",
            "Circuit breaker state per service: 0 closed, 1 half-open, 2 open.",
            labels=["service"],
        )
        failures = GaugeMetricFamily(
            "evalrag_circuit_breaker_failures",
            "Consecutive failures recorded by each circuit breaker.",
            labels=["service"],
        )
        for breaker in registered_breakers():
            state.add_metric(
                [breaker.service_name], _BREAKER_STATE_VALUES.get(breaker.state, 2)
            )
            failures.add_metric([breaker.service_name], breaker.failure_count)
        yield state
        yield failures


REGISTRY.register(CircuitBreakerCollector())


def observe_tokens(model: str, strategy: str, input_tokens: int, output_tokens: int) -> None:
    """Count the tokens of one model call."""
    if input_tokens:
        LLM_TOKENS.labels(model, strategy, "input").inc(input_tokens)
    if output_tokens:
        LLM_TOKENS.labels(model, strategy, "output").inc(output_tokens)


def observe_cost(model: str, strategy: str, cost_usd: float | None) -> None:
    """Count the estimated cost of one model call (None: the model is unpriced)."""
    if cost_usd is None:
        LLM_UNPRICED_CALLS.labels(model, strategy).inc()
    elif cost_usd > 0:
        LLM_COST.labels(model, strategy).inc(cost_usd)


# Stages with their own latency histogram series. "other" is left out: it is
# the remainder, and its distribution says little on its own.
OBSERVED_STAGES = ("retrieval", "rerank", "generation", "citation_validation")


def observe_request_metrics(metrics: Any) -> None:
    """Record a finished request's stage latencies, context size and cost.

    Takes a :class:`app.schemas.RequestMetrics`; typed loosely so this module
    stays free of the schema import.
    """
    strategy = metrics.strategy
    for stage in OBSERVED_STAGES:
        ms = getattr(metrics.latency_ms, stage, 0.0)
        if ms:
            STAGE_LATENCY.labels(strategy, stage).observe(ms / 1000)
    CONTEXT_CHUNKS.labels(strategy).observe(metrics.context_chunks)
    if metrics.estimated_cost_usd is not None:
        REQUEST_COST.labels(strategy).observe(metrics.estimated_cost_usd)


def observe_refusal(strategy: str, reason: str) -> None:
    """Count a refusal, including those returned before any trace opens."""
    REFUSALS.labels(strategy, reason).inc()


def observe_trace(t: Trace) -> None:
    """Record latency and refusals for a finished trace."""
    STRATEGY_LATENCY.labels(t.strategy, t.outcome).observe(t.duration_seconds)
    if t.outcome != "ok":
        observe_refusal(t.strategy, t.refusal_reason or t.outcome)


def observe_dropped(exporter: str, reason: str, n: int = 1) -> None:
    TELEMETRY_DROPPED.labels(exporter, reason).inc(n)


def observe_exported(exporter: str, n: int = 1) -> None:
    TELEMETRY_EXPORTED.labels(exporter).inc(n)


def route_template(scope: dict) -> str:
    """The matched route's path template, e.g. ``/traces/{trace_id}``.

    Rebuilt from the request path and its path parameters rather than read off
    the route object: with ``include_router(prefix=...)`` the route object's
    own path lacks the prefix on some FastAPI versions.
    """
    if "route" not in scope and "endpoint" not in scope:
        return "unmatched"
    params = {str(v): k for k, v in (scope.get("path_params") or {}).items()}
    segments = scope.get("path", "").split("/")
    return "/".join(f"{{{params[seg]}}}" if seg in params else seg for seg in segments)


class MetricsMiddleware:
    """Pure ASGI middleware counting requests and their latency per route.

    The label is the route *template* (``/traces/{trace_id}``), never the raw
    path, so ids in URLs cannot explode the series count. Requests that match
    no route share the label ``unmatched``.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        status = {"code": 500}

        async def send_wrapper(message: dict) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
            await send(message)

        start = time.perf_counter()
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            label = route_template(scope)
            method = scope.get("method", "GET")
            HTTP_REQUESTS.labels(method, label, str(status["code"])).inc()
            HTTP_LATENCY.labels(method, label).observe(time.perf_counter() - start)
