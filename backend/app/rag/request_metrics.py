"""Build the cost and latency summary of one strategy run.

Inputs are what the orchestrator already has at the end of a run: the
strategy's result (token totals, candidate and context counts), the request's
:class:`~app.rag.timing.StageTimer`, and its trace, whose generation spans are
the ledger of model calls. Each call is priced with the model that served it,
so graph entity extraction (a cheaper model) and agent steps are billed at
their own rates.
"""
from __future__ import annotations

from app import pricing
from app.rag.strategies.base import StrategyResult
from app.rag.timing import StageTimer
from app.schemas import RequestLatency, RequestMetrics
from app.tracing.spans import Trace

# Named latency buckets; everything else a timer saw is folded into "other".
LATENCY_STAGES = ("retrieval", "rerank", "generation", "citation_validation")


def _latency(timer: StageTimer, total_ms: float) -> RequestLatency:
    parts = {name: round(max(0.0, timer.get(name)), 3) for name in LATENCY_STAGES}
    total = round(max(total_ms, 0.0), 3)
    attributed = sum(parts.values())
    if attributed > total:
        # Rounding (or a clock step) can push the parts a hair past the wall
        # time; the parts are the measurement, so the total follows them.
        total = round(attributed, 3)
    return RequestLatency(total=total, other=round(total - attributed, 3), **parts)


def _llm_cost(trace: Trace | None) -> tuple[int, float | None, list[str]]:
    """Calls, estimated cost and unpriced models, from the trace's generation spans."""
    if trace is None:
        return 0, None, []
    calls = [s for s in trace.spans if s.as_type == "generation"]
    total = 0.0
    unpriced: list[str] = []
    for s in calls:
        if s.usage is None:  # the call raised: nothing was billed
            continue
        cost = pricing.estimate_cost(s.model, s.usage.get("input", 0), s.usage.get("output", 0))
        if cost is None:
            unpriced.append(s.model or "unknown")
        else:
            total += cost
    if not calls or unpriced:
        return len(calls), None, sorted(set(unpriced))
    return len(calls), round(total, 8), []


def build_request_metrics(
    result: StrategyResult,
    *,
    strategy: str,
    model: str | None,
    provider: str | None,
    total_ms: float,
    timer: StageTimer,
    trace: Trace | None,
) -> RequestMetrics:
    """Summarise one finished run.

    The cost is None when any call used a model without a listed price (a
    partial sum would understate it) or when no model call was recorded at all.
    """
    llm_calls, cost, unpriced = _llm_cost(trace)
    if llm_calls == 0 and not (result.input_tokens or result.output_tokens):
        cost = 0.0  # nothing was called (e.g. a refusal before generation)
    return RequestMetrics(
        strategy=strategy,
        model=model,
        provider=provider,
        input_tokens=max(0, result.input_tokens),
        output_tokens=max(0, result.output_tokens),
        estimated_cost_usd=cost,
        unpriced_models=unpriced,
        latency_ms=_latency(timer, total_ms),
        retrieved_chunks=max(0, result.candidate_count),
        context_chunks=max(0, result.context_count),
        llm_calls=llm_calls,
    )


def response_attributes(result: StrategyResult, metrics: RequestMetrics) -> dict:
    """Attributes of the ``response`` span: outcome, counts, latency, tokens, cost."""
    return {
        "status": result.status,
        "grounded": result.grounded,
        "refusal": result.refusal,
        "citation_count": result.citation_count,
        "invalid_count": len(result.invalid_citations),
        "latency_ms": metrics.latency_ms.total,
        "input_tokens": metrics.input_tokens,
        "output_tokens": metrics.output_tokens,
        "estimated_cost_usd": metrics.estimated_cost_usd,
        "llm_calls": metrics.llm_calls,
        "context_chunks": metrics.context_chunks,
        "retrieved_chunks": metrics.retrieved_chunks,
    }
