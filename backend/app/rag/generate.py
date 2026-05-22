"""Orchestrator: takes a question and routes it to a RAG strategy.

The strategy does the actual work; this file just:
- picks a default provider/model when the caller didn't,
- starts a trace,
- catches provider errors uniformly,
- times the call (latency_ms) for the comparison UI,
- adapts the strategy's `StrategyResult` to the public `Answer` shape.
"""
from __future__ import annotations

import time

from app.config import settings
from app.rag.providers import MissingKeyError, ProviderError
from app.rag.store import count as store_count
from app.rag.strategies import get_strategy
from app.schemas import Answer, Source
from app.tracing import start_trace


def _refusal(
    question: str,
    message: str,
    sources: list[Source] | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> Answer:
    return Answer(
        question=question,
        answer=message,
        sources=sources or [Source(chunk_id="none", quote="")],
        confidence=0.0,
        refusal=True,
        provider=provider,
        model=model,
    )


def _default_model(provider: str, strategy: str) -> str:
    if provider == "local":
        return settings.ollama_model
    if provider == "openai":
        return settings.openai_generator_model
    if strategy == "agentic":
        return settings.agentic_model
    return settings.generator_model


async def answer_question(
    question: str,
    top_k: int = 8,
    provider: str | None = None,
    model: str | None = None,
    prompt_version: str | None = None,
    strategy: str = "classic",
) -> Answer:
    # The three new strategies all rely on Anthropic features (tool use, cheap
    # graph extraction). We ignore `provider` for graph/agentic and always use
    # Anthropic — surface that to the caller via the returned `provider` field.
    if strategy in ("graph", "agentic"):
        effective_provider = "anthropic"
    else:
        effective_provider = provider or settings.generator_provider
    model = model or _default_model(effective_provider, strategy)
    prompt_version = prompt_version or "default"

    if store_count() == 0:
        return _refusal(
            question,
            "No documents indexed yet. Upload files in the Ingest tab to start asking questions.",
            provider=effective_provider,
            model=model,
        )

    with start_trace(
        name=f"ask:{strategy}",
        inputs={"question": question, "strategy": strategy, "provider": effective_provider, "model": model},
    ) as trace:
        try:
            strat = get_strategy(strategy)
        except ValueError as e:
            return _refusal(question, str(e), provider=effective_provider, model=model)

        t0 = time.perf_counter()
        try:
            result = await strat.run(
                question,
                top_k=top_k,
                model=model,
                prompt_version=prompt_version,
            )
        except (MissingKeyError, ProviderError) as e:
            return _refusal(question, str(e), provider=effective_provider, model=model)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        result.latency_ms = latency_ms

        for event in result.trace:
            trace.log(event.get("step", "step"), event)
        trace.log(
            "result",
            {
                "latency_ms": latency_ms,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "iterations": result.iterations,
            },
        )

        return Answer(
            question=question,
            answer=result.answer,
            sources=result.sources,
            confidence=result.confidence,
            refusal=result.refusal,
            provider=effective_provider,
            model=model,
        )


async def run_strategy_raw(
    question: str,
    *,
    strategy: str,
    model: str | None = None,
    top_k: int = 8,
    prompt_version: str = "default",
):
    """Internal: returns the full `StrategyResult` + telemetry (used by /compare/strategies)."""
    model = model or _default_model("anthropic", strategy)
    if store_count() == 0:
        return None, "No documents indexed yet."
    try:
        strat = get_strategy(strategy)
    except ValueError as e:
        return None, str(e)
    t0 = time.perf_counter()
    try:
        result = await strat.run(
            question, top_k=top_k, model=model, prompt_version=prompt_version
        )
    except (MissingKeyError, ProviderError) as e:
        return None, str(e)
    result.latency_ms = int((time.perf_counter() - t0) * 1000)
    return result, None
