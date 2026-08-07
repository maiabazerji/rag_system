"""Orchestrator: routes questions to RAG strategies with uniform error handling.

This module is the entry point for question answering. It:
1. Validates that documents are indexed
2. Selects a strategy (classic/graph/agentic) based on caller preference
3. Picks a default model/provider if not specified
4. Handles provider errors uniformly (missing API key, service outages)
5. Times execution and collects telemetry
6. Adapts StrategyResult to the public Answer schema for the API

The actual retrieval and generation logic lives in strategy subclasses.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from app.config import settings
from app.logging_config import get_structured_logger
from app.rag.providers import MissingKeyError, ProviderError
from app.rag.response_clean import clean_response
from app.rag.store import count as store_count
from app.rag.strategies import get_strategy
from app.schemas import Answer, Source
from app.tracing import start_trace

logger = get_structured_logger(__name__)


def _refusal(
    question: str,
    message: str,
    sources: Optional[list[Source]] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> Answer:
    """Create a refusal Answer for error cases.

    Constructs an Answer that indicates the system could not answer due to
    missing documents, provider errors, or other issues.

    Args:
        question: The original question that could not be answered.
        message: Human-readable error message explaining the refusal.
        sources: List of sources (empty for refusals). Defaults to
            [Source(chunk_id="none", quote="")].
        provider: Provider name (e.g., "anthropic", "openai"). Defaults to None.
        model: Model ID used (e.g., "claude-opus-4-7"). Defaults to None.

    Returns:
        Answer with refusal=True, confidence=0.0, and the provided message.
    """
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
    """Select a default model based on provider and strategy.

    Different strategies have different model requirements:
    - Agentic RAG requires tool-use capability (expensive models)
    - Classic/Graph RAG work with any generation model

    Providers have different model lineups:
    - Anthropic: Claude models with varying capabilities
    - OpenAI: GPT models
    - Local: Ollama-served models (usually smaller, faster)

    Args:
        provider: Provider identifier (e.g., "anthropic", "openai", "local").
        strategy: RAG strategy name (e.g., "classic", "agentic", "graph").

    Returns:
        Recommended model ID for this provider/strategy combination,
        from settings (defaults if caller didn't specify).
    """
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
    provider: Optional[str] = None,
    model: Optional[str] = None,
    prompt_version: Optional[str] = None,
    strategy: str = "classic",
) -> Answer:
    """Answer a question using the specified RAG strategy.

    This is the primary public API for question answering. It orchestrates:
    1. Document availability checks
    2. Provider/model selection
    3. Strategy execution with error handling
    4. Telemetry collection (latency, token counts)
    5. Response normalization and cleanup

    Callers can specify strategy (classic/graph/agentic), provider (anthropic/openai/local),
    and model. Defaults are filled from settings for any omitted parameters.

    Graph and Agentic strategies require Anthropic's Claude due to tool use and
    entity extraction features. If requested with another provider, they will use
    Anthropic anyway and this is reflected in the returned Answer.provider field.

    Args:
        question: The user's question to answer. Required.
        top_k: Max chunks to include in context (default 8). Ignored by agentic RAG
            (model controls retrieval).
        provider: Provider to use ("anthropic", "openai", "local"). If None, uses
            settings.generator_provider. Ignored for graph/agentic (always Anthropic).
        model: Model ID to use (e.g., "claude-opus-4-7", "gpt-4o-mini"). If None,
            uses defaults based on provider and strategy.
        prompt_version: Prompt template version (e.g., "default", "v2-structured").
            If None, defaults to "default". Ignored by agentic RAG.
        strategy: RAG strategy to use ("classic", "graph", "agentic").
            Defaults to "classic".

    Returns:
        Answer with question, answer text, sources, confidence (0-1), refusal flag,
        provider, and model. Always returns a valid Answer (never raises).

    Raises:
        No exceptions are raised. Provider and strategy errors are caught and
        converted to refusal Answers with error messages.

    Example:
        >>> result = await answer_question(
        ...     "What is the capital of France?",
        ...     strategy="classic",
        ...     model="claude-opus-4-7",
        ... )
        >>> print(result.answer)
        >>> print(f"Confidence: {result.confidence}")
    """
    # Graph and agentic strategies require Anthropic features (tool use, entity extraction).
    # Override provider selection and surface this to caller via returned provider field.
    if strategy in ("graph", "agentic"):
        effective_provider = "anthropic"
    else:
        effective_provider = provider or settings.generator_provider
    model = model or _default_model(effective_provider, strategy)
    prompt_version = prompt_version or "default"

    try:
        doc_count = await store_count()
        if doc_count == 0:
            logger.info(
                "No documents indexed",
                extra_fields={
                    "strategy": strategy,
                    "provider": effective_provider,
                    "model": model,
                },
            )
            return _refusal(
                question,
                "No documents uploaded yet. Go to Ingest to upload files, then I can answer your questions.",
                provider=effective_provider,
                model=model,
            )
    except Exception as e:
        # Qdrant might be slow or unreachable; proceed anyway and let the strategy handle it
        logger.warning(
            f"Document count check failed: {type(e).__name__}: {e}",
            exc_info=False,
            extra_fields={
                "error_type": type(e).__name__,
                "strategy": strategy,
            },
        )

    with start_trace(
        name=f"ask:{strategy}",
        inputs={"question": question, "strategy": strategy, "provider": effective_provider, "model": model},
    ) as trace:
        try:
            strat = get_strategy(strategy)
        except ValueError as e:
            logger.warning(
                f"Unknown strategy: {strategy}",
                extra_fields={"error_type": "invalid_strategy"},
            )
            return _refusal(question, str(e), provider=effective_provider, model=model)

        logger.info(
            "Strategy started",
            extra_fields={
                "strategy": strategy,
                "provider": effective_provider,
                "model": model,
                "question_len": len(question),
                "top_k": top_k,
            },
        )

        t0 = time.perf_counter()
        try:
            result = await strat.run(
                question,
                top_k=top_k,
                model=model,
                prompt_version=prompt_version,
            )
        except (MissingKeyError, ProviderError) as e:
            logger.warning(
                f"Provider error: {type(e).__name__}",
                extra_fields={
                    "error_type": type(e).__name__,
                    "strategy": strategy,
                    "provider": effective_provider,
                },
            )
            return _refusal(question, str(e), provider=effective_provider, model=model)
        except RuntimeError as e:
            # Handle circuit breaker and service unavailable errors
            if "circuit breaker open" in str(e).lower() or "unavailable" in str(e).lower():
                logger.warning(
                    "Service unavailable",
                    extra_fields={
                        "error_type": "service_unavailable",
                        "provider": effective_provider,
                        "strategy": strategy,
                    },
                )
                return _refusal(
                    question,
                    f"The {effective_provider} service is temporarily unavailable. Please try again shortly.",
                    provider=effective_provider,
                    model=model,
                )
            raise
        latency_ms = int((time.perf_counter() - t0) * 1000)
        result.latency_ms = latency_ms

        logger.info(
            "Strategy completed",
            extra_fields={
                "strategy": strategy,
                "provider": effective_provider,
                "model": model,
                "latency_ms": latency_ms,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "iterations": result.iterations,
                "refusal": result.refusal,
                "confidence": result.confidence,
            },
        )

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
            answer=clean_response(result.answer),
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
    model: Optional[str] = None,
    top_k: int = 8,
    prompt_version: str = "default",
) -> tuple[Optional[any], Optional[str]]:
    """Execute a strategy and return raw StrategyResult + telemetry.

    Internal function used by the /compare/strategies endpoint to run multiple
    strategies side-by-side and return full telemetry for comparison UI.

    Unlike answer_question, this returns the raw StrategyResult (before cleanup)
    so callers can access strategy-specific trace and extra data.

    Args:
        question: The question to answer.
        strategy: Strategy name to execute ("classic", "graph", "agentic").
        model: Model to use. If None, defaults to Anthropic model for the strategy.
        top_k: Max chunks in context. Defaults to 8.
        prompt_version: Prompt template version. Defaults to "default".

    Returns:
        Tuple of (StrategyResult or None, error_message or None).
        On success: (StrategyResult, None)
        On error: (None, error_message_string)

    Example:
        >>> result, error = await run_strategy_raw(
        ...     "What is X?",
        ...     strategy="classic",
        ... )
        >>> if error:
        ...     print(f"Strategy failed: {error}")
        >>> else:
        ...     print(f"Answer: {result.answer}")
        ...     print(f"Trace: {result.trace}")
    """
    model = model or _default_model("anthropic", strategy)
    try:
        doc_count = await store_count()
        if doc_count == 0:
            logger.info(
                "No documents indexed for strategy comparison",
                extra_fields={"strategy": strategy},
            )
            return None, "No documents indexed yet."
    except Exception as e:
        # Qdrant might be slow; proceed anyway
        logger.warning(
            f"Document count check failed in strategy comparison: {type(e).__name__}: {e}",
            exc_info=False,
            extra_fields={
                "error_type": type(e).__name__,
                "strategy": strategy,
                "context": "strategy_comparison",
            },
        )
    try:
        strat = get_strategy(strategy)
    except ValueError as e:
        logger.warning(
            f"Unknown strategy in comparison: {strategy}",
            extra_fields={
                "error_type": "invalid_strategy",
                "strategy": strategy,
            },
        )
        return None, str(e)

    logger.info(
        "Strategy comparison started",
        extra_fields={
            "strategy": strategy,
            "model": model,
            "question_len": len(question),
            "top_k": top_k,
        },
    )

    t0 = time.perf_counter()
    try:
        result = await strat.run(
            question, top_k=top_k, model=model, prompt_version=prompt_version
        )
    except (MissingKeyError, ProviderError) as e:
        logger.warning(
            f"Provider error in strategy comparison: {type(e).__name__}",
            extra_fields={
                "error_type": type(e).__name__,
                "strategy": strategy,
                "context": "strategy_comparison",
            },
        )
        return None, str(e)
    except RuntimeError as e:
        # Handle circuit breaker and service unavailable errors
        if "circuit breaker open" in str(e).lower() or "unavailable" in str(e).lower():
            logger.warning(
                "Service unavailable in strategy comparison",
                extra_fields={
                    "error_type": "service_unavailable",
                    "strategy": strategy,
                    "context": "strategy_comparison",
                },
            )
            return None, f"Service temporarily unavailable: {str(e)}"
        return None, str(e)
    result.latency_ms = int((time.perf_counter() - t0) * 1000)

    logger.info(
        "Strategy comparison completed",
        extra_fields={
            "strategy": strategy,
            "model": model,
            "latency_ms": result.latency_ms,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "iterations": result.iterations,
            "refusal": result.refusal,
        },
    )

    return result, None
