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

import time

from app import monitoring
from app.config import settings
from app.logging_config import get_structured_logger
from app.prompts.loader import UnknownPromptVersionError
from app.rag.providers import MissingKeyError, ProviderError
from app.rag.response_clean import clean_response
from app.rag.store import count as store_count
from app.rag.strategies import get_strategy
from app.rag.strategies.base import StrategyResult
from app.schemas import Answer, Source
from app.tracing import start_trace

logger = get_structured_logger(__name__)

# Shown instead of a provider exception's text, which can carry upstream
# response bodies, hostnames or request details. The details go to the logs.
PROVIDER_UNAVAILABLE_MESSAGE = (
    "The language model provider is unavailable right now. Try again shortly; "
    "the details are in the backend logs."
)


def _is_store_unavailable(exc: BaseException) -> bool:
    """True for the vector store's own "unavailable" error, which the app maps to 503.

    Matched by name so this module does not depend on where the store defines it.
    """
    return type(exc).__name__ == "StoreUnavailable"


def public_provider_error(exc: ProviderError) -> str:
    """The client-safe message for a provider failure.

    A missing API key is a configuration fault whose message this codebase
    writes itself and which tells the operator exactly what to fix, so it is
    passed through. Every other provider error gets a generic message.
    """
    if isinstance(exc, MissingKeyError):
        return str(exc)
    return PROVIDER_UNAVAILABLE_MESSAGE


def _refusal(
    question: str,
    message: str,
    sources: list[Source] | None = None,
    provider: str | None = None,
    model: str | None = None,
    trace_id: str | None = None,
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
        model: Model ID used (e.g., "claude-sonnet-5"). Defaults to None.

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
        trace_id=trace_id,
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
    top_k: int | None = None,
    provider: str | None = None,
    model: str | None = None,
    prompt_version: str | None = None,
    strategy: str = "classic",
) -> Answer:
    """Answer a question using the specified RAG strategy.

    Thin wrapper over :func:`answer_question_detailed` for callers that only
    need the public :class:`Answer`.

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
        model: Model ID to use (e.g., "claude-sonnet-5", "gpt-4o-mini"). If None,
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
        ...     model="claude-sonnet-5",
        ... )
        >>> print(result.answer)
        >>> print(f"Confidence: {result.confidence}")
    """
    answer, _ = await answer_question_detailed(
        question,
        top_k=top_k,
        provider=provider,
        model=model,
        prompt_version=prompt_version,
        strategy=strategy,
    )
    return answer


async def answer_question_detailed(
    question: str,
    top_k: int | None = None,
    provider: str | None = None,
    model: str | None = None,
    prompt_version: str | None = None,
    strategy: str = "classic",
) -> tuple[Answer, StrategyResult | None]:
    """Like :func:`answer_question`, but also return the raw StrategyResult.

    The evaluation harness needs what the public Answer leaves out: the full
    context the generator saw and the ranked retrieved chunk ids (both in
    ``StrategyResult.extra``).

    Returns:
        ``(answer, result)``; ``result`` is ``None`` when the strategy never ran
        or failed.
    """
    # Graph and agentic strategies require Anthropic features (tool use, entity extraction).
    # Override provider selection and surface this to caller via returned provider field.
    if strategy in ("graph", "agentic"):
        effective_provider = "anthropic"
    else:
        effective_provider = provider or settings.generator_provider
    model = model or _default_model(effective_provider, strategy)
    prompt_version = prompt_version or "default"
    top_k = settings.rerank_top_k if top_k is None else top_k

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
            monitoring.observe_refusal(strategy, "no_documents")
            return _refusal(
                question,
                "No documents uploaded yet. Go to Ingest to upload files, then I can answer your questions.",
                provider=effective_provider,
                model=model,
            ), None
    except Exception as e:
        if _is_store_unavailable(e):
            raise
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
            return trace.finish(
                _refusal(
                    question, str(e), provider=effective_provider, model=model, trace_id=trace.id
                ),
                reason="invalid_strategy",
            ), None
        # Only classic honours a non-Anthropic provider; see Strategy.provider.
        strat.provider = effective_provider

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
                f"Provider error: {type(e).__name__}: {e}",
                extra_fields={
                    "error_type": type(e).__name__,
                    "strategy": strategy,
                    "provider": effective_provider,
                },
            )
            return trace.finish(
                _refusal(
                    question,
                    public_provider_error(e),
                    provider=effective_provider,
                    model=model,
                    trace_id=trace.id,
                ),
                reason="provider_error",
            ), None
        except UnknownPromptVersionError as e:
            return trace.finish(
                _refusal(
                    question, str(e), provider=effective_provider, model=model, trace_id=trace.id
                ),
                reason="invalid_prompt_version",
            ), None
        except Exception as e:
            logger.exception(
                f"Strategy '{strategy}' failed: {type(e).__name__}: {e}",
                extra_fields={
                    "error_type": type(e).__name__,
                    "strategy": strategy,
                    "provider": effective_provider,
                    "model": model,
                },
            )
            return trace.finish(
                _refusal(
                    question,
                    "Something went wrong answering this question. "
                    "The details are in the backend logs.",
                    provider=effective_provider,
                    model=model,
                    trace_id=trace.id,
                ),
                reason="error",
            ), None
        latency_ms = int((time.perf_counter() - t0) * 1000)
        result.latency_ms = latency_ms
        result.trace_id = trace.id

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
        # Which chunks grounded the answer, so an erasure can find this trace.
        trace.log("sources", {"chunk_ids": [src.chunk_id for src in result.sources]})
        trace.log(
            "result",
            {
                "latency_ms": latency_ms,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "iterations": result.iterations,
            },
        )

        answer = Answer(
            question=question,
            answer=clean_response(result.answer),
            sources=result.sources,
            confidence=result.confidence,
            refusal=result.refusal,
            provider=effective_provider,
            model=model,
            latency_ms=latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            trace_id=trace.id,
        )
        return trace.finish(answer), result


async def run_strategy_raw(
    question: str,
    *,
    strategy: str,
    model: str | None = None,
    top_k: int | None = None,
    prompt_version: str = "default",
) -> tuple[StrategyResult | None, str | None]:
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
    top_k = settings.rerank_top_k if top_k is None else top_k
    try:
        doc_count = await store_count()
        if doc_count == 0:
            logger.info(
                "No documents indexed for strategy comparison",
                extra_fields={"strategy": strategy},
            )
            return None, "No documents indexed yet."
    except Exception as e:
        if _is_store_unavailable(e):
            raise
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

    with start_trace(
        name=f"compare:{strategy}",
        inputs={"question": question, "strategy": strategy, "provider": "anthropic", "model": model},
    ) as trace:
        t0 = time.perf_counter()
        try:
            result = await strat.run(
                question, top_k=top_k, model=model, prompt_version=prompt_version
            )
        except (MissingKeyError, ProviderError) as e:
            logger.warning(
                f"Provider error in strategy comparison: {type(e).__name__}: {e}",
                extra_fields={
                    "error_type": type(e).__name__,
                    "strategy": strategy,
                    "context": "strategy_comparison",
                },
            )
            trace.log("error", {"error_type": type(e).__name__})
            trace.fail(public_provider_error(e), reason="provider_error")
            return None, public_provider_error(e)
        except UnknownPromptVersionError as e:
            trace.fail(str(e), reason="invalid_prompt_version")
            return None, str(e)
        except Exception as e:
            logger.exception(
                f"Strategy '{strategy}' failed during comparison: {type(e).__name__}: {e}",
                extra_fields={
                    "error_type": type(e).__name__,
                    "strategy": strategy,
                    "context": "strategy_comparison",
                },
            )
            trace.log("error", {"error_type": type(e).__name__})
            trace.fail(type(e).__name__)
            return None, f"This strategy failed ({type(e).__name__}). See the backend logs."

        result.latency_ms = int((time.perf_counter() - t0) * 1000)
        result.trace_id = trace.id
        for event in result.trace:
            trace.log(event.get("step", "step"), event)
        trace.log(
            "result",
            {
                "latency_ms": result.latency_ms,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "iterations": result.iterations,
            },
        )
        trace.finish(result)

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
