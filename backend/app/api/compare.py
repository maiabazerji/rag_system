import asyncio

from fastapi import APIRouter, Depends

from app.auth import record_tokens, require_api_key
from app.logging_config import get_structured_logger
from app.rag.generate import answer_question, run_strategy_raw
from app.schemas import (
    CompareRequest,
    CompareStrategiesRequest,
    Source,
    StrategyComparison,
)

logger = get_structured_logger(__name__)
router = APIRouter(dependencies=[Depends(require_api_key)])


def _failed_comparison(strategy: str, question: str, message: str) -> StrategyComparison:
    """Build a zero-scored comparison row describing why a strategy failed."""
    return StrategyComparison(
        strategy=strategy,
        question=question,
        answer=message,
        sources=[Source(chunk_id="none", quote="")],
        refusal=True,
        confidence=0.0,
        latency_ms=0,
        input_tokens=0,
        output_tokens=0,
        iterations=0,
    )


@router.post(
    "",
    summary="Compare model and prompt variants",
    description="Runs one question through several provider/model/prompt variants in parallel.",
)
async def compare(req: CompareRequest, auth: dict = Depends(require_api_key)) -> dict:
    """Answer the same question with several variants and return them side by side.

    Args:
        req: Question plus the variants to compare.
        auth: Authenticated principal.

    Returns:
        The question and one result per variant, in request order. A variant
        that fails becomes a refusal row rather than failing the whole request.
    """
    results = await asyncio.gather(
        *(
            answer_question(
                question=req.question,
                provider=v.provider,
                model=v.model,
                prompt_version=v.prompt_version,
                strategy=v.strategy,
            )
            for v in req.variants
        ),
        return_exceptions=True,
    )

    safe_results = []
    total_in = total_out = 0
    for variant, r in zip(req.variants, results, strict=True):
        if isinstance(r, BaseException):
            logger.warning(
                f"Compare variant failed: {type(r).__name__}: {r}",
                extra_fields={
                    "error_type": type(r).__name__,
                    "strategy": variant.strategy,
                    "model": variant.model,
                },
            )
            safe_results.append(
                {
                    "question": req.question,
                    "answer": "This variant could not be run. Check the backend logs.",
                    "sources": [],
                    "confidence": 0.0,
                    "refusal": True,
                    "provider": variant.provider,
                    "model": variant.model,
                    "latency_ms": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            )
        else:
            total_in += r.input_tokens
            total_out += r.output_tokens
            safe_results.append(r.model_dump())

    record_tokens(auth, tokens_input=total_in, tokens_output=total_out)
    return {"question": req.question, "results": safe_results}


@router.post(
    "/strategies",
    summary="Compare RAG strategies",
    description=(
        "Runs the same question through each requested strategy and returns the "
        "answer, sources, latency, token cost and reasoning trace for each."
    ),
)
async def compare_strategies(
    req: CompareStrategiesRequest, auth: dict = Depends(require_api_key)
) -> dict:
    """Run one question through several RAG strategies.

    Strategies run sequentially: the agentic strategy issues many model calls,
    and running all three at once reliably trips provider rate limits.

    Args:
        req: Question, strategies to run, and an optional model override.
        auth: Authenticated principal.

    Returns:
        The question and one comparison row per strategy, in request order.
    """
    results: list[StrategyComparison] = []
    total_in = total_out = 0

    for name in req.strategies:
        try:
            result, err = await run_strategy_raw(
                req.question, strategy=name, model=req.model
            )
        except Exception as e:
            logger.exception(
                f"Strategy {name} raised: {type(e).__name__}: {e}",
                extra_fields={"error_type": type(e).__name__, "strategy": name},
            )
            results.append(
                _failed_comparison(
                    name, req.question, "This strategy failed. Check the backend logs."
                )
            )
            continue

        if err or result is None:
            results.append(
                _failed_comparison(name, req.question, err or "No result returned.")
            )
            continue

        total_in += result.input_tokens
        total_out += result.output_tokens
        results.append(
            StrategyComparison(
                strategy=name,
                question=req.question,
                answer=result.answer,
                sources=result.sources,
                refusal=result.refusal,
                confidence=result.confidence,
                latency_ms=result.latency_ms,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                iterations=result.iterations,
                trace=result.trace,
                extra=result.extra,
            )
        )

    record_tokens(auth, tokens_input=total_in, tokens_output=total_out)
    return {"question": req.question, "results": [r.model_dump() for r in results]}
