import asyncio

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.access import Principal
from app.audit import audited
from app.auth import charge, record_tokens, require_principal, strategy_units
from app.eval.online import (
    GoldenNotFound,
    Reference,
    evaluate_rows,
    evaluation_units,
    resolve_reference,
)
from app.logging_config import get_structured_logger
from app.rag.generate import answer_question, grounding_fields, run_strategy_raw
from app.schemas import (
    CompareRequest,
    CompareStrategiesRequest,
    Source,
    StrategyComparison,
)

logger = get_structured_logger(__name__)
router = APIRouter(dependencies=[Depends(require_principal)])


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
async def compare(
    req: CompareRequest, principal: Principal = Depends(require_principal)
) -> dict:
    """Answer the same question with several variants and return them side by side.

    Args:
        req: Question plus the variants to compare.
        principal: Authenticated caller; retrieval is limited to its scope.

    Returns:
        The question and one result per variant, in request order. A variant
        that fails becomes a refusal row rather than failing the whole request.

    Raises:
        HTTPException: 429 if the variants together exceed the key's rate limit.
    """
    strategies = ",".join(dict.fromkeys(v.strategy for v in req.variants))
    with audited(principal, "compare", strategy=strategies, question=req.question) as event:
        await run_in_threadpool(
            charge, principal, sum(strategy_units(v.strategy) for v in req.variants)
        )
        results = await asyncio.gather(
            *(
                answer_question(
                    question=req.question,
                    provider=v.provider,
                    model=v.model,
                    prompt_version=v.prompt_version,
                    strategy=v.strategy,
                    access=principal.scope(),
                )
                for v in req.variants
            ),
            return_exceptions=True,
        )
        for r in results:
            if not isinstance(r, BaseException):
                event.add_sources(r.sources)

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

    await run_in_threadpool(
        record_tokens, principal, tokens_input=total_in, tokens_output=total_out
    )
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
    req: CompareStrategiesRequest, principal: Principal = Depends(require_principal)
) -> dict:
    """Run one question through several RAG strategies.

    Strategies run sequentially: the agentic strategy issues many model calls,
    and running all three at once reliably trips provider rate limits.

    With ``evaluate=true`` each row also gets an ``evaluation``: LLM-judge
    scores (answer correctness only with a reference answer) and, when the
    relevant documents are known, ranked retrieval metrics. The reference comes
    from ``reference`` or from a golden example (``golden``). Judge calls run
    concurrently after every strategy has answered.

    Args:
        req: Question, strategies to run, an optional model override and the
            optional evaluation settings.
        principal: Authenticated caller; retrieval is limited to its scope.

    Returns:
        The question and one comparison row per strategy, in request order.

    Raises:
        HTTPException: 404 if ``golden`` names a missing dataset or example;
            429 if the strategies (plus one unit per judged strategy) exceed
            the key's rate limit (agentic counts as three).
    """
    reference: Reference | None = None
    if req.evaluate:
        try:
            reference = await run_in_threadpool(resolve_reference, req.reference, req.golden)
        except GoldenNotFound as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    units = sum(strategy_units(s) for s in req.strategies)
    if reference is not None:
        units += evaluation_units(len(req.strategies), reference)

    with audited(
        principal, "compare", strategy=",".join(req.strategies), question=req.question
    ) as event:
        await run_in_threadpool(charge, principal, units)
        results, failed, total_in, total_out = await _run_strategies(req, principal)
        for r in results:
            event.add_sources(r.sources)
        if reference is not None:
            evaluations = await evaluate_rows(results, failed, reference)
            for row, evaluation in zip(results, evaluations, strict=True):
                row.evaluation = evaluation

    await run_in_threadpool(
        record_tokens,
        principal,
        tokens_input=total_in,
        tokens_output=total_out,
        model=req.model,
    )
    return {"question": req.question, "results": [r.model_dump() for r in results]}


async def _run_strategies(
    req: CompareStrategiesRequest, principal: Principal
) -> tuple[list[StrategyComparison], list[bool], int, int]:
    """Run each requested strategy in turn.

    Returns:
        The rows, whether each row is a failure placeholder, and token totals.
    """
    results: list[StrategyComparison] = []
    failed: list[bool] = []
    total_in = total_out = 0

    for name in req.strategies:
        # run_strategy_raw opens (and finishes) the trace for this row.
        try:
            result, err = await run_strategy_raw(
                req.question, strategy=name, model=req.model, access=principal.scope()
            )
        except Exception as e:
            logger.exception(
                f"Strategy {name} raised: {type(e).__name__}: {e}",
                extra_fields={"error_type": type(e).__name__, "strategy": name},
            )
            result, err = None, "This strategy failed. Check the backend logs."

        if err or result is None:
            results.append(
                _failed_comparison(name, req.question, err or "No result returned.")
            )
            failed.append(True)
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
                trace_id=result.trace_id,
                retrieval=result.extra.get("retrieval"),
                metrics=result.metrics,
                **grounding_fields(result),
            )
        )
        failed.append(False)
    return results, failed, total_in, total_out
