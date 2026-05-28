import asyncio

from fastapi import APIRouter

from app.rag.generate import answer_question, run_strategy_raw
from app.schemas import CompareRequest, CompareStrategiesRequest, StrategyComparison, Source

router = APIRouter()


@router.post("")
async def compare(req: CompareRequest) -> dict:
    """Original endpoint — compares prompt/model variants of the *classic* strategy."""
    tasks = [
        answer_question(
            question=req.question,
            provider=v.get("provider"),
            model=v.get("model"),
            prompt_version=v.get("prompt_version"),
            strategy=v.get("strategy", "classic"),
        )
        for v in req.variants
    ]
    results = await asyncio.gather(*tasks)
    return {"question": req.question, "results": [r.model_dump() for r in results]}


@router.post("/strategies", response_model=dict)
async def compare_strategies(req: CompareStrategiesRequest) -> dict:
    """Run the same question through classic / graph / agentic in parallel.

    Returns the full `StrategyResult` for each — answer, sources, latency,
    token usage, iterations, and the reasoning trace. The frontend renders
    these as side-by-side cards.
    """
    async def _one(name: str) -> StrategyComparison:
        result, err = await run_strategy_raw(
            req.question, strategy=name, model=req.model
        )
        if err:
            return StrategyComparison(
                strategy=name,
                question=req.question,
                answer=err,
                sources=[Source(chunk_id="none", quote="")],
                refusal=True,
                confidence=0.0,
                latency_ms=0,
                input_tokens=0,
                output_tokens=0,
                iterations=0,
            )
        return StrategyComparison(
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

    results = await asyncio.gather(*[_one(s) for s in req.strategies])
    return {"question": req.question, "results": [r.model_dump() for r in results]}
