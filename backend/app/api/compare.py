import asyncio
import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.rag.generate import answer_question, run_strategy_raw
from app.schemas import CompareRequest, CompareStrategiesRequest, StrategyComparison, Source

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("")
async def compare(req: CompareRequest) -> dict:
    try:
        if not req.question or not req.question.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "Question is required"},
            )

        if not req.variants:
            return JSONResponse(
                status_code=400,
                content={"error": "At least one variant is required"},
            )

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
        results = await asyncio.gather(*tasks, return_exceptions=True)

        safe_results = []
        for r in results:
            if isinstance(r, Exception):
                logger.exception(f"Compare error: {r}")
                safe_results.append({
                    "answer": "An error occurred processing this variant.",
                    "sources": [],
                    "confidence": 0.0,
                    "refusal": True,
                })
            else:
                safe_results.append(r.model_dump())

        return {"question": req.question, "results": safe_results}
    except Exception as e:
        logger.exception(f"Compare endpoint error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to compare variants. Please try again."},
        )


@router.post("/strategies", response_model=dict)
async def compare_strategies(req: CompareStrategiesRequest) -> dict:
    try:
        if not req.question or not req.question.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "Question is required"},
            )

        if not req.strategies:
            return JSONResponse(
                status_code=400,
                content={"error": "At least one strategy is required"},
            )

        async def _one(name: str) -> StrategyComparison:
            try:
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
            except Exception as e:
                error_msg = str(e)
                logger.exception(f"Error running strategy {name}: {error_msg}")
                if "timeout" in error_msg.lower():
                    msg = "API timeout - try again in a moment"
                elif "rate" in error_msg.lower():
                    msg = "Rate limited - running too many requests at once"
                else:
                    msg = f"Error: {error_msg[:100]}"
                return StrategyComparison(
                    strategy=name,
                    question=req.question,
                    answer=msg,
                    sources=[Source(chunk_id="none", quote="")],
                    refusal=True,
                    confidence=0.0,
                    latency_ms=0,
                    input_tokens=0,
                    output_tokens=0,
                    iterations=0,
                )

        results = []
        for s in req.strategies:
            result = await _one(s)
            results.append(result)
            await asyncio.sleep(0.5)
        return {"question": req.question, "results": [r.model_dump() for r in results]}
    except Exception as e:
        logger.exception(f"Compare strategies endpoint error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to compare strategies. Please try again."},
        )
