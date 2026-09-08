from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth import require_api_key
from app.eval.human_ratings import load_all as load_ratings
from app.eval.human_ratings import save as save_rating
from app.eval.metrics import run_evaluation
from app.eval.regression import get_run, list_run_summaries, load_regressions
from app.schemas import EvalRunRequest, HumanRating, HumanRatingRequest

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post(
    "/run",
    summary="Run an evaluation",
    description=(
        "Runs a golden dataset through the RAG pipeline and scores each answer "
        "with the LLM judge. Examples the judge could not score are reported "
        "separately and excluded from the aggregate."
    ),
)
async def run(req: EvalRunRequest) -> dict:
    """Evaluate a golden dataset.

    Args:
        req: Dataset name and optional provider/model/prompt overrides.

    Returns:
        Run summary with judge scores, deterministic retrieval scores, cost,
        per-example detail, and the count of examples that could not be scored.
    """
    return await run_evaluation(
        dataset=req.dataset,
        strategy=req.strategy,
        provider=req.provider,
        model=req.model,
        prompt_version=req.prompt_version,
    )


@router.get("/runs", summary="List evaluation runs")
async def runs() -> list[dict]:
    """List past runs as summaries, newest first, without per-example detail."""
    return list_run_summaries()


@router.get("/runs/{run_id}", summary="Get one evaluation run")
async def run_detail(run_id: str) -> dict:
    """Return a single run including its per-example scores.

    Args:
        run_id: Run identifier from GET /eval/runs.

    Raises:
        HTTPException: 404 if no run has that id.
    """
    result = get_run(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No evaluation run '{run_id}'")
    return result


@router.get("/regressions", summary="List detected regressions")
async def regressions(
    threshold: float = Query(
        default=0.05, ge=0, le=1, description="Minimum drop counted as a regression"
    ),
) -> list[dict]:
    """List metric drops between consecutive comparable runs.

    Runs are only compared when they share a dataset, model, provider and
    prompt version, so unrelated runs are never reported as a regression.
    """
    return load_regressions(threshold=threshold)


@router.post("/human-rate", response_model=HumanRating, summary="Record a human rating")
async def human_rate(req: HumanRatingRequest) -> HumanRating:
    """Persist a 1-5 human rating for an answer."""
    return save_rating(req)


@router.get(
    "/human-ratings", response_model=list[HumanRating], summary="List human ratings"
)
async def human_ratings(provider: str | None = None) -> list[HumanRating]:
    """List stored human ratings, optionally filtered by provider."""
    return load_ratings(provider=provider)
