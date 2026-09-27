from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from app.auth import charge, record_tokens, require_api_key, strategy_units
from app.eval.human_ratings import load_all as load_ratings
from app.eval.human_ratings import save as save_rating
from app.eval.metrics import load_dataset, run_evaluation
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
async def run(req: EvalRunRequest, auth: dict = Depends(require_api_key)) -> dict:
    """Evaluate a golden dataset.

    Charged one rate-limit unit per example (three for agentic), since every
    example is a full RAG run plus a judge call.

    Args:
        req: Dataset name and optional provider/model/prompt overrides.
        auth: Authenticated principal.

    Returns:
        Run summary with judge scores, deterministic retrieval scores, cost,
        per-example detail, and the count of examples that could not be scored.

    Raises:
        HTTPException: 429 if the run would exceed the key's rate limit.
    """
    n_examples = len(await run_in_threadpool(load_dataset, req.dataset))
    await run_in_threadpool(charge, auth, n_examples * strategy_units(req.strategy))

    result = await run_evaluation(
        dataset=req.dataset,
        strategy=req.strategy,
        provider=req.provider,
        model=req.model,
        prompt_version=req.prompt_version,
    )
    cost = result.get("cost") or {}
    await run_in_threadpool(
        record_tokens,
        auth,
        tokens_input=cost.get("total_input_tokens", 0),
        tokens_output=cost.get("total_output_tokens", 0),
        model=result.get("model"),
    )
    return result


@router.get("/runs", summary="List evaluation runs")
def runs() -> list[dict]:
    """List past runs as summaries, newest first, without per-example detail."""
    return list_run_summaries()


@router.get("/runs/{run_id}", summary="Get one evaluation run")
def run_detail(run_id: str) -> dict:
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
def regressions(
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
def human_rate(req: HumanRatingRequest) -> HumanRating:
    """Persist a 1-5 human rating for an answer."""
    return save_rating(req)


@router.get(
    "/human-ratings", response_model=list[HumanRating], summary="List human ratings"
)
def human_ratings(provider: str | None = None) -> list[HumanRating]:
    """List stored human ratings, optionally filtered by provider."""
    return load_ratings(provider=provider)
