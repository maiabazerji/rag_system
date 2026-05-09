from fastapi import APIRouter

from app.eval.human_ratings import load_all as load_ratings
from app.eval.human_ratings import save as save_rating
from app.eval.regression import list_runs, load_regressions
from app.eval.metrics import run_evaluation
from app.schemas import EvalRunRequest, HumanRating, HumanRatingRequest

router = APIRouter()


@router.post("/run")
async def run(req: EvalRunRequest) -> dict:
    return await run_evaluation(
        dataset=req.dataset,
        provider=req.provider,
        model=req.model,
        prompt_version=req.prompt_version,
    )


@router.get("/runs")
async def runs() -> list[dict]:
    return list_runs()


@router.get("/regressions")
async def regressions() -> list[dict]:
    return load_regressions()


@router.post("/human-rate", response_model=HumanRating)
async def human_rate(req: HumanRatingRequest) -> HumanRating:
    return save_rating(req)


@router.get("/human-ratings", response_model=list[HumanRating])
async def human_ratings(provider: str | None = None) -> list[HumanRating]:
    return load_ratings(provider=provider)
