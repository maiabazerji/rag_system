"""Strategy advisor routes: recommend a RAG strategy, then validate it on real questions."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool

from app.advisor.profile import build_profile
from app.advisor.schemas import (
    AdviseRequest,
    AdviseResponse,
    NextStep,
    ValidateRequest,
    ValidateResponse,
)
from app.advisor.scoring import compliance_notes, hybrid_routing, score_strategies
from app.advisor.validate import default_strategies, run_validation
from app.auth import charge, record_tokens, require_api_key, strategy_units
from app.config import settings

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post(
    "",
    summary="Recommend a RAG strategy for a project",
    description=(
        "Reads a free-text project description (any language), extracts a project "
        "profile, and ranks the classic, graph and agentic strategies with reasons, "
        "tradeoffs and a suggested starting configuration."
    ),
)
async def advise(req: AdviseRequest, auth: dict = Depends(require_api_key)) -> AdviseResponse:
    """Profile the project and score the strategies.

    Profile extraction uses Claude when a key is configured and falls back to
    keyword rules otherwise (``profile.source`` says which). Scoring is
    deterministic.
    """
    profile, tokens_in, tokens_out = await build_profile(req.description, req.overrides)
    await run_in_threadpool(
        record_tokens,
        auth,
        tokens_input=tokens_in,
        tokens_output=tokens_out,
        model=settings.generator_model if profile.source == "llm" else None,
    )

    ranked = score_strategies(profile)
    top_two = [r.strategy for r in ranked[:2]]
    examples = profile.example_questions[:3] or ["<one of your real user questions>"]
    return AdviseResponse(
        profile=profile,
        recommendations=ranked,
        top_strategy=ranked[0].strategy,
        hybrid_routing=hybrid_routing(profile),
        compliance_notes=compliance_notes(profile),
        next_step=NextStep(
            summary=(
                "This ranking is a prior, not a measurement. Ingest a representative "
                "sample of your documents (POST /ingest), then send 10-20 real user "
                "questions, ideally with the answer you expect, to POST /advise/validate. "
                f"It runs them through {' and '.join(top_two)} and reports refusals, "
                "latency, tokens and quality, and names the measured winner."
            ),
            suggested_strategies=top_two,
            example_payload={
                "questions": [{"question": q, "ideal_answer": None} for q in examples],
                "strategies": top_two,
            },
        ),
    )


@router.post(
    "/validate",
    summary="Validate strategies on your own questions",
    description=(
        "Runs up to 20 questions through up to 3 strategies (default: the top two "
        "recommended) against the indexed documents and returns a scorecard and the "
        "measured winner."
    ),
)
async def validate(
    req: ValidateRequest, auth: dict = Depends(require_api_key)
) -> ValidateResponse:
    """Run the questions and aggregate refusals, latency, tokens and quality."""
    strategies = req.strategies or default_strategies(req.profile)
    await run_in_threadpool(
        charge, auth, len(req.questions) * sum(strategy_units(s) for s in strategies)
    )
    response, tokens_in, tokens_out = await run_validation(req)
    await run_in_threadpool(
        record_tokens, auth, tokens_input=tokens_in, tokens_output=tokens_out
    )
    return response
