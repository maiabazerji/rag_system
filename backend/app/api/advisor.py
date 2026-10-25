"""Strategy advisor routes: recommend a RAG strategy, then validate it on real questions."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Path
from fastapi.concurrency import run_in_threadpool

from app.access import AccessScope, Principal
from app.advisor.followup import confidence, follow_up_questions
from app.advisor.messages import msg
from app.advisor.profile import build_profile
from app.advisor.schemas import (
    WORKSPACE_PATTERN,
    AdviseRequest,
    AdviseResponse,
    NextStep,
    ValidateRequest,
    ValidateResponse,
)
from app.advisor.scoring import compliance_notes, hybrid_routing, score_strategies
from app.advisor.validate import default_strategies, run_validation
from app.audit import audited
from app.auth import charge, record_tokens, require_principal, strategy_units
from app.config import settings
from app.privacy import erasure

router = APIRouter(dependencies=[Depends(require_principal)])


def workspace_scope(principal: Principal, workspace: str | None) -> AccessScope:
    """The read scope for a validation: the caller's, or one private workspace.

    A workspace scope holds that single group (not even ``public``), so only
    the documents uploaded for the test are retrievable. Labelling documents
    with a new group needs admin rights (the local principal has them), so a
    non-admin may only use a workspace it is already a member of.
    """
    if workspace is None:
        return principal.scope()
    if not principal.is_admin and workspace not in principal.scope().groups:
        raise HTTPException(status_code=403, detail="Not a member of this workspace.")
    return AccessScope(
        tenant=principal.tenant, groups=frozenset({workspace}), principal_id=principal.id
    )


@router.post(
    "",
    summary="Recommend a RAG strategy for a project",
    description=(
        "Reads a free-text project description (any language), extracts a project "
        "profile, and ranks the classic, graph and agentic strategies with reasons, "
        "tradeoffs and a suggested starting configuration."
    ),
)
async def advise(
    req: AdviseRequest, principal: Principal = Depends(require_principal)
) -> AdviseResponse:
    """Profile the project and score the strategies.

    Profile extraction uses Claude when a key is configured and falls back to
    keyword rules otherwise (``profile.source`` says which). Scoring is
    deterministic.
    """
    with audited(principal, "advise"):
        profile, tokens_in, tokens_out = await build_profile(req.description, req.overrides)
    await run_in_threadpool(
        record_tokens,
        principal,
        tokens_input=tokens_in,
        tokens_output=tokens_out,
        model=settings.generator_model if profile.source == "llm" else None,
    )

    lang = req.language
    ranked = score_strategies(profile, lang)
    questions = follow_up_questions(profile, ranked)
    top_two = [r.strategy for r in ranked[:2]]
    examples = profile.example_questions[:3] or [msg("next_step.example_question", lang)]
    return AdviseResponse(
        profile=profile,
        recommendations=ranked,
        top_strategy=ranked[0].strategy,
        hybrid_routing=hybrid_routing(profile, lang),
        compliance_notes=compliance_notes(profile, lang),
        next_step=NextStep(
            summary=msg(
                "next_step.summary", lang, strategies=msg("next_step.and", lang).join(top_two)
            ),
            suggested_strategies=top_two,
            example_payload={
                "questions": [{"question": q, "ideal_answer": None} for q in examples],
                "strategies": top_two,
            },
        ),
        follow_up_questions=questions,
        confidence=confidence(ranked, questions),
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
    req: ValidateRequest, principal: Principal = Depends(require_principal)
) -> ValidateResponse:
    """Run the questions and aggregate refusals, latency, tokens and quality.

    Retrieval is limited to the documents the caller may read, and with
    ``workspace`` to only the documents ingested under that group.
    """
    strategies = req.strategies or default_strategies(req.profile)
    scope = workspace_scope(principal, req.workspace)
    with audited(principal, "advise", strategy=",".join(strategies)) as event:
        await run_in_threadpool(
            charge, principal, len(req.questions) * sum(strategy_units(s) for s in strategies)
        )
        response, tokens_in, tokens_out = await run_validation(req, access=scope)
        event.detail = f"validate questions={len(req.questions)}"
    await run_in_threadpool(
        record_tokens, principal, tokens_input=tokens_in, tokens_output=tokens_out
    )
    return response


@router.delete(
    "/workspace/{workspace}",
    summary="Delete a test workspace's documents",
    description=(
        "Erases every document uploaded into a private test workspace: its chunks, "
        "graph triples, traces and eval-run text."
    ),
)
async def delete_workspace(
    workspace: str = Path(..., pattern=WORKSPACE_PATTERN),
    principal: Principal = Depends(require_principal),
) -> dict:
    """Clear a workspace once the client has seen the measurements.

    Same rule as validation: admins (including the local principal) and the
    workspace's members only.
    """
    scope = workspace_scope(principal, workspace)
    with audited(principal, "delete") as event:
        request = await erasure.resolve_group(scope.tenant, workspace)
        report = await erasure.erase(request)
        event.detail = f"workspace documents={len(request.doc_ids)}"
    return report.as_dict()
