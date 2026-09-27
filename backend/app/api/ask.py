from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool

from app.auth import charge, record_tokens, require_api_key, strategy_units
from app.logging_config import get_structured_logger
from app.rag.generate import answer_question
from app.schemas import Answer, AskRequest

logger = get_structured_logger(__name__)
router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post(
    "",
    response_model=Answer,
    summary="Answer a question with RAG",
    description=(
        "Runs the question through the chosen RAG strategy and returns a grounded "
        "answer with its sources, latency and token counts."
    ),
)
async def ask(req: AskRequest, auth: dict = Depends(require_api_key)) -> Answer:
    """Answer a question using RAG.

    Requires `Authorization: Bearer <key>` when REQUIRE_API_KEY is enabled.

    Args:
        req: Question and per-request overrides.
        auth: Authenticated principal, injected by the dependency.

    Returns:
        An Answer. Errors surface as a refusal rather than an exception, so the
        UI always has something to render.
    """
    await run_in_threadpool(charge, auth, strategy_units(req.strategy))
    result = await answer_question(
        question=req.question,
        top_k=req.top_k,
        provider=req.provider,
        model=req.model,
        prompt_version=req.prompt_version,
        strategy=req.strategy,
    )

    await run_in_threadpool(
        record_tokens,
        auth,
        tokens_input=result.input_tokens,
        tokens_output=result.output_tokens,
        model=result.model,
    )
    return result
