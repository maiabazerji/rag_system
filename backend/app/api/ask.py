import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.rag.generate import answer_question
from app.schemas import Answer, AskRequest, Source

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("", response_model=Answer)
async def ask(req: AskRequest) -> Answer:
    try:
        if not req.question or not req.question.strip():
            return Answer(
                question="",
                answer="Please provide a question.",
                sources=[Source(chunk_id="none", quote="")],
                confidence=0.0,
                refusal=True,
            )

        return await answer_question(
            question=req.question,
            top_k=req.top_k,
            provider=req.provider,
            model=req.model,
            prompt_version=req.prompt_version,
            strategy=req.strategy,
        )
    except ValueError as e:
        logger.warning(f"Invalid request: {e}")
        return Answer(
            question=req.question,
            answer=f"Invalid request: {str(e)}",
            sources=[Source(chunk_id="none", quote="")],
            confidence=0.0,
            refusal=True,
        )
    except Exception as e:
        logger.exception(f"Ask error: {e}")
        return Answer(
            question=req.question,
            answer="An error occurred while processing your question. Please try again.",
            sources=[Source(chunk_id="none", quote="")],
            confidence=0.0,
            refusal=True,
        )
