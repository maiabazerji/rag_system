from fastapi import APIRouter

from app.rag.generate import answer_question
from app.schemas import Answer, AskRequest

router = APIRouter()


@router.post("", response_model=Answer)
async def ask(req: AskRequest) -> Answer:
    return await answer_question(
        question=req.question,
        top_k=req.top_k,
        provider=req.provider,
        model=req.model,
        prompt_version=req.prompt_version,
    )
