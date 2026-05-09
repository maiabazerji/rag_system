import asyncio

from fastapi import APIRouter

from app.rag.generate import answer_question
from app.schemas import CompareRequest

router = APIRouter()


@router.post("")
async def compare(req: CompareRequest) -> dict:
    tasks = [
        answer_question(
            question=req.question,
            provider=v.get("provider"),
            model=v.get("model"),
            prompt_version=v.get("prompt_version"),
        )
        for v in req.variants
    ]
    results = await asyncio.gather(*tasks)
    return {"question": req.question, "results": [r.model_dump() for r in results]}
