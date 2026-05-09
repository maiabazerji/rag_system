from __future__ import annotations

from openai import AsyncOpenAI

from app.config import settings
from app.rag.providers.base import MissingKeyError


async def generate(*, model: str, prompt: str, max_tokens: int = 1024) -> str:
    if not settings.openai_api_key:
        raise MissingKeyError("OPENAI_API_KEY is not configured. Add it to .env and restart.")
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    resp = await client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return (resp.choices[0].message.content or "").strip()
