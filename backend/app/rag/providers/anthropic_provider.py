from __future__ import annotations

from anthropic import AsyncAnthropic

from app.config import settings
from app.rag.providers.base import MissingKeyError


async def generate(*, model: str, prompt: str, max_tokens: int = 1024) -> str:
    if not settings.anthropic_api_key:
        raise MissingKeyError("ANTHROPIC_API_KEY is not configured. Add it to .env and restart.")
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    resp = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
