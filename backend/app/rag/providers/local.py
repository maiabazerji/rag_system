from __future__ import annotations

import httpx

from app.config import settings
from app.rag.providers.base import ProviderError


async def generate(*, model: str, prompt: str, max_tokens: int = 1024) -> str:
    url = f"{settings.ollama_host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens},
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        raise ProviderError(
            f"Ollama at {settings.ollama_host} is unreachable or slow to respond "
            f"(first call cold-loads weights, can take 60s+). "
            f"Verify with: curl {settings.ollama_host}/api/tags  ·  pulled model: {model}. ({type(e).__name__}: {e})"
        ) from e
    return (data.get("response") or "").strip()
