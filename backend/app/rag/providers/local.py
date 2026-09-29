from __future__ import annotations

import httpx

from app.config import settings
from app.logging_config import get_structured_logger
from app.rag.providers.base import ProviderError

logger = get_structured_logger(__name__)


async def generate(*, model: str, prompt: str, max_tokens: int = 1024) -> str:
    result = await generate_with_usage(model=model, prompt=prompt, max_tokens=max_tokens)
    return result["text"]


async def generate_with_usage(
    *,
    model: str,
    prompt: str,
    max_tokens: int = 1024,
    system: str | None = None,
    temperature: float | None = None,
) -> dict:
    """Generate with Ollama and report token usage.

    Returns:
        Dict with ``text``, ``input_tokens`` and ``output_tokens`` (Ollama's
        ``prompt_eval_count`` / ``eval_count``; 0 when it omits them).

    Raises:
        ProviderError: If Ollama is unreachable or returns an error.
    """
    url = f"{settings.ollama_host.rstrip('/')}/api/generate"
    options: dict = {"num_predict": max_tokens}
    if temperature is not None:
        options["temperature"] = temperature
    payload: dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }
    if system:
        payload["system"] = system
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
    except httpx.HTTPError as e:
        logger.warning(
            f"Ollama call failed: {type(e).__name__}: {e}",
            extra_fields={"error_type": type(e).__name__, "model": model},
        )
        raise ProviderError(
            "The local model server (Ollama) is unreachable or slow to respond; "
            "the first call cold-loads weights and can take 60s+. Check that the "
            "model is pulled and OLLAMA_HOST is reachable."
        ) from e
    try:
        data = resp.json()
    except ValueError:
        data = None
    if not isinstance(data, dict):
        # A proxy error page or another service on OLLAMA_HOST, answering 200.
        logger.warning(
            "Ollama reply is not a JSON object",
            extra_fields={"error_type": "invalid_response", "model": model},
        )
        raise ProviderError(
            "The local model server returned a reply that is not a JSON object. "
            "Check that OLLAMA_HOST points at Ollama."
        )
    return {
        "text": (data.get("response") or "").strip(),
        "input_tokens": int(data.get("prompt_eval_count") or 0),
        "output_tokens": int(data.get("eval_count") or 0),
    }
