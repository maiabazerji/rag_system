from __future__ import annotations

from app.config import settings
from app.rag.providers.base import MissingKeyError, ProviderError


async def generate(*, model: str, prompt: str, max_tokens: int = 1024) -> str:
    """Generate text with the OpenAI API.

    Args:
        model: OpenAI model ID.
        prompt: The user prompt.
        max_tokens: Maximum tokens to generate.

    Returns:
        The generated text, stripped.

    Raises:
        MissingKeyError: If OPENAI_API_KEY is not configured.
        ProviderError: If the openai package is not installed or the call fails.
    """
    if not settings.openai_api_key:
        raise MissingKeyError(
            "OPENAI_API_KEY is not configured. Add it to .env and restart."
        )

    try:
        # Imported lazily so the OpenAI SDK stays an optional dependency.
        from openai import AsyncOpenAI
    except ImportError as e:  # pragma: no cover
        raise ProviderError(
            "The 'openai' package is not installed. Install it with "
            "`pip install openai` to use GENERATOR_PROVIDER=openai."
        ) from e

    try:
        async with AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.provider_timeout_seconds,
        ) as client:
            resp = await client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
    except Exception as e:
        raise ProviderError(f"OpenAI request failed ({type(e).__name__}: {e})") from e

    return (resp.choices[0].message.content or "").strip()
