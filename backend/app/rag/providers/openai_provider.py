from __future__ import annotations

from app.config import settings
from app.logging_config import get_structured_logger
from app.rag.providers.base import MissingKeyError, ProviderError

logger = get_structured_logger(__name__)


async def generate(*, model: str, prompt: str, max_tokens: int = 1024) -> str:
    """Generate text with the OpenAI API. See :func:`generate_with_usage`."""
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
    """Generate text with the OpenAI API and report token usage.

    Args:
        model: OpenAI model ID.
        prompt: The user prompt.
        max_tokens: Maximum tokens to generate.
        system: Optional system prompt.
        temperature: Optional sampling temperature.

    Returns:
        Dict with ``text``, ``input_tokens`` and ``output_tokens``.

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

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    kwargs: dict = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if temperature is not None:
        kwargs["temperature"] = temperature

    try:
        async with AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.provider_timeout_seconds,
        ) as client:
            resp = await client.chat.completions.create(**kwargs)
    except Exception as e:
        logger.warning(
            f"OpenAI call failed: {type(e).__name__}: {e}",
            extra_fields={"error_type": type(e).__name__, "model": model},
        )
        raise ProviderError(f"OpenAI request failed ({type(e).__name__}).") from e

    usage = getattr(resp, "usage", None)
    return {
        "text": (resp.choices[0].message.content or "").strip(),
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }
