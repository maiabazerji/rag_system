from __future__ import annotations


class ProviderError(RuntimeError):
    pass


class MissingKeyError(ProviderError):
    pass


async def generate(provider: str, *, model: str, prompt: str, max_tokens: int = 1024) -> str:
    result = await generate_with_usage(
        provider, model=model, prompt=prompt, max_tokens=max_tokens
    )
    return result["text"]


async def generate_with_usage(
    provider: str,
    *,
    model: str,
    prompt: str,
    max_tokens: int = 1024,
    system: str | None = None,
    temperature: float | None = None,
) -> dict:
    """Generate with the selected provider and report token usage.

    Args:
        provider: ``anthropic``, ``openai`` or ``local``.
        model: Model ID for that provider.
        prompt: The user prompt.
        max_tokens: Maximum tokens to generate.
        system: Optional system prompt.
        temperature: Optional sampling temperature.

    Returns:
        Dict with ``text``, ``input_tokens`` and ``output_tokens``.

    Raises:
        ProviderError: For an unknown provider or a failed call.
    """
    if provider == "local":
        from app.rag.providers.local import generate_with_usage as fn
    elif provider == "anthropic":
        from app.rag.providers.anthropic_provider import generate_with_usage as fn
    elif provider == "openai":
        from app.rag.providers.openai_provider import generate_with_usage as fn
    else:
        raise ProviderError(f"unknown generator provider: {provider!r}")
    return await fn(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        system=system,
        temperature=temperature,
    )
