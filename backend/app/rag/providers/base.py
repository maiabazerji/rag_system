from __future__ import annotations

import json


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


async def generate_structured(
    provider: str,
    *,
    model: str,
    prompt: str,
    tool: dict,
    system: str | None = None,
    max_tokens: int = 1024,
) -> dict:
    """Ask the selected provider for a structured answer shaped like `tool`.

    Anthropic receives `tool` as a real tool definition and returns its input.
    The other providers have no tool-use path here, so they get the schema as
    an instruction to reply with a single JSON object, and the caller parses
    ``text`` (``structured`` is None).

    Returns:
        Dict with ``structured`` (dict or None), ``text``, ``input_tokens`` and
        ``output_tokens``.

    Raises:
        ProviderError: For an unknown provider or a failed call.
    """
    if provider == "anthropic":
        from app.rag.providers.anthropic_provider import generate_structured as fn

        return await fn(
            model=model, prompt=prompt, tool=tool, system=system, max_tokens=max_tokens
        )
    json_instruction = (
        "Reply with a single JSON object and nothing else. It must match this JSON "
        f"schema:\n{json.dumps(tool['input_schema'], ensure_ascii=False)}"
    )
    out = await generate_with_usage(
        provider,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        system=f"{system}\n\n{json_instruction}" if system else json_instruction,
    )
    return {**out, "structured": None}
