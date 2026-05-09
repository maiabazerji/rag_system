from __future__ import annotations


class ProviderError(RuntimeError):
    pass


class MissingKeyError(ProviderError):
    pass


async def generate(provider: str, *, model: str, prompt: str, max_tokens: int = 1024) -> str:
    if provider == "local":
        from app.rag.providers.local import generate as fn
    elif provider == "anthropic":
        from app.rag.providers.anthropic_provider import generate as fn
    elif provider == "openai":
        from app.rag.providers.openai_provider import generate as fn
    else:
        raise ProviderError(f"unknown generator provider: {provider!r}")
    return await fn(model=model, prompt=prompt, max_tokens=max_tokens)
