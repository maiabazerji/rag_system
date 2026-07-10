from __future__ import annotations

from typing import Any

from anthropic import AsyncAnthropic

from app.config import settings
from app.rag.providers.base import MissingKeyError


def _require_key() -> str:
    if not settings.anthropic_api_key:
        raise MissingKeyError("ANTHROPIC_API_KEY is not configured. Add it to .env and restart.")
    return settings.anthropic_api_key


async def generate(*, model: str, prompt: str, max_tokens: int = 1024) -> str:
    async with AsyncAnthropic(api_key=_require_key()) as client:
        resp = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()


async def generate_with_usage(
    *, model: str, prompt: str, max_tokens: int = 1024, system: str | None = None
) -> dict:
    import httpx
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    try:
        async with AsyncAnthropic(api_key=_require_key(), timeout=60.0) as client:
            resp = await client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
        return {
            "text": text,
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        }
    except (httpx.ConnectTimeout, httpx.ReadTimeout, TimeoutError) as e:
        raise TimeoutError(f"API request timed out: {str(e)}") from e


async def tool_use_loop(
    *,
    model: str,
    system: str,
    user_message: str,
    tools: list[dict],
    tool_handlers: dict,
    max_iters: int = 6,
    max_tokens: int = 1024,
) -> dict:
    """Run a Claude tool-use loop. tool_handlers maps name -> async callable(input_dict)."""
    import httpx
    messages: list[dict] = [{"role": "user", "content": user_message}]
    trace: list[dict] = []
    total_in = total_out = 0

    async with AsyncAnthropic(api_key=_require_key(), timeout=90.0) as client:
        for step in range(max_iters):
            resp = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                tools=tools,
                messages=messages,
            )
            total_in += resp.usage.input_tokens
            total_out += resp.usage.output_tokens

            messages.append({"role": "assistant", "content": resp.content})

            tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
            text_blocks = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
            trace.append(
                {
                    "step": step,
                    "stop_reason": resp.stop_reason,
                    "thoughts": " ".join(text_blocks)[:600],
                    "tool_calls": [{"name": t.name, "input": t.input} for t in tool_uses],
                }
            )

            if resp.stop_reason != "tool_use" or not tool_uses:
                return {
                    "text": " ".join(text_blocks).strip(),
                    "trace": trace,
                    "input_tokens": total_in,
                    "output_tokens": total_out,
                    "iterations": step + 1,
                }

            tool_results: list[dict] = []
            for tu in tool_uses:
                handler = tool_handlers.get(tu.name)
                if handler is None:
                    content = f"error: unknown tool {tu.name!r}"
                else:
                    try:
                        content = await handler(tu.input)
                    except Exception as e:  # surface to model so it can recover
                        content = f"error: {type(e).__name__}: {e}"
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": tu.id, "content": str(content)}
                )
            messages.append({"role": "user", "content": tool_results})

    return {
        "text": "(agent exceeded max iterations without finishing)",
        "trace": trace,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "iterations": max_iters,
    }
