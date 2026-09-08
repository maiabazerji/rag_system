"""Anthropic provider: generation, generation-with-usage, and the tool-use loop.

One `AsyncAnthropic` client is shared across the process so connections are
pooled instead of renegotiating TLS on every call. Retries live in exactly one
place: the SDK's own retry is disabled and `with_retry` owns the policy, so
attempts do not multiply.
"""
from __future__ import annotations

from typing import Any

from anthropic import AsyncAnthropic

from app.config import settings
from app.logging_config import get_structured_logger
from app.rag.providers.base import MissingKeyError, ProviderError
from app.resilience import CircuitBreaker, async_timeout_wrapper, with_retry

logger = get_structured_logger(__name__)

_anthropic_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=30.0,
    service_name="Anthropic",
)

_client: AsyncAnthropic | None = None


def _require_key() -> str:
    """Return the configured Anthropic key.

    Raises:
        MissingKeyError: If ANTHROPIC_API_KEY is not set.
    """
    if not settings.anthropic_api_key:
        raise MissingKeyError(
            "ANTHROPIC_API_KEY is not configured. Add it to .env and restart."
        )
    return settings.anthropic_api_key


def get_client() -> AsyncAnthropic:
    """Return the shared Anthropic client, creating it on first use.

    `max_retries=0` because retries are owned by `with_retry`; leaving the SDK's
    default of 2 in place would multiply with ours.

    Raises:
        MissingKeyError: If ANTHROPIC_API_KEY is not set.
    """
    global _client
    key = _require_key()
    if _client is None:
        _client = AsyncAnthropic(
            api_key=key,
            timeout=settings.provider_timeout_seconds,
            max_retries=0,
        )
    return _client


async def close_client() -> None:
    """Close the shared client. Called from the app lifespan on shutdown."""
    global _client
    if _client is not None:
        await _client.close()
        _client = None


def _guard_circuit(operation: str, model: str) -> None:
    """Reject the call when the Anthropic circuit breaker is open.

    Raises:
        ProviderError: If the breaker is open.
    """
    if not _anthropic_breaker.can_execute():
        logger.error(
            "Anthropic circuit breaker is open",
            extra_fields={
                "error_type": "circuit_breaker_open",
                "model": model,
                "operation": operation,
            },
        )
        raise ProviderError(
            "The Anthropic API is temporarily unavailable (circuit breaker open). "
            "Retry in about 30 seconds."
        )


def _text_of(response: Any) -> str:
    """Concatenate the text blocks of a message response."""
    return "".join(
        b.text for b in response.content if getattr(b, "type", None) == "text"
    ).strip()


@with_retry(max_retries=settings.provider_max_retries, backoff_factor=2.0, jitter=True)
async def _create_message(**kwargs: Any) -> Any:
    """Issue one Messages API call with a timeout, tracking the circuit breaker.

    Raises:
        ProviderError: If the circuit breaker is open.
        Exception: Whatever the SDK raises, after recording the failure.
    """
    _guard_circuit(kwargs.get("_operation", "message"), kwargs.get("model", "unknown"))
    kwargs.pop("_operation", None)

    client = get_client()
    try:
        response = await async_timeout_wrapper(
            client.messages.create(**kwargs),
            timeout=settings.provider_timeout_seconds,
            service_name="Anthropic",
        )
    except Exception:
        _anthropic_breaker.record_failure()
        raise
    _anthropic_breaker.record_success()
    return response


async def generate(*, model: str, prompt: str, max_tokens: int = 1024) -> str:
    """Generate text from a single prompt.

    Args:
        model: Anthropic model ID.
        prompt: The user prompt.
        max_tokens: Maximum tokens to generate.

    Returns:
        The generated text.

    Raises:
        MissingKeyError: If no API key is configured.
        ProviderError: If the service is unavailable.
    """
    result = await generate_with_usage(model=model, prompt=prompt, max_tokens=max_tokens)
    return result["text"]


async def generate_with_usage(
    *,
    model: str,
    prompt: str,
    max_tokens: int = 1024,
    system: str | None = None,
) -> dict:
    """Generate text and report token usage.

    Args:
        model: Anthropic model ID.
        prompt: The user prompt.
        max_tokens: Maximum tokens to generate.
        system: Optional system prompt.

    Returns:
        Dict with ``text``, ``input_tokens`` and ``output_tokens``.

    Raises:
        MissingKeyError: If no API key is configured.
        ProviderError: If the service is unavailable.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "_operation": "generate_with_usage",
    }
    if system:
        kwargs["system"] = system

    logger.info(
        "Anthropic call started",
        extra_fields={
            "model": model,
            "max_tokens": max_tokens,
            "prompt_len": len(prompt),
            "has_system": system is not None,
        },
    )

    resp = await _create_message(**kwargs)
    text = _text_of(resp)

    logger.info(
        "Anthropic call succeeded",
        extra_fields={
            "model": model,
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "response_len": len(text),
            "stop_reason": resp.stop_reason,
        },
    )
    return {
        "text": text,
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
    }


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
    """Run a Claude tool-use loop until the model stops calling tools.

    Args:
        model: Anthropic model ID.
        system: System prompt.
        user_message: The opening user message.
        tools: Tool definitions passed to the API.
        tool_handlers: Maps tool name to an async callable taking the tool input.
        max_iters: Maximum round trips before giving up.
        max_tokens: Maximum tokens per response.

    Returns:
        Dict with ``text``, ``trace``, ``input_tokens``, ``output_tokens`` and
        ``iterations``. Always returns; failures are reported in ``text``.
    """
    messages: list[dict] = [{"role": "user", "content": user_message}]
    trace: list[dict] = []
    total_in = total_out = 0

    logger.info(
        "Tool-use loop started",
        extra_fields={
            "model": model,
            "max_iters": max_iters,
            "num_tools": len(tools),
            "user_message_len": len(user_message),
        },
    )

    for step in range(max_iters):
        try:
            resp = await _create_message(
                model=model,
                max_tokens=max_tokens,
                system=system,
                tools=tools,
                messages=messages,
                _operation="tool_use_loop",
            )
        except Exception as e:
            logger.error(
                f"Tool-use step {step} failed: {type(e).__name__}: {e}",
                extra_fields={
                    "error_type": type(e).__name__,
                    "step": step,
                    "model": model,
                },
            )
            return {
                "text": f"The agent stopped early after a provider error ({type(e).__name__}).",
                "trace": trace,
                "input_tokens": total_in,
                "output_tokens": total_out,
                "iterations": step,
            }

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
            logger.info(
                "Tool-use loop completed",
                extra_fields={
                    "iterations": step + 1,
                    "stop_reason": resp.stop_reason,
                    "total_input_tokens": total_in,
                    "total_output_tokens": total_out,
                    "model": model,
                },
            )
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
                content, is_error = f"unknown tool {tu.name!r}", True
            else:
                try:
                    content, is_error = await handler(tu.input), False
                except Exception as e:  # surfaced to the model so it can recover
                    content, is_error = f"{type(e).__name__}: {e}", True
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": str(content),
                    "is_error": is_error,
                }
            )
        messages.append({"role": "user", "content": tool_results})

    logger.info(
        "Tool-use loop hit its iteration limit",
        extra_fields={
            "max_iters": max_iters,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "model": model,
        },
    )
    return {
        "text": f"The agent reached its {max_iters}-step limit without finishing.",
        "trace": trace,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "iterations": max_iters,
    }
