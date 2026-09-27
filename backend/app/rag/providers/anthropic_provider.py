"""Anthropic provider: generation, generation-with-usage, and the tool-use loop.

One `AsyncAnthropic` client is shared across the process so connections are
pooled instead of renegotiating TLS on every call. Retries live in exactly one
place: the SDK's own retry is disabled and `with_retry` owns the policy, so
attempts do not multiply.
"""
from __future__ import annotations

import asyncio
from typing import Any

import anthropic
from anthropic import AsyncAnthropic

from app.config import settings
from app.logging_config import get_structured_logger
from app.rag.providers.base import MissingKeyError, ProviderError
from app.resilience import CircuitBreaker, async_timeout_wrapper, with_retry
from app.tracing import instrument

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


def _is_outage(exc: BaseException) -> bool:
    """True when an exception says the Anthropic service itself is unhealthy.

    Only these count toward opening the circuit breaker: 5xx responses,
    timeouts and connection failures. A 400/404 is our request's fault and a
    429 is back-pressure (retried with backoff, see `with_retry`); neither
    means the service is down, so neither may open the breaker for everyone.
    """
    if isinstance(
        exc,
        (
            anthropic.APIConnectionError,  # includes APITimeoutError
            anthropic.InternalServerError,
            TimeoutError,
            asyncio.TimeoutError,
            ConnectionError,
        ),
    ):
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and status >= 500


def _is_rate_limit(exc: BaseException) -> bool:
    """True for a 429, which is neither an outage nor proof of health."""
    return isinstance(exc, anthropic.RateLimitError) or getattr(exc, "status_code", None) == 429


# Models that reject sampling parameters (`temperature`, `top_p`, `top_k`) with
# a 400. On these, determinism is not configurable, so the parameter is dropped.
_NO_SAMPLING_PREFIXES = (
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-",
    "claude-mythos-",
    "claude-opus-4-7",
    "claude-opus-4-8",
)


def supports_temperature(model: str) -> bool:
    """Whether `model` accepts the `temperature` parameter."""
    return not model.startswith(_NO_SAMPLING_PREFIXES)


def _text_of(response: Any) -> str:
    """Concatenate the text blocks of a message response."""
    return "".join(
        b.text for b in response.content if getattr(b, "type", None) == "text"
    ).strip()


@instrument.llm_call
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
    except Exception as e:
        if _is_outage(e):
            _anthropic_breaker.record_failure()
        elif _is_rate_limit(e):
            _anthropic_breaker.release_probe()
        else:
            # The service answered (a 4xx for our request), so it is up.
            _anthropic_breaker.record_success()
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
    temperature: float | None = None,
) -> dict:
    """Generate text and report token usage.

    Args:
        model: Anthropic model ID.
        prompt: The user prompt.
        max_tokens: Maximum tokens to generate.
        system: Optional system prompt.
        temperature: Optional sampling temperature. Silently omitted for models
            that reject sampling parameters (see `supports_temperature`).

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
    if temperature is not None and supports_temperature(model):
        kwargs["temperature"] = temperature

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
    terminal_tools: frozenset[str] | set[str] = frozenset(),
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
        terminal_tools: Tool names that end the loop once they run successfully
            (e.g. ``finish``), so no further model call is made after them.

    Returns:
        Dict with ``text``, ``trace``, ``input_tokens``, ``output_tokens``,
        ``iterations``, ``stop_reason`` and ``error``. ``stop_reason`` is one of
        ``"end_turn"`` (the model stopped calling tools), ``"terminal_tool"``,
        ``"max_iters"`` or ``"provider_error"``; ``error`` is True for the last
        two, in which case ``text`` is a diagnostic, never an answer.

    Raises:
        MissingKeyError: If no API key is configured (a configuration fault,
            not a transient one, so it is not folded into ``stop_reason``).
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
        except MissingKeyError:
            raise
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
                "stop_reason": "provider_error",
                "error": True,
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
                "stop_reason": "end_turn",
                "error": False,
            }

        tool_results: list[dict] = []
        finished = False
        for tu in tool_uses:
            handler = tool_handlers.get(tu.name)
            with instrument.tool_call(tu.name, tu.input) as tool_span:
                if handler is None:
                    content, is_error = f"unknown tool {tu.name!r}", True
                else:
                    try:
                        content, is_error = await handler(tu.input), False
                    except Exception as e:  # surfaced to the model so it can recover
                        content, is_error = f"{type(e).__name__}: {e}", True
                if tool_span is not None:
                    tool_span.output = str(content)
                    tool_span.error = "tool_error" if is_error else None
            if tu.name in terminal_tools and not is_error:
                finished = True
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": str(content),
                    "is_error": is_error,
                }
            )
        if finished:
            logger.info(
                "Tool-use loop ended by a terminal tool",
                extra_fields={
                    "iterations": step + 1,
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
                "stop_reason": "terminal_tool",
                "error": False,
            }
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
        "stop_reason": "max_iters",
        "error": True,
    }
