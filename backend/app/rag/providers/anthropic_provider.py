from __future__ import annotations

import logging
from typing import Any

from anthropic import AsyncAnthropic

from app.config import settings
from app.logging_config import get_structured_logger
from app.rag.providers.base import MissingKeyError
from app.resilience import CircuitBreaker, async_timeout_wrapper, with_retry

logger = get_structured_logger(__name__)

_anthropic_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=30.0,
    service_name="Anthropic",
)


def _require_key() -> str:
    if not settings.anthropic_api_key:
        raise MissingKeyError("ANTHROPIC_API_KEY is not configured. Add it to .env and restart.")
    return settings.anthropic_api_key


@with_retry(max_retries=3, backoff_factor=2.0, jitter=True, retryable_http_status=(429, 500, 502, 503, 504))
async def _generate_impl(*, model: str, prompt: str, max_tokens: int = 1024) -> str:
    """Internal implementation with retry logic."""
    if not _anthropic_breaker.can_execute():
        logger.error(
            "Anthropic circuit breaker is open",
            extra_fields={
                "error_type": "circuit_breaker_open",
                "model": model,
            },
        )
        raise RuntimeError("Anthropic service is unavailable (circuit breaker open)")
    try:
        logger.info(
            "Anthropic API call started",
            extra_fields={
                "model": model,
                "max_tokens": max_tokens,
                "prompt_len": len(prompt),
            },
        )
        async with AsyncAnthropic(api_key=_require_key(), timeout=30.0) as client:
            resp = await async_timeout_wrapper(
                client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=30.0,
                service_name="Anthropic",
            )
        _anthropic_breaker.record_success()
        result = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
        logger.info(
            "Anthropic API call succeeded",
            extra_fields={
                "model": model,
                "response_len": len(result),
            },
        )
        return result
    except Exception as e:
        logger.error(
            f"Anthropic generate failed: {type(e).__name__}: {e}",
            extra_fields={
                "error_type": type(e).__name__,
                "model": model,
            },
        )
        _anthropic_breaker.record_failure()
        raise


async def generate(*, model: str, prompt: str, max_tokens: int = 1024) -> str:
    """Generate text using Anthropic API with resilience."""
    return await _generate_impl(model=model, prompt=prompt, max_tokens=max_tokens)


@with_retry(max_retries=3, backoff_factor=2.0, jitter=True, retryable_http_status=(429, 500, 502, 503, 504))
async def _generate_with_usage_impl(
    *, model: str, prompt: str, max_tokens: int = 1024, system: str | None = None
) -> dict:
    """Internal implementation with retry logic."""
    if not _anthropic_breaker.can_execute():
        logger.error(
            "Anthropic circuit breaker is open",
            extra_fields={
                "error_type": "circuit_breaker_open",
                "model": model,
                "operation": "generate_with_usage",
            },
        )
        raise RuntimeError("Anthropic service is unavailable (circuit breaker open)")
    try:
        logger.info(
            "Anthropic API call started",
            extra_fields={
                "model": model,
                "max_tokens": max_tokens,
                "prompt_len": len(prompt),
                "operation": "generate_with_usage",
                "has_system": system is not None,
            },
        )
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        async with AsyncAnthropic(api_key=_require_key(), timeout=30.0) as client:
            resp = await async_timeout_wrapper(
                client.messages.create(**kwargs),
                timeout=30.0,
                service_name="Anthropic",
            )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
        _anthropic_breaker.record_success()
        logger.info(
            "Anthropic API call succeeded",
            extra_fields={
                "model": model,
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "response_len": len(text),
                "operation": "generate_with_usage",
            },
        )
        return {
            "text": text,
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        }
    except TimeoutError as e:
        logger.error(
            f"Anthropic generate_with_usage timed out: {e}",
            extra_fields={
                "error_type": "timeout",
                "model": model,
                "operation": "generate_with_usage",
            },
        )
        _anthropic_breaker.record_failure()
        raise TimeoutError(f"API request timed out: {str(e)}") from e
    except Exception as e:
        logger.error(
            f"Anthropic generate_with_usage failed: {type(e).__name__}: {e}",
            extra_fields={
                "error_type": type(e).__name__,
                "model": model,
                "operation": "generate_with_usage",
            },
        )
        _anthropic_breaker.record_failure()
        raise


async def generate_with_usage(
    *, model: str, prompt: str, max_tokens: int = 1024, system: str | None = None
) -> dict:
    """Generate text with usage stats using Anthropic API with resilience."""
    return await _generate_with_usage_impl(
        model=model, prompt=prompt, max_tokens=max_tokens, system=system
    )


@with_retry(max_retries=2, backoff_factor=2.0, jitter=True, retryable_http_status=(429, 500, 502, 503, 504))
async def _tool_use_step(
    client: AsyncAnthropic,
    model: str,
    max_tokens: int,
    system: str,
    tools: list[dict],
    messages: list[dict],
) -> Any:
    """Execute a single tool-use step with retry logic."""
    return await async_timeout_wrapper(
        client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            messages=messages,
        ),
        timeout=30.0,
        service_name="Anthropic",
    )


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
    """Run a Claude tool-use loop with resilience. tool_handlers maps name -> async callable(input_dict)."""
    if not _anthropic_breaker.can_execute():
        logger.error(
            "Anthropic circuit breaker is open",
            extra_fields={
                "error_type": "circuit_breaker_open",
                "model": model,
                "operation": "tool_use_loop",
            },
        )
        return {
            "text": "Anthropic service is unavailable (circuit breaker open)",
            "trace": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "iterations": 0,
        }

    messages: list[dict] = [{"role": "user", "content": user_message}]
    trace: list[dict] = []
    total_in = total_out = 0

    logger.info(
        "Tool-use loop started",
        extra_fields={
            "model": model,
            "max_iters": max_iters,
            "max_tokens": max_tokens,
            "user_message_len": len(user_message),
            "num_tools": len(tools),
        },
    )

    try:
        async with AsyncAnthropic(api_key=_require_key(), timeout=30.0) as client:
            for step in range(max_iters):
                try:
                    logger.debug(
                        "Tool-use step started",
                        extra_fields={
                            "step": step,
                            "model": model,
                            "accumulated_input_tokens": total_in,
                            "accumulated_output_tokens": total_out,
                        },
                    )
                    resp = await _tool_use_step(client, model, max_tokens, system, tools, messages)
                except Exception as e:
                    logger.error(
                        f"Tool-use step {step} failed: {type(e).__name__}: {e}",
                        extra_fields={
                            "error_type": type(e).__name__,
                            "step": step,
                            "model": model,
                            "operation": "tool_use_loop",
                        },
                    )
                    _anthropic_breaker.record_failure()
                    # Return best-effort result with available iterations
                    logger.info(
                        "Tool-use loop failed mid-execution",
                        extra_fields={
                            "step": step,
                            "total_iterations": max_iters,
                            "accumulated_input_tokens": total_in,
                            "accumulated_output_tokens": total_out,
                            "model": model,
                        },
                    )
                    return {
                        "text": "(tool-use loop failed mid-execution)",
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

                logger.debug(
                    "Tool-use step completed",
                    extra_fields={
                        "step": step,
                        "stop_reason": resp.stop_reason,
                        "step_input_tokens": resp.usage.input_tokens,
                        "step_output_tokens": resp.usage.output_tokens,
                        "tool_calls_count": len(tool_uses),
                        "model": model,
                    },
                )

                if resp.stop_reason != "tool_use" or not tool_uses:
                    _anthropic_breaker.record_success()
                    logger.info(
                        "Tool-use loop completed",
                        extra_fields={
                            "step": step + 1,
                            "total_iterations": max_iters,
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

        _anthropic_breaker.record_success()
    except Exception as e:
        logger.error(
            f"Tool-use loop failed: {type(e).__name__}: {e}",
            extra_fields={
                "error_type": type(e).__name__,
                "model": model,
                "operation": "tool_use_loop",
                "total_iterations": max_iters,
                "accumulated_input_tokens": total_in,
                "accumulated_output_tokens": total_out,
            },
        )
        _anthropic_breaker.record_failure()

    logger.info(
        "Tool-use loop exhausted max iterations",
        extra_fields={
            "total_iterations": max_iters,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "model": model,
        },
    )

    return {
        "text": "(agent exceeded max iterations without finishing)",
        "trace": trace,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "iterations": max_iters,
    }
