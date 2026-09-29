"""Span decorators for the RAG pipeline's steps.

Kept here rather than in the pipeline modules so that instrumenting a step is a
one-line decorator there, and what gets recorded for each step is visible in
one place. Content (queries, prompts, completions) is recorded on the span but
only leaves the process through the exporter's redaction and content policy.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any

from app import monitoring, pricing
from app.tracing.spans import Span, current_strategy, span, traced


def _chunk_ids(chunks: Any) -> list[str]:
    return [getattr(c, "id", str(c)) for c in (chunks or [])]


def _describe_retrieval(s: Span, args: tuple, kwargs: dict, result: Any) -> None:
    s.input = kwargs.get("query", args[0] if args else None)
    s.output = _chunk_ids(result)
    s.metadata = {
        "top_k": kwargs.get("top_k", args[1] if len(args) > 1 else None),
        "returned": len(result or []),
    }


def _describe_rerank(s: Span, args: tuple, kwargs: dict, result: Any) -> None:
    candidates = kwargs.get("chunks", args[1] if len(args) > 1 else [])
    s.input = kwargs.get("query", args[0] if args else None)
    s.output = _chunk_ids(result)
    s.metadata = {"candidates": len(candidates or []), "kept": len(result or [])}


def _describe_rerank_scored(s: Span, args: tuple, kwargs: dict, result: Any) -> None:
    kept = getattr(result, "chunks", None)
    _describe_rerank(s, args, kwargs, kept)
    s.metadata["reranker"] = getattr(result, "reranker", None)


retrieval = traced("retrieve", as_type="retriever", describe=_describe_retrieval)
rerank = traced("rerank", describe=_describe_rerank)
# The same span for a reranker call that returns a RerankOutcome.
rerank_scored = traced("rerank", describe=_describe_rerank_scored)


def _message_text(content: Any) -> Any:
    """Plain text or JSON-able structure for a message's content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for block in content:
            if isinstance(block, dict):
                out.append(block)
            elif getattr(block, "type", None) == "text":
                out.append({"type": "text", "text": block.text})
            elif getattr(block, "type", None) == "tool_use":
                out.append({"type": "tool_use", "name": block.name, "input": block.input})
        return out
    return str(content)


def record_usage(
    s: Span | None,
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    provider: str = "anthropic",
) -> dict[str, float] | None:
    """Count one model call's tokens and cost, and put them on its span.

    The call is priced with the model that served it, so a judge or entity
    extraction call made during a request is billed at its own rate. Metrics
    are recorded even outside a trace (``strategy="none"``).

    Returns:
        The cost breakdown, or None when the model has no listed price.
    """
    strategy = current_strategy()
    monitoring.observe_tokens(model, strategy, input_tokens, output_tokens)
    cost = pricing.cost_breakdown(model, input_tokens, output_tokens)
    monitoring.observe_cost(model, strategy, cost["total"] if cost else None)
    if s is not None:
        s.model = model
        s.usage = {"input": input_tokens, "output": output_tokens}
        s.cost = cost
        s.metadata["provider"] = provider
        s.metadata["cost_usd"] = cost["total"] if cost else None
    return cost


def _record_generation(s: Span | None, kwargs: dict, resp: Any) -> None:
    model = str(kwargs.get("model") or "unknown")
    usage = getattr(resp, "usage", None)
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    record_usage(s, model=model, input_tokens=input_tokens, output_tokens=output_tokens)
    if s is None:
        return

    operation = kwargs.get("_operation", "message")
    messages = kwargs.get("messages") or []
    s.name = "agent_step" if operation == "tool_use_loop" else "generate"
    # The latest message only: in an agent loop the full history is re-sent on
    # every step, and exporting it each time would grow quadratically.
    s.input = {
        "system": kwargs.get("system"),
        "message": _message_text(messages[-1]["content"]) if messages else None,
    }
    s.output = _message_text(getattr(resp, "content", None))
    s.metadata.update(
        {
            "operation": operation,
            "stop_reason": getattr(resp, "stop_reason", None),
            "max_tokens": kwargs.get("max_tokens"),
        }
    )


def llm_call(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Wrap one Messages API call: a generation span plus token metrics.

    Token metrics are recorded even outside a trace, so judge and graph
    extraction spend shows up on /metrics under ``strategy="none"``.
    """

    @wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        with span("generate", as_type="generation") as s:
            resp = await fn(*args, **kwargs)
            _record_generation(s, kwargs, resp)
            return resp

    return wrapper


def tool_call(name: str, tool_input: Any):
    """Span for one agent tool invocation (use as a context manager)."""
    return span(f"tool:{name}", as_type="tool", input=tool_input)
