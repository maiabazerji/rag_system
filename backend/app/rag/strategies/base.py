from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from app.schemas import Source


@dataclass
class StrategyResult:
    """Uniform output across classic/graph/agentic so /compare can render side-by-side."""

    answer: str
    sources: list[Source]
    refusal: bool = False
    confidence: float = 0.85
    # Telemetry — what makes the comparison page interesting.
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    iterations: int = 1
    # Strategy-specific reasoning trace (tool calls, graph walk, …) for the UI to inspect.
    trace: list[dict] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


class Strategy(ABC):
    """A RAG strategy turns a question into a grounded answer.

    Subclasses implement `run`. The orchestrator in `generate.py` handles timing
    and provider-error wrapping uniformly.
    """

    name: str = "base"

    @abstractmethod
    async def run(
        self,
        question: str,
        *,
        top_k: int,
        model: str,
        prompt_version: str,
    ) -> StrategyResult: ...
