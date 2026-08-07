"""Strategy base classes and interfaces for RAG pipelines.

This module defines the abstract interface that all RAG strategies (classic, graph, agentic)
must implement. It also provides the unified StrategyResult dataclass that ensures
consistent output formatting across different strategies for comparison and evaluation.

Key concepts:
    - Strategy: Abstract base class defining the contract for RAG implementations
    - StrategyResult: Normalized output with answer, sources, confidence, and telemetry
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from app.schemas import Source


@dataclass
class StrategyResult:
    """Uniform output across classic/graph/agentic so /compare can render side-by-side.

    This dataclass standardizes the response format from all RAG strategies, enabling
    side-by-side comparison in the UI. It includes both the answer/sources and rich
    telemetry data for analysis.

    Attributes:
        answer: The generated answer text from the RAG pipeline.
        sources: List of Source objects citing specific chunks that support the answer.
        refusal: Whether the strategy refused to answer (no relevant context found).
            Defaults to False.
        confidence: Model's confidence in the answer (0.0-1.0). Defaults to 0.85.
        latency_ms: Total execution time in milliseconds. Defaults to 0.
        input_tokens: Number of tokens consumed for input. Defaults to 0.
        output_tokens: Number of tokens generated. Defaults to 0.
        iterations: Number of retrieval/reasoning iterations (relevant for agentic RAG).
            Defaults to 1.
        trace: List of dicts containing strategy-specific reasoning trace (tool calls,
            graph walks, retrieval steps) for UI inspection. Defaults to empty list.
        extra: Dict for strategy-specific metadata and extra information. Defaults to
            empty dict.

    Example:
        >>> result = StrategyResult(
        ...     answer="The Earth orbits the Sun.",
        ...     sources=[Source(chunk_id="doc1_chunk5", quote="Earth revolves around the Sun...")],
        ...     confidence=0.95,
        ...     input_tokens=150,
        ...     output_tokens=25,
        ... )
    """

    answer: str
    sources: list[Source]
    refusal: bool = False
    confidence: float = 0.85
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    iterations: int = 1
    trace: list[dict] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


class Strategy(ABC):
    """Abstract base class for RAG strategies.

    A RAG strategy turns a user question into a grounded, sourced answer by:
    1. Retrieving relevant chunks from indexed documents
    2. Optionally processing/ranking/reasoning over those chunks
    3. Generating an answer using an LLM, grounded in the retrieved context

    Subclasses must implement the `run` method. The orchestrator in `generate.py`
    handles timing, provider-error wrapping, and telemetry collection uniformly
    across all strategies.

    Attributes:
        name: Human-readable name of the strategy (e.g., "classic", "graph", "agentic").
            Override in subclasses.

    Example:
        >>> class CustomRAG(Strategy):
        ...     name = "custom"
        ...     async def run(self, question: str, *, top_k: int, model: str,
        ...                   prompt_version: str) -> StrategyResult:
        ...         # Implement retrieval and generation logic
        ...         return StrategyResult(...)
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
    ) -> StrategyResult:
        """Execute the RAG strategy to answer a question.

        Args:
            question: The user's question to answer.
            top_k: Maximum number of chunks to include in context for answer generation.
                Strategies may retrieve more chunks initially then rerank to this value.
            model: Name/ID of the language model to use for generation
                (e.g., "claude-opus-4-7", "gpt-4").
            prompt_version: Version/name of the prompt template to use for system/user
                messages (e.g., "default", "v2"). Versioning allows A/B testing.

        Returns:
            StrategyResult containing the answer, sources, confidence, and telemetry.

        Raises:
            Exception subclasses may be raised by underlying provider/retrieval APIs.
                These are caught and wrapped uniformly by generate.py.
        """
        ...
