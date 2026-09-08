"""Classic RAG: the textbook 3-step vector search + rerank + LLM pipeline.

    question ──► embed ──► vector search (top-50)
                                │
                                ▼
                          cross-encoder rerank (top-8)
                                │
                                ▼
                  prompt = system + chunks + question
                                │
                                ▼
                            Claude → answer

Strengths:
    - Simple, predictable, low-cost pipeline
    - Fast execution with minimal latency
    - Well-understood failure modes and debugging
    - Good for factual questions with clear lexical/semantic matches

Weaknesses:
    - Multi-hop questions ("how is X connected to Y?") may underperform
    - Complex synthesis across many documents often missed by vector search
    - No reasoning over retrieved information, only direct context injection
    - Struggles with ambiguous or domain-specific terminology

Use cases:
    - FAQ-style factual lookups
    - Single-document reference queries
    - Baseline/control comparison in RAG evaluation
"""
from __future__ import annotations

from app.config import settings
from app.logging_config import get_structured_logger
from app.prompts import render_prompt
from app.rag.providers.anthropic_provider import generate_with_usage
from app.rag.rerank import rerank_async
from app.rag.retrieve import dense_search
from app.rag.strategies.base import Strategy, StrategyResult
from app.schemas import Source

logger = get_structured_logger(__name__)


class ClassicRAG(Strategy):
    """Classic RAG: retrieve relevant chunks then generate answer from context.

    Implements the canonical three-step RAG pipeline:
    1. Vector search to retrieve candidate chunks (sparse + semantic)
    2. Cross-encoder reranking to pick top-k most relevant
    3. LLM generation with context window containing top chunks

    This strategy is deterministic and repeatable for a given question/corpus.
    It does not reason over retrieved information, only injects them into
    the prompt context for the LLM to use.

    Attributes:
        name: Strategy identifier, always "classic".
    """

    name: str = "classic"

    async def run(
        self,
        question: str,
        *,
        top_k: int,
        model: str,
        prompt_version: str,
    ) -> StrategyResult:
        """Execute classic RAG pipeline: retrieve, rerank, and generate.

        Args:
            question: The user's question to answer.
            top_k: Number of top chunks to include in context for generation.
            model: Language model ID for generation (e.g., "claude-sonnet-5").
            prompt_version: Prompt template version to use for formatting the context.

        Returns:
            StrategyResult with answer, sources, token counts, latency, and trace.
                The trace includes retrieval counts and generation metadata.

        Raises:
            Exceptions from retrieve, rerank, or generate are propagated to generate.py
                for uniform error handling.
        """
        logger.info(
            "Classic RAG strategy started",
            extra_fields={
                "strategy": "classic",
                "question_len": len(question),
                "top_k": top_k,
                "model": model,
                "prompt_version": prompt_version,
            },
        )

        candidates = await dense_search(question, top_k=settings.retrieval_top_k)
        logger.debug(
            "Retrieval completed",
            extra_fields={
                "strategy": "classic",
                "candidates_count": len(candidates),
                "retrieval_top_k": settings.retrieval_top_k,
            },
        )

        context = await rerank_async(question, candidates, top_k=top_k)
        logger.debug(
            "Reranking completed",
            extra_fields={
                "strategy": "classic",
                "reranked_count": len(context),
                "requested_top_k": top_k,
            },
        )

        if not context:
            logger.warning(
                "No relevant context found after reranking",
                extra_fields={
                    "strategy": "classic",
                    "candidates_count": len(candidates),
                    "refusal": True,
                },
            )
            return StrategyResult(
                answer="No relevant context found in the indexed documents.",
                sources=[Source(chunk_id="none", quote="")],
                refusal=True,
                confidence=0.0,
                trace=[{"step": "retrieve", "result": "empty"}],
            )

        ctx_block = "\n\n".join(f"[{c.id}]\n{c.text}" for c in context)
        logger.debug(
            "Context block created",
            extra_fields={
                "strategy": "classic",
                "context_block_size": len(ctx_block),
                "num_chunks": len(context),
            },
        )

        user_msg = render_prompt(prompt_version, question=question, context=ctx_block)
        logger.debug(
            "Prompt rendered",
            extra_fields={
                "strategy": "classic",
                "prompt_len": len(user_msg),
                "prompt_version": prompt_version,
            },
        )

        out = await generate_with_usage(model=model, prompt=user_msg, max_tokens=settings.max_answer_tokens)

        logger.info(
            "Classic RAG strategy completed",
            extra_fields={
                "strategy": "classic",
                "model": model,
                "input_tokens": out["input_tokens"],
                "output_tokens": out["output_tokens"],
                "answer_len": len(out["text"]),
                "sources_count": len(context[:5]),
            },
        )

        return StrategyResult(
            answer=out["text"] or "(empty response)",
            sources=[
                Source(
                    chunk_id=c.id,
                    quote=c.text[:280],
                    document=c.metadata.get("filename"),
                )
                for c in context[:5]
            ],
            input_tokens=out["input_tokens"],
            output_tokens=out["output_tokens"],
            trace=[
                {"step": "retrieve", "candidates": len(candidates), "kept": len(context)},
                {"step": "generate", "model": model, "chars": len(out["text"])},
            ],
            extra={"context_chunks": [c.id for c in context]},
        )
