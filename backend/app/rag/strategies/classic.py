"""Classic RAG: the textbook 3-step vector search + rerank + LLM pipeline.

    question ──► dense (top-50) + BM25 (top-50) ──► RRF fusion
                                │
                                ▼
                          cross-encoder rerank (top-8)
                                │
                                ▼
          prompt = system + [S1..Sn] chunks + question
                                │
                                ▼
              Claude → submit_answer(answer, claims, status)
                                │
                                ▼
            validate citations → cited sources, grounded, evidence score

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

from app.access import AccessScope
from app.config import settings
from app.i18n import localized
from app.logging_config import get_structured_logger
from app.prompts import render_prompt
from app.rag.grounding import (
    SUBMIT_ANSWER_TOOL,
    cite_chunks,
    format_context,
    ground,
    grounded_system,
    grounding_extra,
    validation_attributes,
)
from app.rag.providers.base import generate_structured
from app.rag.retrieve import hybrid_search
from app.rag.strategies.base import Strategy, StrategyResult
from app.rag.timing import stage
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
        access: AccessScope | None = None,
    ) -> StrategyResult:
        """Execute classic RAG pipeline: retrieve, rerank, and generate.

        Args:
            question: The user's question to answer.
            top_k: Number of top chunks to include in context for generation.
            model: Language model ID for generation (e.g., "claude-sonnet-5").
            prompt_version: Prompt template version to use for formatting the context.
            access: The caller's read scope; retrieval is restricted to it.

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

        retrieval = await hybrid_search(question, access, final_k=top_k)
        context = retrieval.chunks
        diagnostics = retrieval.diagnostics.model_dump()
        candidates = retrieval.diagnostics.counts.fused
        logger.debug(
            "Retrieval completed",
            extra_fields={
                "strategy": "classic",
                "mode": retrieval.diagnostics.mode,
                "candidates_count": candidates,
                "context_count": len(context),
                "requested_top_k": top_k,
            },
        )

        if not context:
            logger.warning(
                "No relevant context found after reranking",
                extra_fields={
                    "strategy": "classic",
                    "candidates_count": candidates,
                    "refusal": True,
                },
            )
            return StrategyResult(
                answer=localized("no_context", question),
                sources=[Source(chunk_id="none", quote="")],
                refusal=True,
                confidence=0.0,
                trace=[{"step": "retrieve", "result": "empty"}],
                extra={"retrieval": diagnostics},
                candidate_count=candidates,
            )

        with stage("context_selection", span_name="context_selection") as s:
            cited = cite_chunks(context)
            ctx_block = format_context(cited)
            user_msg = render_prompt(prompt_version, question=question, context=ctx_block)
            if s is not None:
                s.metadata.update(
                    {
                        "candidates": candidates,
                        "selected": len(cited),
                        "context_chars": len(ctx_block),
                        "prompt_version": prompt_version,
                    }
                )
        logger.debug(
            "Context block created",
            extra_fields={
                "strategy": "classic",
                "context_block_size": len(ctx_block),
                "num_chunks": len(context),
            },
        )
        logger.debug(
            "Prompt rendered",
            extra_fields={
                "strategy": "classic",
                "prompt_len": len(user_msg),
                "prompt_version": prompt_version,
            },
        )

        # Classic is the one strategy that runs on any provider, so generation
        # goes through the dispatcher rather than straight to Anthropic.
        with stage("generation"):
            out = await generate_structured(
                self.provider,
                model=model,
                prompt=user_msg,
                tool=SUBMIT_ANSWER_TOOL,
                system=grounded_system(self.provider),
                max_tokens=settings.max_answer_tokens,
            )
        with stage("citation_validation", span_name="citation_validation") as s:
            grounded, raw = ground(question, out, cited)
            if s is not None:
                s.metadata.update(validation_attributes(grounded))

        logger.info(
            "Classic RAG strategy completed",
            extra_fields={
                "strategy": "classic",
                "provider": self.provider,
                "model": model,
                "input_tokens": out["input_tokens"],
                "output_tokens": out["output_tokens"],
                "answer_len": len(grounded.answer),
                "sources_count": grounded.citation_count,
                "grounded": grounded.grounded,
                "status": grounded.status,
                "invalid_citations": len(grounded.invalid_citations),
            },
        )

        return StrategyResult(
            **grounded.result_fields(),
            input_tokens=out["input_tokens"],
            output_tokens=out["output_tokens"],
            candidate_count=candidates,
            context_count=len(context),
            trace=[
                {"step": "retrieve", "candidates": candidates, "kept": len(context)},
                {
                    "step": "generate",
                    "provider": self.provider,
                    "model": model,
                    "mode": grounded.mode,
                    "chars": len(out.get("text") or ""),
                },
                {
                    "step": "validate_citations",
                    "status": grounded.status,
                    "grounded": grounded.grounded,
                    "cited": [c.handle for c in grounded.cited],
                    "invalid": grounded.invalid_citations,
                },
            ],
            extra={
                "context_chunks": [c.id for c in context],
                "retrieved_ids": [c.id for c in context],
                "retrieved_docs": [c.metadata.get("filename") for c in context],
                "context_text": ctx_block,
                "retrieval": diagnostics,
                **grounding_extra(grounded, raw, cited),
            },
        )
