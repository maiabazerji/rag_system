"""Classic RAG: the textbook 3-step pipeline.

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

Strengths: simple, predictable, cheap, fast.
Weaknesses: multi-hop questions ("how is X connected to Y?") and questions
that need synthesis across many documents tend to underperform.
"""
from __future__ import annotations

from app.config import settings
from app.prompts import load_prompt
from app.rag.providers.anthropic_provider import generate_with_usage
from app.rag.rerank import compress_context, rerank
from app.rag.retrieve import hybrid_search
from app.rag.strategies.base import Strategy, StrategyResult
from app.schemas import Source


class ClassicRAG(Strategy):
    name = "classic"

    async def run(
        self,
        question: str,
        *,
        top_k: int,
        model: str,
        prompt_version: str,
    ) -> StrategyResult:
        prompt = load_prompt(prompt_version)

        candidates = hybrid_search(question, top_k=settings.retrieval_top_k)
        reranked = rerank(question, candidates, top_k=top_k)
        context = compress_context(question, reranked)

        if not context:
            return StrategyResult(
                answer="No relevant context found in the indexed documents.",
                sources=[Source(chunk_id="none", quote="")],
                refusal=True,
                confidence=0.0,
                trace=[{"step": "retrieve", "result": "empty"}],
            )

        ctx_block = "\n\n".join(f"[{c.id}]\n{c.text}" for c in context)
        user_msg = prompt.template.replace("{question}", question).replace("{context}", ctx_block)

        out = await generate_with_usage(model=model, prompt=user_msg, max_tokens=1024)

        return StrategyResult(
            answer=out["text"] or "(empty response)",
            sources=[Source(chunk_id=c.id, quote=c.text[:280]) for c in context[:5]],
            input_tokens=out["input_tokens"],
            output_tokens=out["output_tokens"],
            trace=[
                {"step": "retrieve", "candidates": len(candidates), "kept": len(context)},
                {"step": "generate", "model": model, "chars": len(out["text"])},
            ],
            extra={"context_chunks": [c.id for c in context]},
        )
