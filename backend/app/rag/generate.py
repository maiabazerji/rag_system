from __future__ import annotations

from app.config import settings
from app.prompts import load_prompt
from app.rag.providers import MissingKeyError, ProviderError
from app.rag.providers import generate as provider_generate
from app.rag.rerank import compress_context, rerank
from app.rag.retrieve import hybrid_search
from app.rag.store import count as store_count
from app.schemas import Answer, Source
from app.tracing import start_trace


def _refusal(
    question: str,
    message: str,
    sources: list[Source] | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> Answer:
    return Answer(
        question=question,
        answer=message,
        sources=sources or [Source(chunk_id="none", quote="")],
        confidence=0.0,
        refusal=True,
        provider=provider,
        model=model,
    )


def _default_model(provider: str) -> str:
    if provider == "local":
        return settings.ollama_model
    if provider == "openai":
        return settings.openai_generator_model
    return settings.generator_model


async def answer_question(
    question: str,
    top_k: int = 8,
    provider: str | None = None,
    model: str | None = None,
    prompt_version: str | None = None,
) -> Answer:
    provider = provider or settings.generator_provider
    model = model or _default_model(provider)
    prompt = load_prompt(prompt_version or "default")

    with start_trace(
        name="ask",
        inputs={"question": question, "provider": provider, "model": model},
    ) as trace:
        candidates = hybrid_search(question, top_k=settings.retrieval_top_k)
        reranked = rerank(question, candidates, top_k=top_k)
        context = compress_context(question, reranked)
        trace.log("retrieval", {"candidates": len(candidates), "final": len(context)})

        if not context:
            indexed = store_count()
            msg = (
                "No documents indexed yet. Upload files in the Ingest tab to start asking questions."
                if indexed == 0
                else "I couldn't find anything relevant in the indexed documents."
            )
            return _refusal(question, msg, provider=provider, model=model)

        ctx_block = "\n\n".join(f"[{c.id}]\n{c.text}" for c in context)
        user_msg = prompt.template.replace("{question}", question).replace("{context}", ctx_block)

        try:
            text = await provider_generate(
                provider, model=model, prompt=user_msg, max_tokens=1024
            )
        except (MissingKeyError, ProviderError) as e:
            return _refusal(
                question,
                str(e),
                [Source(chunk_id=c.id, quote=c.text[:240]) for c in context[:3]],
                provider=provider,
                model=model,
            )

        trace.log("generation", {"provider": provider, "model": model, "chars": len(text)})

        return Answer(
            question=question,
            answer=text or "(empty response)",
            sources=[Source(chunk_id=c.id, quote=c.text[:280]) for c in context[:5]],
            confidence=0.85,
            refusal=False,
            provider=provider,
            model=model,
        )
