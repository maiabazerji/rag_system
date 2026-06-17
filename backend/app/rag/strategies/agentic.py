"""Agentic RAG: Claude drives search and fetch tools in a loop.

Exposes search (semantic), fetch_chunk (by id), and finish (answer+citations) tools.
Model decides when to retrieve, how to refine queries, and when to stop. Costs more tokens
but handles multi-step reasoning and ambiguous queries better than single-pass retrieval.
"""
from __future__ import annotations

import json

from app.config import settings
from app.rag.embed import embed_query
from app.rag.providers.anthropic_provider import tool_use_loop
from app.rag.store import search as vector_search
from app.rag.strategies.base import Strategy, StrategyResult
from app.schemas import Source

_RESERVED = {"chunk_id", "doc_id", "text"}

_SYSTEM = (
    "You are a research agent answering questions strictly from a private document corpus.\n"
    "You MUST ground every claim in retrieved chunks. Never use outside knowledge.\n\n"
    "Workflow:\n"
    "  1. Call `search` with a focused query. Read the previews.\n"
    "  2. If a preview looks promising but is truncated, call `fetch_chunk` for the full text.\n"
    "  3. If your first search misses, try a different phrasing (synonyms, related concepts).\n"
    "  4. When you have enough evidence, call `finish` with the answer and the chunk_ids you used.\n"
    "  5. If the corpus genuinely doesn't cover the question, call `finish` with refusal=true and a brief reason.\n\n"
    f"Hard limit: {settings.agentic_max_iters} tool calls total. Be efficient."
)

_TOOLS = [
    {
        "name": "search",
        "description": "Semantic search the indexed documents. Returns up to N chunks "
        "with id and short preview.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "what to search for"},
                "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 12},
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_chunk",
        "description": "Read the full text of a specific chunk by its id "
        "(only use ids returned by `search`).",
        "input_schema": {
            "type": "object",
            "properties": {"chunk_id": {"type": "string"}},
            "required": ["chunk_id"],
        },
    },
    {
        "name": "finish",
        "description": "Emit the final answer. Call exactly once when you have evidence.",
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "citations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "chunk_ids that support the answer",
                },
                "refusal": {"type": "boolean", "default": False},
            },
            "required": ["answer", "citations"],
        },
    },
]


class AgenticRAG(Strategy):
    name = "agentic"

    async def run(
        self,
        question: str,
        *,
        top_k: int,
        model: str,
        prompt_version: str,
    ) -> StrategyResult:
        # Closure-captured state the tool handlers populate.
        seen_chunks: dict[str, dict] = {}
        final: dict = {}

        async def _tool_search(args: dict) -> str:
            q = args.get("query", "").strip()
            if not q:
                return json.dumps([], ensure_ascii=False)
            k = int(args.get("top_k", 5))
            vec = embed_query(q)
            hits = await vector_search(vec, top_k=k)
            previews = []
            for h in hits:
                cid = h.payload.get("chunk_id")
                text = h.payload.get("text")
                doc_id = h.payload.get("doc_id")
                if cid and text and doc_id:
                    seen_chunks[cid] = {"doc_id": doc_id, "text": text}
                    previews.append({"chunk_id": cid, "preview": text[:300]})
            return json.dumps(previews, ensure_ascii=False)

        async def _tool_fetch(args: dict) -> str:
            cid = args.get("chunk_id", "").strip()
            if not cid:
                return "error: chunk_id is required"
            cached = seen_chunks.get(cid)
            if cached:
                return cached["text"]
            vec = embed_query(" ")
            hits = await vector_search(vec, top_k=512)
            for h in hits:
                if h.payload.get("chunk_id") == cid:
                    text = h.payload.get("text")
                    doc_id = h.payload.get("doc_id")
                    if text and doc_id:
                        seen_chunks[cid] = {"doc_id": doc_id, "text": text}
                        return text
            return f"error: chunk {cid!r} not found"

        async def _tool_finish(args: dict) -> str:
            final["answer"] = args.get("answer", "")
            final["citations"] = args.get("citations", []) or []
            final["refusal"] = bool(args.get("refusal", False))
            return "ok"

        out = await tool_use_loop(
            model=model,
            system=_SYSTEM,
            user_message=f"Question: {question}",
            tools=_TOOLS,
            tool_handlers={
                "search": _tool_search,
                "fetch_chunk": _tool_fetch,
                "finish": _tool_finish,
            },
            max_iters=settings.agentic_max_iters,
            max_tokens=1024,
        )

        if final:
            citations = [c for c in final["citations"] if c in seen_chunks]
            sources = [
                Source(chunk_id=c, quote=seen_chunks[c]["text"][:280]) for c in citations
            ] or [Source(chunk_id="none", quote="")]
            return StrategyResult(
                answer=final["answer"] or "(empty)",
                sources=sources,
                refusal=final.get("refusal", False),
                confidence=0.6 if final.get("refusal") else 0.9,
                input_tokens=out["input_tokens"],
                output_tokens=out["output_tokens"],
                iterations=out["iterations"],
                trace=out["trace"],
                extra={"chunks_explored": list(seen_chunks.keys())},
            )

        # Agent never called finish — return what we have.
        return StrategyResult(
            answer=out["text"] or "(agent stopped without finishing)",
            sources=[Source(chunk_id=c, quote=v["text"][:280]) for c, v in list(seen_chunks.items())[:5]]
            or [Source(chunk_id="none", quote="")],
            refusal=True,
            confidence=0.3,
            input_tokens=out["input_tokens"],
            output_tokens=out["output_tokens"],
            iterations=out["iterations"],
            trace=out["trace"],
        )
