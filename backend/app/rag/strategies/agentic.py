"""Agentic RAG: Claude as a research agent driving iterative search and reasoning.

Agentic RAG delegates retrieval decisions to Claude via a tool-use loop. The model
is given semantic search, chunk fetching, and finishing tools, and decides:
- When to search and with what queries
- Which truncated previews warrant full-text fetches
- Whether to refine the search or move to answering
- When it has enough information to answer or should refuse

This approach trades increased token cost for better multi-step reasoning,
ambiguity handling, and result quality on complex queries.

Strengths:
    - Multi-step reasoning and query refinement
    - Handles ambiguous/vague questions by asking implicitly through search
    - Can discover connections between documents
    - Adaptive -- model decides retrieval strategy
    - Better for open-ended research queries

Weaknesses:
    - Higher token cost (multiple search rounds + reasoning)
    - Slower than single-pass retrieval (iterative loop)
    - Model may refine endlessly or get stuck
    - Hallucination risk if model uses outside knowledge

Use cases:
    - Open-ended research questions
    - Ambiguous user queries that need clarification via search
    - Complex multi-step questions with implicit entity relationships
    - Exploratory document analysis
"""
from __future__ import annotations

import json
import logging
from typing import Callable, Optional

from app.config import settings
from app.logging_config import get_structured_logger
from app.rag.embed import embed_query
from app.rag.providers.anthropic_provider import tool_use_loop
from app.rag.store import search as vector_search
from app.rag.strategies.base import Strategy, StrategyResult
from app.schemas import Source

logger = get_structured_logger(__name__)

_RESERVED = {"chunk_id", "doc_id", "text"}

_SYSTEM = (
    "You are a research agent answering questions strictly from a private document corpus.\n"
    "You MUST ground every claim in retrieved chunks. Never use outside knowledge.\n\n"
    "Workflow:\n"
    "  1. Call `search` with a focused query. Read the previews.\n"
    "  2. If a preview looks promising but is truncated, call `fetch_chunk` for the full text.\n"
    "  3. If your first search misses, try a different phrasing (synonyms, related concepts).\n"
    "  4. When you have found relevant chunks that answer the question, call `finish` with refusal=false, the answer, and chunk_ids.\n"
    "  5. ONLY call `finish` with refusal=true if you've tried multiple searches and the corpus genuinely has NO relevant information.\n\n"
    f"Hard limit: {settings.agentic_max_iters} tool calls total. Be efficient. Default to refusal=false when you have evidence."
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
    """Agentic RAG: Claude decides how to search and refine queries iteratively.

    Claude acts as a research agent with access to three tools:
    1. search(query, top_k) - Semantic vector search returning chunk previews
    2. fetch_chunk(chunk_id) - Retrieve full text of a specific chunk
    3. finish(answer, citations, refusal) - Emit final answer with citations

    The model decides when to search, with what queries, whether to fetch full text,
    and when to answer. The system enforces a maximum iteration limit to prevent
    infinite loops.

    Tool descriptions and constraints are embedded in _SYSTEM and _TOOLS.

    Attributes:
        name: Strategy identifier, always "agentic".

    Example:
        >>> strategy = AgenticRAG()
        >>> result = await strategy.run(
        ...     question="Who were the founders of the company?",
        ...     top_k=8,
        ...     model="claude-opus-4-7",
        ...     prompt_version="default",
        ... )
        >>> print(f"Answer: {result.answer}")
        >>> print(f"Confidence: {result.confidence}")
        >>> print(f"Iterations: {result.iterations}")
    """

    name: str = "agentic"

    async def run(
        self,
        question: str,
        *,
        top_k: int,
        model: str,
        prompt_version: str,
    ) -> StrategyResult:
        """Execute agentic RAG: run Claude in a tool-use loop to answer the question.

        Initializes internal tool handlers that manage state (seen_chunks, final answer),
        then runs Claude through a tool-use loop where it calls search/fetch/finish tools.

        Args:
            question: The user's question to research and answer.
            top_k: Not used directly; controls context size in final generation.
            model: Language model for Claude (must support tool use).
            prompt_version: Not used for agentic RAG (uses _SYSTEM instead).

        Returns:
            StrategyResult with answer from tool calls, sources, confidence,
            iteration count, and detailed trace of tool calls.

        Trace events include:
            - Each tool call with inputs and results
            - Search queries attempted
            - Chunks fetched
            - Final answer/refusal from finish tool

        Raises:
            RuntimeError if model doesn't support tool use or max iterations exceeded.
        """
        logger.info(
            "Agentic RAG strategy started",
            extra_fields={
                "strategy": "agentic",
                "question_len": len(question),
                "model": model,
                "max_iters": settings.agentic_max_iters,
            },
        )

        # Closure-captured state the tool handlers populate.
        seen_chunks: dict[str, dict] = {}
        final: dict = {}
        search_count = 0
        fetch_count = 0

        async def _tool_search(args: dict) -> str:
            nonlocal search_count
            search_count += 1
            q = args.get("query", "").strip()
            if not q:
                logger.debug(
                    "Search called with empty query",
                    extra_fields={
                        "strategy": "agentic",
                        "search_number": search_count,
                    },
                )
                return json.dumps([], ensure_ascii=False)
            k = int(args.get("top_k", 5))
            logger.debug(
                "Search tool invoked",
                extra_fields={
                    "strategy": "agentic",
                    "search_number": search_count,
                    "query_len": len(q),
                    "top_k": k,
                },
            )
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
            logger.debug(
                "Search results",
                extra_fields={
                    "strategy": "agentic",
                    "search_number": search_count,
                    "results_count": len(previews),
                    "new_chunks": len(previews),
                },
            )
            return json.dumps(previews, ensure_ascii=False)

        async def _tool_fetch(args: dict) -> str:
            nonlocal fetch_count
            fetch_count += 1
            cid = args.get("chunk_id", "").strip()
            if not cid:
                logger.debug(
                    "Fetch called with empty chunk_id",
                    extra_fields={
                        "strategy": "agentic",
                        "fetch_number": fetch_count,
                    },
                )
                return "error: chunk_id is required"
            logger.debug(
                "Fetch tool invoked",
                extra_fields={
                    "strategy": "agentic",
                    "fetch_number": fetch_count,
                    "chunk_id": cid,
                },
            )
            cached = seen_chunks.get(cid)
            if cached:
                logger.debug(
                    "Chunk retrieved from cache",
                    extra_fields={
                        "strategy": "agentic",
                        "fetch_number": fetch_count,
                        "chunk_id": cid,
                        "source": "cache",
                    },
                )
                return cached["text"]
            vec = embed_query(" ")
            hits = await vector_search(vec, top_k=512)
            for h in hits:
                if h.payload.get("chunk_id") == cid:
                    text = h.payload.get("text")
                    doc_id = h.payload.get("doc_id")
                    if text and doc_id:
                        seen_chunks[cid] = {"doc_id": doc_id, "text": text}
                        logger.debug(
                            "Chunk fetched from vector store",
                            extra_fields={
                                "strategy": "agentic",
                                "fetch_number": fetch_count,
                                "chunk_id": cid,
                                "source": "vector_store",
                                "text_len": len(text),
                            },
                        )
                        return text
            logger.debug(
                "Chunk not found",
                extra_fields={
                    "strategy": "agentic",
                    "fetch_number": fetch_count,
                    "chunk_id": cid,
                    "source": "not_found",
                },
            )
            return f"error: chunk {cid!r} not found"

        async def _tool_finish(args: dict) -> str:
            final["answer"] = args.get("answer", "")
            final["citations"] = args.get("citations", []) or []
            final["refusal"] = bool(args.get("refusal", False))
            logger.debug(
                "Finish tool called",
                extra_fields={
                    "strategy": "agentic",
                    "answer_len": len(final["answer"]),
                    "citations_count": len(final["citations"]),
                    "refusal": final["refusal"],
                },
            )
            return "ok"

        logger.debug(
            "Tool-use loop setup complete",
            extra_fields={
                "strategy": "agentic",
                "model": model,
                "max_iters": settings.agentic_max_iters,
            },
        )

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
            logger.info(
                "Agentic RAG strategy completed with finish",
                extra_fields={
                    "strategy": "agentic",
                    "model": model,
                    "answer_len": len(final["answer"]),
                    "citations_count": len(citations),
                    "refusal": final.get("refusal", False),
                    "input_tokens": out["input_tokens"],
                    "output_tokens": out["output_tokens"],
                    "iterations": out["iterations"],
                    "search_calls": search_count,
                    "fetch_calls": fetch_count,
                    "chunks_explored": len(seen_chunks),
                },
            )
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

        # Agent never called finish but has substantial output
        has_answer = out["text"] and len(out["text"].strip()) > 50
        logger.info(
            "Agentic RAG strategy completed without finish",
            extra_fields={
                "strategy": "agentic",
                "model": model,
                "has_answer": has_answer,
                "input_tokens": out["input_tokens"],
                "output_tokens": out["output_tokens"],
                "iterations": out["iterations"],
                "search_calls": search_count,
                "fetch_calls": fetch_count,
                "chunks_explored": len(seen_chunks),
            },
        )
        return StrategyResult(
            answer=out["text"] or "(agent stopped without finishing)",
            sources=[Source(chunk_id=c, quote=v["text"][:280]) for c, v in list(seen_chunks.items())[:5]]
            or [Source(chunk_id="none", quote="")],
            refusal=not has_answer,
            confidence=0.8 if has_answer else 0.3,
            input_tokens=out["input_tokens"],
            output_tokens=out["output_tokens"],
            iterations=out["iterations"],
            trace=out["trace"],
        )
