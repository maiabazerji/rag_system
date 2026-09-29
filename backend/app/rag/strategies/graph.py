"""Graph RAG: retrieve by walking a knowledge graph for multi-hop reasoning.

Graph RAG augments vector search with structured entity relationships extracted from
the corpus. The approach:

1. Offline: Extract entity triples (subject, predicate, object) from documents using
   Claude, building a knowledge graph of relationships
2. At query time: Extract entities from the question, walk the graph 1 hop to find
   related entities and their associated chunks
3. Combine graph results with vector search results for better coverage
4. Rerank combined results and inject both graph structure and passages into context

Strengths:
    - Captures multi-hop relationships ("how is X connected to Y through Z?")
    - Surface relevant chunks that vector search might miss due to entity synonymy
    - Explicit graph structure helps LLM reason about connections
    - Better for knowledge-base style queries with complex entity relationships

Weaknesses:
    - Requires offline graph extraction (extra cost + latency during setup)
    - Graph quality depends on entity extraction accuracy
    - More token consumption due to enriched context
    - Graph walk limited to single hop (could extend but trade-off with performance)

Use cases:
    - Knowledge graph queries ("who did X work with?", "what products do companies Y make?")
    - Multi-document synthesis with entity relationships
    - Domain-specific fact databases with clear entities/relationships
"""
from __future__ import annotations

from app.access import AccessScope
from app.config import settings
from app.i18n import localized
from app.logging_config import get_structured_logger
from app.prompts import render_prompt
from app.rag.graph_extract import extract_question_entities
from app.rag.graph_store import describe_subgraph, neighbors
from app.rag.graph_store import load as load_graph
from app.rag.grounding import (
    SUBMIT_ANSWER_TOOL,
    cite_chunks,
    format_context,
    ground,
    grounded_system,
    grounding_extra,
    validation_attributes,
)
from app.rag.providers.anthropic_provider import generate_structured
from app.rag.rerank import rerank_async
from app.rag.retrieve import hybrid_search
from app.rag.store import fetch_chunks, readable_doc_ids
from app.rag.strategies.base import Strategy, StrategyResult
from app.rag.timing import stage
from app.schemas import Chunk, Source

logger = get_structured_logger(__name__)

_RESERVED = {"chunk_id", "doc_id", "text"}


async def _fetch_chunks_by_id(
    chunk_ids: set[str], access: AccessScope | None = None
) -> list[Chunk]:
    """Fetch full Chunk objects from the vector store by their IDs.

    Looks the points up directly by id, so every requested chunk that exists is
    found, however large the corpus.

    Args:
        chunk_ids: Set of chunk IDs to fetch.
        access: Fetch only chunks this scope may read.

    Returns:
        List of Chunk objects found in the store. IDs not found are silently skipped.
            Chunks are reconstructed from vector store payloads (chunk_id, doc_id, text,
            metadata).

    Raises:
        StoreUnavailable: If the vector store cannot be reached.
    """
    if not chunk_ids:
        return []
    fetched: list[Chunk] = []
    hits = await fetch_chunks(chunk_ids, access=access)
    by_id = {h.payload["chunk_id"]: h for h in hits if h.payload.get("chunk_id") in chunk_ids}
    for cid, h in by_id.items():
        doc_id = h.payload.get("doc_id")
        text = h.payload.get("text")
        if doc_id is not None and text is not None:
            fetched.append(
                Chunk(
                    id=cid,
                    doc_id=doc_id,
                    text=text,
                    tokens=len(text.split()),
                    metadata={k: v for k, v in h.payload.items() if k not in _RESERVED},
                )
            )

    logger.debug(
        "Fetched chunks by ID",
        extra_fields={
            "requested_count": len(chunk_ids),
            "found_count": len(fetched),
            "search_breadth": 512,
        },
    )
    return fetched


class GraphRAG(Strategy):
    """Graph-based RAG: entity-aware retrieval using knowledge graph structure.

    Combines vector search with knowledge graph traversal to surface relevant chunks
    that might be missed by pure semantic similarity. The graph encodes entity
    relationships extracted during corpus ingestion, enabling multi-hop reasoning.

    Graph construction is offline (happens during ingest via extract_entities),
    while query-time execution walks the graph from extracted question entities
    to find related passages.

    Attributes:
        name: Strategy identifier, always "graph".

    Example:
        >>> strategy = GraphRAG()
        >>> result = await strategy.run(
        ...     question="What companies did Steve Jobs work for?",
        ...     top_k=8,
        ...     model="claude-sonnet-5",
        ...     prompt_version="default",
        ... )
        >>> print(f"Answer: {result.answer}")
        >>> print(f"Entities found: {result.extra['entities']}")
    """

    name: str = "graph"

    async def run(
        self,
        question: str,
        *,
        top_k: int,
        model: str,
        prompt_version: str,
        access: AccessScope | None = None,
    ) -> StrategyResult:
        """Execute graph RAG: extract entities, walk graph, combine with vector search.

        Args:
            question: The user's question to answer.
            top_k: Number of chunks to include in context for generation.
            model: Language model ID for generation.
            prompt_version: Prompt template version.
            access: The caller's read scope. Vector search, chunk fetches and
                the graph walk (edges, entities, subgraph text) are all
                restricted to documents it may read.

        Returns:
            StrategyResult with answer, sources, and rich trace including:
                - Extracted entities from question
                - Graph walk results (related entities)
                - Vector search results
                - Reranking scores
                - Subgraph structure (for UI visualization)

        Raises:
            Returns a refusal StrategyResult if:
                - Graph has not been built (no triples in store)
                - No entities extracted from question
                - No relevant chunks found after combining graph + vector results
        """
        logger.info(
            "Graph RAG strategy started",
            extra_fields={
                "strategy": "graph",
                "question_len": len(question),
                "top_k": top_k,
                "model": model,
            },
        )

        graph = load_graph()

        trace: list[dict] = []

        if not graph.triples:
            logger.warning(
                "Graph not built",
                extra_fields={
                    "strategy": "graph",
                    "triples_count": 0,
                    "refusal": True,
                },
            )
            return StrategyResult(
                answer=localized("graph_not_built", question),
                sources=[Source(chunk_id="none", quote="")],
                refusal=True,
                confidence=0.0,
                trace=[{"step": "load_graph", "triples": 0}],
            )

        logger.debug(
            "Graph loaded",
            extra_fields={
                "strategy": "graph",
                "triples_count": len(graph.triples),
            },
        )

        # Entity extraction is query analysis for the graph walk, so its time
        # counts toward retrieval; its tokens and cost count like any model call.
        with stage("retrieval", span_name="graph.entity_extraction") as s:
            extracted = await extract_question_entities(question)
            if s is not None:
                s.metadata["entities"] = len(extracted)
        # Entity extraction is a model call too; its tokens belong in the totals.
        extract_in = int(getattr(extracted, "input_tokens", 0) or 0)
        extract_out = int(getattr(extracted, "output_tokens", 0) or 0)
        entities = list(extracted)
        logger.debug(
            "Question entities extracted",
            extra_fields={
                "strategy": "graph",
                "entities_count": len(entities),
                "entities": entities,
            },
        )
        trace.append({"step": "extract_entities", "entities": entities})

        # The graph is built from the whole corpus; walk only the part the
        # caller may read, or its triples would leak other documents' facts.
        with stage("retrieval", span_name="graph.walk") as s:
            visible_docs = await readable_doc_ids(access) if access is not None else None

            graph_chunks: set[str] = set()
            related_entities: set[str] = set()
            for e in entities:
                chunks, neigh = neighbors(e, hops=1, doc_ids=visible_docs)
                graph_chunks |= chunks
                related_entities |= neigh
            if s is not None:
                s.metadata.update(
                    {"chunks": len(graph_chunks), "related_entities": len(related_entities)}
                )

        logger.debug(
            "Graph walk completed",
            extra_fields={
                "strategy": "graph",
                "graph_chunks_count": len(graph_chunks),
                "related_entities_count": len(related_entities),
            },
        )
        trace.append(
            {
                "step": "graph_walk",
                "chunks_from_graph": len(graph_chunks),
                "related_entities": sorted(related_entities)[:10],
            }
        )

        # Hybrid (dense + BM25, fused) candidates, left unreranked: the graph's
        # own chunks join them below and the whole pool is reranked together.
        retrieval = await hybrid_search(
            question, access, final_k=settings.dense_top_k, rerank=False
        )
        vector_chunks = retrieval.chunks
        vector_ids = {c.id for c in vector_chunks}
        logger.debug(
            "Vector search completed",
            extra_fields={
                "strategy": "graph",
                "mode": retrieval.diagnostics.mode,
                "vector_chunks_count": len(vector_chunks),
                "dense_top_k": settings.dense_top_k,
            },
        )
        trace.append({"step": "vector_search", "chunks_from_vectors": len(vector_chunks)})

        # Pull full chunks for graph hits that vector search missed.
        graph_only = graph_chunks - vector_ids
        logger.debug(
            "Graph-only chunks identified",
            extra_fields={
                "strategy": "graph",
                "graph_only_count": len(graph_only),
                "overlap_count": len(vector_ids & graph_chunks),
            },
        )

        with stage("retrieval", log=False):
            extra = await _fetch_chunks_by_id(graph_only, access=access)
        all_chunks = vector_chunks + extra
        logger.debug(
            "Combined chunks from graph and vector search",
            extra_fields={
                "strategy": "graph",
                "total_chunks": len(all_chunks),
                "vector_chunks": len(vector_chunks),
                "graph_extra_chunks": len(extra),
            },
        )

        with stage("rerank"):
            ranked = await rerank_async(question, all_chunks, top_k=top_k)
        logger.debug(
            "Reranking completed",
            extra_fields={
                "strategy": "graph",
                "reranked_count": len(ranked),
                "requested_top_k": top_k,
            },
        )
        trace.append({"step": "rerank", "kept": len(ranked)})

        if not ranked:
            logger.warning(
                "No ranked chunks found",
                extra_fields={
                    "strategy": "graph",
                    "refusal": True,
                    "total_chunks": len(all_chunks),
                },
            )
            return StrategyResult(
                answer=localized("graph_no_match", question),
                sources=[Source(chunk_id="none", quote="")],
                refusal=True,
                confidence=0.0,
                input_tokens=extract_in,
                output_tokens=extract_out,
                trace=trace,
                candidate_count=len(all_chunks),
            )

        with stage("context_selection", span_name="context_selection") as s:
            subgraph_text = describe_subgraph(
                entities + sorted(related_entities)[:5], doc_ids=visible_docs
            )
            cited = cite_chunks(ranked)
            ctx_block = format_context(cited)
            augmented_ctx = (
                f"# Knowledge graph (extracted from your documents)\n{subgraph_text}\n\n"
                f"# Relevant passages\n{ctx_block}"
            )
            user_msg = render_prompt(prompt_version, question=question, context=augmented_ctx)
            if s is not None:
                s.metadata.update(
                    {
                        "candidates": len(all_chunks),
                        "selected": len(cited),
                        "context_chars": len(augmented_ctx),
                        "prompt_version": prompt_version,
                    }
                )
        logger.debug(
            "Context prepared",
            extra_fields={
                "strategy": "graph",
                "augmented_context_size": len(augmented_ctx),
                "subgraph_size": len(subgraph_text),
                "passages_count": len(ranked),
            },
        )
        logger.debug(
            "Prompt rendered",
            extra_fields={
                "strategy": "graph",
                "prompt_len": len(user_msg),
                "prompt_version": prompt_version,
            },
        )

        with stage("generation"):
            out = await generate_structured(
                model=model,
                prompt=user_msg,
                tool=SUBMIT_ANSWER_TOOL,
                system=grounded_system("anthropic"),
                max_tokens=settings.max_answer_tokens,
            )
        with stage("citation_validation", span_name="citation_validation") as s:
            grounded, raw = ground(question, out, cited)
            if s is not None:
                s.metadata.update(validation_attributes(grounded))
        trace.append(
            {"step": "generate", "mode": grounded.mode, "chars": len(out.get("text") or "")}
        )
        trace.append(
            {
                "step": "validate_citations",
                "status": grounded.status,
                "grounded": grounded.grounded,
                "cited": [c.handle for c in grounded.cited],
                "invalid": grounded.invalid_citations,
            }
        )

        logger.info(
            "Graph RAG strategy completed",
            extra_fields={
                "strategy": "graph",
                "model": model,
                "input_tokens": out["input_tokens"],
                "output_tokens": out["output_tokens"],
                "answer_len": len(grounded.answer),
                "sources_count": grounded.citation_count,
                "grounded": grounded.grounded,
                "status": grounded.status,
                "entities_found": len(entities),
            },
        )

        return StrategyResult(
            **grounded.result_fields(),
            input_tokens=out["input_tokens"] + extract_in,
            output_tokens=out["output_tokens"] + extract_out,
            candidate_count=len(all_chunks),
            context_count=len(ranked),
            trace=trace,
            extra={
                "entities": entities,
                "related_entities": sorted(related_entities)[:10],
                "subgraph": subgraph_text,
                "retrieved_ids": [c.id for c in ranked],
                "retrieved_docs": [c.metadata.get("filename") for c in ranked],
                "context_text": augmented_ctx,
                "retrieval": retrieval.diagnostics_for(ranked).model_dump(),
                **grounding_extra(grounded, raw, cited),
            },
        )
