"""Graph RAG: retrieve by walking a knowledge graph, not just by similarity.

How it works
============
1. (Offline) For each indexed chunk, Claude Haiku extracts (s, p, o) triples
   → stored in `data/graph/triples.jsonl` and indexed in `graph_store`.
2. At query time:
     a. Claude Haiku pulls candidate entities out of the question.
     b. We walk the graph 1 hop from each entity → set of related chunks.
     c. We also do a normal vector search.
     d. Final context = union(graph chunks, vector chunks), deduped.
     e. The prompt is augmented with a textual rendering of the subgraph,
        so the model can reason about *connections* and not just adjacency.

Why bother?
===========
Classic RAG retrieves chunks that are *lexically/semantically close* to the
question. Graph RAG also pulls in chunks that are *structurally close* — e.g.
"how does X relate to Y" surfaces edges connecting X and Y even if no single
chunk happens to mention them together.
"""
from __future__ import annotations

from app.config import settings
from app.prompts import load_prompt
from app.rag.graph_extract import extract_question_entities
from app.rag.graph_store import describe_subgraph, load as load_graph, neighbors
from app.rag.providers.anthropic_provider import generate_with_usage
from app.rag.rerank import rerank
from app.rag.retrieve import hybrid_search
from app.rag.store import search as vector_search
from app.rag.embed import embed_query
from app.rag.strategies.base import Strategy, StrategyResult
from app.schemas import Chunk, Source

_RESERVED = {"chunk_id", "doc_id", "text"}


def _fetch_chunks_by_id(chunk_ids: set[str]) -> list[Chunk]:
    """Pull full chunks back from Qdrant by chunk_id (payload field)."""
    if not chunk_ids:
        return []
    # Qdrant doesn't index payload by default; cheapest path is a wide vector
    # search and filter client-side. For demos this is fine.
    fetched: list[Chunk] = []
    # Use the embedding of the first id as a "neutral" query — doesn't matter,
    # we just need a vector to ride along with a filter. Simpler: re-embed a
    # constant. We rely on hybrid_search elsewhere for actual scoring.
    probe = embed_query(" ")
    hits = vector_search(probe, top_k=512)
    by_id = {h.payload["chunk_id"]: h for h in hits if h.payload.get("chunk_id") in chunk_ids}
    for cid, h in by_id.items():
        fetched.append(
            Chunk(
                id=cid,
                doc_id=h.payload["doc_id"],
                text=h.payload["text"],
                tokens=len(h.payload["text"].split()),
                metadata={k: v for k, v in h.payload.items() if k not in _RESERVED},
            )
        )
    return fetched


class GraphRAG(Strategy):
    name = "graph"

    async def run(
        self,
        question: str,
        *,
        top_k: int,
        model: str,
        prompt_version: str,
    ) -> StrategyResult:
        prompt = load_prompt(prompt_version)
        graph = load_graph()

        trace: list[dict] = []

        if not graph.triples:
            return StrategyResult(
                answer=(
                    "Graph RAG is not ready yet — no triples have been extracted. "
                    "Call POST /graph/build (after ingesting documents) to populate the graph."
                ),
                sources=[Source(chunk_id="none", quote="")],
                refusal=True,
                confidence=0.0,
                trace=[{"step": "load_graph", "triples": 0}],
            )

        entities = await extract_question_entities(question)
        trace.append({"step": "extract_entities", "entities": entities})

        graph_chunks: set[str] = set()
        related_entities: set[str] = set()
        for e in entities:
            chunks, neigh = neighbors(e, hops=1)
            graph_chunks |= chunks
            related_entities |= neigh
        trace.append(
            {
                "step": "graph_walk",
                "chunks_from_graph": len(graph_chunks),
                "related_entities": sorted(related_entities)[:10],
            }
        )

        vector_chunks = await hybrid_search(question, top_k=settings.retrieval_top_k)
        vector_ids = {c.id for c in vector_chunks}
        trace.append({"step": "vector_search", "chunks_from_vectors": len(vector_chunks)})

        # Pull full chunks for graph hits that vector search missed.
        graph_only = graph_chunks - vector_ids
        extra = _fetch_chunks_by_id(graph_only)
        all_chunks = vector_chunks + extra
        ranked = rerank(question, all_chunks, top_k=top_k)
        trace.append({"step": "rerank", "kept": len(ranked)})

        if not ranked:
            return StrategyResult(
                answer="Graph RAG found neither matching entities nor similar chunks.",
                sources=[Source(chunk_id="none", quote="")],
                refusal=True,
                confidence=0.0,
                trace=trace,
            )

        subgraph_text = describe_subgraph(entities + sorted(related_entities)[:5])
        ctx_block = "\n\n".join(f"[{c.id}]\n{c.text}" for c in ranked)
        augmented_ctx = (
            f"# Knowledge graph (extracted from your documents)\n{subgraph_text}\n\n"
            f"# Relevant passages\n{ctx_block}"
        )
        user_msg = prompt.template.replace("{question}", question).replace(
            "{context}", augmented_ctx
        )

        out = await generate_with_usage(model=model, prompt=user_msg, max_tokens=1024)
        trace.append({"step": "generate", "chars": len(out["text"])})

        return StrategyResult(
            answer=out["text"] or "(empty response)",
            sources=[Source(chunk_id=c.id, quote=c.text[:280]) for c in ranked[:5]],
            input_tokens=out["input_tokens"],
            output_tokens=out["output_tokens"],
            trace=trace,
            extra={
                "entities": entities,
                "related_entities": sorted(related_entities)[:10],
                "subgraph": subgraph_text,
            },
        )
