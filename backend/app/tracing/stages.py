"""The request pipeline's stages, as recorded by spans.

Span names come from the code that records them (``retrieval.dense``,
``generate``, ``agent_step``...). This module maps them onto the stable stage
names the trace API and dashboards use, in pipeline order::

    preprocess → dense → sparse → fusion → rerank → context_selection
               → generation → citation_validation → response

Graph RAG adds ``entity_extraction`` and ``graph_walk`` before retrieval.
Only a span's own metadata (counts, flags, timings, token and cost figures)
is ever surfaced here, never its input or output content.
"""
from __future__ import annotations

from typing import Any

from app.tracing.spans import Span, Trace

PIPELINE_ORDER = (
    "entity_extraction",
    "graph_walk",
    "preprocess",
    "dense",
    "sparse",
    "fusion",
    "rerank",
    "context_selection",
    "generation",
    "citation_validation",
    "response",
)

SPAN_STAGES: dict[str, str] = {
    "graph.entity_extraction": "entity_extraction",
    "graph.walk": "graph_walk",
    "retrieval.preprocess": "preprocess",
    "retrieval.dense": "dense",
    "retrieval.sparse": "sparse",
    "retrieval.fusion": "fusion",
    "retrieval.rerank": "rerank",
    "rerank": "rerank",
    "context_selection": "context_selection",
    "generate": "generation",
    "agent_step": "generation",
    "citation_validation": "citation_validation",
    "response": "response",
}


def stage_of(span: Span) -> str | None:
    """The pipeline stage a span represents, or None for a helper span."""
    return SPAN_STAGES.get(span.name)


def pipeline_stages(trace: Trace) -> list[dict[str, Any]]:
    """The trace's stage spans in start order, for a waterfall view.

    A stage span nested inside another stage span (the reranker's own span
    inside ``retrieval.rerank``, the extraction call inside
    ``graph.entity_extraction``) is part of its parent and is left out, so
    each stage appears once per time it ran. Agentic runs repeat stages: one
    ``generation`` per agent step, one retrieval sequence per search.
    """
    by_id = {s.id: s for s in trace.spans}

    def inside_stage(s: Span) -> bool:
        parent = by_id.get(s.parent_id) if s.parent_id else None
        while parent is not None:
            if stage_of(parent) is not None:
                return True
            parent = by_id.get(parent.parent_id) if parent.parent_id else None
        return False

    rows = []
    for s in sorted(trace.spans, key=lambda x: x.start_ns):
        stage = stage_of(s)
        if stage is None or inside_stage(s):
            continue
        end = s.end_ns if s.end_ns is not None else s.start_ns
        row: dict[str, Any] = {
            "stage": stage,
            "span": s.name,
            "start_ms": round((s.start_ns - trace.start_ns) / 1e6, 3),
            "duration_ms": round((end - s.start_ns) / 1e6, 3),
            "attributes": dict(s.metadata),
        }
        if s.model:
            row["model"] = s.model
        if s.usage:
            row["usage"] = dict(s.usage)
        if s.cost:
            row["cost_usd"] = s.cost.get("total")
        if s.error:
            row["error"] = s.error
        rows.append(row)
    return rows
