"""Deterministic retrieval metrics, scored against a golden example's expected sources.

These need no model call. That matters twice over: they cost nothing to compute,
and they still produce numbers when the LLM judge is unavailable -- so an eval
run always yields *some* measurement.

They are also the metrics that separate retrieval strategies most cleanly. The
judge scores the answer, which mixes retrieval quality with generation quality;
precision and recall here score retrieval alone.
"""
from __future__ import annotations

from app.schemas import RetrievalScore, Source


def _normalize(name: str) -> str:
    """Reduce a document reference to a comparable key.

    Golden datasets list bare filenames (`rag_overview.md`) while a source may
    carry a path or differ in case, so both sides are reduced to a lowercase
    basename before comparison.
    """
    return name.strip().replace("\\", "/").rsplit("/", 1)[-1].lower()


def retrieved_documents(sources: list[Source]) -> list[str]:
    """Documents behind an answer's sources, best-ranked first, deduplicated.

    Args:
        sources: The answer's sources, in rank order.

    Returns:
        Document names in first-seen order. Sources with no document (older
        answers, or the placeholder used for refusals) are skipped.
    """
    ordered: dict[str, None] = {}
    for source in sources:
        if source.document and source.chunk_id != "none":
            ordered.setdefault(source.document, None)
    return list(ordered)


def score_retrieval(
    expected_sources: list[str] | None, sources: list[Source]
) -> RetrievalScore | None:
    """Score retrieval for one example.

    Args:
        expected_sources: Documents the golden example expects. When absent or
            empty there is nothing to score against.
        sources: The answer's sources, in rank order.

    Returns:
        A :class:`RetrievalScore`, or ``None`` when the example declares no
        expected sources -- the same "not measured" convention the judge uses.

    Example:
        >>> score = score_retrieval(["rag_overview.md"], answer.sources)
        >>> score.recall  # did retrieval surface the document at all
        1.0
    """
    if not expected_sources:
        return None

    retrieved = retrieved_documents(sources)
    expected_keys = {_normalize(e) for e in expected_sources}
    retrieved_keys = [_normalize(r) for r in retrieved]

    if not retrieved_keys:
        return RetrievalScore(
            precision=0.0,
            recall=0.0,
            hit=False,
            mrr=0.0,
            retrieved=[],
            expected=list(expected_sources),
        )

    matched = [k for k in retrieved_keys if k in expected_keys]

    # Reciprocal rank of the first expected document, 1-indexed.
    mrr = 0.0
    for rank, key in enumerate(retrieved_keys, start=1):
        if key in expected_keys:
            mrr = 1.0 / rank
            break

    return RetrievalScore(
        precision=len(matched) / len(retrieved_keys),
        recall=len(set(matched)) / len(expected_keys),
        hit=bool(matched),
        mrr=mrr,
        retrieved=retrieved,
        expected=list(expected_sources),
    )


def aggregate_retrieval(scores: list[dict]) -> dict | None:
    """Average retrieval scores across examples.

    Args:
        scores: Serialized :class:`RetrievalScore` dicts. Callers filter out
            examples that had nothing to score first.

    Returns:
        Mean precision, recall, hit rate and MRR, plus the number of examples
        the average covers -- or ``None`` if there were none.
    """
    if not scores:
        return None
    n = len(scores)
    return {
        "precision": sum(s["precision"] for s in scores) / n,
        "recall": sum(s["recall"] for s in scores) / n,
        "hit_rate": sum(1 for s in scores if s["hit"]) / n,
        "mrr": sum(s["mrr"] for s in scores) / n,
        "n": n,
    }
