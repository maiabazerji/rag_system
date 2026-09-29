"""Deterministic retrieval metrics, scored against a golden example's expected sources.

These need no model call. That matters twice over: they cost nothing to compute,
and they still produce numbers when the LLM judge is unavailable -- so an eval
run always yields *some* measurement.

They are also the metrics that separate retrieval strategies most cleanly. The
judge scores the answer, which mixes retrieval quality with generation quality;
precision and recall here score retrieval alone.

Ranked metrics
--------------
Relevance is binary and judged at *document* level: a retrieved document is
relevant when its key (see :func:`doc_key`) is in the example's relevant set
``G``. Retrieval output is the ranked, deduplicated document list
``r_1 .. r_n`` and ``rel_i = 1`` when ``r_i`` is in ``G``, else ``0``.
For a cutoff ``K``:

* ``Precision@K = (sum_{i<=K} rel_i) / K`` -- the denominator is ``K`` even
  when fewer than ``K`` documents came back (trec_eval convention), so
  returning less is not rewarded.
* ``Recall@K    = |{r_i in G : i <= K}| / |G|``
* ``HitRate@K   = 1 if any rel_i = 1 for i <= K else 0``
* ``MRR         = 1 / min{i : rel_i = 1}`` over the whole list, ``0`` if none.
* ``nDCG@K      = DCG@K / IDCG@K`` with ``DCG@K = sum_{i<=K} rel_i / log2(i + 1)``
  and ``IDCG@K = sum_{i=1}^{min(|G|, K)} 1 / log2(i + 1)`` (the DCG of a
  perfect ranking).

When an example has no relevant documents (``G`` empty) every metric is
undefined -- not zero -- so :func:`ranked_retrieval_metrics` returns ``None``
and the example is excluded from aggregates and counted separately.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from app.schemas import RetrievalScore, Source

#: Default rank cutoffs reported for every @K metric.
DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 5, 10)

#: Per-cutoff metric families, reported as ``<family>@<K>``.
AT_K_FAMILIES: tuple[str, ...] = ("recall", "precision", "hit_rate", "ndcg")

_DOC_EXTENSIONS = frozenset(
    {"md", "markdown", "txt", "pdf", "html", "htm", "docx", "doc", "pptx", "odt", "rst", "csv"}
)


def _normalize(name: str) -> str:
    """Reduce a document reference to a comparable key.

    Golden datasets list bare filenames (`rag_overview.md`) or bare document
    ids (`rag_overview`) while a source may carry a path or differ in case, so
    both sides are reduced to a lowercase basename with any document extension
    removed before comparison.
    """
    base = name.strip().replace("\\", "/").rsplit("/", 1)[-1].lower()
    stem, dot, ext = base.rpartition(".")
    if dot and stem and ext in _DOC_EXTENSIONS:
        return stem
    return base


def doc_key(name: str) -> str:
    """Public alias of the document-key normalisation used for matching."""
    return _normalize(name)


def relevant_documents(example: Mapping[str, Any]) -> list[str] | None:
    """The documents a golden example declares relevant.

    ``relevant_doc_ids`` takes precedence over the older ``expected_sources``
    when both are present and non-empty: a dataset that introduced the new
    field did so to correct or replace the old labels.

    Returns:
        The labels as written, or ``None`` when the example declares none.
    """
    for field in ("relevant_doc_ids", "expected_sources"):
        value = example.get(field)
        if value:
            return [str(v) for v in value]
    return None


def _check_k_values(k_values: Iterable[int]) -> tuple[int, ...]:
    ks = tuple(sorted(set(k_values)))
    if not ks or any(not isinstance(k, int) or k < 1 for k in ks):
        raise ValueError(f"K values must be positive integers, got {ks!r}")
    return ks


def precision_at_k(relevance: Sequence[int], k: int) -> float:
    """``(sum of rel_i for i <= k) / k``."""
    return sum(relevance[:k]) / k


def recall_at_k(relevance: Sequence[int], k: int, n_relevant: int) -> float:
    """``(relevant documents in the top k) / n_relevant``.

    ``relevance`` must come from a deduplicated ranking, so each relevant
    document contributes at most once.
    """
    return min(1.0, sum(relevance[:k]) / n_relevant)


def hit_rate_at_k(relevance: Sequence[int], k: int) -> float:
    """``1.0`` when any of the top ``k`` is relevant, else ``0.0``."""
    return 1.0 if any(relevance[:k]) else 0.0


def reciprocal_rank(relevance: Sequence[int]) -> float:
    """``1 / rank`` of the first relevant document (1-indexed), ``0.0`` if none."""
    for rank, rel in enumerate(relevance, start=1):
        if rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevance: Sequence[int], k: int, n_relevant: int) -> float:
    """Binary-relevance nDCG with a ``log2(rank + 1)`` discount.

    ``IDCG`` is computed from ``n_relevant`` (the size of the label set), not
    from what was retrieved, so a ranking that misses relevant documents is
    penalised even if what it did return was perfectly ordered.
    """
    dcg = sum(rel / math.log2(i + 1) for i, rel in enumerate(relevance[:k], start=1))
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(n_relevant, k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def ranked_retrieval_metrics(
    relevant: Iterable[str] | None,
    retrieved: Sequence[str],
    k_values: Iterable[int] = DEFAULT_K_VALUES,
) -> dict[str, float] | None:
    """Every ranked metric for one example, keyed ``recall@5``, ``mrr`` etc.

    Args:
        relevant: Relevant document labels (filenames or ids). Matched through
            :func:`doc_key`, so ``a.md``, ``docs/A.md`` and ``a`` are equal.
        retrieved: Retrieved documents, best first. Duplicates are collapsed
            to their first (best) position before scoring.
        k_values: Rank cutoffs.

    Returns:
        ``{"mrr": ..., "recall@K": ..., "precision@K": ..., "hit_rate@K": ...,
        "ndcg@K": ...}`` for each ``K``, or ``None`` when ``relevant`` is empty
        (every metric is undefined without a label).

    Raises:
        ValueError: If a K value is not a positive integer.
    """
    ks = _check_k_values(k_values)
    relevant_keys = {doc_key(r) for r in relevant or () if str(r).strip()}
    if not relevant_keys:
        return None

    ranked: dict[str, None] = {}
    for doc in retrieved:
        if doc:
            ranked.setdefault(doc_key(doc), None)
    relevance = [1 if key in relevant_keys else 0 for key in ranked]
    n_rel = len(relevant_keys)

    out: dict[str, float] = {"mrr": reciprocal_rank(relevance)}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(relevance, k, n_rel)
        out[f"precision@{k}"] = precision_at_k(relevance, k)
        out[f"hit_rate@{k}"] = hit_rate_at_k(relevance, k)
        out[f"ndcg@{k}"] = ndcg_at_k(relevance, k, n_rel)
    return out


def ranked_metric_names(k_values: Iterable[int] = DEFAULT_K_VALUES) -> list[str]:
    """Names produced by :func:`ranked_retrieval_metrics`, in report order."""
    ks = _check_k_values(k_values)
    return ["mrr"] + [f"{family}@{k}" for family in AT_K_FAMILIES for k in ks]


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
    return score_retrieval_documents(expected_sources, retrieved_documents(sources))


def score_retrieval_documents(
    expected_sources: list[str] | None, retrieved: list[str]
) -> RetrievalScore | None:
    """Score retrieval for one example from a ranked list of documents.

    Args:
        expected_sources: Documents the golden example expects.
        retrieved: Documents behind the retrieved context, best-ranked first
            and deduplicated.

    Returns:
        A :class:`RetrievalScore`, or ``None`` when nothing is expected.
    """
    if not expected_sources:
        return None

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
        the average covers -- or ``None`` if there were none. Ranked ``@K``
        metrics (see :func:`ranked_retrieval_metrics`) present in the dicts are
        averaged too, over the examples that carry them.
    """
    if not scores:
        return None
    n = len(scores)
    out: dict[str, Any] = {
        "precision": sum(s["precision"] for s in scores) / n,
        "recall": sum(s["recall"] for s in scores) / n,
        "hit_rate": sum(1 for s in scores if s["hit"]) / n,
        "mrr": sum(s["mrr"] for s in scores) / n,
        "n": n,
    }
    at_k = sorted({k for s in scores for k in s if "@" in k})
    for key in at_k:
        values = [s[key] for s in scores if s.get(key) is not None]
        if values:
            out[key] = sum(values) / len(values)
    return out


def score_example_retrieval(
    example: Mapping[str, Any],
    retrieved: Sequence[str],
    k_values: Iterable[int] = DEFAULT_K_VALUES,
) -> dict[str, Any] | None:
    """Full retrieval record for one golden example.

    Combines the unranked :class:`RetrievalScore` (kept for existing
    consumers) with every ranked metric from :func:`ranked_retrieval_metrics`.

    Returns:
        A JSON-ready dict, or ``None`` when the example has no relevance labels
        (it is then excluded from retrieval aggregates).
    """
    relevant = relevant_documents(example)
    base = score_retrieval_documents(relevant, list(retrieved))
    ranked = ranked_retrieval_metrics(relevant, retrieved, k_values)
    if base is None or ranked is None:
        return None
    return {**base.model_dump(), **ranked}
