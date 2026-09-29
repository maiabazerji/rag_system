"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009).

Dense and BM25 scores live on unrelated scales (cosine similarity against an
unbounded sum of IDF-weighted terms), so they cannot be added. RRF ignores the
scores and fuses the *ranks*::

    score(d) = sum over retrievers i of  1 / (k + rank_i(d))

with 1-based ranks and a document absent from a list contributing nothing. The
constant ``k`` (60 in the paper) damps the head of each list, so one
retriever's first place does not drown out broad agreement lower down.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from app.schemas import Chunk

DEFAULT_RRF_K = 60


@dataclass
class FusedCandidate:
    """One chunk after fusion.

    Attributes:
        chunk: The chunk, taken from the list that ranked it best.
        fused_score: Its RRF score.
        ranks: Its 1-based rank in each list it appeared in, by list name.
    """

    chunk: Chunk
    fused_score: float = 0.0
    ranks: dict[str, int] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return self.chunk.id

    @property
    def best_rank(self) -> int:
        return min(self.ranks.values())

    @property
    def dense_rank(self) -> int | None:
        return self.ranks.get("dense")

    @property
    def sparse_rank(self) -> int | None:
        return self.ranks.get("sparse")


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[Chunk]], k: int = DEFAULT_RRF_K
) -> list[FusedCandidate]:
    """Fuse ranked chunk lists with RRF.

    Args:
        rankings: Named ranked lists, best first, e.g. ``{"dense": [...],
            "sparse": [...]}``. A chunk repeated within one list counts once,
            at its best rank.
        k: The RRF constant; must be positive.

    Returns:
        One candidate per distinct chunk id, ordered by fused score
        (descending), then best individual rank (ascending), then chunk id --
        a total order, so the result does not depend on input order or on
        hash iteration order.

    Raises:
        ValueError: If ``k`` is not positive.

    Example:
        >>> fused = reciprocal_rank_fusion({"dense": [a, b], "sparse": [b, c]}, k=60)
        >>> [c.chunk_id for c in fused]  # b: 1/62 + 1/61 beats a: 1/61
        ['b', 'a', 'c']
    """
    if k <= 0:
        raise ValueError(f"RRF k must be positive, got {k}")

    by_id: dict[str, FusedCandidate] = {}
    # Sorted names: floating-point sums are then accumulated in the same
    # order whichever way the caller built the mapping.
    for name in sorted(rankings):
        for rank, chunk in enumerate(rankings[name], start=1):
            cand = by_id.get(chunk.id)
            if cand is None:
                by_id[chunk.id] = FusedCandidate(chunk=chunk, ranks={name: rank})
                continue
            if name in cand.ranks:
                continue  # duplicate within this list: its first rank stands
            if rank < cand.best_rank:
                cand.chunk = chunk  # keep the payload from the best ranking
            cand.ranks[name] = rank

    for cand in by_id.values():
        cand.fused_score = sum(1.0 / (k + r) for _, r in sorted(cand.ranks.items()))

    return sorted(by_id.values(), key=lambda c: (-c.fused_score, c.best_rank, c.chunk_id))
