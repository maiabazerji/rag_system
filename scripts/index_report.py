"""Report how the corpus is distributed across chunks in the vector store.

Retrieval can only surface a document if that document has chunks competing in
the similarity search. A document short enough to become a single chunk is
structurally disadvantaged against one that became a dozen -- so when retrieval
keeps missing the same documents, this is the first place to look.

Usage:
    python scripts/index_report.py
    python scripts/index_report.py --only evaluation_metrics.md observability_wandb.md
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import settings  # noqa: E402
from app.rag.store import client  # noqa: E402


async def chunk_counts() -> collections.Counter:
    """Count indexed chunks per source document."""
    qdrant = await client()
    counts: collections.Counter = collections.Counter()
    offset = None
    while True:
        points, offset = await qdrant.scroll(
            collection_name=settings.qdrant_collection,
            limit=256,
            offset=offset,
            with_payload=True,
        )
        for point in points:
            counts[point.payload.get("filename") or "(unknown)"] += 1
        if offset is None:
            break
    return counts


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--only", nargs="*", default=None, help="Report only these documents"
    )
    args = parser.parse_args()

    counts = await chunk_counts()
    if not counts:
        print("Nothing is indexed. Run scripts/ingest.py first.")
        return 1

    total = sum(counts.values())
    print(f"{len(counts)} documents, {total} chunks "
          f"(chunk size {settings.chunk_size_tokens}, overlap {settings.chunk_overlap_tokens})\n")

    rows = (
        [(d, counts.get(d, 0)) for d in args.only]
        if args.only
        else counts.most_common()
    )

    width = max(len(d) for d, _ in rows)
    for doc, n in rows:
        share = n / total
        print(f"  {n:>3}  {'#' * round(share * 60):<20}  {doc:<{width}}")

    singles = [d for d, n in counts.items() if n == 1]
    if singles and not args.only:
        print(
            f"\n  {len(singles)} document(s) produced a single chunk. Those compete "
            "against multi-chunk documents in every search and are the first "
            "candidates for a smaller chunk size or a lexical retrieval signal."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
