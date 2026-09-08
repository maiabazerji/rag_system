"""Measure retrieval quality against a golden dataset. Costs nothing.

Runs the retrieve-and-rerank path for every golden question and scores the
documents it surfaces against that example's `expected_sources`. No generation,
no judge, no API key -- the embedding and cross-encoder models run locally, so
this is free to run as often as you like.

That makes it the cheapest useful signal in the project: it isolates retrieval
quality from generation quality, and it is the number to move first, because no
amount of prompt work rescues an answer built on the wrong documents.

Usage:
    python scripts/measure_retrieval.py
    python scripts/measure_retrieval.py --top-k 5 --show-misses
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import settings  # noqa: E402
from app.eval.metrics import load_dataset  # noqa: E402
from app.eval.retrieval import aggregate_retrieval, score_retrieval  # noqa: E402
from app.rag.rerank import rerank_async  # noqa: E402
from app.rag.retrieve import dense_search  # noqa: E402
from app.schemas import Source  # noqa: E402


async def _measure(example: dict, top_k: int, semaphore: asyncio.Semaphore):
    """Retrieve for one question and score what came back."""
    async with semaphore:
        candidates = await dense_search(
            example["question"], top_k=settings.retrieval_top_k
        )
        ranked = await rerank_async(example["question"], candidates, top_k=top_k)

    sources = [
        Source(
            chunk_id=c.id,
            quote=c.text[:280],
            document=c.metadata.get("filename"),
        )
        for c in ranked
    ]
    return example, score_retrieval(example.get("expected_sources"), sources)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="golden_v1")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Chunks kept after reranking (default: RERANK_TOP_K)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=4, help="Questions measured in parallel"
    )
    parser.add_argument(
        "--show-misses",
        action="store_true",
        help="List the questions where retrieval never surfaced the expected document",
    )
    args = parser.parse_args()

    top_k = args.top_k or settings.rerank_top_k
    examples = load_dataset(args.dataset)
    if not examples:
        print(f"No examples in dataset '{args.dataset}'.")
        return 1

    print(f"Measuring retrieval on {len(examples)} questions (top_k={top_k}) ...")
    semaphore = asyncio.Semaphore(args.concurrency)
    outcomes = await asyncio.gather(
        *(_measure(ex, top_k, semaphore) for ex in examples)
    )

    scored = [(ex, sc) for ex, sc in outcomes if sc is not None]
    if not scored:
        print("\nNo example declares expected_sources, so there is nothing to score.")
        return 1

    agg = aggregate_retrieval([sc.model_dump() for _, sc in scored])
    assert agg is not None

    print(f"\nRetrieval quality  ({agg['n']} of {len(examples)} examples scored)\n")
    for key, label, meaning in [
        ("hit_rate", "Hit rate", "the expected document appears at all"),
        ("mrr", "MRR", "how near the top it lands"),
        ("recall", "Recall", "share of expected documents found"),
        ("precision", "Precision", "share of returned documents that were wanted"),
    ]:
        bar = "#" * round(agg[key] * 30)
        print(f"  {label:<10} {agg[key]:.3f}  {bar:<30}  {meaning}")

    misses = [(ex, sc) for ex, sc in scored if not sc.hit]
    print(f"\n  {len(misses)} of {agg['n']} questions never surfaced their expected document.")

    if misses and args.show_misses:
        print("\nMisses:")
        for ex, sc in misses:
            print(f"\n  Q  {ex['question']}")
            print(f"  want  {', '.join(sc.expected)}")
            print(f"  got   {', '.join(sc.retrieved[:4]) or '(nothing)'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
