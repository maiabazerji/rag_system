"""Evaluate a golden dataset and report the scores.

Exits non-zero when nothing could be measured, so this is usable as a CI gate.

Usage:
    python scripts/run_eval.py                          # classic only
    python scripts/run_eval.py --strategy graph
    python scripts/run_eval.py --all                    # all three, side by side
    python scripts/run_eval.py --all --no-judge         # retrieval only, ~half the cost
    python scripts/run_eval.py --json > run.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.eval.judge import DIMENSIONS  # noqa: E402
from app.eval.metrics import run_evaluation  # noqa: E402
from app.schemas import Strategy  # noqa: E402

STRATEGIES: list[Strategy] = ["classic", "graph", "agentic"]


def _bar(value: float, width: int = 28) -> str:
    return "#" * round(value * width)


def _print_run(result: dict) -> None:
    """Print a readable digest of one run."""
    print(f"\nDataset      {result['dataset']}")
    print(f"Strategy     {result['strategy']}")
    print(f"Model        {result['model'] or '(default)'}")
    print(f"Examples     {result['n']}")

    if result.get("n_refused"):
        print(f"Refused      {result['n_refused']}")

    retrieval = result.get("retrieval_aggregate")
    if retrieval:
        print(f"\nRetrieval (deterministic, {retrieval['n']} examples):")
        for key in ("precision", "recall", "hit_rate", "mrr"):
            print(f"  {key:<16} {retrieval[key]:.3f}  {_bar(retrieval[key])}")

    unscored = result.get("n_unscored", 0)
    if unscored:
        print(f"\nUnscored     {unscored}  (judge failed; excluded from the aggregate)")

    if not result.get("judged", True):
        print("\nAnswer quality   skipped (--no-judge)")

    aggregate = result.get("aggregate")
    if aggregate:
        print(f"\nAnswer quality (LLM judge, {result['n_scored']} examples):")
        width = max(len(d) for d in DIMENSIONS)
        for dimension in DIMENSIONS:
            value = aggregate[dimension]
            print(f"  {dimension:<{width}} {value:.3f}  {_bar(value)}")
    elif not retrieval:
        print("\nNothing could be measured.")
        print("Check that ANTHROPIC_API_KEY is set and documents are indexed.")

    cost = result.get("cost") or {}
    if cost:
        total = cost.get("total_input_tokens", 0) + cost.get("total_output_tokens", 0)
        print(f"\nCost         {cost.get('mean_latency_ms', 0)} ms/question, {total} tokens total")


def _print_comparison(results: list[dict]) -> None:
    """Print the side-by-side table across strategies."""
    print("\n" + "=" * 78)
    print("STRATEGY COMPARISON".center(78))
    print("=" * 78)

    def cell(result: dict, group: str, key: str) -> str:
        block = result.get(group)
        if not block or block.get(key) is None:
            return "  -  "
        return f"{block[key]:.3f}"

    rows = [
        ("Retrieval precision", "retrieval_aggregate", "precision"),
        ("Retrieval recall", "retrieval_aggregate", "recall"),
        ("Retrieval hit rate", "retrieval_aggregate", "hit_rate"),
        ("Retrieval MRR", "retrieval_aggregate", "mrr"),
        ("Faithfulness", "aggregate", "faithfulness"),
        ("Answer relevance", "aggregate", "answer_relevance"),
        ("Context precision", "aggregate", "context_precision"),
        ("Context recall", "aggregate", "context_recall"),
    ]

    header = f"{'Metric':<22}" + "".join(f"{r['strategy']:>13}" for r in results)
    print(f"\n{header}")
    print("-" * len(header))
    for label, group, key in rows:
        line = f"{label:<22}" + "".join(f"{cell(r, group, key):>13}" for r in results)
        print(line)

    print("-" * len(header))
    latency = f"{'Latency (ms/q)':<22}" + "".join(
        f"{(r.get('cost') or {}).get('mean_latency_ms', 0):>13,}" for r in results
    )
    tokens = f"{'Tokens (total)':<22}" + "".join(
        f"{(r.get('cost') or {}).get('total_input_tokens', 0) + (r.get('cost') or {}).get('total_output_tokens', 0):>13,}"
        for r in results
    )
    refused = f"{'Refused':<22}" + "".join(
        f"{r.get('n_refused', 0):>13}" for r in results
    )
    print(latency)
    print(tokens)
    print(refused)
    print()


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="golden_v1", help="Golden dataset name")
    parser.add_argument(
        "--strategy",
        default="classic",
        choices=STRATEGIES,
        help="Strategy to evaluate (default: classic)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate every strategy and print a comparison table",
    )
    parser.add_argument("--provider", default=None, help="Override the LLM provider")
    parser.add_argument("--model", default=None, help="Override the model")
    parser.add_argument("--prompt-version", default=None, help="Prompt template version")
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip the LLM judge. Retrieval is still scored; roughly halves cost.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print full results as JSON instead"
    )
    args = parser.parse_args()

    targets: list[Strategy] = STRATEGIES if args.all else [args.strategy]
    results = []

    for strategy in targets:
        if not args.json:
            print(f"\n>>> Evaluating '{strategy}' on '{args.dataset}' ...")
        result = await run_evaluation(
            dataset=args.dataset,
            strategy=strategy,
            provider=args.provider,
            model=args.model,
            prompt_version=args.prompt_version,
            judge_answers=not args.no_judge,
        )
        if result.get("error"):
            print(result["error"])
            return 1
        results.append(result)
        if not args.json:
            _print_run(result)

    if args.json:
        print(json.dumps(results if args.all else results[0], indent=2))
    elif len(results) > 1:
        _print_comparison(results)

    # Success means something was measured -- retrieval alone is enough.
    measured = any(r.get("aggregate") or r.get("retrieval_aggregate") for r in results)
    return 0 if measured else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
