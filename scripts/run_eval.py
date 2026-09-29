"""Evaluate a golden dataset, report the scores and gate on regressions.

Exit codes (usable as a CI gate):
    0  something was measured (and, with --fail-on-regression, nothing regressed)
    1  nothing could be measured, the dataset is missing, or --dry-run found errors
    2  --fail-on-regression and at least one metric FAILed its threshold

Every run is saved under data/eval_runs/ and compared with its baseline: the
pinned baseline for its configuration (data/eval_runs/baselines/<config>.json,
set with --set-baseline) or else the previous run of the same configuration.
The regression report is saved next to the run as <run>.regression.json/.md.

Usage:
    python scripts/run_eval.py                          # classic only
    python scripts/run_eval.py --strategy graph
    python scripts/run_eval.py --all                    # all three, side by side
    python scripts/run_eval.py --all --no-judge         # retrieval only, ~half the cost
    python scripts/run_eval.py --json > run.json
    python scripts/run_eval.py --k 1,5,20               # retrieval cutoffs
    python scripts/run_eval.py --fail-on-regression     # CI gate
    python scripts/run_eval.py --threshold faithfulness=-0.05 --threshold latency_p50_ms=+30%
    python scripts/run_eval.py --set-baseline           # run, then pin this run as baseline
    python scripts/run_eval.py --set-baseline data/eval_runs/<run>.json   # pin, no run
    python scripts/run_eval.py --dry-run --all-datasets # validate datasets, no LLM calls
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import settings  # noqa: E402
from app.eval import metrics as eval_metrics  # noqa: E402
from app.eval import regression  # noqa: E402
from app.eval.dataset import validate_dataset_file  # noqa: E402
from app.eval.judge import DIMENSIONS  # noqa: E402
from app.eval.metrics import run_evaluation  # noqa: E402
from app.eval.retrieval import DEFAULT_K_VALUES  # noqa: E402
from app.schemas import Strategy  # noqa: E402

STRATEGIES: list[Strategy] = ["classic", "graph", "agentic"]

EXIT_OK, EXIT_ERROR, EXIT_REGRESSION = 0, 1, 2


def _bar(value: float, width: int = 28) -> str:
    return "#" * round(max(0.0, min(1.0, value)) * width)


def _mean(result: dict, metric: str) -> float | None:
    block = (result.get("aggregates") or {}).get(metric) or {}
    return block.get("mean")


def _print_run(result: dict) -> None:
    """Print a readable digest of one run."""
    print(f"\nDataset      {result['dataset']}")
    print(f"Strategy     {result['strategy']}")
    print(f"Model        {result['model'] or '(default)'}")
    print(f"Judge        {result.get('judge_model') or '(none)'}  rubric {result.get('rubric_version') or '-'}")
    print(f"Retrieval    {result.get('retrieval_mode')}  config {result.get('config_hash')}")
    print(
        f"Examples     {result['n']}  scored {result.get('n_scored', 0)}"
        f"  judge-failed {result.get('n_judge_failed', 0)}"
        f"  generation-failed {result.get('n_generation_failed', 0)}"
    )
    if result.get("n_refused"):
        print(f"Refused      {result['n_refused']}")

    aggregates = result.get("aggregates") or {}
    retrieval_keys = [k for k in aggregates if k == "mrr" or "@" in k]
    if retrieval_keys:
        n = aggregates[retrieval_keys[0]]["n"]
        print(f"\nRetrieval (deterministic, {n} labelled examples):")
        for key in retrieval_keys:
            value = aggregates[key]["mean"]
            print(f"  {key:<16} {value:.3f}  {_bar(value)}")

    if not result.get("judged", True):
        print("\nAnswer quality   skipped (--no-judge)")
    elif any(d in aggregates for d in DIMENSIONS):
        print(f"\nAnswer quality (LLM judge, {result['n_scored']} scored examples):")
        width = max(len(d) for d in DIMENSIONS)
        for dimension in DIMENSIONS:
            block = aggregates.get(dimension)
            if block is None:
                # answer_correctness needs ideal answers in the dataset.
                print(f"  {dimension:<{width}}   -    (not scored)")
                continue
            std = f"sd {block['std']:.3f}" if block.get("std") is not None else "sd -"
            print(
                f"  {dimension:<{width}} {block['mean']:.3f}  {std}  n={block['n']:<4}"
                f" {_bar(block['mean'])}"
            )
    elif not retrieval_keys:
        print("\nNothing could be measured.")
        print("Check that ANTHROPIC_API_KEY is set and documents are indexed.")

    for key in ("correct_refusal", "false_refusal"):
        block = aggregates.get(key)
        if block:
            print(f"  {key:<18} {block['mean']:.3f}  n={block['n']}")

    failures = result.get("failures") or []
    if failures:
        print(f"\nFailures ({len(failures)}, excluded from the aggregates):")
        for f in failures[:10]:
            print(f"  [{f['stage']}] {f['id']}: {f['error_type']}: {str(f['error'])[:100]}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more (see the run file)")

    by_type = result.get("by_question_type") or {}
    if len(by_type) > 1:
        print("\nBy question type:")
        for name, block in by_type.items():
            agg = block["aggregates"]
            parts = [
                f"{m} {agg[m]['mean']:.2f}"
                for m in ("faithfulness", "recall@5", "correct_refusal")
                if m in agg
            ]
            print(f"  {name:<20} n={block['n_examples']:<4} {'  '.join(parts)}")

    cost = result.get("cost") or {}
    if cost:
        total = cost.get("total_input_tokens", 0) + cost.get("total_output_tokens", 0)
        print(
            f"\nCost         p50 {cost.get('p50_latency_ms')} ms/question, "
            f"{total} tokens total, judge {cost.get('judge_input_tokens', 0)}"
            f"+{cost.get('judge_output_tokens', 0)} tokens"
        )


def _print_comparison(results: list[dict]) -> None:
    """Print the side-by-side table across strategies."""
    print("\n" + "=" * 78)
    print("STRATEGY COMPARISON".center(78))
    print("=" * 78)

    def cell(result: dict, key: str) -> str:
        value = _mean(result, key)
        return "  -  " if value is None else f"{value:.3f}"

    rows = [
        ("Retrieval recall@5", "recall@5"),
        ("Retrieval nDCG@10", "ndcg@10"),
        ("Retrieval hit rate@5", "hit_rate@5"),
        ("Retrieval MRR", "mrr"),
        ("Faithfulness", "faithfulness"),
        ("Answer relevance", "answer_relevance"),
        ("Context precision", "context_precision"),
        ("Context recall", "context_recall"),
        ("Answer correctness", "answer_correctness"),
        ("Correct refusal", "correct_refusal"),
    ]

    header = f"{'Metric':<22}" + "".join(f"{r['strategy']:>13}" for r in results)
    print(f"\n{header}")
    print("-" * len(header))
    for label, key in rows:
        print(f"{label:<22}" + "".join(f"{cell(r, key):>13}" for r in results))

    print("-" * len(header))

    def cost(r: dict, key: str) -> Any:
        return (r.get("cost") or {}).get(key)

    print(f"{'Latency p50 (ms)':<22}" + "".join(f"{cost(r, 'p50_latency_ms') or 0:>13,.0f}" for r in results))
    print(
        f"{'Tokens / question':<22}"
        + "".join(f"{cost(r, 'mean_tokens_per_question') or 0:>13,.0f}" for r in results)
    )
    print(f"{'Failed':<22}" + "".join(f"{len(r.get('failures') or []):>13}" for r in results))
    print()


def _parse_k(text: str) -> list[int]:
    try:
        ks = sorted({int(p) for p in text.split(",") if p.strip()})
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"--k expects comma-separated integers: {text!r}") from e
    if not ks or ks[0] < 1:
        raise argparse.ArgumentTypeError("--k values must be positive integers")
    return ks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", default="golden_v1", help="Golden dataset name")
    parser.add_argument(
        "--strategy", default="classic", choices=STRATEGIES,
        help="Strategy to evaluate (default: classic)",
    )
    parser.add_argument(
        "--all", action="store_true", help="Evaluate every strategy and print a comparison table"
    )
    parser.add_argument("--provider", default=None, help="Override the LLM provider")
    parser.add_argument("--model", default=None, help="Override the model")
    parser.add_argument("--prompt-version", default=None, help="Prompt template version")
    parser.add_argument(
        "--no-judge", action="store_true",
        help="Skip the LLM judge. Retrieval is still scored; roughly halves cost.",
    )
    parser.add_argument(
        "--k", type=_parse_k, default=list(DEFAULT_K_VALUES), metavar="K1,K2,...",
        help="Rank cutoffs for @K retrieval metrics (default: 1,3,5,10)",
    )
    parser.add_argument("--json", action="store_true", help="Print full results as JSON instead")

    gate = parser.add_argument_group("regression gate")
    gate.add_argument(
        "--fail-on-regression", action="store_true",
        help="Exit 2 when any metric FAILs its threshold against the baseline",
    )
    gate.add_argument(
        "--thresholds", default=None, metavar="PATH",
        help="Thresholds TOML (default: $REGRESSION_THRESHOLDS_PATH or eval/regression_thresholds.toml)",
    )
    gate.add_argument(
        "--threshold", action="append", default=[], metavar="METRIC=VALUE",
        help="Override one threshold: -0.05 (min delta), +25%% (max increase), off. Repeatable.",
    )
    gate.add_argument(
        "--min-examples", type=int, default=None, metavar="N",
        help="Minimum examples behind a metric for it to be checked",
    )
    gate.add_argument(
        "--baseline", default=None, metavar="RUN_FILE",
        help="Compare against this run file instead of the pinned/previous baseline",
    )
    gate.add_argument(
        "--set-baseline", nargs="?", const=True, default=None, metavar="RUN_FILE",
        help="Pin a baseline: with a run file, pin it and exit; alone, pin the run just made",
    )

    dry = parser.add_argument_group("dataset validation")
    dry.add_argument(
        "--dry-run", action="store_true",
        help="Validate dataset files (schema, referenced documents) without calling any model",
    )
    dry.add_argument(
        "--all-datasets", action="store_true", help="With --dry-run: validate every golden file"
    )
    dry.add_argument(
        "--corpus-dir", default=None, metavar="DIR",
        help="Documents that labels must reference (default: <DATA_DIR>/docs)",
    )
    return parser


def dry_run(args: argparse.Namespace) -> int:
    """Validate dataset files; exit 1 if any has errors."""
    golden_dir = eval_metrics.GOLDEN_DIR
    if args.all_datasets:
        paths = sorted(golden_dir.glob("*.jsonl"))
    else:
        candidate = Path(args.dataset)
        paths = [candidate if candidate.suffix == ".jsonl" else golden_dir / f"{args.dataset}.jsonl"]
    corpus = Path(args.corpus_dir) if args.corpus_dir else settings.data_path / "docs"

    if not paths:
        print(f"No dataset files in {golden_dir}")
        return EXIT_ERROR

    reports = [validate_dataset_file(p, corpus) for p in paths]
    if args.json:
        print(json.dumps({"corpus_dir": str(corpus), "datasets": [r.to_dict() for r in reports]}, indent=2))
    else:
        print(f"Corpus: {corpus}")
        for r in reports:
            status = "OK" if r.ok else "INVALID"
            print(
                f"\n{Path(r.path).name}: {status}  examples={r.n_examples}"
                f" with_reference={r.n_with_reference} labelled={r.n_with_relevance}"
                f" expect_refusal={r.n_expect_refusal}"
            )
            if r.question_types:
                print("  question types: " + ", ".join(f"{k}={v}" for k, v in r.question_types.items()))
            if r.difficulties:
                print("  difficulty:     " + ", ".join(f"{k}={v}" for k, v in r.difficulties.items()))
            for e in r.errors:
                print(f"  ERROR   {e}")
            for w in r.warnings[:20]:
                print(f"  warning {w}")
            if len(r.warnings) > 20:
                print(f"  ... and {len(r.warnings) - 20} more warnings")
        n_bad = sum(1 for r in reports if not r.ok)
        print(f"\n{len(reports) - n_bad} of {len(reports)} dataset files valid.")
    return EXIT_OK if all(r.ok for r in reports) else EXIT_ERROR


def gate_run(result: dict, config: regression.ThresholdConfig, baseline_file: str | None) -> dict:
    """Compare a saved run with its baseline, save the report, update the run file."""
    if baseline_file:
        baseline: dict | None = regression.load_run_file(Path(baseline_file))
        source = "explicit"
    else:
        baseline, found = regression.find_baseline(result)
        source = found or "previous"
    report = regression.compare_runs(baseline, result, config, baseline_source=source)

    run_path = eval_metrics.RUNS_DIR / f"{result['id']}.json"
    json_path, md_path = regression.save_report(report, result["id"], run_path.parent)
    result["regression"] = regression.report_summary(report, json_path, md_path)
    if run_path.is_file():
        saved = json.loads(run_path.read_text(encoding="utf-8"))
        saved["regression"] = result["regression"]
        run_path.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    return report


async def run(args: argparse.Namespace) -> int:
    try:
        config = regression.load_thresholds(args.thresholds, args.threshold, args.min_examples)
    except (OSError, ValueError) as e:
        print(f"Invalid regression thresholds: {e}", file=sys.stderr)
        return EXIT_ERROR

    targets: list[Strategy] = STRATEGIES if args.all else [args.strategy]
    results: list[dict] = []
    reports: list[dict] = []

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
            k_values=args.k,
        )
        if result.get("error"):
            print(result["error"], file=sys.stderr)
            return EXIT_ERROR

        try:
            report = gate_run(result, config, args.baseline)
        except (OSError, ValueError) as e:
            print(f"Regression check failed: {e}", file=sys.stderr)
            return EXIT_ERROR
        results.append(result)
        reports.append(report)

        if args.set_baseline is True:
            pinned = regression.pin_baseline(eval_metrics.RUNS_DIR / f"{result['id']}.json")
            if not args.json:
                print(f"Pinned {result['id']} as the baseline: {pinned}")

        if not args.json:
            _print_run(result)
            print()
            print(regression.render_markdown(report))

    if args.json:
        print(json.dumps(results if args.all else results[0], indent=2))
    elif len(results) > 1:
        _print_comparison(results)

    # Success means something was measured -- retrieval alone is enough.
    measured = any(r.get("aggregates") or r.get("aggregate") for r in results)
    if not measured:
        return EXIT_ERROR
    if args.fail_on_regression and any(r["status"] == regression.FAIL for r in reports):
        print("Regression gate: FAIL", file=sys.stderr)
        return EXIT_REGRESSION
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        return dry_run(args)
    if isinstance(args.set_baseline, str):
        try:
            pinned = regression.pin_baseline(Path(args.set_baseline))
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return EXIT_ERROR
        print(f"Pinned {args.set_baseline} as the baseline: {pinned}")
        return EXIT_OK
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
