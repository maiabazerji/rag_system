"""Compare two saved eval runs offline and print a regression report.

Needs no API key and runs nothing: it reads two run files (as saved under
data/eval_runs/ by scripts/run_eval.py or POST /eval/run), checks every
thresholded metric, and prints a markdown table (or JSON with --json).

Exit codes:
    0  report produced (and, with --fail-on-regression, nothing FAILed)
    1  a run file or the thresholds could not be read
    2  --fail-on-regression and at least one metric FAILed

Usage:
    python scripts/regression_report.py BASELINE.json CURRENT.json
    python scripts/regression_report.py a.json b.json --json
    python scripts/regression_report.py a.json b.json --threshold mrr=-0.05 --min-examples 10
    python scripts/regression_report.py a.json b.json --output-dir reports/ --fail-on-regression
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.eval import regression  # noqa: E402

EXIT_OK, EXIT_ERROR, EXIT_REGRESSION = 0, 1, 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("baseline", help="Baseline run file (JSON)")
    parser.add_argument("current", help="Current run file (JSON)")
    parser.add_argument("--thresholds", default=None, metavar="PATH", help="Thresholds TOML")
    parser.add_argument(
        "--threshold", action="append", default=[], metavar="METRIC=VALUE",
        help="Override one threshold: -0.05, +25%%, off. Repeatable.",
    )
    parser.add_argument("--min-examples", type=int, default=None, metavar="N")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON")
    parser.add_argument(
        "--output-dir", default=None, metavar="DIR",
        help="Also save <current>.regression.json and .md into DIR",
    )
    parser.add_argument(
        "--fail-on-regression", action="store_true", help="Exit 2 when any metric FAILs"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = regression.load_thresholds(args.thresholds, args.threshold, args.min_examples)
        baseline = regression.load_run_file(Path(args.baseline))
        current = regression.load_run_file(Path(args.current))
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ERROR

    report = regression.compare_runs(baseline, current, config, baseline_source="explicit")
    if args.output_dir:
        json_path, md_path = regression.save_report(report, current["id"], Path(args.output_dir))
        print(f"Saved {json_path} and {md_path}", file=sys.stderr)

    print(json.dumps(report, indent=2) if args.json else regression.render_markdown(report))
    if args.fail_on_regression and report["status"] == regression.FAIL:
        return EXIT_REGRESSION
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
