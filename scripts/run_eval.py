"""Run evaluation on a golden dataset and print aggregate scores."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.eval.metrics import run_evaluation  # noqa: E402


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="golden_v1")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--prompt-version", default=None)
    args = ap.parse_args()

    result = await run_evaluation(
        dataset=args.dataset,
        provider=args.provider,
        model=args.model,
        prompt_version=args.prompt_version,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
