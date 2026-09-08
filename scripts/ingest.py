"""Ingest the project's documents into the vector store.

Picks up everything under data/docs/, plus the top-level README and EvalRAG.md
so the system can answer questions about itself.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --path data/docs --concurrency 8
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.rag.ingest import SUPPORTED_SUFFIXES, enqueue_document  # noqa: E402

DEFAULT_DOCS = ROOT / "data" / "docs"
META_FILES = [ROOT / "README.md", ROOT / "EvalRAG.md", ROOT / "LEARN.md"]


def _collect(root: Path, include_meta: bool) -> list[Path]:
    """Gather ingestible files, skipping unsupported types quietly."""
    paths = [
        p
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    if include_meta:
        paths += [p for p in META_FILES if p.is_file()]
    return paths


async def _ingest_one(path: Path, semaphore: asyncio.Semaphore) -> tuple[Path, str | None]:
    """Ingest one file. Returns (path, error) so one failure cannot stop the run."""
    async with semaphore:
        try:
            result = await enqueue_document(path.name, path.read_bytes())
            print(f"  ok    {path.name}  ({result['chunks']} chunks)")
            return path, None
        except Exception as e:
            message = f"{type(e).__name__}: {e}"
            print(f"  FAIL  {path.name}  {message}")
            return path, message


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--path", type=Path, default=DEFAULT_DOCS, help="Directory to ingest"
    )
    parser.add_argument(
        "--concurrency", type=int, default=4, help="Files embedded in parallel"
    )
    parser.add_argument(
        "--no-meta", action="store_true", help="Skip the top-level README/LEARN docs"
    )
    args = parser.parse_args()

    if not args.path.exists():
        print(f"No such directory: {args.path}")
        return 1

    paths = _collect(args.path, include_meta=not args.no_meta)
    if not paths:
        print(f"Nothing to ingest under {args.path}")
        print(f"Supported types: {', '.join(sorted(SUPPORTED_SUFFIXES))}")
        return 1

    print(f"Ingesting {len(paths)} file(s) from {args.path} ...\n")
    semaphore = asyncio.Semaphore(args.concurrency)
    outcomes = await asyncio.gather(*(_ingest_one(p, semaphore) for p in paths))

    failures = [(p, err) for p, err in outcomes if err]
    print(f"\n{len(outcomes) - len(failures)}/{len(outcomes)} files ingested.")
    if failures:
        print(f"{len(failures)} failed:")
        for path, err in failures:
            print(f"  {path.name}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
