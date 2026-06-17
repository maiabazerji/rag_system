"""Ingest project docs into the vector store.

Picks up:
- everything under data/docs/
- top-level README.md and EvalRAG.md (so the RAG can answer questions about itself)
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.rag.ingest import enqueue_document  # noqa: E402

DOCS = ROOT / "data" / "docs"
META_FILES = [ROOT / "README.md", ROOT / "EvalRAG.md"]


async def _ingest(path: Path) -> None:
    result = await enqueue_document(path.name, path.read_bytes())
    print(f"queued {path.name} -> {result}")


async def main() -> None:
    tasks = []
    for p in DOCS.rglob("*"):
        if p.is_file():
            tasks.append(_ingest(p))
    for p in META_FILES:
        if p.is_file():
            tasks.append(_ingest(p))
    if tasks:
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
