from __future__ import annotations

from app.schemas import Chunk


def rerank(query: str, chunks: list[Chunk], top_k: int = 8) -> list[Chunk]:
    return chunks[:top_k]
