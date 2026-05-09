from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

from app.rag.embed import embed_texts
from app.rag.store import upsert
from app.schemas import Chunk


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def load_document(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(content))
        return "\n\n".join(p.extract_text() or "" for p in reader.pages)
    return content.decode("utf-8", errors="ignore")


def chunk_text(text: str, size: int = 600, overlap: int = 80) -> list[str]:
    words = text.split()
    chunks, step = [], max(1, size - overlap)
    for i in range(0, len(words), step):
        window = words[i : i + size]
        if not window:
            break
        chunks.append(" ".join(window))
    return chunks


def enqueue_document(filename: str, content: bytes) -> dict:
    doc_id = _hash(filename + str(len(content)))
    text = load_document(filename, content)
    chunks = [
        Chunk(
            id=f"{doc_id}:{i}",
            doc_id=doc_id,
            text=c,
            tokens=len(c.split()),
            metadata={"filename": filename},
        )
        for i, c in enumerate(chunk_text(text))
    ]
    if chunks:
        vectors = embed_texts([c.text for c in chunks])
        upsert(chunks, vectors)
    return {"doc_id": doc_id, "filename": filename, "chunks": len(chunks)}
