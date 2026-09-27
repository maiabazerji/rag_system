"""Document ingestion: parse, chunk, embed and index a file.

Parsing lives in `app.rag.parsers` and chunking in `app.rag.chunking`.

Document ids are derived from the file's normalised text, so re-uploading an
unchanged file is idempotent. Because a changed file gets a *new* id, and so a
new set of point ids, indexing it cannot overwrite the previous revision in
place; the old chunks are deleted explicitly after the new ones land.
"""
from __future__ import annotations

import asyncio
import hashlib

from app.logging_config import get_structured_logger
from app.rag import graph_store, parsers
from app.rag.chunking import chunk_structured
from app.rag.embed import embed_texts_async
from app.rag.store import delete_stale_revisions, upsert
from app.schemas import Chunk

logger = get_structured_logger(__name__)


def _doc_id(text: str) -> str:
    """Derive a stable document id from a document's normalised text.

    Hashing the extracted text rather than the raw bytes means a file whose
    only change is whitespace (line endings rewritten by a checkout, a
    reflowed paragraph, a stripped trailing newline) keeps its id and
    re-indexes onto the same points instead of being stored a second time.
    Chunking collapses whitespace anyway, so two files that normalise alike
    really do produce byte-identical chunks.
    """
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()[:16]


async def enqueue_document(filename: str, content: bytes) -> dict:
    """Chunk, embed and index a document.

    Args:
        filename: Original filename, recorded as chunk metadata.
        content: Raw file bytes.

    Returns:
        The document id, filename, number of chunks indexed, and how much of a
        superseded revision was cleared from the vector store and the graph.

    Raises:
        ValueError: If the file cannot be parsed or contains no extractable text.
    """
    # Parsing is CPU-bound (and OCR can take seconds a page): keep it off the loop.
    parsed = await asyncio.to_thread(parsers.extract, filename, content)
    text = parsed.text
    doc_id = _doc_id(text)
    doc_metadata = parsed.metadata.as_dict()

    chunks = [
        Chunk(
            id=f"{doc_id}:{i}",
            doc_id=doc_id,
            text=c.text,
            tokens=len(c.text.split()),
            metadata={
                "filename": filename,
                "heading_path": c.heading_path,
                "doc_metadata": doc_metadata,
            },
        )
        for i, c in enumerate(chunk_structured(text))
    ]

    if not chunks:
        raise ValueError(
            f"No text could be extracted from '{filename}'. "
            "If it is a scanned PDF, enable OCR (OCR_ENABLED, with tesseract "
            "and poppler installed) or run OCR on it first."
        )

    vectors = await embed_texts_async([c.text for c in chunks])
    await upsert(chunks, vectors)
    # Index the new revision first, then drop the old one, so the document is
    # never briefly absent from search.
    stale = await delete_stale_revisions(filename, doc_id)
    # The graph is keyed to the chunks that are going away, so it has to follow.
    stale_triples = graph_store.remove_docs(stale.doc_ids)

    logger.info(
        "Document indexed",
        extra_fields={
            "doc_id": doc_id,
            "filename": filename,
            "chunks": len(chunks),
            "stale_chunks_removed": stale.points,
            "stale_triples_removed": stale_triples,
        },
    )
    return {
        "doc_id": doc_id,
        "filename": filename,
        "chunks": len(chunks),
        "stale_chunks_removed": stale.points,
        "stale_triples_removed": stale_triples,
    }
