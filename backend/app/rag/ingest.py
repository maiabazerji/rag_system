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
from datetime import UTC, datetime
from pathlib import Path

from app.config import settings
from app.logging_config import get_structured_logger
from app.privacy.pii import redact
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


def _redact_metadata(metadata: dict[str, str | int]) -> dict[str, str | int]:
    """Mask personal data in string metadata unless PII handling is off.

    Metadata is masked even in ``reject`` mode: a sender address in an email
    header is expected and should not refuse the whole document.
    """
    if settings.pii_mode_ingest == "off":
        return metadata
    return {
        k: redact(v, "mask").text if isinstance(v, str) else v for k, v in metadata.items()
    }


def source_key(owner: str, path: str) -> str:
    """Identify a document by who indexed it and where it came from.

    Revisions of one document share this key and nothing else does, so it is
    what stale-revision cleanup matches on. A bare filename is not enough:
    two users, or two folders, can each hold a ``notes.md``.

    Args:
        owner: Who is indexing, e.g. ``key:<api key id>``, or ``local``.
        path: The document's path relative to its source root (for an upload,
            the uploaded filename).
    """
    return f"{owner}:{Path(path).as_posix()}"


async def enqueue_document(filename: str, content: bytes, owner: str = "local") -> dict:
    """Chunk, embed and index a document.

    Args:
        filename: Original filename, or the path relative to the ingest root,
            recorded as chunk metadata.
        content: Raw file bytes.
        owner: Who is indexing it. Together with ``filename`` it forms the
            document's ``source_key``, which scopes stale-revision cleanup.

    Returns:
        The document id, filename, number of chunks indexed, how much of a
        superseded revision was cleared from the vector store and the graph,
        and how many pieces of personal data of each type were masked.

    Raises:
        ValueError: If the file cannot be parsed or contains no extractable text.
        PIIRejected: If it contains personal data and PII_MODE_INGEST is 'reject'.
    """
    # Parsing is CPU-bound (and OCR can take seconds a page): keep it off the loop.
    parsed = await asyncio.to_thread(parsers.extract, filename, content)
    # Personal data is masked (or the document refused) before the text is
    # hashed, chunked, embedded or stored anywhere. Only counts are kept.
    # File metadata (an email's From/To, a document's author) is masked too.
    pii = redact(parsed.text, settings.pii_mode_ingest)
    text = pii.text
    doc_id = _doc_id(text)
    key = source_key(owner, filename)
    ingested_at = datetime.now(UTC).isoformat(timespec="seconds")
    doc_metadata = _redact_metadata(parsed.metadata.as_dict())

    chunks = [
        Chunk(
            id=f"{doc_id}:{i}",
            doc_id=doc_id,
            text=c.text,
            tokens=len(c.text.split()),
            metadata={
                "filename": filename,
                "source_key": key,
                "owner": owner,
                "ingested_at": ingested_at,
                "pii_counts": pii.counts,
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
    stale = await delete_stale_revisions(key, doc_id)
    # The graph is keyed to the chunks that are going away, so it has to follow.
    stale_triples = await asyncio.to_thread(graph_store.remove_docs, stale.doc_ids)

    logger.info(
        "Document indexed",
        extra_fields={
            "doc_id": doc_id,
            "filename": filename,
            "chunks": len(chunks),
            "stale_chunks_removed": stale.points,
            "stale_triples_removed": stale_triples,
            "pii_counts": pii.counts,
        },
    )
    return {
        "doc_id": doc_id,
        "filename": filename,
        "chunks": len(chunks),
        "stale_chunks_removed": stale.points,
        "stale_triples_removed": stale_triples,
        "pii_counts": pii.counts,
    }
