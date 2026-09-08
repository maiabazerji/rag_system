"""Document ingestion: load, chunk, embed and index a file.

Document ids are derived from the file's normalised text, so re-uploading an
unchanged file is idempotent. Because a changed file gets a *new* id, and so a
new set of point ids, indexing it cannot overwrite the previous revision in
place; the old chunks are deleted explicitly after the new ones land.
"""
from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

from app.config import settings
from app.logging_config import get_structured_logger
from app.rag import graph_store
from app.rag.embed import embed_texts_async
from app.rag.store import delete_stale_revisions, upsert
from app.schemas import Chunk

logger = get_structured_logger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown", ".rst", ".csv", ".json"}


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


def load_document(filename: str, content: bytes) -> str:
    """Extract plain text from an uploaded file.

    Args:
        filename: Original filename, used to pick a parser by extension.
        content: Raw file bytes.

    Returns:
        The extracted text.

    Raises:
        ValueError: If the extension is unsupported, the PDF cannot be parsed,
            or the file is not valid UTF-8 text.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file type '{suffix or filename}'. "
            f"Supported types: {', '.join(sorted(SUPPORTED_SUFFIXES))}."
        )

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(BytesIO(content))
            return "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            raise ValueError(
                f"Could not read '{filename}' as a PDF ({type(e).__name__}). "
                "If it is a scanned document, run OCR on it first."
            ) from e

    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(
            f"'{filename}' is not valid UTF-8 text. Re-save it as UTF-8 and retry."
        ) from e


def chunk_text(
    text: str, size: int | None = None, overlap: int | None = None
) -> list[str]:
    """Split text into overlapping word windows.

    Args:
        text: The document text.
        size: Words per chunk. Defaults to CHUNK_SIZE_TOKENS.
        overlap: Words shared between neighbouring chunks. Defaults to
            CHUNK_OVERLAP_TOKENS.

    Returns:
        The chunks, in document order. Empty if the text has no words.

    Raises:
        ValueError: If overlap is not smaller than size, which would make
            chunking loop forever.
    """
    size = settings.chunk_size_tokens if size is None else size
    overlap = settings.chunk_overlap_tokens if overlap is None else overlap
    if overlap >= size:
        raise ValueError(f"chunk overlap ({overlap}) must be smaller than size ({size})")

    words = text.split()
    step = size - overlap
    chunks = []
    for i in range(0, len(words), step):
        window = words[i : i + size]
        if not window:
            break
        chunks.append(" ".join(window))
    return chunks


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
    text = load_document(filename, content)
    doc_id = _doc_id(text)

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

    if not chunks:
        raise ValueError(
            f"No text could be extracted from '{filename}'. "
            "If it is a scanned PDF, run OCR on it first."
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
