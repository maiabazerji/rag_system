"""Document ingestion: parse, chunk, embed and index a file.

Parsing lives in `app.rag.parsers` and chunking in `app.rag.chunking`.

Document ids are derived from the file's normalised text, so re-uploading an
unchanged file is idempotent. Because a changed file gets a *new* id, and so a
new set of point ids, indexing it cannot overwrite the previous revision in
place; the old chunks are deleted explicitly after the new ones land.

Every chunk carries the document's ``tenant`` and ``acl_groups``, which
retrieval filters on (see ``app.access``).
"""
from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from pathlib import Path

from app.access import PUBLIC_GROUP, normalize_groups
from app.config import settings
from app.logging_config import get_structured_logger
from app.privacy.pii import redact
from app.rag import graph_store, parsers
from app.rag.chunking import chunk_structured
from app.rag.embed import embed_texts_async
from app.rag.store import delete_stale_revisions, upsert
from app.schemas import Chunk

logger = get_structured_logger(__name__)


def _doc_id(text: str, namespace: str = "") -> str:
    """Derive a stable document id from a document's normalised text.

    Hashing the extracted text rather than the raw bytes means a file whose
    only change is whitespace (line endings rewritten by a checkout, a
    reflowed paragraph, a stripped trailing newline) keeps its id and
    re-indexes onto the same points instead of being stored a second time.
    Chunking collapses whitespace anyway, so two files that normalise alike
    really do produce byte-identical chunks.

    ``namespace`` separates copies of the same text indexed under different
    access scopes. Point ids derive from the doc id, so without it a second
    tenant uploading the same file would overwrite the first tenant's points
    (and their ACL). The default-tenant public scope uses an empty namespace,
    which keeps the ids of documents indexed before ACLs existed.
    """
    normalised = " ".join(text.split())
    if namespace:
        normalised = f"{namespace}\x00{normalised}"
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:16]


def _acl_namespace(tenant: str, acl_groups: list[str]) -> str:
    """The doc-id namespace for an access scope; empty for default-tenant public."""
    if tenant == settings.default_tenant and acl_groups == [PUBLIC_GROUP]:
        return ""
    return f"{tenant}\x00{','.join(acl_groups)}"


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


async def enqueue_document(
    filename: str,
    content: bytes,
    owner: str = "local",
    *,
    tenant: str | None = None,
    acl_groups: list[str] | None = None,
) -> dict:
    """Chunk, embed and index a document.

    Args:
        filename: Original filename, or the path relative to the ingest root,
            recorded as chunk metadata.
        content: Raw file bytes.
        owner: Who is indexing it. Together with ``filename`` it forms the
            document's ``source_key``, which scopes stale-revision cleanup.
        tenant: Tenant the document belongs to. Defaults to DEFAULT_TENANT.
        acl_groups: Groups allowed to read it. Defaults to ``["public"]``.
            Routes resolve both with ``app.access.resolve_document_acl``.

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
    tenant = tenant or settings.default_tenant
    groups = list(normalize_groups(acl_groups)) or [PUBLIC_GROUP]
    doc_id = _doc_id(text, _acl_namespace(tenant, groups))
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
                "tenant": tenant,
                "acl_groups": groups,
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
        "tenant": tenant,
        "acl_groups": groups,
        "chunks": len(chunks),
        "stale_chunks_removed": stale.points,
        "stale_triples_removed": stale_triples,
        "pii_counts": pii.counts,
    }
