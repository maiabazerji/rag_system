from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile

from app.access import Principal, resolve_document_acl
from app.audit import audited
from app.auth import require_principal
from app.config import settings
from app.logging_config import get_structured_logger
from app.rag.ingest import SUPPORTED_SUFFIXES, enqueue_document
from app.rag.store import count as store_count

logger = get_structured_logger(__name__)
router = APIRouter(dependencies=[Depends(require_principal)])


def _split_groups(values: list[str] | None) -> list[str] | None:
    """Accept ``groups`` as repeated form fields, comma-separated, or both."""
    if values is None:
        return None
    return [g for v in values for g in v.split(",")]


@router.post(
    "",
    summary="Ingest a document",
    description=(
        "Chunks, embeds and indexes an uploaded document. "
        f"Accepts {', '.join(sorted(SUPPORTED_SUFFIXES))}. The document belongs "
        "to the caller's tenant and is readable by the groups in `groups` "
        "(default: the caller's own groups)."
    ),
)
async def ingest_document(
    file: UploadFile,
    groups: list[str] | None = Form(
        default=None,
        description=(
            "Groups allowed to read the document, as repeated fields or comma-"
            "separated. Must be groups the caller belongs to (or 'public'), "
            "unless the caller is an admin."
        ),
    ),
    principal: Principal = Depends(require_principal),
) -> dict:
    """Index an uploaded document.

    Args:
        file: The uploaded file. Must have a supported extension and be within
            the MAX_UPLOAD_MB limit.
        groups: Groups allowed to read it. Omitted: the caller's own groups,
            or ``public`` in local mode / with ``ACL_DEFAULT_PUBLIC``.
        principal: The caller (resolved once per request with the router's
            own dependency). Scopes which earlier revisions a re-upload
            replaces, and fixes the document's tenant.

    Returns:
        The document id, filename, number of chunks written, and the new total
        number of indexed chunks.

    Raises:
        HTTPException: 400 for a missing name, empty file or unsupported type;
            403 for a group the caller is not a member of; 413 if the file
            exceeds MAX_UPLOAD_MB; 422 if it holds no text or ``groups`` is
            malformed. 503 (via the StoreUnavailable handler) if Qdrant is
            unreachable.
    """
    with audited(principal, "ingest") as event:
        event.detail = file.filename
        result = await _ingest(file, _split_groups(groups), principal)
        event.add_doc_ids([result.get("doc_id")])
    return result


async def _ingest(file: UploadFile, groups: list[str] | None, principal: Principal) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a name.")

    suffix = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{suffix or file.filename}'. "
                f"Supported types: {', '.join(sorted(SUPPORTED_SUFFIXES))}."
            ),
        )

    limit = settings.max_upload_bytes
    if file.size is not None and file.size > limit:
        raise HTTPException(
            status_code=413,
            detail=f"File is larger than the {settings.max_upload_mb} MB limit.",
        )

    content = await file.read(limit + 1)
    if len(content) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"File is larger than the {settings.max_upload_mb} MB limit.",
        )
    if not content:
        raise HTTPException(status_code=400, detail="File is empty.")

    tenant, acl_groups = resolve_document_acl(principal, groups)
    try:
        result = await enqueue_document(
            filename=file.filename,
            content=content,
            owner=principal.id,
            tenant=tenant,
            acl_groups=acl_groups,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    total = await store_count(principal.scope())
    logger.info(
        "Document ingested",
        extra_fields={
            "filename": file.filename,
            "chunks": result["chunks"],
            "tenant": tenant,
            "acl_groups": acl_groups,
            "indexed_total": total,
        },
    )
    return {**result, "indexed_total": total}


@router.get("/stats", summary="Index statistics")
async def stats(principal: Principal = Depends(require_principal)) -> dict:
    """Return the number of indexed chunks the caller may read."""
    return {"indexed_chunks": await store_count(principal.scope())}
