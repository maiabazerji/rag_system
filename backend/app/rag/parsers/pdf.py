"""PDF parser: pypdf text layer, with OCR for pages that have none."""
from __future__ import annotations

import re
from io import BytesIO
from typing import Any

from app.config import settings
from app.logging_config import get_structured_logger
from app.rag.parsers import ocr
from app.rag.parsers.base import (
    DocumentMetadata,
    EncryptedDocumentError,
    ParsedDocument,
    ParseError,
    clean_str,
    iso_date,
)

logger = get_structured_logger(__name__)

_BLANK_RUNS = re.compile(r"\n{3,}")


def parse(filename: str, content: bytes, depth: int = 0) -> ParsedDocument:
    """Extract a PDF's text page by page, OCR'ing pages with no text layer.

    Raises:
        EncryptedDocumentError: If the PDF needs a password to open.
        ParseError: If pypdf cannot read the file at all.
    """
    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(content))
        if reader.is_encrypted and not reader.decrypt(""):
            raise EncryptedDocumentError(
                f"'{filename}' is password-protected. Remove the password and retry."
            )
        pages = [_page_text(page) for page in reader.pages]
    except ParseError:
        raise
    except Exception as e:
        raise ParseError(
            f"Could not read '{filename}' as a PDF ({type(e).__name__})."
        ) from e

    metadata = _metadata(reader, len(pages))
    warnings = _ocr_sparse_pages(filename, content, pages, metadata)
    text = _BLANK_RUNS.sub("\n\n", "\n\n".join(p.strip("\n") for p in pages))
    return ParsedDocument(text=text, metadata=metadata, warnings=warnings)


def _page_text(page: Any) -> str:
    """A page's text, in layout mode so table columns stay aligned."""
    try:
        text = page.extract_text(extraction_mode="layout") or ""
    except Exception:
        # Layout mode is newer and stricter; plain mode reads more PDFs.
        text = page.extract_text() or ""
    return "\n".join(line.rstrip() for line in text.splitlines())


def _metadata(reader: Any, page_count: int) -> DocumentMetadata:
    meta = DocumentMetadata(source_format="pdf", page_count=page_count)
    try:
        info = reader.metadata
        if info is not None:
            meta.title = clean_str(info.title)
            meta.author = clean_str(info.author)
            meta.created = iso_date(info.creation_date)
            meta.modified = iso_date(info.modification_date)
        meta.language = clean_str(reader.trailer["/Root"].get("/Lang"))
    except Exception as e:  # malformed dates or info dictionaries are common
        logger.debug("Unreadable PDF metadata", extra_fields={"error": str(e)})
    return meta


def _ocr_sparse_pages(
    filename: str, content: bytes, pages: list[str], metadata: DocumentMetadata
) -> list[str]:
    """Replace the text of near-empty pages with OCR output, in place.

    Returns:
        Warnings to surface when OCR was needed but could not run.
    """
    sparse = [
        i for i, text in enumerate(pages)
        if len(text.strip()) < settings.ocr_min_chars_per_page
    ]
    if not sparse or not settings.ocr_enabled:
        return []

    if not ocr.ocr_available():
        warning = (
            f"{len(sparse)} page(s) of '{filename}' have no text layer and look "
            "scanned, but OCR is unavailable (tesseract and poppler must be "
            "installed); indexing only the text that could be extracted."
        )
        logger.warning(warning, extra_fields={"filename": filename, "pages": len(sparse)})
        return [warning]

    warnings: list[str] = []
    if len(sparse) > settings.ocr_max_pages:
        warning = (
            f"'{filename}' has {len(sparse)} scanned pages; only the first "
            f"{settings.ocr_max_pages} were OCR'd (OCR_MAX_PAGES)."
        )
        logger.warning(warning, extra_fields={"filename": filename})
        warnings.append(warning)

    done = 0
    for i in sparse[: settings.ocr_max_pages]:
        try:
            recognised = ocr.ocr_pdf_page(content, i)
        except Exception as e:
            logger.warning(
                "OCR failed on a page; keeping its extracted text",
                extra_fields={"filename": filename, "page": i + 1, "error": str(e)},
            )
            continue
        if len(recognised.strip()) > len(pages[i].strip()):
            pages[i] = recognised
            done += 1

    if done:
        metadata.extra["ocr_pages"] = done
        logger.info("OCR'd scanned pages", extra_fields={"filename": filename, "pages": done})
    return warnings
