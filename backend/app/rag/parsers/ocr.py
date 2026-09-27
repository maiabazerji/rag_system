"""Optional OCR for scanned PDF pages, via tesseract and poppler.

Both are system binaries, not Python packages, so OCR is available only where
they are installed (the backend Docker image installs them). Everywhere else
ingestion keeps whatever text the PDF's own text layer gave and logs why OCR
was skipped.
"""
from __future__ import annotations

import shutil

from app.config import settings

# Scanned text needs roughly 300 dpi for tesseract to read small print.
OCR_DPI = 300


def ocr_available() -> bool:
    """Whether the OCR libraries and the tesseract and poppler binaries are present."""
    try:
        import pdf2image  # noqa: F401
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    return bool(shutil.which("tesseract") and shutil.which("pdftoppm"))


def ocr_status() -> dict:
    """OCR configuration and availability, for /ingest/formats and logs."""
    available = ocr_available()
    return {
        "enabled": settings.ocr_enabled,
        "available": available,
        "active": settings.ocr_enabled and available,
        "languages": settings.ocr_languages,
        "max_pages": settings.ocr_max_pages,
    }


def ocr_pdf_page(content: bytes, page_index: int) -> str:
    """OCR one page of a PDF.

    Args:
        content: The raw PDF bytes.
        page_index: Zero-based page number.

    Returns:
        The recognised text.
    """
    import pytesseract
    from pdf2image import convert_from_bytes

    images = convert_from_bytes(
        content, dpi=OCR_DPI, first_page=page_index + 1, last_page=page_index + 1
    )
    return "\n".join(
        pytesseract.image_to_string(image, lang=settings.ocr_languages) for image in images
    )
