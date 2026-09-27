"""Document parsers: raw file bytes to markdown-ish text plus metadata.

One module per format; `registry` holds the single list of supported
extensions and MIME types that the API, the ingest script and ingestion all
read. Call `extract(filename, content)` to parse a file.
"""
from app.rag.parsers.base import (
    ArchiveTooLargeError,
    DocumentMetadata,
    EncryptedDocumentError,
    ParsedDocument,
    ParseError,
    Section,
    UnsupportedFormatError,
)
from app.rag.parsers.ocr import ocr_available, ocr_status
from app.rag.parsers.registry import (
    FORMATS,
    SUPPORTED_EXTENSIONS,
    SUPPORTED_MIME_TYPES,
    detect_format,
    extension_of,
    extract,
    is_supported,
    supported_formats,
    unsupported_message,
)

__all__ = [
    "FORMATS",
    "SUPPORTED_EXTENSIONS",
    "SUPPORTED_MIME_TYPES",
    "ArchiveTooLargeError",
    "DocumentMetadata",
    "EncryptedDocumentError",
    "ParseError",
    "ParsedDocument",
    "Section",
    "UnsupportedFormatError",
    "detect_format",
    "extension_of",
    "extract",
    "is_supported",
    "ocr_available",
    "ocr_status",
    "supported_formats",
    "unsupported_message",
]
