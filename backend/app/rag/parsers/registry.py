"""The format registry: the one list of what can be ingested, and dispatch to it.

A file's format comes from its extension, cross-checked against its leading
bytes: a PDF or Office file saved under the wrong extension is still parsed as
what it really is. When there is no usable extension, the MIME type and then
the content decide.
"""
from __future__ import annotations

import re
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from app.logging_config import get_structured_logger
from app.rag.parsers import docx, eml, html, odf, pdf, pptx, text
from app.rag.parsers.base import ParsedDocument, UnsupportedFormatError
from app.rag.parsers.structure import split_sections

logger = get_structured_logger(__name__)

Parser = Callable[[str, bytes, int], ParsedDocument]


@dataclass(frozen=True)
class Format:
    """One ingestible format.

    Attributes:
        name: Recorded as the document's ``source_format``.
        extensions: Lower-case suffixes, with the dot.
        mime_types: MIME types that identify the format on upload.
        parse: ``parse(filename, content, depth) -> ParsedDocument``.
        binary: Whether the content can be recognised from its magic bytes,
            which then wins over a contradicting extension.
    """

    name: str
    extensions: tuple[str, ...]
    mime_types: tuple[str, ...]
    parse: Parser
    binary: bool = False


FORMATS: tuple[Format, ...] = (
    Format("pdf", (".pdf",), ("application/pdf",), pdf.parse, binary=True),
    Format(
        "docx",
        (".docx",),
        ("application/vnd.openxmlformats-officedocument.wordprocessingml.document",),
        docx.parse,
        binary=True,
    ),
    Format(
        "pptx",
        (".pptx",),
        ("application/vnd.openxmlformats-officedocument.presentationml.presentation",),
        pptx.parse,
        binary=True,
    ),
    Format("odt", (".odt",), ("application/vnd.oasis.opendocument.text",), odf.parse, binary=True),
    Format(
        "ods", (".ods",), ("application/vnd.oasis.opendocument.spreadsheet",), odf.parse, binary=True
    ),
    Format("eml", (".eml",), ("message/rfc822",), eml.parse),
    Format("html", (".html", ".htm"), ("text/html", "application/xhtml+xml"), html.parse),
    Format(
        "markdown", (".md", ".markdown"), ("text/markdown", "text/x-markdown"),
        text.parser_for("markdown"),
    ),
    Format("rst", (".rst",), ("text/x-rst", "text/prs.fallenstein.rst"), text.parser_for("rst")),
    Format("csv", (".csv",), ("text/csv", "application/csv"), text.parse_csv),
    Format("json", (".json",), ("application/json",), text.parser_for("json")),
    Format("text", (".txt",), ("text/plain",), text.parser_for("text")),
)

_BY_NAME = {f.name: f for f in FORMATS}
_BY_EXTENSION = {ext: f for f in FORMATS for ext in f.extensions}
_BY_MIME = {mime: f for f in FORMATS for mime in f.mime_types}

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(_BY_EXTENSION)
SUPPORTED_MIME_TYPES: frozenset[str] = frozenset(_BY_MIME)

_ZIP_MARKERS = {
    "word/document.xml": "docx",
    "ppt/presentation.xml": "pptx",
}
_ODF_MIMETYPES = {
    b"application/vnd.oasis.opendocument.text": "odt",
    b"application/vnd.oasis.opendocument.spreadsheet": "ods",
}
_EMAIL_START = re.compile(
    rb"^(received|return-path|from|to|subject|date|message-id|mime-version|delivered-to)"
    rb":",
    re.IGNORECASE,
)


def extension_of(filename: str) -> str:
    """The lower-case suffix of a filename, with its dot, or ""."""
    return Path(filename).suffix.lower()


def is_supported(filename: str, mime_type: str | None = None) -> bool:
    """Whether an upload's extension or declared MIME type is ingestible."""
    return (
        extension_of(filename) in SUPPORTED_EXTENSIONS
        or _normalise_mime(mime_type) in SUPPORTED_MIME_TYPES
    )


def supported_formats() -> list[dict[str, object]]:
    """The registry as JSON-ready rows, for /ingest/formats."""
    return [
        {"name": f.name, "extensions": list(f.extensions), "mime_types": list(f.mime_types)}
        for f in FORMATS
    ]


def unsupported_message(filename: str) -> str:
    """The error shown for a file no parser accepts."""
    return (
        f"Unsupported file type '{extension_of(filename) or filename}'. "
        f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
    )


def detect_format(filename: str, content: bytes, mime_type: str | None = None) -> Format | None:
    """Pick the format for a file from its extension, MIME type and content."""
    by_extension = _BY_EXTENSION.get(extension_of(filename))
    sniffed = _sniff_binary(content)

    if sniffed and (by_extension is None or (sniffed is not by_extension)):
        if by_extension is not None:
            logger.info(
                "File content contradicts its extension; parsing by content",
                extra_fields={"filename": filename, "detected": sniffed.name},
            )
        return sniffed
    if by_extension is not None:
        return by_extension
    by_mime = _BY_MIME.get(_normalise_mime(mime_type))
    if by_mime is not None or extension_of(filename):
        # An unknown extension is a statement about the file; only a
        # recognisable binary signature (handled above) overrides it.
        return by_mime
    return _sniff_text(content)


def extract(
    filename: str, content: bytes, *, mime_type: str | None = None, depth: int = 0
) -> ParsedDocument:
    """Parse a file into text, metadata and heading sections.

    Args:
        filename: Original filename; its extension selects the parser.
        content: Raw file bytes.
        mime_type: Declared MIME type, consulted when the extension is unknown.
        depth: Nesting depth, for documents found inside other documents
            (email attachments). Top-level callers leave it at 0.

    Returns:
        The parsed document.

    Raises:
        UnsupportedFormatError: If no parser accepts the file.
        EncryptedDocumentError: If it is password-protected.
        ArchiveTooLargeError: If a zip-based file fails the zip-bomb checks.
        ParseError: If the parser cannot read it.
    """
    fmt = detect_format(filename, content, mime_type)
    if fmt is None:
        raise UnsupportedFormatError(unsupported_message(filename))
    document = fmt.parse(filename, content, depth)
    document.sections = split_sections(document.text)
    return document


def _normalise_mime(mime_type: str | None) -> str:
    return (mime_type or "").split(";", 1)[0].strip().lower()


def _sniff_binary(content: bytes) -> Format | None:
    """Recognise PDF and zip-based office formats from their bytes."""
    if content[:1024].lstrip().startswith(b"%PDF-"):
        return _BY_NAME["pdf"]
    if not content.startswith(b"PK\x03\x04"):
        return None
    try:
        # Only the central directory is read here; members are not inflated.
        with zipfile.ZipFile(BytesIO(content)) as archive:
            names = set(archive.namelist())
            for marker, name in _ZIP_MARKERS.items():
                if marker in names:
                    return _BY_NAME[name]
            info = archive.getinfo("mimetype") if "mimetype" in names else None
            if info is not None and info.file_size < 256:
                return _BY_NAME.get(_ODF_MIMETYPES.get(archive.read(info).strip(), ""))
    except (zipfile.BadZipFile, OSError, KeyError):
        return None
    return None


def _sniff_text(content: bytes) -> Format | None:
    """Recognise HTML, email or plain text, for files with no extension."""
    head = content[:1024].lstrip()
    lowered = head.lower()
    if lowered.startswith((b"<!doctype html", b"<html")):
        return _BY_NAME["html"]
    if _EMAIL_START.match(head):
        return _BY_NAME["eml"]
    try:
        content.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return None if b"\x00" in content else _BY_NAME["text"]
