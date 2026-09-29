"""Types shared by every document parser.

A parser turns raw bytes into markdown-ish text: headings as ``#`` lines,
tables as pipe tables, lists as ``-`` items. Keeping one text dialect across
formats is what lets a single structure-aware chunker serve all of them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


class ParseError(ValueError):
    """A document could not be turned into text.

    Subclasses ValueError so the ingest route's existing ValueError -> 422
    mapping covers every parser failure without a new except clause.
    """


class UnsupportedFormatError(ParseError):
    """The file type is not one the registry knows how to parse."""


class EncryptedDocumentError(ParseError):
    """The document is password-protected, so its text is unreadable."""


class ArchiveTooLargeError(ParseError):
    """A zip-based document would inflate past the configured ceiling."""


@dataclass
class DocumentMetadata:
    """File-level metadata a parser could recover.

    Attributes:
        source_format: The registry format that parsed the file ("pdf", "docx", ...).
        title: Document title from file properties, or the email subject.
        author: Author from file properties, or the email sender.
        created: Creation date, ISO 8601.
        modified: Last modification date, ISO 8601.
        page_count: Pages (or slides) when the format records them.
        language: Language hint from the file's own metadata, not detected.
        extra: Format-specific details (email recipients, OCR'd pages, ...).
    """

    source_format: str
    title: str | None = None
    author: str | None = None
    created: str | None = None
    modified: str | None = None
    page_count: int | None = None
    language: str | None = None
    extra: dict[str, str | int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, str | int]:
        """The metadata without empty fields, flattened for a vector payload."""
        fields: dict[str, str | int | None] = {
            "source_format": self.source_format,
            "title": self.title,
            "author": self.author,
            "created": self.created,
            "modified": self.modified,
            "page_count": self.page_count,
            "language": self.language,
            **self.extra,
        }
        return {k: v for k, v in fields.items() if v is not None and v != ""}


@dataclass(frozen=True)
class Section:
    """A span of document text under one heading.

    Attributes:
        heading_path: Headings from the outermost down to this one.
        start: Offset of the section's first character (its heading line).
        end: Offset one past the section's last character.
        body_start: Offset where the section's own content begins. Equal to
            ``start`` when the heading line also carries body text, as in
            "Art. 12. - Le présent décret...".
    """

    heading_path: tuple[str, ...]
    start: int
    end: int
    body_start: int

    @property
    def breadcrumb(self) -> str:
        """The heading path as "Chapitre 2 > Article 5"."""
        return " > ".join(self.heading_path)


@dataclass(frozen=True)
class PageSpan:
    """Where one page (or slide) sits in a document's text.

    Attributes:
        number: 1-based page number.
        start: Offset of the page's first character.
        end: Offset one past its last character.
    """

    number: int
    start: int
    end: int


@dataclass
class ParsedDocument:
    """The result of parsing one file.

    Attributes:
        text: Markdown-ish text of the whole document.
        metadata: File-level metadata.
        sections: Heading spans over ``text``, in document order.
        warnings: Non-fatal problems worth surfacing (e.g. OCR unavailable).
        pages: Page spans over ``text``, in order, for formats that have
            pages (PDF pages, presentation slides). Empty otherwise. Pages
            with no text have no span.
    """

    text: str
    metadata: DocumentMetadata
    sections: list[Section] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    pages: list[PageSpan] = field(default_factory=list)


def join_pages(pages: list[str], separator: str = "\n\n") -> tuple[str, list[PageSpan]]:
    """Join page texts with ``separator``, recording where each page lands.

    The text is exactly ``separator.join(pages)``. Pages that are blank get
    no span, so every span holds some text.
    """
    spans: list[PageSpan] = []
    offset = 0
    for number, page in enumerate(pages, start=1):
        if number > 1:
            offset += len(separator)
        if page.strip():
            spans.append(PageSpan(number, offset, offset + len(page)))
        offset += len(page)
    return separator.join(pages), spans


def iso_date(value: object) -> str | None:
    """Normalise a date from file metadata to ISO 8601, or None if unusable."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    text = str(value).strip()
    return text or None


def clean_str(value: object) -> str | None:
    """Strip a metadata string, mapping blanks to None."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def markdown_table(rows: list[list[str]]) -> str:
    """Render rows as a pipe table, using the first row as the header.

    Cells are flattened to one line and pipes escaped, so a row can never
    spill across lines: the chunker relies on one line per row to avoid
    splitting inside one.
    """
    rows = [[_cell(c) for c in row] for row in rows]
    rows = [row for row in rows if any(row)]
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    lines = [_row(rows[0]), _row(["---"] * width)]
    lines += [_row(row) for row in rows[1:]]
    return "\n".join(lines)


def _cell(value: str) -> str:
    return " ".join(value.split()).replace("|", "\\|")


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"
