"""Word .docx parser (python-docx).

The body is read in document order, paragraphs and tables interleaved.
Heading styles become markdown headings, list paragraphs ``-`` items and
tables pipe tables. Page headers and footers are skipped: they repeat on every
page and would pollute every chunk they landed in.
"""
from __future__ import annotations

import re
from io import BytesIO
from typing import Any

from app.rag.parsers.archive import open_zip
from app.rag.parsers.base import (
    DocumentMetadata,
    ParsedDocument,
    ParseError,
    clean_str,
    iso_date,
    markdown_table,
)

_HEADING_STYLE = re.compile(r"^(?:heading|titre)\s*(\d)$", re.IGNORECASE)
_PAGES = re.compile(r"<Pages>(\d+)</Pages>")


def parse(filename: str, content: bytes, depth: int = 0) -> ParsedDocument:
    """Parse a .docx into markdown-ish text with its core properties.

    Raises:
        EncryptedDocumentError: If the document is password-protected.
        ArchiveTooLargeError: If the archive fails the zip-bomb checks.
        ParseError: If python-docx cannot read it.
    """
    archive = open_zip(filename, content)
    try:
        from docx import Document
        from docx.oxml.ns import qn
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        document = Document(BytesIO(content))
        blocks: list[str] = []
        for element in document.element.body.iterchildren():
            if element.tag == qn("w:p"):
                blocks.append(_paragraph(Paragraph(element, document)))
            elif element.tag == qn("w:tbl"):
                blocks.append(_table(Table(element, document)))
    except Exception as e:
        raise ParseError(f"Could not read '{filename}' as a Word document ({type(e).__name__}).") from e

    text = "\n\n".join(b for b in blocks if b.strip())
    return ParsedDocument(text=text, metadata=_metadata(document, archive))


def _paragraph(paragraph: Any) -> str:
    text = paragraph.text.strip()
    if not text:
        return ""
    style = (paragraph.style.name if paragraph.style is not None else "") or ""
    if style.lower() == "title":
        return f"# {text}"
    heading = _HEADING_STYLE.match(style)
    if heading:
        return "#" * min(int(heading.group(1)), 6) + " " + text
    level = _list_level(paragraph)
    if level is not None or style.lower().startswith("list"):
        return "  " * (level or 0) + "- " + text
    return text


def _list_level(paragraph: Any) -> int | None:
    """Nesting level of a numbered/bulleted paragraph, or None if not in a list."""
    ppr = paragraph._p.pPr
    if ppr is None or ppr.numPr is None:
        return None
    ilvl = ppr.numPr.ilvl
    return int(ilvl.val) if ilvl is not None and ilvl.val is not None else 0


def _table(table: Any) -> str:
    return markdown_table([_row_texts(row) for row in table.rows])


def _row_texts(row: Any) -> list[str]:
    """Cell texts of a row, once per cell.

    python-docx repeats a horizontally merged cell once per grid column it
    spans; the repeats share one underlying <w:tc> element.
    """
    texts: list[str] = []
    previous = None
    for cell in row.cells:
        if cell._tc is not previous:
            texts.append(cell.text)
        previous = cell._tc
    return texts


def _metadata(document: Any, archive: Any) -> DocumentMetadata:
    props = document.core_properties
    meta = DocumentMetadata(
        source_format="docx",
        title=clean_str(props.title),
        author=clean_str(props.author),
        created=iso_date(props.created),
        modified=iso_date(props.modified),
        language=clean_str(props.language),
    )
    # Word records the page count it last laid out in the extended properties.
    try:
        match = _PAGES.search(archive.read("docProps/app.xml").decode("utf-8", "replace"))
        meta.page_count = int(match.group(1)) if match else None
    except KeyError:
        pass
    return meta
