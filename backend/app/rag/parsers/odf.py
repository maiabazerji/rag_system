"""OpenDocument .odt and .ods parser (odfpy).

LibreOffice is the default office suite across much of the French public
sector, so these formats turn up alongside .docx. Text documents keep their
headings (by outline level), lists and tables; spreadsheets become one pipe
table per sheet under a heading with the sheet's name.
"""
from __future__ import annotations

from io import BytesIO
from typing import Any

from app.rag.parsers.archive import open_zip
from app.rag.parsers.base import (
    DocumentMetadata,
    EncryptedDocumentError,
    ParsedDocument,
    ParseError,
    clean_str,
    iso_date,
    markdown_table,
)

_TEXT_NS = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
_TABLE_NS = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
_DC_NS = "http://purl.org/dc/elements/1.1/"
_META_NS = "urn:oasis:names:tc:opendocument:xmlns:meta:1.0"

# Spreadsheets mark runs of identical (usually empty) cells and rows with a
# repeat count, and a blank sheet can declare a million of them. Repeats of
# non-empty content are honoured up to these caps; empty ones are dropped.
_MAX_REPEAT = 100
_MAX_COLUMNS = 256


def parse(filename: str, content: bytes, depth: int = 0) -> ParsedDocument:
    """Parse an .odt or .ods file.

    Raises:
        EncryptedDocumentError: If the document is password-protected.
        ArchiveTooLargeError: If the archive fails the zip-bomb checks.
        ParseError: If odfpy cannot read it.
    """
    archive = open_zip(filename, content)
    try:
        manifest = archive.read("META-INF/manifest.xml")
    except KeyError:
        manifest = b""
    if b"encryption-data" in manifest:
        raise EncryptedDocumentError(
            f"'{filename}' is password-protected. Remove the password and retry."
        )

    try:
        from odf.opendocument import load

        document = load(BytesIO(content))
        is_sheet = document.mimetype.endswith("spreadsheet")
        body = document.spreadsheet if is_sheet else document.text
        blocks = _blocks(body, sheet_headings=is_sheet)
    except Exception as e:
        raise ParseError(
            f"Could not read '{filename}' as an OpenDocument file ({type(e).__name__})."
        ) from e

    text = "\n\n".join(b for b in blocks if b.strip())
    return ParsedDocument(
        text=text, metadata=_metadata(document, "ods" if is_sheet else "odt")
    )


def _blocks(element: Any, sheet_headings: bool = False, list_depth: int = 0) -> list[str]:
    """Walk an element's children in order, rendering each block."""
    from odf import teletype

    blocks: list[str] = []
    for child in element.childNodes:
        if child.nodeType != child.ELEMENT_NODE:
            continue
        ns, name = child.qname
        if ns == _TEXT_NS and name == "h":
            level = _int(child.getAttrNS(_TEXT_NS, "outline-level"), default=1)
            text = " ".join(teletype.extractText(child).split())
            if text:
                blocks.append("#" * min(max(level, 1), 6) + " " + text)
        elif ns == _TEXT_NS and name == "p":
            blocks.append(teletype.extractText(child).strip())
        elif ns == _TEXT_NS and name == "list":
            blocks.append("\n".join(_list_items(child, list_depth)))
        elif ns == _TABLE_NS and name == "table":
            if sheet_headings:
                sheet = child.getAttrNS(_TABLE_NS, "name")
                if sheet:
                    blocks.append(f"## {sheet}")
            blocks.append(markdown_table(_table_rows(child)))
        else:
            # Sections, frames and the like just wrap more blocks.
            blocks.extend(_blocks(child, sheet_headings, list_depth))
    return blocks


def _list_items(element: Any, depth: int) -> list[str]:
    from odf import teletype

    lines: list[str] = []
    for item in element.childNodes:
        if item.nodeType != item.ELEMENT_NODE:
            continue
        for child in item.childNodes:
            if child.nodeType != child.ELEMENT_NODE:
                continue
            if child.qname == (_TEXT_NS, "list"):
                lines.extend(_list_items(child, depth + 1))
            else:
                text = " ".join(teletype.extractText(child).split())
                if text:
                    lines.append("  " * depth + "- " + text)
    return lines


def _table_rows(table: Any) -> list[list[str]]:
    from odf import teletype

    rows: list[list[str]] = []
    for row in _descendants(table, "table-row"):
        cells: list[str] = []
        for cell in row.childNodes:
            if cell.nodeType != cell.ELEMENT_NODE or cell.qname[1] not in (
                "table-cell",
                "covered-table-cell",
            ):
                continue
            text = " ".join(teletype.extractText(cell).split())
            repeat = _int(cell.getAttrNS(_TABLE_NS, "number-columns-repeated"), default=1)
            cells.extend([text] * min(repeat, _MAX_REPEAT if text else _MAX_COLUMNS))
            if len(cells) >= _MAX_COLUMNS:
                break
        while cells and not cells[-1]:
            cells.pop()
        if not cells:
            continue
        repeat = _int(row.getAttrNS(_TABLE_NS, "number-rows-repeated"), default=1)
        rows.extend([cells] * min(repeat, _MAX_REPEAT))
    return rows


def _descendants(element: Any, name: str) -> list[Any]:
    """Table rows, including those inside header-rows and row-group wrappers."""
    found: list[Any] = []
    for child in element.childNodes:
        if child.nodeType != child.ELEMENT_NODE or child.qname[0] != _TABLE_NS:
            continue
        if child.qname[1] == name:
            found.append(child)
        elif child.qname[1] != "table":  # a nested table is its own table
            found.extend(_descendants(child, name))
    return found


def _metadata(document: Any, source_format: str) -> DocumentMetadata:
    fields: dict[tuple[str, str], str] = {}
    stats: Any = None
    for child in document.meta.childNodes:
        if child.nodeType != child.ELEMENT_NODE:
            continue
        if child.qname == (_META_NS, "document-statistic"):
            stats = child
        else:
            fields[child.qname] = str(child)

    page_count = None
    if stats is not None:
        page_count = _int(stats.getAttrNS(_META_NS, "page-count"), default=0) or None
    return DocumentMetadata(
        source_format=source_format,
        title=clean_str(fields.get((_DC_NS, "title"))),
        author=clean_str(
            fields.get((_DC_NS, "creator")) or fields.get((_META_NS, "initial-creator"))
        ),
        created=iso_date(fields.get((_META_NS, "creation-date"))),
        modified=iso_date(fields.get((_DC_NS, "date"))),
        page_count=page_count,
        language=clean_str(fields.get((_DC_NS, "language"))),
    )


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
