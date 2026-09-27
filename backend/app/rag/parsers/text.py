"""Plain-text formats: .txt, .md, .markdown, .rst, .json and .csv."""
from __future__ import annotations

import csv
import io

from app.rag.parsers.base import DocumentMetadata, ParsedDocument, ParseError, markdown_table


def decode_utf8(filename: str, content: bytes) -> str:
    """Decode UTF-8, dropping a byte-order mark.

    Raises:
        ParseError: If the bytes are not UTF-8. Decoding with errors='ignore'
            would index mojibake instead.
    """
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise ParseError(
            f"'{filename}' is not valid UTF-8 text. Re-save it as UTF-8 and retry."
        ) from e


def parser_for(source_format: str):
    """A parser that returns the file's text verbatim, tagged with `source_format`."""

    def parse(filename: str, content: bytes, depth: int = 0) -> ParsedDocument:
        return ParsedDocument(
            text=decode_utf8(filename, content),
            metadata=DocumentMetadata(source_format=source_format),
        )

    return parse


def parse_csv(filename: str, content: bytes, depth: int = 0) -> ParsedDocument:
    """Render a CSV as a markdown table, so chunks keep (and repeat) its header.

    A file the csv module cannot make sense of is indexed as plain text.
    """
    text = decode_utf8(filename, content)
    try:
        dialect: type[csv.Dialect] | csv.Dialect = csv.Sniffer().sniff(
            text[:4096], delimiters=",;\t|"
        )
    except csv.Error:
        dialect = csv.excel
    try:
        rows = list(csv.reader(io.StringIO(text), dialect))
    except csv.Error:
        rows = []
    table = markdown_table(rows)
    return ParsedDocument(
        text=table or text, metadata=DocumentMetadata(source_format="csv")
    )
