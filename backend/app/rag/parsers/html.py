"""HTML to markdown-ish text, on the standard library's html.parser.

The converter never executes or fetches anything: it only walks tags. The
content of script, style and other non-text elements is dropped, headings
become ``#`` lines, list items ``-`` lines and tables pipe tables, which is
what the structure-aware chunker keys on.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

from app.rag.parsers.base import DocumentMetadata, ParsedDocument, clean_str, markdown_table
from app.rag.parsers.text import decode_utf8

_SKIP = {"script", "style", "noscript", "template", "svg", "math", "iframe", "object", "head"}
_BLOCK = {
    "p", "div", "section", "article", "header", "footer", "main", "aside", "nav",
    "blockquote", "figure", "figcaption", "form", "fieldset", "address", "dl", "dt", "dd",
    "ul", "ol", "hr", "details", "summary",
}  # fmt: skip
_HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
# Void elements never get an end tag, so they must not open a skip region.
_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "wbr"}
_BLANK_RUNS = re.compile(r"\n{3,}")
_CHARSET = re.compile(rb"<meta[^>]+charset=[\"']?([\w-]+)", re.IGNORECASE)


@dataclass
class _Table:
    rows: list[list[str]] = field(default_factory=list)
    cell: list[str] | None = None

    def close_cell(self) -> None:
        if self.cell is not None:
            self.rows[-1].append(" ".join("".join(self.cell).split()))
            self.cell = None


class _HtmlToText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.line: list[str] = []
        # Open elements whose content is dropped. A stack rather than a count
        # so an unclosed <head> (legal HTML) is closed by <body>.
        self.skip: list[str] = []
        self.pre_depth = 0
        self.lists: list[str] = []
        self.heading: int | None = None
        # Tables being read, innermost last.
        self.tables: list[_Table] = []
        self.title: list[str] = []
        self.in_title = False
        self.lang: str | None = None
        self.meta: dict[str, str] = {}

    # --- tags ---------------------------------------------------------------

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {k: v or "" for k, v in attrs}
        if tag == "html" and attributes.get("lang"):
            self.lang = attributes["lang"]
        elif tag == "meta" and attributes.get("name") and attributes.get("content"):
            self.meta[attributes["name"].lower()] = attributes["content"]
        elif tag == "title":
            self.in_title = True
        if tag in _VOID:
            if tag in ("br", "hr"):
                self._newline()
            return
        if tag == "body":
            self.skip.clear()
        if self.skip or tag in _SKIP:
            if tag in _SKIP:
                self.skip.append(tag)
            return

        if tag == "table":
            if not self.tables:
                self._block()
            self.tables.append(_Table())
        elif self.tables:
            table = self.tables[-1]
            if tag in ("tr", "td", "th"):
                table.close_cell()  # </td> and </tr> are optional in HTML
            if tag == "tr":
                table.rows.append([])
            elif tag in ("td", "th"):
                if not table.rows:
                    table.rows.append([])
                table.cell = []
            # Any other block structure inside a table flattens to its text.
        elif tag in _HEADINGS:
            self._block()
            self.heading = _HEADINGS[tag]
        elif tag in ("ul", "ol"):
            self._list_break()
            self.lists.append(tag)
        elif tag == "li":
            self._newline()
            indent = "  " * max(len(self.lists) - 1, 0)
            marker = "1." if self.lists and self.lists[-1] == "ol" else "-"
            self.line.append(f"{indent}{marker} ")
        elif tag == "pre":
            self._block()
            self.pre_depth += 1
        elif tag in _BLOCK:
            self._block()

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.in_title = False
        if tag in _VOID:
            return
        if self.skip:
            if tag in self.skip:
                while self.skip.pop() != tag:
                    pass
            return

        if tag == "table" and self.tables:
            table = self.tables.pop()
            table.close_cell()
            self._end_table(table)
        elif self.tables:
            if tag in ("td", "th"):
                self.tables[-1].close_cell()
        elif tag in _HEADINGS and self.heading:
            text = " ".join("".join(self.line).split())
            self.line = []
            if text:
                self.out.append("#" * self.heading + " " + text)
            self.heading = None
            self._block()
        elif tag in ("ul", "ol"):
            if self.lists:
                self.lists.pop()
            self._list_break()
        elif tag == "pre":
            self.pre_depth = max(self.pre_depth - 1, 0)
            self._block()
        elif tag in _BLOCK or tag == "li":
            self._newline()

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title.append(data)
        if self.skip:
            return
        if self.tables:
            if self.tables[-1].cell is not None:
                self.tables[-1].cell.append(data)
        elif self.pre_depth:
            self.line.append(data)
        else:
            text = re.sub(r"\s+", " ", data)
            if not self.line or self.line[-1].endswith((" ", "\n")):
                text = text.lstrip()
            self.line.append(text)

    # --- output -------------------------------------------------------------

    def _end_table(self, table: _Table) -> None:
        texts = [" ".join(cell for cell in row if cell) for row in table.rows]
        if self.tables and self.tables[-1].cell is not None:
            # A nested table flattens into the cell that holds it.
            self.tables[-1].cell.append(" " + " ".join(texts) + " ")
        elif max((len(row) for row in table.rows), default=0) <= 1:
            # One column: a layout table (common in HTML email), not data.
            for text in texts:
                if text:
                    self.out.extend([text, ""])
        else:
            self.out.append(markdown_table(table.rows))
            self._block()

    def _newline(self) -> None:
        text = "".join(self.line)
        self.line = []
        # Only trailing space goes: leading space is a nested list's indent.
        text = text if self.pre_depth else text.rstrip()
        if text.strip():
            self.out.append(text)

    def _list_break(self) -> None:
        """A nested list continues its parent's lines; a top-level one is a block."""
        if self.lists:
            self._newline()
        else:
            self._block()

    def _block(self) -> None:
        self._newline()
        if self.out and self.out[-1] != "":
            self.out.append("")

    def text(self) -> str:
        self._newline()
        return _BLANK_RUNS.sub("\n\n", "\n".join(self.out)).strip()


def html_to_text(markup: str) -> str:
    """Convert HTML to markdown-ish text, dropping scripts and styles."""
    parser = _HtmlToText()
    parser.feed(markup)
    parser.close()
    return parser.text()


def parse(filename: str, content: bytes, depth: int = 0) -> ParsedDocument:
    """Parse an HTML page, taking title, author and language from its markup."""
    markup = _decode(filename, content)
    parser = _HtmlToText()
    parser.feed(markup)
    parser.close()
    metadata = DocumentMetadata(
        source_format="html",
        title=clean_str("".join(parser.title)),
        author=clean_str(parser.meta.get("author")),
        language=clean_str(parser.lang or parser.meta.get("language")),
    )
    return ParsedDocument(text=parser.text(), metadata=metadata)


def _decode(filename: str, content: bytes) -> str:
    """Decode by the page's declared charset, falling back to strict UTF-8."""
    declared = _CHARSET.search(content[:2048])
    if declared:
        try:
            return content.decode(declared.group(1).decode("ascii"))
        except (LookupError, UnicodeDecodeError):
            pass
    return decode_utf8(filename, content)
