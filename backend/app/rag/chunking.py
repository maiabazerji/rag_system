"""Split document text into chunks.

`chunk_document` is what ingestion uses. With the default ``structured``
strategy (`chunk_structured`) it cuts at headings first, so a chunk never
straddles two articles or chapters, and records each chunk's heading path so
a citation can say where in the document it came from. Within a section it
packs whole paragraphs; only a paragraph larger than a chunk is split, between
sentences, and only a sentence larger than a chunk is split between words.
Overlap between neighbouring chunks is made of whole trailing sentences. It
never cuts a markdown table row in half: a long table is split into groups of
rows, each group repeating the header so it still reads as a table on its own.

The ``fixed`` strategy is the legacy word-window chunker (`chunk_text`), kept
for comparison.

Every chunk records the span of the document text it was cut from
(``start``/``end``). A structured chunk's text *is* that span,
``text[start:end]``, except that a table row group after the first carries
the table header in front of it; so ``chunk.text.endswith(text[start:end])``
always holds. A fixed chunk's text is the span's words joined by single spaces.
"""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, replace
from typing import Literal

from app.config import settings
from app.rag.parsers.base import PageSpan, Section
from app.rag.parsers.structure import split_sections

_TABLE_SEPARATOR = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_WORD = re.compile(r"\S+")

# Sentence-final punctuation, any closing quotes or brackets after it (French
# puts a space before "»"), then the whitespace before the next sentence.
_SENTENCE_END = re.compile(r"(?P<punct>[.!?…]+)(?:\s?[»”\"'’)\]])*\s+")
_SENTENCE_OPENERS = "«“\"'‘([   "
# A line that starts a list item starts a new sentence, punctuated or not.
_LIST_ITEM = re.compile(r"\n[ \t]*(?=(?:[-*•]|\d{1,3}[.)])\s)")
# Words that end with a period without ending the sentence ("M. Dupont",
# "art. 5", "p. 12"), lower-cased and without the period. Single letters
# (initials, "L. 121-1") are recognised without being listed.
_ABBREVIATIONS = frozenset(
    {
        "m", "mm", "mme", "mmes", "mlle", "mlles", "mr", "mrs", "ms", "dr", "pr",
        "me", "st", "ste", "jr", "sr", "art", "arts", "al", "p", "pp", "cf", "ex",
        "fig", "vol", "chap", "n", "no", "nos", "num", "env", "av", "bd", "réf",
        "ref", "op", "cit", "vs", "approx", "sect", "suiv", "éd", "ed",
    }
)

# How an atom may be packed with its neighbours. A "window_cont" atom is a
# later window of one over-long sentence: it already overlaps the window
# before it, so it must start a chunk of its own and must not receive more
# overlap.
_AtomKind = Literal["para", "sentence", "window_first", "window_cont", "table"]


@dataclass(frozen=True)
class TextChunk:
    """One chunk of a document and where it sits in the document.

    Attributes:
        text: The chunk text.
        heading_path: Breadcrumb of the section it belongs to, "A > B".
        start: Offset in the document text where the chunk's span begins.
        end: Offset one past the span's end.
        headings: The heading path as a tuple, outermost first.
        page_start: 1-based page (or slide) the chunk starts on, when the
            document has pages.
        page_end: Page it ends on, when the document has pages.
    """

    text: str
    heading_path: str
    start: int = 0
    end: int = 0
    headings: tuple[str, ...] = ()
    page_start: int | None = None
    page_end: int | None = None

    @property
    def section(self) -> str:
        """The innermost heading, or "" before the first heading."""
        return self.headings[-1] if self.headings else ""

    @property
    def token_count(self) -> int:
        """Words in the chunk, the unit CHUNK_SIZE_TOKENS counts in."""
        return len(self.text.split())


@dataclass(frozen=True)
class _Atom:
    """A unit the packer never splits.

    ``start``/``end`` is its span of the document. ``prefix`` is text shown
    before that span (a repeated table header). ``breaks`` are the offsets
    where a sentence starts inside it: the places overlap may begin.
    """

    words: int
    kind: _AtomKind
    start: int
    end: int
    breaks: tuple[int, ...] = ()
    prefix: str = ""


@dataclass(frozen=True)
class _Piece:
    """A packed chunk: ``prefix`` followed by the document span ``start:end``."""

    prefix: str
    start: int
    end: int


def chunk_document(
    text: str,
    *,
    strategy: str | None = None,
    size: int | None = None,
    overlap: int | None = None,
    pages: list[PageSpan] | None = None,
) -> list[TextChunk]:
    """Chunk a parsed document with the configured strategy.

    Args:
        text: The document text, as produced by the parsers.
        strategy: "structured" or "fixed". Defaults to CHUNK_STRATEGY.
        size: Words per chunk. Defaults to CHUNK_SIZE_TOKENS, capped by
            CHUNK_MAX_MODEL_TOKENS.
        overlap: Overlap between neighbouring chunks, in words. Defaults to
            CHUNK_OVERLAP_TOKENS.
        pages: Page spans over ``text``; when given, each chunk gets the
            pages it starts and ends on.

    Returns:
        The chunks in document order.

    Raises:
        ValueError: If the strategy is unknown, or overlap is not smaller
            than size.
    """
    strategy = (strategy or settings.chunk_strategy).lower()
    if strategy == "structured":
        chunks = chunk_structured(text, size, overlap)
    elif strategy == "fixed":
        chunks = _chunk_fixed(text, size, overlap)
    else:
        raise ValueError(f"Unknown chunk strategy '{strategy}'; expected structured or fixed")
    if pages:
        chunks = [_with_pages(c, pages) for c in chunks]
    return chunks


def chunk_text(
    text: str, size: int | None = None, overlap: int | None = None
) -> list[str]:
    """Split text into overlapping word windows.

    Args:
        text: The document text.
        size: Words per chunk. Defaults to CHUNK_SIZE_TOKENS, capped by
            CHUNK_MAX_MODEL_TOKENS.
        overlap: Words shared between neighbouring chunks. Defaults to
            CHUNK_OVERLAP_TOKENS.

    Returns:
        The chunks, in document order. Empty if the text has no words.

    Raises:
        ValueError: If overlap is not smaller than size, which would make
            chunking loop forever.
    """
    size, overlap = _resolve(size, overlap)
    words = text.split()
    step = size - overlap
    chunks = []
    for i in range(0, len(words), step):
        window = words[i : i + size]
        if not window:
            break
        chunks.append(" ".join(window))
    return chunks


def chunk_structured(
    text: str, size: int | None = None, overlap: int | None = None
) -> list[TextChunk]:
    """Split text at headings, then into paragraphs and sentences, by size.

    Args:
        text: Markdown-ish document text, as produced by the parsers.
        size: Words per chunk. Defaults to CHUNK_SIZE_TOKENS, capped by
            CHUNK_MAX_MODEL_TOKENS.
        overlap: Most words of trailing whole sentences a prose chunk repeats
            from the one before it. Defaults to CHUNK_OVERLAP_TOKENS.

    Returns:
        The chunks in document order. A section holding nothing but its own
        heading yields no chunk; its heading still appears in the paths of the
        sections beneath it.

    Raises:
        ValueError: If overlap is not smaller than size.
    """
    size, overlap = _resolve(size, overlap)
    chunks: list[TextChunk] = []
    for section in split_sections(text):
        if not text[section.body_start : section.end].strip():
            continue
        chunks.extend(
            TextChunk(
                piece.prefix + text[piece.start : piece.end],
                section.breadcrumb,
                piece.start,
                piece.end,
                section.heading_path,
            )
            for piece in _chunk_section(text, section, size, overlap)
        )
    return chunks


def _chunk_fixed(text: str, size: int | None, overlap: int | None) -> list[TextChunk]:
    """Legacy word windows (the text `chunk_text` makes), with spans and headings."""
    size, overlap = _resolve(size, overlap)
    sections = split_sections(text)
    starts = [s.start for s in sections]
    chunks = []
    for start, end in _windows(_word_spans(text, 0, len(text)), size, overlap):
        section = sections[max(0, bisect.bisect_right(starts, start) - 1)]
        chunks.append(
            TextChunk(
                " ".join(text[start:end].split()),
                section.breadcrumb,
                start,
                end,
                section.heading_path,
            )
        )
    return chunks


def _with_pages(chunk: TextChunk, pages: list[PageSpan]) -> TextChunk:
    """Record the pages a chunk's span starts and ends on."""
    starts = [p.start for p in pages]

    def page_at(offset: int) -> int:
        # Text between two pages (the separator) counts as the earlier page.
        return pages[max(0, bisect.bisect_right(starts, offset) - 1)].number

    first = page_at(chunk.start)
    last = page_at(max(chunk.start, chunk.end - 1))
    return replace(chunk, page_start=first, page_end=max(first, last))


def _chunk_section(doc: str, section: Section, size: int, overlap: int) -> list[_Piece]:
    """Chunk one section, keeping its heading line on the first chunk.

    The body is chunked to leave room for the heading, so the heading never
    ends up alone in a chunk of its own.
    """
    raw = doc[section.start : section.body_start]
    heading = raw.strip()
    room = size - len(heading.split())
    if not heading or room <= overlap:
        return _pack(doc, _atoms(doc, section.start, section.end, size, overlap), size, overlap)
    pieces = _pack(doc, _atoms(doc, section.body_start, section.end, room, overlap), room, overlap)
    heading_start = section.start + len(raw) - len(raw.lstrip())
    pieces[0] = _Piece("", heading_start, pieces[0].end)
    return pieces


def _resolve(size: int | None, overlap: int | None) -> tuple[int, int]:
    # An explicit size is taken as given; the configured one is capped so a
    # chunk fits the embedder's and reranker's input (CHUNK_MAX_MODEL_TOKENS).
    size = settings.chunk_word_budget if size is None else size
    overlap = settings.chunk_overlap_tokens if overlap is None else overlap
    if overlap >= size:
        raise ValueError(f"chunk overlap ({overlap}) must be smaller than size ({size})")
    return size, overlap


def _count(text: str) -> int:
    return len(text.split())


def _atoms(doc: str, start: int, end: int, size: int, overlap: int) -> list[_Atom]:
    """Break ``doc[start:end]`` into units of at most `size` words that must not be split."""
    atoms: list[_Atom] = []
    for kind, lines in _blocks(doc, start, end):
        if kind == "table":
            atoms.extend(_table_atoms(doc, lines, size))
            continue
        block = doc[lines[0][0] : lines[-1][1]]
        first = lines[0][0] + len(block) - len(block.lstrip())
        atoms.extend(_prose_atoms(doc, first, first + len(block.strip()), size, overlap))
    return atoms


def _prose_atoms(doc: str, start: int, end: int, size: int, overlap: int) -> list[_Atom]:
    """A paragraph as one atom; if too long, its sentences; failing that, word windows."""
    sentences = _sentence_spans(doc, start, end)
    words = _count(doc[start:end])
    if words <= size:
        return [_Atom(words, "para", start, end, tuple(s for s, _ in sentences))]

    atoms: list[_Atom] = []
    for s, e in sentences:
        n = _count(doc[s:e])
        if n <= size:
            atoms.append(_Atom(n, "sentence", s, e, (s,)))
            continue
        for i, (ws, we) in enumerate(_windows(_word_spans(doc, s, e), size, overlap)):
            kind: _AtomKind = "window_cont" if i else "window_first"
            atoms.append(_Atom(_count(doc[ws:we]), kind, ws, we, () if i else (ws,)))
    return atoms


def _sentence_spans(doc: str, start: int, end: int) -> list[tuple[int, int]]:
    """Cut ``doc[start:end]`` into sentences, as spans without surrounding whitespace.

    A sentence ends at ".", "!", "?" or "…" (plus any closing quote or
    bracket) followed by whitespace and a capital letter, possibly behind an
    opening "«". A period after a known abbreviation or a single letter does
    not end one. A new list item always starts a sentence.
    """
    segment = doc[start:end]
    starts = {0}
    for match in _SENTENCE_END.finditer(segment):
        after = match.end()
        if (
            after < len(segment)
            and _opens_sentence(segment, after)
            and not (
                match.group("punct") == "."
                and _after_abbreviation(segment, match.start())
            )
        ):
            starts.add(after)
    for match in _LIST_ITEM.finditer(segment):
        line = segment[match.end() :]
        starts.add(match.end() + len(line) - len(line.lstrip()))

    ordered = sorted(starts)
    spans = []
    for i, s in enumerate(ordered):
        e = ordered[i + 1] if i + 1 < len(ordered) else len(segment)
        spans.append((start + s, start + s + len(segment[s:e].rstrip())))
    return [(s, e) for s, e in spans if e > s]


def _opens_sentence(segment: str, at: int) -> bool:
    rest = segment[at:].lstrip(_SENTENCE_OPENERS)
    return bool(rest) and rest[0].isupper()


def _after_abbreviation(segment: str, period: int) -> bool:
    """Whether the lone period at ``period`` closes an abbreviation, not a sentence."""
    token = re.search(r"\S*$", segment[:period])
    word = token.group(0).lstrip(_SENTENCE_OPENERS).lower() if token else ""
    word = re.split(r"['’]", word)[-1]  # elision: "l'art." is "art."
    if not word:
        return False
    if len(word) == 1 and word.isalpha():
        return True  # an initial: "J. Dupont", "L. 121-1"
    if "." in word:  # "e.g", "U.S"
        return all(len(part) <= 2 for part in word.split("."))
    return word in _ABBREVIATIONS


def _word_spans(doc: str, start: int, end: int) -> list[tuple[int, int]]:
    return [(start + m.start(), start + m.end()) for m in _WORD.finditer(doc[start:end])]


def _windows(words: list[tuple[int, int]], size: int, overlap: int) -> list[tuple[int, int]]:
    """Overlapping windows of `size` words, as spans: the windows `chunk_text` makes."""
    return [
        (words[i][0], words[min(i + size, len(words)) - 1][1])
        for i in range(0, len(words), size - overlap)
    ]


def _blocks(doc: str, start: int, end: int) -> list[tuple[str, list[tuple[int, int]]]]:
    """Group the lines of ``doc[start:end]`` into paragraphs and pipe tables.

    Paragraphs are split on blank lines, except inside code fences. Each line
    is a span with its trailing whitespace left out.
    """
    blocks: list[tuple[str, list[tuple[int, int]]]] = []
    current: list[tuple[int, int]] = []
    current_kind = "text"
    in_fence = False

    def flush() -> None:
        nonlocal current
        if any(e > s for s, e in current):
            blocks.append((current_kind, current))
        current = []

    offset = start
    for line in doc[start:end].splitlines(keepends=True):
        line_start, offset = offset, offset + len(line)
        content = line.rstrip()
        if _FENCE.match(line):
            in_fence = not in_fence
        is_table = not in_fence and line.lstrip().startswith("|")
        kind = "table" if is_table else "text"
        if kind != current_kind or (not in_fence and not content):
            flush()
            current_kind = kind
        if content or in_fence:
            current.append((line_start, line_start + len(content)))
    flush()
    return blocks


def _table_atoms(doc: str, lines: list[tuple[int, int]], size: int) -> list[_Atom]:
    """Split a pipe table into row groups of at most `size` words, header repeated."""
    has_header = len(lines) > 1 and bool(
        _TABLE_SEPARATOR.match(doc[lines[1][0] : lines[1][1]].strip())
    )
    header = lines[:2] if has_header else []
    rows = lines[2:] if has_header else lines
    header_text = "\n".join(doc[s:e] for s, e in header)
    header_words = _count(header_text)

    groups: list[list[tuple[int, int]]] = []
    group: list[tuple[int, int]] = []
    words = header_words
    for row in rows:
        n = _count(doc[row[0] : row[1]])
        if group and words + n > size:
            groups.append(group)
            group, words = [], header_words
        group.append(row)
        words += n
    if group or not groups:
        groups.append(group)

    atoms = []
    for i, group in enumerate(groups):
        words = header_words + sum(_count(doc[s:e]) for s, e in group)
        if i == 0:
            span = header + group
            atoms.append(_Atom(words, "table", span[0][0], span[-1][1]))
        else:
            prefix = header_text + "\n" if header else ""
            atoms.append(_Atom(words, "table", group[0][0], group[-1][1], prefix=prefix))
    return atoms


def _pack(doc: str, atoms: list[_Atom], size: int, overlap: int) -> list[_Piece]:
    """Greedily join atoms into chunks of at most `size` words.

    When a chunk closes between two pieces of prose, the next chunk opens
    with the previous chunk's trailing whole sentences, as many as fit in
    `overlap` words, so context carries across the boundary without a
    sentence ever being cut.
    """
    pieces: list[_Piece] = []
    chunk: list[_Atom] = []
    start = words = 0

    def close() -> None:
        prefix = chunk[0].prefix if start == chunk[0].start else ""
        pieces.append(_Piece(prefix, start, chunk[-1].end))

    for atom in atoms:
        if chunk and (atom.kind == "window_cont" or words + atom.words > size):
            close()
            seed = _overlap_start(doc, chunk, atom, size, overlap)
            start = atom.start if seed is None else seed
            words = 0 if seed is None else _count(doc[seed : chunk[-1].end])
            chunk = []
        elif not chunk:
            start, words = atom.start, 0
        chunk.append(atom)
        words += atom.words

    if chunk:
        close()
    return pieces


def _overlap_start(
    doc: str, chunk: list[_Atom], atom: _Atom, size: int, overlap: int
) -> int | None:
    """Where the chunk after ``chunk`` should start so it repeats whole sentences.

    Returns the earliest sentence start in the prose at the end of ``chunk``
    from which at most `overlap` words remain and that still leaves room for
    ``atom``; None when no whole sentence fits, or overlap does not apply
    (before a table, or a window that carries its own overlap).
    """
    if not overlap or atom.kind not in ("para", "sentence", "window_first"):
        return None
    end = chunk[-1].end
    best = None
    for previous in reversed(chunk):
        if previous.kind == "table":
            break
        for candidate in reversed(previous.breaks):
            n = _count(doc[candidate:end])
            if n > overlap or n + atom.words > size:
                return best
            best = candidate
    return best
