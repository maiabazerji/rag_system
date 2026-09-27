"""Split document text into chunks.

`chunk_structured` is what ingestion uses. It cuts at headings first, so a
chunk never straddles two articles or chapters, and records each chunk's
heading path so a citation can say where in the document it came from. Only
then does it split by size, with overlap, and it never cuts a markdown table
row in half: a long table is split into groups of rows, each group repeating
the header so it still reads as a table on its own.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.config import settings
from app.rag.parsers.structure import split_sections

_TABLE_SEPARATOR = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?$")
_FENCE = re.compile(r"^\s*(```|~~~)")

# How an atom may be packed with its neighbours. A "window_cont" atom is a
# later window of one long paragraph: it already overlaps the window before
# it, so it must start a chunk of its own and must not receive more overlap.
_AtomKind = Literal["para", "window_first", "window_cont", "table"]


@dataclass(frozen=True)
class TextChunk:
    """One chunk of a document and where it sits in the heading tree."""

    text: str
    heading_path: str


@dataclass(frozen=True)
class _Atom:
    text: str
    words: int
    kind: _AtomKind


def chunk_text(
    text: str, size: int | None = None, overlap: int | None = None
) -> list[str]:
    """Split text into overlapping word windows.

    Args:
        text: The document text.
        size: Words per chunk. Defaults to CHUNK_SIZE_TOKENS.
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
    """Split text at headings, then by size, keeping tables row-intact.

    Args:
        text: Markdown-ish document text, as produced by the parsers.
        size: Words per chunk. Defaults to CHUNK_SIZE_TOKENS.
        overlap: Words shared between neighbouring prose chunks. Defaults to
            CHUNK_OVERLAP_TOKENS.

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
        body = text[section.body_start : section.end]
        if not body.strip():
            continue
        heading = text[section.start : section.body_start].strip()
        chunks.extend(
            TextChunk(piece, section.breadcrumb)
            for piece in _chunk_section(heading, body, size, overlap)
        )
    return chunks


def _chunk_section(heading: str, body: str, size: int, overlap: int) -> list[str]:
    """Chunk one section, keeping its heading line on the first chunk.

    The body is chunked to leave room for the heading, so the heading never
    ends up alone in a chunk of its own.
    """
    room = size - len(heading.split())
    if not heading or room <= overlap:
        text = f"{heading}\n\n{body}" if heading else body
        return _pack(_atoms(text, size, overlap), size, overlap)
    pieces = _pack(_atoms(body, room, overlap), room, overlap)
    pieces[0] = f"{heading}\n\n{pieces[0]}"
    return pieces


def _resolve(size: int | None, overlap: int | None) -> tuple[int, int]:
    size = settings.chunk_size_tokens if size is None else size
    overlap = settings.chunk_overlap_tokens if overlap is None else overlap
    if overlap >= size:
        raise ValueError(f"chunk overlap ({overlap}) must be smaller than size ({size})")
    return size, overlap


def _atoms(body: str, size: int, overlap: int) -> list[_Atom]:
    """Break a section into units no larger than `size` that must not be split."""
    atoms: list[_Atom] = []
    for kind, lines in _blocks(body):
        if kind == "table":
            atoms.extend(_table_atoms(lines, size))
            continue
        block = "\n".join(lines).strip()
        n = len(block.split())
        if n <= size:
            atoms.append(_Atom(block, n, "para"))
            continue
        for i, window in enumerate(chunk_text(block, size, overlap)):
            atoms.append(_Atom(window, len(window.split()), "window_cont" if i else "window_first"))
    return atoms


def _blocks(body: str) -> list[tuple[str, list[str]]]:
    """Group lines into paragraphs (split on blank lines) and pipe tables."""
    blocks: list[tuple[str, list[str]]] = []
    current: list[str] = []
    current_kind = "text"
    in_fence = False

    def flush() -> None:
        nonlocal current
        if any(line.strip() for line in current):
            blocks.append((current_kind, current))
        current = []

    for line in body.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        is_table = not in_fence and line.lstrip().startswith("|")
        kind = "table" if is_table else "text"
        if kind != current_kind or (not in_fence and not line.strip()):
            flush()
            current_kind = kind
        if line.strip() or in_fence:
            current.append(line.rstrip())
    flush()
    return blocks


def _table_atoms(lines: list[str], size: int) -> list[_Atom]:
    """Split a pipe table into row groups of at most `size` words, header repeated."""
    has_header = len(lines) > 1 and bool(_TABLE_SEPARATOR.match(lines[1].strip()))
    header = lines[:2] if has_header else []
    rows = lines[2:] if has_header else lines
    header_words = sum(len(line.split()) for line in header)

    atoms: list[_Atom] = []
    group: list[str] = []
    words = header_words
    for row in rows:
        n = len(row.split())
        if group and words + n > size:
            atoms.append(_Atom("\n".join(header + group), words, "table"))
            group, words = [], header_words
        group.append(row)
        words += n
    if group or not atoms:
        atoms.append(_Atom("\n".join(header + group), words, "table"))
    return atoms


def _pack(atoms: list[_Atom], size: int, overlap: int) -> list[str]:
    """Greedily join atoms into chunks of at most `size` words.

    When a chunk closes between two pieces of prose, the next chunk opens
    with the last `overlap` words of the previous one, as plain windowing
    did, so a sentence cut at the boundary is still whole in one chunk.
    """
    chunks: list[str] = []
    current: list[str] = []
    words = 0
    previous: _Atom | None = None

    for atom in atoms:
        must_break = atom.kind == "window_cont" or words + atom.words > size
        if current and must_break:
            chunks.append("\n\n".join(current))
            current, words = [], 0
            if previous and overlap and _takes_overlap(previous, atom, size, overlap):
                seed = " ".join(previous.text.split()[-overlap:])
                current, words = [seed], len(seed.split())
        current.append(atom.text)
        words += atom.words
        previous = atom

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _takes_overlap(previous: _Atom, atom: _Atom, size: int, overlap: int) -> bool:
    return (
        previous.kind != "table"
        and atom.kind in ("para", "window_first")
        and atom.words + overlap <= size
    )
