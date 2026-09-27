"""Find the heading structure of parsed text.

Two kinds of heading are recognised:

* Markdown ATX headings (``# Title`` to ``###### Title``), which is what every
  parser emits for headings it finds in the file's own markup.
* French legal divisions written as plain lines, as in codes, décrets and
  arrêtés: "Titre I", "Chapitre 2 : Dispositions générales", "Section 3",
  "Article L. 121-1", "Article 5", "Art. 12. - Le présent décret...".

Markdown headings nest by level among themselves. Legal divisions nest by
their kind ("Article" under "Chapitre" under "Titre"), including against a
markdown heading that names a division: "# Chapitre 2" followed by a bare
"Article 5" nests as "Chapitre 2 > Article 5", and a later bare "Chapitre 3"
closes both. A bare legal line never closes a markdown heading that is not a
division, so "# Code civil > Article 5" holds too.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.rag.parsers.base import Section

# Rank of each legal division, outermost first.
_LEGAL_RANKS = {
    "partie": 1,
    "livre": 2,
    "titre": 3,
    "chapitre": 4,
    "section": 5,
    "sous-section": 6,
    "paragraphe": 7,
    "article": 8,
    "art.": 8,
}

_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")

_DIVISION_NUMBER = (
    r"(?:premier|première|préliminaire|preliminaire|unique"
    r"|[IVXLCDM]+|\d+(?:[.-]\d+)*|[A-Z])"
    r"(?:\s*(?:er|ère|bis|ter|quater))?"
)
_ARTICLE_NUMBER = (
    r"(?:(?:LO|L|R|D|A)\.?\s*\*?\s*)?\d+(?:[-.]\d+)*"
    r"(?:\s*(?:er|bis|ter|quater|quinquies|sexies|septies|octies))?"
    r"|premier|préliminaire|unique"
)
_DIVISION = re.compile(
    r"^(?P<kind>partie|livre|titre|chapitre|sous-section|section|paragraphe)\s+"
    rf"(?P<num>{_DIVISION_NUMBER})\b(?P<rest>.*)$",
    re.IGNORECASE,
)
_ARTICLE = re.compile(
    rf"^(?P<kind>article|art\.)\s*(?P<num>{_ARTICLE_NUMBER})(?![\w-])(?P<rest>.*)$",
    re.IGNORECASE,
)
_SEPARATOR = re.compile(r"^\s*[.:\-–—]+\s*")

# A heading line longer than this is prose that happens to start with
# "Section 3" or "Article 5", not a heading.
_MAX_HEADING_WORDS = 20
_MAX_ARTICLE_TITLE_WORDS = 10
_MAX_LABEL_CHARS = 120


@dataclass(frozen=True)
class Heading:
    """A heading found on one line.

    Attributes:
        label: The text shown in a breadcrumb.
        level: Markdown heading level (1-6), None for a bare line.
        legal_rank: Rank of the legal division it names, lower is outer;
            None when it names none.
        inline_body: True when the rest of the line is body text, not title.
    """

    label: str
    level: int | None
    legal_rank: int | None
    inline_body: bool = False

    def closes(self, other: Heading) -> bool:
        """Whether this heading ends the section `other` opened."""
        is_bare = self.level is None or other.level is None
        if is_bare and self.legal_rank and other.legal_rank:
            return other.legal_rank >= self.legal_rank
        if self.level is not None and other.level is not None:
            return other.level >= self.level
        # A markdown heading ends bare legal structure; a bare legal line
        # nests under a markdown heading that is not itself a division.
        return self.level is not None


def detect_heading(line: str) -> Heading | None:
    """Classify one line of text as a heading, or None if it is not one."""
    stripped = line.strip()
    if not stripped or stripped.startswith("|"):
        return None

    md = _MARKDOWN_HEADING.match(stripped)
    if md:
        label = md.group(2).strip().strip("*_").strip()
        legal = _legal_heading(label)
        return Heading(_truncate(label), len(md.group(1)), legal.legal_rank if legal else None)

    return _legal_heading(stripped)


def split_sections(text: str) -> list[Section]:
    """Cut text into sections at every heading, tracking the heading path.

    Text before the first heading forms a section with an empty path. Lines
    inside fenced code blocks are never headings.

    Returns:
        Sections covering the whole text, in order. Empty for blank text.
    """
    if not text.strip():
        return []

    sections: list[Section] = []
    stack: list[Heading] = []
    path: tuple[str, ...] = ()
    start = body_start = 0
    offset = 0
    in_fence = False

    for line in text.splitlines(keepends=True):
        line_start, offset = offset, offset + len(line)
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        heading = None if in_fence else detect_heading(line)
        if heading is None:
            continue

        if line_start > start:
            sections.append(Section(path, start, line_start, body_start))
        while stack and heading.closes(stack[-1]):
            stack.pop()
        stack.append(heading)
        path = tuple(h.label for h in stack)
        start = line_start
        body_start = line_start if heading.inline_body else offset

    sections.append(Section(path, start, len(text), body_start))
    return sections


def _legal_heading(line: str) -> Heading | None:
    """Recognise a French legal division line."""
    if not line[0].isupper():
        return None

    article = _ARTICLE.match(line)
    if article:
        label = f"{_kind_label(article.group('kind'))} {' '.join(article.group('num').split())}"
        rest = article.group("rest")
        if not rest.strip():
            return Heading(label, None, _LEGAL_RANKS["article"])
        sep = _SEPARATOR.match(rest)
        if not sep:
            return None  # "Article 5 dispose que..." is a sentence
        title = rest[sep.end() :].strip()
        if not title:
            return Heading(label, None, _LEGAL_RANKS["article"])
        if len(title.split()) <= _MAX_ARTICLE_TITLE_WORDS and title[-1] not in ".;:":
            return Heading(_truncate(f"{label} : {title}"), None, _LEGAL_RANKS["article"])
        # "Art. 12. - Le présent décret entre en vigueur..." : the line is body.
        return Heading(label, None, _LEGAL_RANKS["article"], inline_body=True)

    division = _DIVISION.match(line)
    if not division or len(line.split()) > _MAX_HEADING_WORDS:
        return None
    rest = division.group("rest").strip()
    if rest and not (_SEPARATOR.match(rest) or rest.isupper()):
        return None  # "Section 3 du présent chapitre..." is a sentence
    return Heading(
        _truncate(" ".join(line.split())), None, _LEGAL_RANKS[division.group("kind").lower()]
    )


def _kind_label(kind: str) -> str:
    return "Art." if kind.lower() == "art." else "Article"


def _truncate(label: str) -> str:
    return label if len(label) <= _MAX_LABEL_CHARS else label[: _MAX_LABEL_CHARS - 1] + "…"
