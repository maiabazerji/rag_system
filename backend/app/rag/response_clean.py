"""Light normalisation of model responses before they reach the API.

This module deliberately does very little. Earlier versions stripped chunk-id
citations, code fences and headings; that removed exactly the grounding the
prompt asks the model to provide, and mangled answers about code. Citations are
now preserved and returned to the client, which renders them against the
`sources` array.
"""
from __future__ import annotations

import json
import re

# Matches the citation form the default prompt asks for, e.g. "[9fa3c1b0e2:4]".
CITATION_RE = re.compile(r"\[([0-9a-f]{6,}:\d+)\]")

# Em dash, en dash and horizontal bar. The prompt already asks the model not to
# use them; this is the deterministic backstop for when it does anyway.
_DASHES = "—–―"

# Fenced blocks and inline spans are left exactly as written: a dash inside code
# or a quoted command is content, not prose punctuation.
_CODE_RE = re.compile(r"```.*?```|~~~.*?~~~|`[^`\n]*`", re.DOTALL)

# Horizontal whitespace only, so line breaks survive and a dash opening a line
# is still recognisable as a list bullet rather than punctuation.
_DASH_RE = re.compile(rf"([ \t]*)([{_DASHES}]+)([ \t]*)")


def extract_citations(text: str) -> list[str]:
    """Return the chunk ids cited in an answer, in order of first appearance.

    Args:
        text: The answer text.

    Returns:
        Unique chunk ids, e.g. ``["9fa3c1b0e2:4"]``. Empty if none are cited.
    """
    seen: dict[str, None] = {}
    for match in CITATION_RE.finditer(text or ""):
        seen.setdefault(match.group(1), None)
    return list(seen)


def strip_ai_dashes(text: str) -> str:
    """Rewrite em/en dash punctuation as ordinary commas.

    The long dash is one of the strongest surface tells of generated prose, and
    it reads as stilted in a short answer. Code fences and inline code spans are
    preserved untouched, numeric ranges become hyphens, and list bullets stay
    bullets; everything else becomes the comma the sentence would have used.

    A comma is used rather than a full stop because it is the one substitution
    that cannot turn a paired parenthetical ("the reranker, a cross-encoder, is
    slow") into two broken sentences.

    Args:
        text: The answer text.

    Returns:
        The text with dash punctuation normalised. Empty input is unchanged.
    """
    if not text:
        return text

    # Resolved up front and matched against the original offsets: replacing in
    # one pass keeps every decision anchored to the real surrounding text, so a
    # dash following an inline code span is not mistaken for a line-leading one.
    code = [(m.start(), m.end()) for m in _CODE_RE.finditer(text)]

    def repl(m: re.Match[str]) -> str:
        indent, dashes, trailing = m.groups()
        if any(a <= m.start() < b for a, b in code):
            return m.group(0)  # code spans are content, copied verbatim

        before = text[: m.start()].rstrip()
        after = text[m.end() :].lstrip()

        # "15-25%", "pages 3-7": a range, so a hyphen is what was meant.
        if before[-1:].isdigit() and after[:1].isdigit():
            return "-"

        # Only whitespace back to the line start: this is a list bullet.
        if not text[text.rfind("\n", 0, m.start()) + 1 : m.start()].strip():
            return f"{indent}-{trailing or ' '}"

        # Dangling at either end: the dash is not joining two clauses.
        if not before or not after:
            return ""
        # Punctuation already does the job on one side; don't double it up.
        if before[-1] in ",;:.!?([{" or after[0] in "([{":
            return " "
        if after[0] in ",;:.!?)]}":
            return ""
        # Don't leave a space stranded at the end of a line.
        return "," if text[m.end() : m.end() + 1] == "\n" else ", "

    return _DASH_RE.sub(repl, text)


def clean_response(text: str) -> str:
    """Normalise a model response for display.

    Unwraps a JSON envelope if the model returned one, rewrites em/en dash
    punctuation as commas, and collapses runs of blank lines. Citations, code
    blocks and headings are left intact.

    Args:
        text: Raw text from the model.

    Returns:
        The cleaned text. Empty input is returned unchanged.
    """
    if not text:
        return text

    # Some prompt variants ask for JSON; unwrap the answer field if we got one.
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            for key in ("answer", "summary"):
                if key in data and isinstance(data[key], str):
                    return clean_response(data[key])

    return re.sub(r"\n{3,}", "\n\n", strip_ai_dashes(text)).strip()
