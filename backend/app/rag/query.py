"""Query preprocessing shared by every retriever.

A question arrives as free text typed or pasted by a user: it may carry
full-width characters, ligatures, non-breaking spaces, stray control characters
or thousands of characters of pasted context. Retrieval works on a normalised
copy; generation keeps the original, so the model answers exactly what was
asked.

The BM25 tokenizer lives here too, because the sparse retriever
(:mod:`app.rag.sparse`) and the reranker's BM25 fallback (:mod:`app.rag.rerank`)
must agree on what a term is.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from app.i18n import STOPWORDS
from app.i18n import tokenize as _unicode_tokens

# Longer queries are cut here. An embedding model reads a few hundred tokens at
# most, and BM25 over a pasted page scores everything a little and nothing well.
MAX_QUERY_CHARS = 1000

# English and French function words carry no topical signal for BM25; dropping
# them keeps "le", "de", "the" from dominating short queries.
BM25_STOPWORDS: frozenset[str] = STOPWORDS["en"] | STOPWORDS["fr"]

# Control and format characters (zero-width spaces, BOMs, soft hyphens) that
# NFKC leaves in place but that split or glue words invisibly.
_INVISIBLE = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f­​-‏⁠﻿]")
_WHITESPACE = re.compile(r"\s+")


def bm25_tokenize(text: str) -> list[str]:
    """Tokenize for BM25: Unicode words, lowercased, accents folded, stopwords dropped.

    Folding makes "requête", "requete" and "REQUÊTE" the same term, and ``\\w+``
    keeps accented letters inside words instead of splitting on them. If a text
    is nothing but stopwords, they are kept so it still has something to match.
    """
    tokens = _unicode_tokens(text)
    content = [t for t in tokens if t not in BM25_STOPWORDS]
    return content or tokens


def normalize_query(text: str, max_chars: int = MAX_QUERY_CHARS) -> tuple[str, bool]:
    """NFKC-normalise, drop invisible characters, collapse whitespace, strip, cap.

    Returns:
        The normalised text and whether it had to be truncated.
    """
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _INVISIBLE.sub("", normalized)
    normalized = _WHITESPACE.sub(" ", normalized).strip()
    if len(normalized) <= max_chars:
        return normalized, False
    cut = normalized[:max_chars]
    # Prefer not to end mid-word when there is a nearby space to cut at.
    space = cut.rfind(" ")
    if space >= max_chars * 0.8:
        cut = cut[:space]
    return cut.rstrip(), True


@dataclass(frozen=True)
class PreparedQuery:
    """A query ready for retrieval.

    Attributes:
        original: The text as the caller gave it; this is what generation sees.
        text: The normalised text dense retrieval embeds.
        tokens: BM25 terms for sparse retrieval.
        truncated: Whether ``text`` was cut to :data:`MAX_QUERY_CHARS`.
    """

    original: str
    text: str
    tokens: tuple[str, ...]
    truncated: bool = False


def preprocess_query(query: str, max_chars: int = MAX_QUERY_CHARS) -> PreparedQuery:
    """Normalise ``query`` for retrieval and tokenize it for BM25."""
    text, truncated = normalize_query(query, max_chars)
    return PreparedQuery(
        original=query,
        text=text,
        tokens=tuple(bm25_tokenize(text)) if text else (),
        truncated=truncated,
    )
