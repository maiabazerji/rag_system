"""Embedding functions for converting text to vector representations.

This module handles text-to-embedding conversion using SentenceTransformers models.
It provides both sync and async APIs, with thread-pooling to avoid blocking on CPU-bound
embedding computation in async contexts.

The embedding model is configured via settings (by default the multilingual
"intfloat/multilingual-e5-small"). All embeddings are normalized to unit length
(L2 norm = 1.0) for cosine similarity metrics.

Some model families are trained with a marker on each input and retrieve
noticeably worse without it: E5 expects "query: " on questions and "passage: "
on documents. :func:`embedding_profile` maps a model id to its markers, and
every embedding call says which side it embeds (``kind="query"`` or
``kind="passage"``) so the right one is applied.

Key concepts:
    - Embeddings are fixed-dimension vectors (typically 384-768 dimensions)
    - Normalized embeddings enable fast similarity search in vector databases
    - Thread pool avoids blocking async event loop during encoding
"""
from __future__ import annotations

import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

from app.config import settings

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4)

EmbedKind = Literal["query", "passage"]


@dataclass(frozen=True)
class EmbeddingProfile:
    """How a model family expects its inputs to be marked."""

    family: str
    query_prefix: str = ""
    passage_prefix: str = ""
    multilingual: bool = False


_E5_INSTRUCT_QUERY = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
    "Query: "
)

# Known families, most specific first. Matched against the model id's last
# path segment, so a local directory named after the model is recognised too.
_PROFILES: tuple[tuple[re.Pattern[str], EmbeddingProfile], ...] = (
    (
        re.compile(r"e5-.*instruct"),
        EmbeddingProfile("e5-instruct", query_prefix=_E5_INSTRUCT_QUERY, multilingual=True),
    ),
    (
        re.compile(r"multilingual-e5-"),
        EmbeddingProfile("e5", "query: ", "passage: ", multilingual=True),
    ),
    (re.compile(r"(^|-)e5-"), EmbeddingProfile("e5", "query: ", "passage: ")),
    # bge-m3 needs no instruction on either side.
    (re.compile(r"^bge-m3"), EmbeddingProfile("bge-m3", multilingual=True)),
    # BGE English v1.5 accepts an optional query instruction; this project has
    # always embedded without one, and keeps doing so for comparable numbers.
    (re.compile(r"^bge-.*-en"), EmbeddingProfile("bge-en")),
    (re.compile(r"multilingual|labse"), EmbeddingProfile("multilingual", multilingual=True)),
)


def embedding_profile(model: str | None = None) -> EmbeddingProfile:
    """Return the input-marking profile for an embedding model.

    Args:
        model: Model id or local path. Defaults to ``settings.embedding_model``.

    Returns:
        The matching profile, with ``EMBEDDING_QUERY_PREFIX`` /
        ``EMBEDDING_PASSAGE_PREFIX`` applied on top when set. Unknown models get
        no prefixes.
    """
    name = (model or settings.embedding_model).rstrip("/").rsplit("/", 1)[-1].lower()
    profile = next(
        (p for pattern, p in _PROFILES if pattern.search(name)),
        EmbeddingProfile("generic"),
    )
    query = settings.embedding_query_prefix
    passage = settings.embedding_passage_prefix
    if query is None and passage is None:
        return profile
    return EmbeddingProfile(
        family=profile.family,
        query_prefix=profile.query_prefix if query is None else query,
        passage_prefix=profile.passage_prefix if passage is None else passage,
        multilingual=profile.multilingual,
    )


def is_multilingual_model(model: str | None = None) -> bool:
    """Whether the embedding model is known to map many languages into one space."""
    return embedding_profile(model).multilingual


def prepare_texts(texts: list[str], kind: EmbedKind) -> list[str]:
    """Add the model family's query or passage marker to each text."""
    profile = embedding_profile()
    prefix = profile.query_prefix if kind == "query" else profile.passage_prefix
    if not prefix:
        return list(texts)
    return [prefix + t for t in texts]


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    """Load and cache the embedding model.

    Uses LRU cache to load the model once and reuse for all embedding operations.
    Downloads the model on first use if not already cached locally.

    Returns:
        Loaded SentenceTransformer model.

    Raises:
        RuntimeError: If model download/initialization fails (e.g., missing API key,
            internet connectivity, invalid model name).
    """
    try:
        # Imported lazily: sentence-transformers pulls in torch, and importing
        # this module should not cost that unless an embedding is actually needed.
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(settings.embedding_model)
    except Exception as e:
        logger.error(f"Failed to load embedding model '{settings.embedding_model}': {e}")
        raise RuntimeError(
            f"Failed to initialize embedding model '{settings.embedding_model}'. "
            f"Ensure the model name is correct and you have internet access to download it."
        ) from e


def embed_texts(texts: list[str], *, kind: EmbedKind = "passage") -> list[list[float]]:
    """Embed a list of texts synchronously.

    Converts each text to a normalized embedding vector. Uses cached model
    to avoid repeated initialization.

    Args:
        texts: List of text strings to embed. Longer texts may be truncated by
            the model (typically to 512 tokens).
        kind: ``"passage"`` for documents being indexed, ``"query"`` for
            questions. Selects the model family's input marker.

    Returns:
        List of embedding vectors (one per input text), each of shape [embedding_dim].
        Vectors are normalized (L2 norm = 1.0) for cosine similarity.
        Empty list if input is empty.

    Raises:
        RuntimeError: If embedding fails (e.g., model not loaded, OOM).

    Example:
        >>> embeddings = embed_texts(["Hello world", "Goodbye"])
        >>> print(len(embeddings))  # 2
        >>> print(len(embeddings[0]))  # 384 (model dimension)
    """
    if not texts:
        return []

    try:
        model = _model()
        embeddings = model.encode(prepare_texts(texts, kind), normalize_embeddings=True)
        return embeddings.tolist()
    except Exception as e:
        logger.error(f"Failed to embed texts: {e}")
        raise RuntimeError(f"Embedding failed: {str(e)}") from e


async def embed_texts_async(
    texts: list[str], *, kind: EmbedKind = "passage"
) -> list[list[float]]:
    """Embed a list of texts asynchronously using thread pool.

    Non-blocking async wrapper around synchronous embedding. Runs the CPU-bound
    encoding operation in a thread pool to avoid blocking the event loop.

    Args:
        texts: List of text strings to embed.
        kind: ``"passage"`` or ``"query"``; see :func:`embed_texts`.

    Returns:
        List of embedding vectors (normalized).
        Empty list if input is empty.

    Raises:
        RuntimeError: If embedding fails.

    Example:
        >>> embeddings = await embed_texts_async(["Hello", "World"])
        >>> print(len(embeddings))  # 2
    """
    if not texts:
        return []

    try:
        prepared = prepare_texts(texts, kind)
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            _executor,
            lambda: _model().encode(prepared, normalize_embeddings=True)
        )
        return embeddings.tolist()
    except Exception as e:
        logger.error(f"Failed to embed texts asynchronously: {e}")
        raise RuntimeError(f"Async embedding failed: {str(e)}") from e


def embed_query(text: str) -> list[float]:
    """Embed a single query text synchronously.

    Convenience function for embedding a single string. Internally calls embed_texts
    with a list of one element.

    Args:
        text: The query string to embed (e.g., a user question).

    Returns:
        Single embedding vector of shape [embedding_dim], normalized.

    Raises:
        RuntimeError: If embedding fails.

    Example:
        >>> query_vec = embed_query("What is machine learning?")
        >>> print(len(query_vec))  # 384 (model dimension)
    """
    try:
        return embed_texts([text], kind="query")[0]
    except Exception as e:
        logger.error(f"Failed to embed query: {e}")
        raise RuntimeError(f"Query embedding failed: {str(e)}") from e


async def embed_query_async(text: str) -> list[float]:
    """Embed a single query text asynchronously.

    Non-blocking async wrapper for single-query embedding.

    Args:
        text: The query string to embed.

    Returns:
        Single embedding vector of shape [embedding_dim], normalized.

    Raises:
        RuntimeError: If embedding fails.

    Example:
        >>> query_vec = await embed_query_async("What is AI?")
    """
    try:
        embeddings = await embed_texts_async([text], kind="query")
        return embeddings[0]
    except Exception as e:
        logger.error(f"Failed to embed query asynchronously: {e}")
        raise RuntimeError(f"Async query embedding failed: {str(e)}") from e


def embedding_dim() -> int:
    """Get the embedding dimension of the configured model.

    Returns the size of the embedding vectors produced by the model.
    Useful for initializing vector stores or verifying model compatibility.

    Returns:
        Dimension of embedding vectors (e.g., 384, 768).

    Raises:
        RuntimeError: If model loading fails.

    Example:
        >>> dim = embedding_dim()
        >>> print(f"Creating index for {dim}-dimensional embeddings")
    """
    try:
        dim = _model().get_sentence_embedding_dimension()
        if dim is None:
            raise RuntimeError(
                f"Model '{settings.embedding_model}' did not report an embedding "
                "dimension. It may not be a sentence-transformers model."
            )
        return int(dim)
    except Exception as e:
        logger.error(f"Failed to get embedding dimension: {e}")
        raise RuntimeError(f"Failed to get embedding dimension: {str(e)}") from e
