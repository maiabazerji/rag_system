from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import settings


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return _model().encode(texts, normalize_embeddings=True).tolist()


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]


def embedding_dim() -> int:
    return int(_model().get_sentence_embedding_dimension())
