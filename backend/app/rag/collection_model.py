"""Record and verify which embedding model built a Qdrant collection.

Vectors from two embedding models live in unrelated spaces even when their
dimensions agree (``bge-small-en-v1.5`` and ``multilingual-e5-small`` are both
384-d), so searching one model's index with the other's query vectors returns
plausible-looking garbage and no error. The collection therefore carries the
model id in its metadata, stamped at creation, and the store refuses to use a
collection built by a different model.

Nothing is ever deleted automatically. The error says how to re-ingest.
"""
from __future__ import annotations

import json
from typing import Any

from app.config import settings

# Collection metadata keys.
EMBEDDING_MODEL_KEY = "embedding_model"
EMBEDDING_DIM_KEY = "embedding_dim"


class EmbeddingModelMismatch(RuntimeError):
    """The collection was built by a different embedding model than the configured one."""


def collection_stamp(dim: int) -> dict[str, Any]:
    """Metadata recording the configured model on a collection."""
    return {EMBEDDING_MODEL_KEY: settings.embedding_model, EMBEDDING_DIM_KEY: dim}


def reingest_hint(stored_model: str | None) -> str:
    """How to get out of a mismatch without losing data by accident."""
    name = settings.qdrant_collection
    hint = (
        "Re-ingest with the new model: either point QDRANT_COLLECTION at a new name "
        f"(e.g. {name}_v2) and run `python scripts/ingest.py`, or delete the old "
        f"collection (`curl -X DELETE $QDRANT_URL/collections/{name}`, or "
        "`docker compose -f infra/docker-compose.yml down -v`) and ingest again."
    )
    if stored_model:
        hint += f" To keep the existing vectors instead, set EMBEDDING_MODEL={stored_model}."
    return hint


def check_collection_model(info: Any, dim: int) -> bool:
    """Verify an existing collection against the configured embedding model.

    Args:
        info: The collection description returned by ``get_collection``.
        dim: The vector size the configured model produces.

    Returns:
        True when the collection records no model yet but is empty, so the
        caller should stamp it with :func:`collection_stamp`. False when it
        already records the configured model.

    Raises:
        EmbeddingModelMismatch: On a dimension or model mismatch, or when a
            non-empty collection predates model tracking and so cannot be
            verified.
    """
    name = settings.qdrant_collection
    model = settings.embedding_model
    metadata = getattr(info.config, "metadata", None)
    stored = metadata.get(EMBEDDING_MODEL_KEY) if isinstance(metadata, dict) else None
    current = info.config.params.vectors.size

    if current != dim:
        raise EmbeddingModelMismatch(
            f"Qdrant collection '{name}' has dim={current} "
            f"(model: {stored or 'not recorded'}) but embedding model '{model}' "
            f"produces dim={dim}. " + reingest_hint(stored)
        )
    if stored is not None:
        if stored != model:
            raise EmbeddingModelMismatch(
                f"Qdrant collection '{name}' was built with embedding model '{stored}' "
                f"but EMBEDDING_MODEL is '{model}'. Searching it with the new model "
                "would silently return wrong results. " + reingest_hint(stored)
            )
        return False

    points = getattr(info, "points_count", None)
    if isinstance(points, int) and points == 0:
        return True
    payload = json.dumps({"metadata": {EMBEDDING_MODEL_KEY: model}})
    raise EmbeddingModelMismatch(
        f"Qdrant collection '{name}' holds vectors but does not record which embedding "
        f"model produced them (it predates model tracking), so it cannot be verified "
        f"against '{model}'. " + reingest_hint(None) + " If you are certain it was "
        f"built with '{model}', record that instead: curl -X PATCH "
        f"$QDRANT_URL/collections/{name} -H 'Content-Type: application/json' "
        f"-d '{payload}'"
    )
