"""Shared test fixtures.

Settings are overridden by mutating the shared `settings` singleton rather than
by patching `app.config.settings`. Every module does `from app.config import
settings` at import time and holds its own reference, so rebinding the module
attribute would reach almost nothing.
"""
import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client.http import models as qm

from app.config import settings as app_settings
from app.schemas import Chunk, Source

TEST_SETTINGS = {
    "anthropic_api_key": "sk-ant-test-key-not-real",
    "qdrant_url": "http://localhost:6333",
    "qdrant_collection": "test-evalrag",
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "generator_provider": "anthropic",
    "generator_model": "claude-sonnet-5",
    "judge_model": "claude-opus-5",
    "agentic_max_iters": 3,
    "require_api_key": False,
    "admin_key": "",
    "chunk_size_tokens": 100,
    "chunk_overlap_tokens": 20,
    "rerank_top_k": 8,
    "eval_concurrency": 2,
}


@pytest.fixture(autouse=True, scope="session")
def _no_telemetry_uploads():
    """Keep the eval tests off the network.

    Settings come from the real `.env`, so a developer with `WANDB_MODE=online`
    had every eval test open a live W&B run and upload to it: the suite went
    from ~20s to minutes and would sometimes hang outright.
    """
    os.environ["WANDB_MODE"] = "disabled"


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    """Point the shared settings object at test values for the duration of a test.

    `monkeypatch.setattr` restores the original values automatically, so tests
    that mutate settings cannot leak into their neighbours.
    """
    for key, value in TEST_SETTINGS.items():
        monkeypatch.setattr(app_settings, key, value, raising=True)
    return app_settings


@pytest.fixture(autouse=True)
def isolated_data_dirs(tmp_path, monkeypatch):
    """Redirect every on-disk artifact into tmp_path.

    Without this, a test that reaches a route writing eval runs or human
    ratings quietly deposits files in the repository's data/ directory.
    """
    import app.eval.human_ratings as human_ratings
    import app.eval.metrics as metrics
    import app.eval.regression as regression

    runs = tmp_path / "eval_runs"
    ratings = tmp_path / "human_ratings"
    monkeypatch.setattr(metrics, "RUNS_DIR", runs)
    monkeypatch.setattr(regression, "RUNS_DIR", runs)
    monkeypatch.setattr(human_ratings, "RATINGS_DIR", ratings)
    monkeypatch.setattr(human_ratings, "RATINGS_FILE", ratings / "ratings.jsonl")


@pytest.fixture
def client():
    """A TestClient for the app, with the lifespan skipped.

    The lifespan validates configuration and touches Postgres; tests exercise
    routes, not startup.
    """
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


@pytest.fixture
def fake_chunks():
    """Factory producing deterministic Chunk objects."""

    def _create_chunks(count=3, text_prefix=""):
        return [
            Chunk(
                id=f"chunk_{i}",
                doc_id=f"doc_{i // 2}",
                text=(
                    f"{text_prefix}This is test chunk {i}. "
                    "It contains relevant information about topics."
                ),
                tokens=20,
                section=f"section_{i}",
                metadata={"source": f"doc_{i // 2}.pdf", "page": i},
            )
            for i in range(count)
        ]

    return _create_chunks


@pytest.fixture
def fake_embeddings():
    """Factory producing random embedding vectors."""

    def _create_embeddings(count=3, dim=384):
        import numpy as np

        return [np.random.randn(dim).tolist() for _ in range(count)]

    return _create_embeddings


@pytest.fixture
def fake_sources():
    """Factory producing Source objects with descending scores."""

    def _create_sources(count=3):
        return [
            Source(
                chunk_id=f"chunk_{i}",
                quote=f"Quote {i}: This is a sample quote from chunk {i}.",
                score=0.9 - (i * 0.05),
            )
            for i in range(count)
        ]

    return _create_sources


@pytest.fixture
def mock_qdrant_client():
    """Mock AsyncQdrantClient with the calls store.py makes."""
    mock = AsyncMock()
    mock.get_collections = AsyncMock(
        return_value=MagicMock(collections=[MagicMock(name="test-evalrag")])
    )
    mock.get_collection = AsyncMock(
        return_value=MagicMock(
            config=MagicMock(params=MagicMock(vectors=MagicMock(size=384)))
        )
    )
    mock.create_collection = AsyncMock()
    mock.upsert = AsyncMock()
    mock.query_points = AsyncMock()
    mock.count = AsyncMock(return_value=MagicMock(count=10))
    return mock


@pytest.fixture
def mock_qdrant_search_results():
    """Factory turning Chunks into a Qdrant QueryResponse."""

    def _create_results(chunks, scores=None):
        points = [
            qm.ScoredPoint(
                id=abs(hash(chunk.id)) % (2**31),
                version=1,
                score=scores[i] if scores else 0.95 - (i * 0.1),
                payload={
                    "chunk_id": chunk.id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    **chunk.metadata,
                },
            )
            for i, chunk in enumerate(chunks)
        ]
        return qm.QueryResponse(points=points)

    return _create_results


@pytest.fixture
def mock_anthropic_response():
    """A Messages API response shaped like the SDK's, with usage."""

    def _make(text="Test generated response.", input_tokens=100, output_tokens=50):
        response = MagicMock()
        response.content = [MagicMock(type="text", text=text)]
        response.stop_reason = "end_turn"
        response.usage = MagicMock(
            input_tokens=input_tokens, output_tokens=output_tokens
        )
        return response

    return _make


@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer returning deterministic vectors."""
    mock = MagicMock()

    def encode_side_effect(texts, normalize_embeddings=False):
        import numpy as np

        items = texts if isinstance(texts, list) else [texts]
        return np.array(
            [np.random.RandomState(abs(hash(t)) % (2**31)).randn(384) for t in items]
        )

    mock.encode = encode_side_effect
    mock.get_sentence_embedding_dimension = MagicMock(return_value=384)
    return mock


@pytest.fixture
def mock_cross_encoder():
    """Mock CrossEncoder scoring longer text higher."""
    mock = MagicMock()

    def predict_side_effect(pairs):
        import numpy as np

        return np.array([len(pair[1]) / 1000.0 for pair in pairs])

    mock.predict = predict_side_effect
    return mock
