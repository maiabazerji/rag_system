import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client.http import models as qm

from app.config import Settings
from app.schemas import Chunk, Source


@pytest.fixture(scope="session")
def test_settings():
    """Override settings for test environment."""
    settings = Settings(
        anthropic_api_key="test-key-12345",
        qdrant_url="http://localhost:6333",
        qdrant_collection="test-evalrag",
        embedding_model="BAAI/bge-small-en-v1.5",
        generator_provider="anthropic",
        generator_model="claude-sonnet-4-6",
        judge_model="claude-opus-4-7",
        agentic_max_iters=3,
    )
    return settings


@pytest.fixture
def fake_chunks():
    """Factory fixture for creating fake Chunk objects."""
    def _create_chunks(count=3, text_prefix=""):
        chunks = []
        for i in range(count):
            chunks.append(
                Chunk(
                    id=f"chunk_{i}",
                    doc_id=f"doc_{i // 2}",
                    text=f"{text_prefix}This is test chunk {i}. It contains relevant information about topics.",
                    tokens=20,
                    section=f"section_{i}",
                    metadata={"source": f"doc_{i // 2}.pdf", "page": i},
                )
            )
        return chunks
    return _create_chunks


@pytest.fixture
def fake_embeddings():
    """Factory fixture for creating fake embeddings."""
    def _create_embeddings(count=3, dim=384):
        import numpy as np
        return [np.random.randn(dim).tolist() for _ in range(count)]
    return _create_embeddings


@pytest.fixture
def fake_sources():
    """Factory fixture for creating fake Source objects."""
    def _create_sources(count=3):
        sources = []
        for i in range(count):
            sources.append(
                Source(
                    chunk_id=f"chunk_{i}",
                    quote=f"Quote {i}: This is a sample quote from chunk {i}.",
                    score=0.9 - (i * 0.05),
                )
            )
        return sources
    return _create_sources


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant AsyncQdrantClient."""
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
    """Create mock Qdrant search results."""
    def _create_results(chunks, scores=None):
        points = []
        for i, chunk in enumerate(chunks):
            score = scores[i] if scores else 0.95 - (i * 0.1)
            point = qm.ScoredPoint(
                id=hash(chunk.id) % (2**31),
                version=1,
                score=score,
                payload={
                    "chunk_id": chunk.id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    **chunk.metadata,
                },
            )
            points.append(point)
        return qm.QueryResponse(points=points)
    return _create_results


@pytest.fixture
def mock_anthropic_client():
    """Mock AsyncAnthropic client."""
    mock = AsyncMock()

    def create_message_mock(**kwargs):
        # Create mock response based on input
        response = MagicMock()
        response.content = [MagicMock(type="text", text="Test generated response.")]
        response.stop_reason = "end_turn"
        response.usage = MagicMock(input_tokens=100, output_tokens=50)
        return response

    mock.messages.create = AsyncMock(side_effect=create_message_mock)
    return mock


@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer model."""
    mock = MagicMock()

    def encode_side_effect(texts, normalize_embeddings=False):
        import numpy as np
        # Return deterministic embeddings based on text length
        return np.array([
            np.random.RandomState(hash(t) % (2**31)).randn(384)
            for t in (texts if isinstance(texts, list) else [texts])
        ])

    mock.encode = encode_side_effect
    mock.get_sentence_embedding_dimension = MagicMock(return_value=384)
    return mock


@pytest.fixture
def mock_cross_encoder():
    """Mock CrossEncoder model for reranking."""
    mock = MagicMock()

    def predict_side_effect(pairs):
        import numpy as np
        # Simple scoring: longer text gets higher scores
        return np.array([len(pair[1]) / 1000.0 for pair in pairs])

    mock.predict = predict_side_effect
    return mock


@pytest.fixture(autouse=True)
def override_settings(test_settings):
    """Override app settings for all tests."""
    with patch("app.config.settings", test_settings):
        yield test_settings


@pytest.fixture
def async_client_mock():
    """Create a mock async client for testing."""
    return AsyncMock()


@pytest.fixture
async def mock_embedding_service(mock_sentence_transformer):
    """Mock the embedding service."""
    with patch("app.rag.embed._model", return_value=mock_sentence_transformer):
        yield mock_sentence_transformer


@pytest.fixture
async def mock_qdrant_service(mock_qdrant_client):
    """Mock the Qdrant service."""
    with patch("app.rag.store.client", return_value=mock_qdrant_client):
        yield mock_qdrant_client


@pytest.fixture
async def mock_anthropic_service(mock_anthropic_client):
    """Mock the Anthropic service."""
    with patch("app.rag.providers.anthropic_provider.AsyncAnthropic", return_value=mock_anthropic_client):
        yield mock_anthropic_client
