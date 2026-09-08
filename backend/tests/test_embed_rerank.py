from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.rag.embed import (
    embed_query,
    embed_query_async,
    embed_texts,
    embed_texts_async,
    embedding_dim,
)
from app.rag.rerank import _rerank_with_bm25, _rerank_with_cross_encoder, rerank
from app.schemas import Chunk


class TestEmbedTexts:
    def test_embed_texts_success(self):
        """Test successful embedding of multiple texts."""
        mock_model = MagicMock()
        texts = ["Hello world", "This is a test", "Another text"]
        embeddings = np.random.randn(3, 384)
        mock_model.encode = MagicMock(return_value=embeddings)

        with patch("app.rag.embed._model", return_value=mock_model):
            result = embed_texts(texts)

            assert len(result) == 3
            assert all(len(emb) == 384 for emb in result)
            mock_model.encode.assert_called_once_with(texts, normalize_embeddings=True)

    def test_embed_texts_single_text(self):
        """Test embedding a single text."""
        mock_model = MagicMock()
        texts = ["Single text"]
        embeddings = np.random.randn(1, 384)
        mock_model.encode = MagicMock(return_value=embeddings)

        with patch("app.rag.embed._model", return_value=mock_model):
            result = embed_texts(texts)

            assert len(result) == 1
            assert len(result[0]) == 384

    def test_embed_texts_empty_list(self):
        """Test embedding empty list returns empty list."""
        result = embed_texts([])

        assert result == []

    def test_embed_texts_non_zero_vectors(self):
        """Test that embeddings are non-zero vectors."""
        mock_model = MagicMock()
        texts = ["Test text"]
        embeddings = np.random.randn(1, 384)
        mock_model.encode = MagicMock(return_value=embeddings)

        with patch("app.rag.embed._model", return_value=mock_model):
            result = embed_texts(texts)

            assert len(result) > 0
            for emb in result:
                assert any(x != 0 for x in emb)  # At least one non-zero value

    def test_embed_texts_normalized(self):
        """Test that embeddings are normalized."""
        mock_model = MagicMock()
        texts = ["Normalized text"]
        # Create normalized embeddings
        embeddings = np.random.randn(1, 384)
        embeddings = embeddings / np.linalg.norm(embeddings)
        mock_model.encode = MagicMock(return_value=embeddings)

        with patch("app.rag.embed._model", return_value=mock_model):
            result = embed_texts(texts)

            assert len(result) == 1
            # Check norm is close to 1
            norm = np.linalg.norm(result[0])
            assert abs(norm - 1.0) < 0.01

    def test_embed_texts_error_handling(self):
        """Test embed_texts error handling."""
        mock_model = MagicMock()
        mock_model.encode = MagicMock(side_effect=RuntimeError("Model error"))

        with patch("app.rag.embed._model", return_value=mock_model):
            with pytest.raises(RuntimeError):
                embed_texts(["text"])


@pytest.mark.asyncio
class TestEmbedTextsAsync:
    async def test_embed_texts_async_success(self):
        """Test async embedding of texts."""
        mock_model = MagicMock()
        texts = ["Hello world", "This is a test"]
        embeddings = np.random.randn(2, 384)
        mock_model.encode = MagicMock(return_value=embeddings)

        with patch("app.rag.embed._model", return_value=mock_model):
            result = await embed_texts_async(texts)

            assert len(result) == 2
            assert all(len(emb) == 384 for emb in result)

    async def test_embed_texts_async_empty_list(self):
        """Test async embedding with empty list."""
        result = await embed_texts_async([])

        assert result == []

    async def test_embed_texts_async_error_handling(self):
        """Test async embedding error handling."""
        mock_model = MagicMock()
        mock_model.encode = MagicMock(side_effect=RuntimeError("Error"))

        with patch("app.rag.embed._model", return_value=mock_model):
            with pytest.raises(RuntimeError):
                await embed_texts_async(["text"])


class TestEmbedQuery:
    def test_embed_query_success(self):
        """Test successful query embedding."""
        mock_model = MagicMock()
        query = "What is the answer?"
        embeddings = np.random.randn(1, 384)
        mock_model.encode = MagicMock(return_value=embeddings)

        with patch("app.rag.embed._model", return_value=mock_model):
            result = embed_query(query)

            assert len(result) == 384
            mock_model.encode.assert_called_once()

    def test_embed_query_empty_string(self):
        """Test embedding empty query."""
        mock_model = MagicMock()
        embeddings = np.random.randn(1, 384)
        mock_model.encode = MagicMock(return_value=embeddings)

        with patch("app.rag.embed._model", return_value=mock_model):
            result = embed_query("")

            assert len(result) == 384

    def test_embed_query_error_handling(self):
        """Test query embedding error handling."""
        mock_model = MagicMock()
        mock_model.encode = MagicMock(side_effect=RuntimeError("Error"))

        with patch("app.rag.embed._model", return_value=mock_model):
            with pytest.raises(RuntimeError):
                embed_query("query")


@pytest.mark.asyncio
class TestEmbedQueryAsync:
    async def test_embed_query_async_success(self):
        """Test async query embedding."""
        mock_model = MagicMock()
        query = "What is X?"
        embeddings = np.random.randn(1, 384)
        mock_model.encode = MagicMock(return_value=embeddings)

        with patch("app.rag.embed._model", return_value=mock_model):
            result = await embed_query_async(query)

            assert len(result) == 384

    async def test_embed_query_async_error_handling(self):
        """Test async query embedding error handling."""
        mock_model = MagicMock()
        mock_model.encode = MagicMock(side_effect=RuntimeError("Error"))

        with patch("app.rag.embed._model", return_value=mock_model):
            with pytest.raises(RuntimeError):
                await embed_query_async("query")


class TestEmbeddingDim:
    def test_embedding_dim_success(self):
        """Test getting embedding dimension."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension = MagicMock(return_value=384)

        with patch("app.rag.embed._model", return_value=mock_model):
            result = embedding_dim()

            assert result == 384
            mock_model.get_sentence_embedding_dimension.assert_called_once()

    def test_embedding_dim_different_size(self):
        """Test embedding dimension for different model."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension = MagicMock(return_value=768)

        with patch("app.rag.embed._model", return_value=mock_model):
            result = embedding_dim()

            assert result == 768

    def test_embedding_dim_error_handling(self):
        """Test embedding dim error handling."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension = MagicMock(side_effect=RuntimeError("Error"))

        with patch("app.rag.embed._model", return_value=mock_model):
            with pytest.raises(RuntimeError):
                embedding_dim()


class TestRerank:
    def test_rerank_with_cross_encoder_success(self, fake_chunks):
        """Test reranking with cross-encoder."""
        chunks = fake_chunks(5)
        query = "What is the topic?"
        mock_encoder = MagicMock()
        scores = np.array([0.9, 0.7, 0.8, 0.6, 0.85])
        mock_encoder.predict = MagicMock(return_value=scores)

        with patch("app.rag.rerank._load_cross_encoder", return_value=mock_encoder):
            result = _rerank_with_cross_encoder(query, chunks, top_k=3)

            assert len(result) == 3
            # Verify they're sorted by score
            mock_encoder.predict.assert_called_once()

    def test_rerank_exact_top_k(self, fake_chunks):
        """Test rerank returns exactly top_k results."""
        chunks = fake_chunks(10)
        query = "Query"
        mock_encoder = MagicMock()
        scores = np.random.randn(10)
        mock_encoder.predict = MagicMock(return_value=scores)

        with patch("app.rag.rerank._load_cross_encoder", return_value=mock_encoder):
            for top_k in [1, 3, 5, 8]:
                result = _rerank_with_cross_encoder(query, chunks, top_k=top_k)
                assert len(result) == top_k

    def test_rerank_fewer_chunks_than_top_k(self, fake_chunks):
        """Test rerank when chunks < top_k."""
        chunks = fake_chunks(3)
        query = "Query"
        mock_encoder = MagicMock()
        scores = np.array([0.8, 0.9, 0.7])
        mock_encoder.predict = MagicMock(return_value=scores)

        with patch("app.rag.rerank._load_cross_encoder", return_value=mock_encoder):
            result = _rerank_with_cross_encoder(query, chunks, top_k=5)

            # Should return all 3 chunks
            assert len(result) == 3

    def test_rerank_cross_encoder_fallback_to_bm25(self, fake_chunks):
        """Test rerank falls back to BM25 if cross-encoder fails."""
        chunks = fake_chunks(3)
        query = "Query text"

        with patch("app.rag.rerank._load_cross_encoder", return_value=None):
            result = _rerank_with_cross_encoder(query, chunks, top_k=2)

            assert len(result) == 2

    def test_rerank_empty_chunks(self):
        """Test rerank with empty chunks list."""
        chunks = []
        query = "Query"

        result = rerank(query, chunks, top_k=5)

        assert result == []

    def test_rerank_bm25_sorting(self, fake_chunks):
        """Test that BM25 rerank sorts by relevance."""
        chunks = fake_chunks(4)
        # Override text for specific relevance
        chunks[0].text = "Query query query"
        chunks[1].text = "Something else"
        chunks[2].text = "Query related"
        chunks[3].text = "Unrelated content"

        query = "Query"
        result = _rerank_with_bm25(query, chunks, top_k=3)

        assert len(result) == 3
        # First chunk should rank highest (has most occurrences of "query")
        assert result[0].text.count("query") >= result[1].text.count("query")

    def test_rerank_bm25_top_k_limit(self, fake_chunks):
        """Test BM25 rerank respects top_k limit."""
        chunks = fake_chunks(10)
        query = "Text"

        result = _rerank_with_bm25(query, chunks, top_k=5)

        assert len(result) == 5

    def test_rerank_fallback_when_cross_encoder_unavailable(self, fake_chunks):
        """Test main rerank function with unavailable cross-encoder."""
        chunks = fake_chunks(5)
        query = "Query"

        with patch("app.rag.rerank._load_cross_encoder", return_value=None):
            result = rerank(query, chunks, top_k=3)

            assert len(result) == 3

    def test_rerank_integration_with_real_bm25(self, fake_chunks):
        """Test rerank integration with real BM25 scoring."""
        chunks = fake_chunks(3)
        query = "test chunk"

        # No mocking - use real BM25
        result = _rerank_with_bm25(query, chunks, top_k=2)

        assert len(result) == 2
        assert all(isinstance(c, Chunk) for c in result)

    def test_rerank_handles_exception_from_cross_encoder(self, fake_chunks):
        """Test rerank handles cross-encoder exceptions."""
        chunks = fake_chunks(3)
        query = "Query"
        mock_encoder = MagicMock()
        mock_encoder.predict = MagicMock(side_effect=RuntimeError("Encoder error"))

        with patch("app.rag.rerank._load_cross_encoder", return_value=mock_encoder):
            # Should fall back to BM25
            result = _rerank_with_cross_encoder(query, chunks, top_k=2)

            assert len(result) == 2

    def test_rerank_preserves_chunk_metadata(self, fake_chunks):
        """Test that rerank preserves chunk metadata."""
        chunks = fake_chunks(2)
        query = "Test"
        mock_encoder = MagicMock()
        mock_encoder.predict = MagicMock(return_value=np.array([0.8, 0.9]))

        with patch("app.rag.rerank._load_cross_encoder", return_value=mock_encoder):
            result = _rerank_with_cross_encoder(query, chunks, top_k=2)

            for chunk in result:
                assert chunk.metadata is not None
                assert chunk.doc_id is not None
