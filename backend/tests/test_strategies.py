import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.rag.strategies.agentic import AgenticRAG
from app.rag.strategies.classic import ClassicRAG
from app.rag.strategies.graph import GraphRAG
from app.schemas import Chunk, Source


@pytest.mark.asyncio
class TestClassicRAG:
    async def test_classic_rag_success(self, fake_chunks, mock_qdrant_search_results):
        """Test classic RAG retrieval and generation."""
        chunks = fake_chunks(3)
        mock_results = mock_qdrant_search_results(chunks)

        strategy = ClassicRAG()

        with patch("app.rag.strategies.classic.hybrid_search", new_callable=AsyncMock) as mock_search:
            with patch("app.rag.strategies.classic.rerank") as mock_rerank:
                with patch("app.rag.strategies.classic.generate_with_usage", new_callable=AsyncMock) as mock_gen:
                    mock_search.return_value = chunks
                    mock_rerank.return_value = chunks[:2]
                    mock_gen.return_value = {
                        "text": "Test answer based on the context.",
                        "input_tokens": 150,
                        "output_tokens": 50,
                    }

                    result = await strategy.run(
                        "What is the test topic?",
                        top_k=8,
                        model="claude-sonnet-4-6",
                        prompt_version="v1",
                    )

                    assert result.answer == "Test answer based on the context."
                    assert result.refusal is False
                    assert len(result.sources) == 2
                    assert result.input_tokens == 150
                    assert result.output_tokens == 50
                    assert result.iterations == 1
                    mock_search.assert_called_once()
                    mock_rerank.assert_called_once()
                    mock_gen.assert_called_once()

    async def test_classic_rag_empty_context(self):
        """Test classic RAG with empty context."""
        strategy = ClassicRAG()

        with patch("app.rag.strategies.classic.hybrid_search", new_callable=AsyncMock) as mock_search:
            with patch("app.rag.strategies.classic.rerank") as mock_rerank:
                mock_search.return_value = []
                mock_rerank.return_value = []

                result = await strategy.run(
                    "What is the test topic?",
                    top_k=8,
                    model="claude-sonnet-4-6",
                    prompt_version="v1",
                )

                assert result.refusal is True
                assert result.confidence == 0.0
                assert "No relevant context" in result.answer
                assert result.sources[0].chunk_id == "none"

    async def test_classic_rag_generation_failure(self, fake_chunks):
        """Test classic RAG when generation fails."""
        chunks = fake_chunks(2)
        strategy = ClassicRAG()

        with patch("app.rag.strategies.classic.hybrid_search", new_callable=AsyncMock) as mock_search:
            with patch("app.rag.strategies.classic.rerank") as mock_rerank:
                with patch("app.rag.strategies.classic.generate_with_usage", new_callable=AsyncMock) as mock_gen:
                    mock_search.return_value = chunks
                    mock_rerank.return_value = chunks
                    mock_gen.side_effect = RuntimeError("API timeout")

                    with pytest.raises(RuntimeError):
                        await strategy.run(
                            "What is the test topic?",
                            top_k=8,
                            model="claude-sonnet-4-6",
                            prompt_version="v1",
                        )

    async def test_classic_rag_empty_generation_response(self, fake_chunks):
        """Test classic RAG with empty generation response."""
        chunks = fake_chunks(2)
        strategy = ClassicRAG()

        with patch("app.rag.strategies.classic.hybrid_search", new_callable=AsyncMock) as mock_search:
            with patch("app.rag.strategies.classic.rerank") as mock_rerank:
                with patch("app.rag.strategies.classic.generate_with_usage", new_callable=AsyncMock) as mock_gen:
                    mock_search.return_value = chunks
                    mock_rerank.return_value = chunks
                    mock_gen.return_value = {
                        "text": "",
                        "input_tokens": 100,
                        "output_tokens": 0,
                    }

                    result = await strategy.run(
                        "What is the test topic?",
                        top_k=8,
                        model="claude-sonnet-4-6",
                        prompt_version="v1",
                    )

                    assert result.answer == "(empty response)"


@pytest.mark.asyncio
class TestGraphRAG:
    async def test_graph_rag_no_graph_setup(self):
        """Test graph RAG when graph is not set up."""
        strategy = GraphRAG()

        with patch("app.rag.strategies.graph.load_graph") as mock_load:
            mock_graph = MagicMock()
            mock_graph.triples = []
            mock_load.return_value = mock_graph

            result = await strategy.run(
                "What is connected to X?",
                top_k=8,
                model="claude-sonnet-4-6",
                prompt_version="v1",
            )

            assert result.refusal is True
            assert "Graph RAG needs setup" in result.answer
            assert result.confidence == 0.0

    async def test_graph_rag_success(self, fake_chunks):
        """Test graph RAG successful retrieval."""
        chunks = fake_chunks(4)
        strategy = GraphRAG()

        with patch("app.rag.strategies.graph.load_graph") as mock_load:
            with patch("app.rag.strategies.graph.extract_question_entities", new_callable=AsyncMock) as mock_extract:
                with patch("app.rag.strategies.graph.neighbors") as mock_neighbors:
                    with patch("app.rag.strategies.graph.hybrid_search", new_callable=AsyncMock) as mock_search:
                        with patch("app.rag.strategies.graph._fetch_chunks_by_id", new_callable=AsyncMock) as mock_fetch:
                            with patch("app.rag.strategies.graph.rerank") as mock_rerank:
                                with patch("app.rag.strategies.graph.describe_subgraph") as mock_describe:
                                    with patch("app.rag.strategies.graph.generate_with_usage", new_callable=AsyncMock) as mock_gen:
                                        mock_graph = MagicMock()
                                        mock_graph.triples = [("entity1", "relation", "entity2")]
                                        mock_load.return_value = mock_graph

                                        mock_extract.return_value = ["entity1", "entity2"]
                                        mock_neighbors.return_value = ({"chunk_0", "chunk_1"}, {"entity3"})
                                        mock_search.return_value = chunks[:2]
                                        mock_fetch.return_value = chunks[2:]
                                        mock_rerank.return_value = chunks[:3]
                                        mock_describe.return_value = "Entity1 -> relation -> Entity2"
                                        mock_gen.return_value = {
                                            "text": "Graph-based answer.",
                                            "input_tokens": 200,
                                            "output_tokens": 75,
                                        }

                                        result = await strategy.run(
                                            "How is entity1 connected to entity2?",
                                            top_k=8,
                                            model="claude-sonnet-4-6",
                                            prompt_version="v1",
                                        )

                                        assert result.answer == "Graph-based answer."
                                        assert result.refusal is False
                                        assert "entity1" in result.extra
                                        assert "subgraph" in result.extra
                                        assert result.input_tokens == 200

    async def test_graph_rag_empty_after_rerank(self, fake_chunks):
        """Test graph RAG with empty results after reranking."""
        strategy = GraphRAG()

        with patch("app.rag.strategies.graph.load_graph") as mock_load:
            with patch("app.rag.strategies.graph.extract_question_entities", new_callable=AsyncMock) as mock_extract:
                with patch("app.rag.strategies.graph.neighbors") as mock_neighbors:
                    with patch("app.rag.strategies.graph.hybrid_search", new_callable=AsyncMock) as mock_search:
                        with patch("app.rag.strategies.graph._fetch_chunks_by_id", new_callable=AsyncMock) as mock_fetch:
                            with patch("app.rag.strategies.graph.rerank") as mock_rerank:
                                mock_graph = MagicMock()
                                mock_graph.triples = [("e1", "rel", "e2")]
                                mock_load.return_value = mock_graph

                                mock_extract.return_value = ["entity1"]
                                mock_neighbors.return_value = (set(), set())
                                mock_search.return_value = []
                                mock_fetch.return_value = []
                                mock_rerank.return_value = []

                                result = await strategy.run(
                                    "How is entity1 connected?",
                                    top_k=8,
                                    model="claude-sonnet-4-6",
                                    prompt_version="v1",
                                )

                                assert result.refusal is True
                                assert "found neither matching entities" in result.answer


@pytest.mark.asyncio
class TestAgenticRAG:
    async def test_agentic_rag_success_with_finish(self, fake_chunks):
        """Test agentic RAG that calls finish tool."""
        chunks = fake_chunks(2)
        strategy = AgenticRAG()

        with patch("app.rag.strategies.agentic.tool_use_loop", new_callable=AsyncMock) as mock_loop:
            mock_loop.return_value = {
                "text": "Final answer.",
                "input_tokens": 300,
                "output_tokens": 100,
                "iterations": 2,
                "trace": [
                    {"step": 0, "tool_calls": [{"name": "search"}]},
                    {"step": 1, "tool_calls": [{"name": "finish"}]},
                ],
            }

            result = await strategy.run(
                "Search for information?",
                top_k=8,
                model="claude-sonnet-4-6",
                prompt_version="v1",
            )

            assert result.answer == "Final answer."
            assert result.iterations == 2
            assert result.input_tokens == 300
            assert mock_loop.called

    async def test_agentic_rag_no_finish_call(self):
        """Test agentic RAG when agent doesn't call finish."""
        strategy = AgenticRAG()

        with patch("app.rag.strategies.agentic.tool_use_loop", new_callable=AsyncMock) as mock_loop:
            mock_loop.return_value = {
                "text": "Long response without finishing",
                "input_tokens": 200,
                "output_tokens": 80,
                "iterations": 3,
                "trace": [{"step": 0, "tool_calls": [{"name": "search"}]}],
            }

            result = await strategy.run(
                "Search for information?",
                top_k=8,
                model="claude-sonnet-4-6",
                prompt_version="v1",
            )

            assert "Long response without finishing" in result.answer
            assert result.iterations == 3

    async def test_agentic_rag_tool_handlers(self):
        """Test that agentic RAG registers tool handlers."""
        strategy = AgenticRAG()

        with patch("app.rag.strategies.agentic.tool_use_loop", new_callable=AsyncMock) as mock_loop:
            mock_loop.return_value = {
                "text": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "iterations": 0,
                "trace": [],
            }

            with patch("app.rag.strategies.agentic.embed_query"):
                with patch("app.rag.strategies.agentic.vector_search", new_callable=AsyncMock):
                    await strategy.run(
                        "What?",
                        top_k=8,
                        model="claude-sonnet-4-6",
                        prompt_version="v1",
                    )

                    # Verify tool_use_loop was called with tool handlers
                    call_kwargs = mock_loop.call_args[1]
                    assert "tool_handlers" in call_kwargs
                    handlers = call_kwargs["tool_handlers"]
                    assert "search" in handlers
                    assert "fetch_chunk" in handlers
                    assert "finish" in handlers

    async def test_agentic_rag_empty_response(self):
        """Test agentic RAG with empty response."""
        strategy = AgenticRAG()

        with patch("app.rag.strategies.agentic.tool_use_loop", new_callable=AsyncMock) as mock_loop:
            mock_loop.return_value = {
                "text": "",
                "input_tokens": 100,
                "output_tokens": 0,
                "iterations": 1,
                "trace": [],
            }

            result = await strategy.run(
                "What?",
                top_k=8,
                model="claude-sonnet-4-6",
                prompt_version="v1",
            )

            assert result.refusal is True
            assert "(agent stopped without finishing)" in result.answer
