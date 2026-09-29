from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.rag.rerank import RerankOutcome
from app.rag.retrieve import RetrievalResult
from app.rag.strategies.agentic import AgenticRAG
from app.rag.strategies.classic import ClassicRAG
from app.rag.strategies.graph import GraphRAG
from app.schemas import Chunk


def _hybrid(hits) -> RetrievalResult:
    """A hybrid_search result holding the chunks described by store-hit payloads."""
    chunks = [
        Chunk(
            id=h.payload["chunk_id"],
            doc_id=h.payload["doc_id"],
            text=h.payload["text"],
            tokens=len(h.payload["text"].split()),
            metadata={"filename": h.payload.get("filename")},
        )
        for h in hits
    ]
    return RetrievalResult(chunks=chunks)


@pytest.mark.asyncio
class TestClassicRAG:
    async def test_classic_rag_success(self, fake_chunks, mock_qdrant_search_results):
        """Test classic RAG retrieval and generation."""
        chunks = fake_chunks(3)

        strategy = ClassicRAG()

        with patch("app.rag.retrieve.dense_search", new_callable=AsyncMock) as mock_search:
            with patch("app.rag.retrieve.rerank_scored_async") as mock_rerank:
                with patch("app.rag.strategies.classic.generate_structured", new_callable=AsyncMock) as mock_gen:
                    mock_search.return_value = chunks
                    mock_rerank.return_value = RerankOutcome.unscored(chunks[:2])
                    mock_gen.return_value = {
                        "structured": {
                            "answer": "Test answer based on the context [S2].",
                            "claims": [
                                {
                                    "text": "Test answer based on the context.",
                                    "citations": ["S2"],
                                    "supported": True,
                                }
                            ],
                            "status": "answered",
                            "unsupported_notes": "",
                        },
                        "text": "",
                        "input_tokens": 150,
                        "output_tokens": 50,
                    }

                    result = await strategy.run(
                        "What is the test topic?",
                        top_k=8,
                        model="claude-sonnet-5",
                        prompt_version="default",
                    )

                    assert result.answer == "Test answer based on the context [S2]."
                    assert result.refusal is False
                    assert result.grounded is True
                    assert result.status == "answered"
                    # Only the cited chunk is a source; the full context stays in extra.
                    assert [src.chunk_id for src in result.sources] == ["chunk_1"]
                    assert result.sources[0].handle == "S2"
                    assert result.sources[0].page == 1
                    assert result.sources[0].section == "section_1"
                    assert [c["handle"] for c in result.extra["context_sources"]] == ["S1", "S2"]
                    assert result.claims[0].chunk_ids == ["chunk_1"]
                    assert 0 < result.confidence <= 1
                    prompt = mock_gen.await_args.kwargs["prompt"]
                    assert "[S1]" in prompt and "chunk_0" not in prompt
                    assert result.input_tokens == 150
                    assert result.output_tokens == 50
                    assert result.iterations == 1
                    mock_search.assert_called_once()
                    mock_rerank.assert_called_once()
                    mock_gen.assert_called_once()

    async def test_classic_rag_empty_context(self):
        """Test classic RAG with empty context."""
        strategy = ClassicRAG()

        with patch("app.rag.retrieve.dense_search", new_callable=AsyncMock) as mock_search:
            with patch("app.rag.retrieve.rerank_scored_async") as mock_rerank:
                mock_search.return_value = []
                mock_rerank.return_value = RerankOutcome.unscored([])

                result = await strategy.run(
                    "What is the test topic?",
                    top_k=8,
                    model="claude-sonnet-5",
                    prompt_version="default",
                )

                assert result.refusal is True
                assert result.confidence == 0.0
                assert "No relevant context" in result.answer
                assert result.sources[0].chunk_id == "none"

    async def test_classic_rag_generation_failure(self, fake_chunks):
        """Test classic RAG when generation fails."""
        chunks = fake_chunks(2)
        strategy = ClassicRAG()

        with patch("app.rag.retrieve.dense_search", new_callable=AsyncMock) as mock_search:
            with patch("app.rag.retrieve.rerank_scored_async") as mock_rerank:
                with patch("app.rag.strategies.classic.generate_structured", new_callable=AsyncMock) as mock_gen:
                    mock_search.return_value = chunks
                    mock_rerank.return_value = RerankOutcome.unscored(chunks)
                    mock_gen.side_effect = RuntimeError("API timeout")

                    with pytest.raises(RuntimeError):
                        await strategy.run(
                            "What is the test topic?",
                            top_k=8,
                            model="claude-sonnet-5",
                            prompt_version="default",
                        )

    async def test_classic_rag_empty_generation_response(self, fake_chunks):
        """Test classic RAG with empty generation response."""
        chunks = fake_chunks(2)
        strategy = ClassicRAG()

        with patch("app.rag.retrieve.dense_search", new_callable=AsyncMock) as mock_search:
            with patch("app.rag.retrieve.rerank_scored_async") as mock_rerank:
                with patch("app.rag.strategies.classic.generate_structured", new_callable=AsyncMock) as mock_gen:
                    mock_search.return_value = chunks
                    mock_rerank.return_value = RerankOutcome.unscored(chunks)
                    mock_gen.return_value = {
                        "text": "",
                        "input_tokens": 100,
                        "output_tokens": 0,
                    }

                    result = await strategy.run(
                        "What is the test topic?",
                        top_k=8,
                        model="claude-sonnet-5",
                        prompt_version="default",
                    )

                    assert result.refusal is True
                    assert result.status == "insufficient_context"
                    assert result.answer.startswith("I cannot answer this from the retrieved documents")


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
                model="claude-sonnet-5",
                prompt_version="default",
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
                    with patch("app.rag.retrieve.dense_search", new_callable=AsyncMock) as mock_search:
                        with patch("app.rag.strategies.graph._fetch_chunks_by_id", new_callable=AsyncMock) as mock_fetch:
                            with patch("app.rag.strategies.graph.rerank_async") as mock_rerank:
                                with patch("app.rag.strategies.graph.describe_subgraph") as mock_describe:
                                    with patch("app.rag.strategies.graph.generate_structured", new_callable=AsyncMock) as mock_gen:
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
                                            "text": "Graph-based answer [S3].",
                                            "input_tokens": 200,
                                            "output_tokens": 75,
                                        }

                                        result = await strategy.run(
                                            "How is entity1 connected to entity2?",
                                            top_k=8,
                                            model="claude-sonnet-5",
                                            prompt_version="default",
                                        )

                                        assert result.answer == "Graph-based answer [S3]."
                                        assert result.refusal is False
                                        assert [src.chunk_id for src in result.sources] == ["chunk_2"]
                                        assert result.extra["grounding"]["mode"] == "text"
                                        assert "entity1" in result.extra["entities"]
                                        assert "subgraph" in result.extra
                                        assert result.input_tokens == 200

    async def test_graph_rag_empty_after_rerank(self, fake_chunks):
        """Test graph RAG with empty results after reranking."""
        strategy = GraphRAG()

        with patch("app.rag.strategies.graph.load_graph") as mock_load:
            with patch("app.rag.strategies.graph.extract_question_entities", new_callable=AsyncMock) as mock_extract:
                with patch("app.rag.strategies.graph.neighbors") as mock_neighbors:
                    with patch("app.rag.retrieve.dense_search", new_callable=AsyncMock) as mock_search:
                        with patch("app.rag.strategies.graph._fetch_chunks_by_id", new_callable=AsyncMock) as mock_fetch:
                            with patch("app.rag.strategies.graph.rerank_async") as mock_rerank:
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
                                    model="claude-sonnet-5",
                                    prompt_version="default",
                                )

                                assert result.refusal is True
                                assert "found neither matching entities" in result.answer


@pytest.mark.asyncio
class TestAgenticRAG:
    async def test_agentic_rag_success_with_finish(self):
        """The finish tool's citations are validated against what the agent saw."""
        strategy = AgenticRAG()
        hits = [
            MagicMock(payload={"chunk_id": f"c{i}", "doc_id": "d", "text": f"text {i}", "filename": "f.md"})
            for i in range(2)
        ]

        async def fake_loop(**kwargs):
            tools = kwargs["tool_handlers"]
            await tools["search"]({"query": "x"})
            await tools["finish"](
                {
                    "answer": "Final answer [S2].",
                    "claims": [{"text": "Final answer.", "citations": ["S2"], "supported": True}],
                    "status": "answered",
                    "unsupported_notes": "",
                }
            )
            return {
                "text": "",
                "input_tokens": 300,
                "output_tokens": 100,
                "iterations": 2,
                "stop_reason": "terminal_tool",
                "trace": [
                    {"step": 0, "tool_calls": [{"name": "search"}]},
                    {"step": 1, "tool_calls": [{"name": "finish"}]},
                ],
            }

        with (
            patch("app.rag.strategies.agentic.tool_use_loop", new=fake_loop),
            patch("app.rag.strategies.agentic.hybrid_search", new=AsyncMock(return_value=_hybrid(hits))),
        ):
            result = await strategy.run(
                "Search for information?",
                top_k=8,
                model="claude-sonnet-5",
                prompt_version="default",
            )

        assert result.answer == "Final answer [S2]."
        assert result.grounded is True
        assert [src.chunk_id for src in result.sources] == ["c1"]
        assert result.iterations == 2
        assert result.input_tokens == 300
        assert result.trace[-1]["step"] == "validate_citations"

    async def test_agentic_rag_cannot_cite_unseen_chunks(self):
        """Handles the agent was never given, and raw chunk ids, are rejected."""
        strategy = AgenticRAG()
        hit = MagicMock(payload={"chunk_id": "c0", "doc_id": "d", "text": "seen", "filename": "f.md"})

        async def fake_loop(**kwargs):
            tools = kwargs["tool_handlers"]
            await tools["search"]({"query": "x"})
            await tools["finish"](
                {
                    "answer": "Invented [S7] and [secret:0].",
                    "claims": [
                        {"text": "Invented.", "citations": ["S7", "secret:0"], "supported": True}
                    ],
                    "status": "answered",
                    "unsupported_notes": "",
                }
            )
            return {"text": "", "input_tokens": 1, "output_tokens": 1, "iterations": 2, "trace": []}

        with (
            patch("app.rag.strategies.agentic.tool_use_loop", new=fake_loop),
            patch("app.rag.strategies.agentic.hybrid_search", new=AsyncMock(return_value=_hybrid([hit]))),
        ):
            result = await strategy.run("q", top_k=8, model="claude-sonnet-5", prompt_version="default")

        assert result.refusal is True
        assert result.grounded is False
        assert set(result.invalid_citations) == {"S7", "secret:0"}
        assert result.sources[0].chunk_id == "none"
        assert result.confidence == 0.0

    async def test_agentic_rag_no_finish_call(self):
        """Uncited prose instead of finish is not accepted as an answer."""
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
                model="claude-sonnet-5",
                prompt_version="default",
            )

            assert result.refusal is True
            assert result.extra["grounding"]["refusal_reason"] == "no_valid_citations"
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

            with patch("app.rag.retrieve.embed_query_async"):
                with patch("app.rag.strategies.agentic.hybrid_search", new_callable=AsyncMock):
                    await strategy.run(
                        "What?",
                        top_k=8,
                        model="claude-sonnet-5",
                        prompt_version="default",
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
                model="claude-sonnet-5",
                prompt_version="default",
            )

            assert result.refusal is True
            assert "(agent stopped without finishing)" in result.answer
