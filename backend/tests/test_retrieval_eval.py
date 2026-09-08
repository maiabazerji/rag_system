"""Tests for retrieval scoring and strategy-aware evaluation.

Two capabilities the project's central question depends on and did not have:
the harness could only ever evaluate `classic`, and the golden dataset's
`expected_sources` field was read by nothing.
"""
from unittest.mock import AsyncMock, patch

import pytest

from app.eval.metrics import run_evaluation
from app.eval.retrieval import (
    aggregate_retrieval,
    retrieved_documents,
    score_retrieval,
)
from app.schemas import Answer, Source


def _sources(*docs: str) -> list[Source]:
    """Build ranked sources for the given document names."""
    return [
        Source(chunk_id=f"c{i}", quote=f"quote {i}", document=doc)
        for i, doc in enumerate(docs)
    ]


class TestRetrievedDocuments:
    def test_preserves_rank_order(self):
        assert retrieved_documents(_sources("b.md", "a.md")) == ["b.md", "a.md"]

    def test_deduplicates_keeping_first_position(self):
        """Several chunks from one document count as one retrieved document."""
        assert retrieved_documents(_sources("a.md", "b.md", "a.md")) == ["a.md", "b.md"]

    def test_skips_sources_with_no_document(self):
        sources = [Source(chunk_id="c1", quote="q"), *_sources("a.md")]
        assert retrieved_documents(sources) == ["a.md"]

    def test_skips_the_refusal_placeholder(self):
        placeholder = [Source(chunk_id="none", quote="", document="x.md")]
        assert retrieved_documents(placeholder) == []

    def test_empty_input(self):
        assert retrieved_documents([]) == []


class TestScoreRetrieval:
    def test_perfect_retrieval(self):
        score = score_retrieval(["a.md"], _sources("a.md"))
        assert score.precision == 1.0
        assert score.recall == 1.0
        assert score.hit is True
        assert score.mrr == 1.0

    def test_expected_document_ranked_second(self):
        score = score_retrieval(["a.md"], _sources("b.md", "a.md"))
        assert score.precision == 0.5
        assert score.recall == 1.0
        assert score.mrr == 0.5

    def test_complete_miss(self):
        score = score_retrieval(["a.md"], _sources("b.md", "c.md"))
        assert score.precision == 0.0
        assert score.recall == 0.0
        assert score.hit is False
        assert score.mrr == 0.0

    def test_partial_recall_across_two_expected_docs(self):
        score = score_retrieval(["a.md", "b.md"], _sources("a.md", "z.md"))
        assert score.recall == 0.5
        assert score.precision == 0.5
        assert score.hit is True

    def test_matching_ignores_case_and_path(self):
        score = score_retrieval(["rag_overview.md"], _sources("docs/RAG_Overview.md"))
        assert score.hit is True
        assert score.precision == 1.0

    def test_returns_none_when_the_example_expects_nothing(self):
        """Same 'not measured' convention the judge uses."""
        assert score_retrieval(None, _sources("a.md")) is None
        assert score_retrieval([], _sources("a.md")) is None

    def test_zero_scores_when_nothing_was_retrieved(self):
        score = score_retrieval(["a.md"], [])
        assert score.hit is False
        assert score.precision == 0.0
        assert score.retrieved == []
        assert score.expected == ["a.md"]

    def test_records_both_sides_for_inspection(self):
        score = score_retrieval(["a.md"], _sources("b.md", "a.md"))
        assert score.retrieved == ["b.md", "a.md"]
        assert score.expected == ["a.md"]

    def test_needs_no_model_call(self):
        """The whole point: this is measurable with no API key."""
        with patch("app.rag.providers.anthropic_provider.get_client") as client:
            score_retrieval(["a.md"], _sources("a.md"))
            client.assert_not_called()


class TestAggregateRetrieval:
    def test_averages_and_counts(self):
        perfect = score_retrieval(["a.md"], _sources("a.md")).model_dump()
        miss = score_retrieval(["a.md"], _sources("b.md")).model_dump()

        agg = aggregate_retrieval([perfect, miss])
        assert agg["precision"] == 0.5
        assert agg["recall"] == 0.5
        assert agg["hit_rate"] == 0.5
        assert agg["n"] == 2

    def test_returns_none_for_nothing_scored(self):
        assert aggregate_retrieval([]) is None


def _answer(docs: tuple[str, ...] = ("a.md",), refusal: bool = False) -> Answer:
    return Answer(
        question="q",
        answer="an answer",
        sources=_sources(*docs) or [Source(chunk_id="none", quote="")],
        confidence=0.9,
        refusal=refusal,
        model="claude-sonnet-5",
        latency_ms=1200,
        input_tokens=900,
        output_tokens=120,
    )


@pytest.fixture
def golden(tmp_path, monkeypatch):
    """A two-example dataset with expected_sources, isolated on disk."""
    monkeypatch.setattr("app.eval.metrics.GOLDEN_DIR", tmp_path)
    monkeypatch.setattr("app.eval.metrics.RUNS_DIR", tmp_path / "runs")
    (tmp_path / "d.jsonl").write_text(
        '{"question": "one", "expected_sources": ["a.md"]}\n'
        '{"question": "two", "expected_sources": ["b.md"]}\n',
        encoding="utf-8",
    )
    return tmp_path


@pytest.mark.asyncio
class TestStrategyAwareEvaluation:
    async def test_strategy_reaches_the_pipeline(self, golden):
        """The regression: run_evaluation always evaluated 'classic'."""
        answer = AsyncMock(return_value=_answer())
        with (
            patch("app.eval.metrics.answer_question", new=answer),
            patch("app.eval.metrics.judge", new=AsyncMock(return_value=None)),
        ):
            result = await run_evaluation(dataset="d", strategy="agentic")

        assert result["strategy"] == "agentic"
        assert {c.kwargs["strategy"] for c in answer.await_args_list} == {"agentic"}

    async def test_defaults_to_classic(self, golden):
        answer = AsyncMock(return_value=_answer())
        with (
            patch("app.eval.metrics.answer_question", new=answer),
            patch("app.eval.metrics.judge", new=AsyncMock(return_value=None)),
        ):
            result = await run_evaluation(dataset="d")

        assert result["strategy"] == "classic"

    async def test_retrieval_is_scored_even_when_the_judge_fails(self, golden):
        """Retrieval needs no model call, so it survives a dead judge."""
        answer = AsyncMock(return_value=_answer(("a.md",)))
        with (
            patch("app.eval.metrics.answer_question", new=answer),
            patch("app.eval.metrics.judge", new=AsyncMock(return_value=None)),
        ):
            result = await run_evaluation(dataset="d", strategy="graph")

        assert result["aggregate"] is None
        assert result["n_scored"] == 0
        assert result["retrieval_aggregate"] is not None
        # Both examples cite a.md; only the first expects it.
        assert result["retrieval_aggregate"]["hit_rate"] == 0.5
        assert result["retrieval_aggregate"]["n"] == 2

    async def test_per_example_carries_both_scores(self, golden):
        with (
            patch("app.eval.metrics.answer_question", new=AsyncMock(return_value=_answer())),
            patch("app.eval.metrics.judge", new=AsyncMock(return_value=None)),
        ):
            result = await run_evaluation(dataset="d")

        first = result["per_example"][0]
        assert first["retrieval"]["hit"] is True
        assert first["score"] is None
        assert first["latency_ms"] == 1200

    async def test_cost_block_is_reported(self, golden):
        with (
            patch("app.eval.metrics.answer_question", new=AsyncMock(return_value=_answer())),
            patch("app.eval.metrics.judge", new=AsyncMock(return_value=None)),
        ):
            result = await run_evaluation(dataset="d")

        assert result["cost"]["mean_latency_ms"] == 1200
        assert result["cost"]["total_input_tokens"] == 1800
        assert result["cost"]["total_output_tokens"] == 240

    async def test_refusals_are_counted(self, golden):
        with (
            patch(
                "app.eval.metrics.answer_question",
                new=AsyncMock(return_value=_answer(refusal=True)),
            ),
            patch("app.eval.metrics.judge", new=AsyncMock(return_value=None)),
        ):
            result = await run_evaluation(dataset="d")

        assert result["n_refused"] == 2

    async def test_run_file_is_named_per_strategy(self, golden):
        """Two strategies must not overwrite each other's run."""
        with (
            patch("app.eval.metrics.answer_question", new=AsyncMock(return_value=_answer())),
            patch("app.eval.metrics.judge", new=AsyncMock(return_value=None)),
        ):
            await run_evaluation(dataset="d", strategy="classic")
            await run_evaluation(dataset="d", strategy="agentic")

        names = [p.name for p in (golden / "runs").glob("*.json")]
        assert any("classic" in n for n in names)
        assert any("agentic" in n for n in names)


class TestRegressionGroupsByStrategy:
    """Classic scoring below Agentic is the finding, not a regression."""

    @staticmethod
    def _write(dir_path, name, *, strategy, faithfulness):
        import json

        (dir_path / f"{name}.json").write_text(
            json.dumps(
                {
                    "dataset": "golden_v1",
                    "strategy": strategy,
                    "provider": "anthropic",
                    "model": "m1",
                    "prompt_version": "default",
                    "aggregate": {
                        "faithfulness": faithfulness,
                        "answer_relevance": 0.8,
                        "context_precision": 0.8,
                        "context_recall": 0.8,
                    },
                }
            ),
            encoding="utf-8",
        )

    def test_different_strategies_are_not_compared(self, tmp_path, monkeypatch):
        from app.eval.regression import load_regressions

        monkeypatch.setattr("app.eval.regression.RUNS_DIR", tmp_path)
        self._write(tmp_path, "20260101T000000Z-a", strategy="agentic", faithfulness=0.9)
        self._write(tmp_path, "20260102T000000Z-b", strategy="classic", faithfulness=0.3)

        assert load_regressions() == []

    def test_same_strategy_still_regresses(self, tmp_path, monkeypatch):
        from app.eval.regression import load_regressions

        monkeypatch.setattr("app.eval.regression.RUNS_DIR", tmp_path)
        self._write(tmp_path, "20260101T000000Z-a", strategy="classic", faithfulness=0.9)
        self._write(tmp_path, "20260102T000000Z-b", strategy="classic", faithfulness=0.3)

        found = load_regressions()
        assert len(found) == 1
        assert found[0]["strategy"] == "classic"


def test_eval_run_request_accepts_a_strategy():
    from app.schemas import EvalRunRequest

    assert EvalRunRequest().strategy == "classic"
    assert EvalRunRequest(strategy="graph").strategy == "graph"

    with pytest.raises(ValueError):
        EvalRunRequest(strategy="telepathy")


def test_sources_carry_their_document(fake_chunks):
    """Without this the golden dataset's expected_sources cannot be scored."""
    from app.rag.strategies.classic import ClassicRAG

    chunks = fake_chunks(2)
    for chunk in chunks:
        chunk.metadata["filename"] = "rag_overview.md"

    strategy = ClassicRAG()
    with (
        patch("app.rag.strategies.classic.dense_search", new=AsyncMock(return_value=chunks)),
        patch("app.rag.strategies.classic.rerank_async", new=AsyncMock(return_value=chunks)),
        patch(
            "app.rag.strategies.classic.generate_with_usage",
            new=AsyncMock(
                return_value={"text": "answer", "input_tokens": 10, "output_tokens": 5}
            ),
        ),
    ):
        result = asyncio_run(
            strategy.run("q", top_k=2, model="claude-sonnet-5", prompt_version="default")
        )

    assert all(s.document == "rag_overview.md" for s in result.sources)


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)
