from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.eval.metrics import (
    faithfulness_score,
    answer_relevance_score,
    context_recall_score,
    context_precision_score,
    aggregate,
)
from app.schemas import EvalScore


@pytest.mark.asyncio
class TestFaithfulnessScore:
    async def test_faithfulness_high_score(self):
        """Test high faithfulness when answer is grounded in context."""
        answer = "The answer states that X is Y, which is supported by the context."
        context = [
            "X is Y",
            "Y has property Z",
        ]

        with patch("app.eval.metrics.ragas_faithfulness.ascore", new_callable=AsyncMock) as mock_score:
            mock_score.return_value = 0.95

            score = await faithfulness_score(answer, context)

            assert score == 0.95
            mock_score.assert_called_once()

    async def test_faithfulness_low_score(self):
        """Test low faithfulness when answer is not grounded."""
        answer = "The answer claims X is Y and also states that Q is R."
        context = [
            "X is Y",
            "This document is about something else.",
        ]

        with patch("app.eval.metrics.ragas_faithfulness.ascore", new_callable=AsyncMock) as mock_score:
            mock_score.return_value = 0.4

            score = await faithfulness_score(answer, context)

            assert score == 0.4
            assert score < 0.5  # Low faithfulness

    async def test_faithfulness_empty_context(self):
        """Test faithfulness with empty context."""
        answer = "Some answer"
        context = []

        score = await faithfulness_score(answer, context)

        assert score == 0.0

    async def test_faithfulness_empty_answer(self):
        """Test faithfulness with empty answer."""
        answer = ""
        context = ["Some context"]

        score = await faithfulness_score(answer, context)

        assert score == 0.0

    async def test_faithfulness_both_empty(self):
        """Test faithfulness with both empty."""
        answer = ""
        context = []

        score = await faithfulness_score(answer, context)

        assert score == 0.0

    async def test_faithfulness_score_clamped(self):
        """Test faithfulness score is clamped to [0, 1]."""
        answer = "Answer"
        context = ["Context"]

        with patch("app.eval.metrics.ragas_faithfulness.ascore", new_callable=AsyncMock) as mock_score:
            mock_score.return_value = 1.5  # Out of range

            score = await faithfulness_score(answer, context)

            assert 0.0 <= score <= 1.0
            assert score == 1.0

    async def test_faithfulness_error_handling(self):
        """Test faithfulness handles errors gracefully."""
        answer = "Answer"
        context = ["Context"]

        with patch("app.eval.metrics.ragas_faithfulness.ascore", new_callable=AsyncMock) as mock_score:
            mock_score.side_effect = RuntimeError("RAGAS error")

            score = await faithfulness_score(answer, context)

            assert score == 0.5  # Default fallback
            mock_score.assert_called_once()


@pytest.mark.asyncio
class TestAnswerRelevanceScore:
    async def test_relevance_high_score(self):
        """Test high relevance when answer addresses question."""
        question = "What is the capital of France?"
        answer = "The capital of France is Paris, located in the north-central part of the country."

        with patch("app.eval.metrics.ragas_answer_relevance.ascore", new_callable=AsyncMock) as mock_score:
            mock_score.return_value = 0.98

            score = await answer_relevance_score(question, answer)

            assert score == 0.98
            mock_score.assert_called_once()

    async def test_relevance_low_score(self):
        """Test low relevance when answer is off-topic."""
        question = "What is the capital of France?"
        answer = "Dogs are animals that live in homes and eat food."

        with patch("app.eval.metrics.ragas_answer_relevance.ascore", new_callable=AsyncMock) as mock_score:
            mock_score.return_value = 0.1

            score = await answer_relevance_score(question, answer)

            assert score == 0.1

    async def test_relevance_empty_question(self):
        """Test relevance with empty question."""
        question = ""
        answer = "Some answer"

        score = await answer_relevance_score(question, answer)

        assert score == 0.0

    async def test_relevance_empty_answer(self):
        """Test relevance with empty answer."""
        question = "What is X?"
        answer = ""

        score = await answer_relevance_score(question, answer)

        assert score == 0.0

    async def test_relevance_error_handling(self):
        """Test relevance error handling."""
        question = "What is X?"
        answer = "This is an answer"

        with patch("app.eval.metrics.ragas_answer_relevance.ascore", new_callable=AsyncMock) as mock_score:
            mock_score.side_effect = RuntimeError("RAGAS error")

            score = await answer_relevance_score(question, answer)

            assert score == 0.5


@pytest.mark.asyncio
class TestContextRecallScore:
    async def test_recall_high_score(self):
        """Test high recall when all information is in context."""
        expected = "The answer should mention X and Y and Z"
        retrieved = [
            "X is located in Y",
            "Z is important because of X",
            "More details about Y and Z",
        ]

        with patch("app.eval.metrics.ragas_context_recall.ascore", new_callable=AsyncMock) as mock_score:
            mock_score.return_value = 0.92

            score = await context_recall_score(expected, retrieved)

            assert score == 0.92

    async def test_recall_low_score(self):
        """Test low recall when information is missing."""
        expected = "The answer should mention X, Y, Z, Q, R"
        retrieved = [
            "X is here",
            "General information",
        ]

        with patch("app.eval.metrics.ragas_context_recall.ascore", new_callable=AsyncMock) as mock_score:
            mock_score.return_value = 0.3

            score = await context_recall_score(expected, retrieved)

            assert score == 0.3

    async def test_recall_empty_retrieved(self):
        """Test recall with empty retrieved context."""
        expected = "Important info"
        retrieved = []

        score = await context_recall_score(expected, retrieved)

        assert score == 0.0

    async def test_recall_empty_expected(self):
        """Test recall with empty expected."""
        expected = ""
        retrieved = ["Some context"]

        score = await context_recall_score(expected, retrieved)

        assert score == 0.0

    async def test_recall_with_list_expected(self):
        """Test recall with list of expected information."""
        expected = ["X is important", "Y is relevant"]
        retrieved = [
            "X is important for this reason",
            "Y is relevant in this context",
        ]

        with patch("app.eval.metrics.ragas_context_recall.ascore", new_callable=AsyncMock) as mock_score:
            mock_score.return_value = 0.85

            score = await context_recall_score(expected, retrieved)

            assert score == 0.85

    async def test_recall_error_handling(self):
        """Test recall error handling."""
        expected = "Something"
        retrieved = ["Context"]

        with patch("app.eval.metrics.ragas_context_recall.ascore", new_callable=AsyncMock) as mock_score:
            mock_score.side_effect = RuntimeError("RAGAS error")

            score = await context_recall_score(expected, retrieved)

            assert score == 0.5


@pytest.mark.asyncio
class TestContextPrecisionScore:
    async def test_precision_high_score(self):
        """Test high precision when all context is relevant."""
        question = "What is the capital of France?"
        context = [
            "Paris is the capital of France.",
            "Paris is located in north-central France.",
            "The city has many museums and landmarks.",
        ]
        answer = "The capital is Paris"

        with patch("app.eval.metrics.ragas_context_precision.ascore", new_callable=AsyncMock) as mock_score:
            mock_score.return_value = 0.95

            score = await context_precision_score(question, context, answer)

            assert score == 0.95

    async def test_precision_low_score(self):
        """Test low precision when context is not relevant."""
        question = "What is the capital of France?"
        context = [
            "Dogs are animals.",
            "Cats live in houses.",
            "Birds fly in the sky.",
        ]
        answer = "Paris"

        with patch("app.eval.metrics.ragas_context_precision.ascore", new_callable=AsyncMock) as mock_score:
            mock_score.return_value = 0.1

            score = await context_precision_score(question, context, answer)

            assert score == 0.1

    async def test_precision_empty_context(self):
        """Test precision with empty context."""
        question = "What is X?"
        context = []
        answer = "Answer"

        score = await context_precision_score(question, context, answer)

        assert score == 0.0

    async def test_precision_empty_question(self):
        """Test precision with empty question."""
        question = ""
        context = ["Some context"]
        answer = "Answer"

        score = await context_precision_score(question, context, answer)

        assert score == 0.0

    async def test_precision_mixed_relevance_context(self):
        """Test precision with mixed relevant/irrelevant context."""
        question = "What is capital of France?"
        context = [
            "Paris is the capital.",
            "Dogs are animals.",
            "France has many cities.",
            "Cats meow.",
        ]
        answer = "Paris"

        with patch("app.eval.metrics.ragas_context_precision.ascore", new_callable=AsyncMock) as mock_score:
            mock_score.return_value = 0.5

            score = await context_precision_score(question, context, answer)

            assert score == 0.5

    async def test_precision_error_handling(self):
        """Test precision error handling."""
        question = "What?"
        context = ["Context"]
        answer = "Answer"

        with patch("app.eval.metrics.ragas_context_precision.ascore", new_callable=AsyncMock) as mock_score:
            mock_score.side_effect = RuntimeError("RAGAS error")

            score = await context_precision_score(question, context, answer)

            assert score == 0.5


@pytest.mark.asyncio
class TestAggregate:
    def test_aggregate_single_score(self):
        """Test aggregation with single score."""
        scores = [
            {
                "faithfulness": 0.9,
                "answer_relevance": 0.85,
                "context_precision": 0.8,
                "context_recall": 0.88,
            }
        ]

        result = aggregate(scores)

        assert result.faithfulness == 0.9
        assert result.answer_relevance == 0.85
        assert result.context_precision == 0.8
        assert result.context_recall == 0.88

    def test_aggregate_multiple_scores(self):
        """Test aggregation with multiple scores."""
        scores = [
            {
                "faithfulness": 0.9,
                "answer_relevance": 0.85,
                "context_precision": 0.8,
                "context_recall": 0.88,
            },
            {
                "faithfulness": 0.8,
                "answer_relevance": 0.9,
                "context_precision": 0.75,
                "context_recall": 0.82,
            },
            {
                "faithfulness": 0.85,
                "answer_relevance": 0.88,
                "context_precision": 0.82,
                "context_recall": 0.85,
            },
        ]

        result = aggregate(scores)

        assert abs(result.faithfulness - 0.85) < 0.01
        assert abs(result.answer_relevance - 0.877) < 0.01
        assert abs(result.context_precision - 0.79) < 0.01
        assert abs(result.context_recall - 0.85) < 0.01

    def test_aggregate_empty_list(self):
        """Test aggregation with empty list."""
        scores = []

        result = aggregate(scores)

        assert result is None

    def test_aggregate_returns_eval_score_type(self):
        """Test that aggregate returns EvalScore type."""
        scores = [
            {
                "faithfulness": 0.8,
                "answer_relevance": 0.85,
                "context_precision": 0.9,
                "context_recall": 0.75,
            }
        ]

        result = aggregate(scores)

        assert isinstance(result, EvalScore)

    def test_aggregate_zero_scores(self):
        """Test aggregation with zero scores."""
        scores = [
            {
                "faithfulness": 0.0,
                "answer_relevance": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
            }
        ]

        result = aggregate(scores)

        assert result.faithfulness == 0.0
        assert result.answer_relevance == 0.0
        assert result.context_precision == 0.0
        assert result.context_recall == 0.0

    def test_aggregate_perfect_scores(self):
        """Test aggregation with perfect scores."""
        scores = [
            {
                "faithfulness": 1.0,
                "answer_relevance": 1.0,
                "context_precision": 1.0,
                "context_recall": 1.0,
            }
        ]

        result = aggregate(scores)

        assert result.faithfulness == 1.0
        assert result.answer_relevance == 1.0
        assert result.context_precision == 1.0
        assert result.context_recall == 1.0
