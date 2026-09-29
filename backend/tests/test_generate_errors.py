"""Error handling of the orchestrator (app.rag.generate).

Every failure in the answer path must become a refusal the UI can render,
with a client-safe message in the question's language; only a vector store
outage propagates (the app serves it as a 503).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.i18n import message
from app.prompts.loader import UnknownPromptVersionError
from app.rag import generate
from app.rag.providers import MissingKeyError, ProviderError
from app.rag.store import StoreUnavailable
from app.rag.strategies.base import StrategyResult
from app.schemas import Source

FR_QUESTION = "Quelle est la durée de validité du devis ?"


def _strategy(run=None, result=None):
    """A strategy double; ``run`` is an AsyncMock (side effect or return value)."""
    if run is None:
        run = AsyncMock(
            return_value=result
            or StrategyResult(
                answer="An answer [S1].",
                sources=[Source(chunk_id="d:0", quote="q")],
                confidence=0.8,
                input_tokens=10,
                output_tokens=5,
            )
        )
    return SimpleNamespace(run=run, provider=None)


@pytest.fixture
def docs_indexed():
    with patch.object(generate, "store_count", new=AsyncMock(return_value=3)) as count:
        yield count


def _with_strategy(strategy):
    return patch.object(generate, "get_strategy", return_value=strategy)


class TestDefaultModel:
    @pytest.mark.parametrize(
        "provider, strategy, setting",
        [
            ("local", "classic", "ollama_model"),
            ("openai", "classic", "openai_generator_model"),
            ("anthropic", "agentic", "agentic_model"),
            ("anthropic", "classic", "generator_model"),
            ("anthropic", "graph", "generator_model"),
        ],
    )
    def test_model_follows_provider_and_strategy(self, settings, provider, strategy, setting):
        assert generate._default_model(provider, strategy) == getattr(settings, setting)


class TestAnswerQuestionDetailed:
    async def test_no_documents_is_a_localized_refusal(self):
        with patch.object(generate, "store_count", new=AsyncMock(return_value=0)):
            answer, result = await generate.answer_question_detailed(FR_QUESTION)
        assert result is None
        assert answer.refusal is True
        assert answer.answer == message("no_documents", "fr")

    async def test_store_outage_propagates(self):
        with patch.object(
            generate, "store_count", new=AsyncMock(side_effect=StoreUnavailable("down"))
        ), pytest.raises(StoreUnavailable):
            await generate.answer_question_detailed("q")

    async def test_other_count_failures_do_not_block_the_strategy(self):
        strategy = _strategy()
        with (
            patch.object(generate, "store_count", new=AsyncMock(side_effect=TimeoutError())),
            _with_strategy(strategy),
        ):
            answer, result = await generate.answer_question_detailed("q")
        assert answer.refusal is False
        assert result is not None
        strategy.run.assert_awaited_once()

    async def test_unknown_strategy_is_refused(self, docs_indexed):
        answer, result = await generate.answer_question_detailed("q", strategy="bogus")
        assert result is None
        assert answer.refusal is True
        assert "bogus" in answer.answer
        assert answer.trace_id

    async def test_provider_error_hides_upstream_details(self, docs_indexed):
        leak = ProviderError("HTTP 500 from https://internal-gateway:8443 body={secret}")
        with _with_strategy(_strategy(run=AsyncMock(side_effect=leak))):
            answer, result = await generate.answer_question_detailed(FR_QUESTION)
        assert result is None
        assert answer.refusal is True
        assert answer.answer == message("provider_unavailable", "fr")
        assert "internal-gateway" not in answer.answer
        assert answer.trace_id

    async def test_missing_key_message_is_passed_through(self, docs_indexed):
        err = MissingKeyError("OPENAI_API_KEY is not configured.")
        with _with_strategy(_strategy(run=AsyncMock(side_effect=err))):
            answer, _ = await generate.answer_question_detailed("q", provider="openai")
        assert answer.answer == "OPENAI_API_KEY is not configured."
        assert (answer.provider, answer.refusal) == ("openai", True)

    async def test_unknown_prompt_version_is_reported(self, docs_indexed):
        err = UnknownPromptVersionError("Unknown prompt version 'v9'.")
        with _with_strategy(_strategy(run=AsyncMock(side_effect=err))):
            answer, _ = await generate.answer_question_detailed("q", prompt_version="v9")
        assert answer.refusal is True
        assert answer.answer == "Unknown prompt version 'v9'."

    async def test_unexpected_error_is_a_generic_localized_refusal(self, docs_indexed):
        with _with_strategy(_strategy(run=AsyncMock(side_effect=KeyError("chunk_id")))):
            answer, result = await generate.answer_question_detailed(FR_QUESTION)
        assert result is None
        assert answer.answer == message("internal_error", "fr")
        assert "chunk_id" not in answer.answer

    async def test_graph_and_agentic_always_run_on_anthropic(self, docs_indexed, settings):
        strategy = _strategy()
        with _with_strategy(strategy):
            answer, _ = await generate.answer_question_detailed(
                "q", strategy="agentic", provider="openai"
            )
        assert strategy.provider == "anthropic"
        assert answer.provider == "anthropic"
        assert answer.model == settings.agentic_model

    async def test_success_carries_metrics_and_trace(self, docs_indexed, settings):
        strategy = _strategy()
        with _with_strategy(strategy):
            answer = await generate.answer_question("q", top_k=3)
        assert strategy.run.await_args.kwargs["top_k"] == 3
        assert answer.metrics is not None
        assert (answer.metrics.input_tokens, answer.metrics.output_tokens) == (10, 5)
        assert answer.trace_id


class TestRunStrategyRaw:
    async def test_no_documents(self):
        with patch.object(generate, "store_count", new=AsyncMock(return_value=0)):
            assert await generate.run_strategy_raw("q", strategy="classic") == (
                None,
                "No documents indexed yet.",
            )

    async def test_store_outage_propagates(self):
        with patch.object(
            generate, "store_count", new=AsyncMock(side_effect=StoreUnavailable("down"))
        ), pytest.raises(StoreUnavailable):
            await generate.run_strategy_raw("q", strategy="classic")

    async def test_other_count_failures_do_not_block_the_strategy(self):
        with (
            patch.object(generate, "store_count", new=AsyncMock(side_effect=OSError("slow"))),
            _with_strategy(_strategy()),
        ):
            result, error = await generate.run_strategy_raw("q", strategy="classic")
        assert error is None
        assert result is not None and result.metrics is not None
        assert result.trace_id

    async def test_unknown_strategy(self, docs_indexed):
        result, error = await generate.run_strategy_raw("q", strategy="bogus")
        assert result is None
        assert "bogus" in error

    @pytest.mark.parametrize(
        "exc, expected",
        [
            (ProviderError("upstream 502 at 10.0.0.7"), message("provider_unavailable", "en")),
            (MissingKeyError("ANTHROPIC_API_KEY is not configured."),
             "ANTHROPIC_API_KEY is not configured."),
            (UnknownPromptVersionError("Unknown prompt version 'x'."),
             "Unknown prompt version 'x'."),
            (RuntimeError("boom"), "This strategy failed (RuntimeError). See the backend logs."),
        ],
    )
    async def test_failures_become_client_safe_messages(self, docs_indexed, exc, expected):
        with _with_strategy(_strategy(run=AsyncMock(side_effect=exc))):
            result, error = await generate.run_strategy_raw(
                "What is the notice period?", strategy="classic"
            )
        assert result is None
        assert error == expected


def test_public_provider_error_without_a_question_uses_the_english_default():
    assert generate.public_provider_error(ProviderError("x")) == (
        generate.PROVIDER_UNAVAILABLE_MESSAGE
    )
