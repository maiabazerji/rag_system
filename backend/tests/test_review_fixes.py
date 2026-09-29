"""Regression tests for the audited defects in providers, strategies, eval and auth.

Each class pins one fix: what the old behaviour was is in its docstring.
"""
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app import auth
from app.config import VALID_ANTHROPIC_MODELS, Settings
from app.rag.providers.base import MissingKeyError, ProviderError
from app.rag.rerank import RerankOutcome
from app.rag.retrieve import RetrievalResult
from app.rag.strategies.base import StrategyResult
from app.schemas import (
    Answer,
    AskRequest,
    CompareStrategiesRequest,
    CompareVariant,
    EvalRunRequest,
    Source,
)

# --------------------------------------------------------------------------
# Model allowlist and model IDs
# --------------------------------------------------------------------------


class TestModelIds:
    def test_defaults_are_current_models(self):
        s = Settings(_env_file=None)
        assert s.judge_model == "claude-opus-5-5"
        assert s.graph_extraction_model == "claude-haiku-4-5-20251001"
        for model in (s.generator_model, s.agentic_model, s.judge_model, s.graph_extraction_model):
            assert model in VALID_ANTHROPIC_MODELS

    def test_obsolete_ids_are_gone(self):
        for old in ("claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8", "claude-sonnet-4-6"):
            assert old not in VALID_ANTHROPIC_MODELS

    def test_env_example_uses_current_ids(self):
        env = (Path(__file__).resolve().parents[2] / ".env.example").read_text(encoding="utf-8")
        assert "JUDGE_MODEL=claude-opus-5-5" in env
        assert "GRAPH_EXTRACTION_MODEL=claude-haiku-4-5-20251001" in env


class TestModelAllowlist:
    """`model` used to be free text passed straight to the provider."""

    def test_known_anthropic_model_is_accepted(self):
        assert AskRequest(question="q", model="claude-opus-5-5").model == "claude-opus-5-5"

    def test_unknown_model_is_rejected(self):
        with pytest.raises(ValueError, match="not allowed"):
            AskRequest(question="q", model="claude-made-up-9")

    def test_model_must_match_the_provider(self):
        with pytest.raises(ValueError):
            AskRequest(question="q", model="gpt-4o")  # default provider is anthropic
        assert AskRequest(question="q", provider="openai", model="gpt-4o").model == "gpt-4o"

    def test_graph_and_agentic_always_validate_against_anthropic(self):
        with pytest.raises(ValueError):
            AskRequest(question="q", provider="openai", model="gpt-4o", strategy="agentic")

    def test_local_accepts_the_configured_ollama_model(self, settings):
        req = AskRequest(question="q", provider="local", model=settings.ollama_model)
        assert req.model == settings.ollama_model
        with pytest.raises(ValueError):
            AskRequest(question="q", provider="local", model="something-else")

    def test_compare_variant_and_strategies_and_eval_are_checked(self):
        with pytest.raises(ValueError):
            CompareVariant(model="nope")
        with pytest.raises(ValueError):
            CompareStrategiesRequest(question="q", model="nope")
        with pytest.raises(ValueError):
            EvalRunRequest(model="nope")
        assert CompareStrategiesRequest(question="q", model="claude-sonnet-5").model

    def test_extra_allowed_models_extends_the_list(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "extra_allowed_models", "claude-next-1, other")
        assert AskRequest(question="q", model="claude-next-1").model == "claude-next-1"

    def test_http_request_with_a_bad_model_is_rejected(self, client):
        r = client.post("/ask", json={"question": "q", "model": "evil-model"})
        assert r.status_code == 422


class TestPromptVersion:
    """An unknown prompt_version used to fall back to `default` silently."""

    def test_unknown_version_is_rejected_by_the_schema(self):
        with pytest.raises(ValueError, match="prompt_version"):
            AskRequest(question="q", prompt_version="nope")
        assert AskRequest(question="q", prompt_version="default").prompt_version == "default"

    def test_http_request_with_an_unknown_version_is_a_client_error(self, client):
        r = client.post("/ask", json={"question": "q", "prompt_version": "nope"})
        assert r.status_code == 422

    def test_load_prompt_raises(self):
        from app.prompts import UnknownPromptVersionError, load_prompt

        with pytest.raises(UnknownPromptVersionError):
            load_prompt("nope")
        assert load_prompt("default").template

    def test_path_traversal_is_not_a_version(self):
        from app.prompts.loader import prompt_version_exists

        assert prompt_version_exists("default")
        assert not prompt_version_exists("../prompts/default")
        assert not prompt_version_exists("..")


# --------------------------------------------------------------------------
# Provider routing (classic)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
class TestProviderRouting:
    """Classic used to call Anthropic even when provider=openai/local."""

    async def test_dispatcher_routes_by_provider(self):
        from app.rag.providers import base

        usage = {"text": "t", "input_tokens": 3, "output_tokens": 4}
        with (
            patch(
                "app.rag.providers.openai_provider.generate_with_usage",
                new=AsyncMock(return_value=usage),
            ) as openai_fn,
            patch(
                "app.rag.providers.local.generate_with_usage",
                new=AsyncMock(return_value=usage),
            ) as local_fn,
        ):
            assert await base.generate_with_usage("openai", model="m", prompt="p") == usage
            assert await base.generate_with_usage("local", model="m", prompt="p") == usage
        openai_fn.assert_awaited_once()
        local_fn.assert_awaited_once()

    async def test_unknown_provider_is_an_error(self):
        from app.rag.providers import base

        with pytest.raises(ProviderError):
            await base.generate_with_usage("nope", model="m", prompt="p")

    async def test_answer_question_runs_classic_on_the_requested_provider(self, fake_chunks):
        from app.rag.generate import answer_question

        chunks = fake_chunks(2)
        gen = AsyncMock(return_value={"text": "hi", "input_tokens": 7, "output_tokens": 2})
        with (
            patch("app.rag.generate.store_count", new=AsyncMock(return_value=5)),
            patch("app.rag.retrieve.dense_search", new=AsyncMock(return_value=chunks)),
            patch(
                "app.rag.retrieve.rerank_scored_async",
                new=AsyncMock(return_value=RerankOutcome.unscored(chunks)),
            ),
            patch("app.rag.strategies.classic.generate_with_usage", new=gen),
        ):
            ans = await answer_question("q", provider="local")

        assert gen.await_args.args[0] == "local"
        assert ans.provider == "local"
        assert ans.input_tokens == 7


# --------------------------------------------------------------------------
# Graph token accounting
# --------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGraphTokens:
    """Entity-extraction tokens used to be dropped from the totals."""

    async def test_extraction_reports_usage(self):
        from app.rag import graph_extract

        reply = {"text": '["BM25", "RAG"]', "input_tokens": 30, "output_tokens": 8}
        with patch.object(graph_extract, "generate_with_usage", new=AsyncMock(return_value=reply)):
            ents = await graph_extract.extract_question_entities("q")
        assert ents == ["BM25", "RAG"]
        assert (ents.input_tokens, ents.output_tokens) == (30, 8)

    async def test_extraction_usage_counts_even_when_unparseable(self):
        from app.rag import graph_extract

        reply = {"text": "no json here", "input_tokens": 12, "output_tokens": 3}
        with patch.object(graph_extract, "generate_with_usage", new=AsyncMock(return_value=reply)):
            ents = await graph_extract.extract_question_entities("q")
        assert ents == [] and ents.input_tokens == 12

    async def test_strategy_totals_include_extraction(self, fake_chunks):
        from app.rag.graph_extract import QuestionEntities
        from app.rag.strategies.graph import GraphRAG

        chunks = fake_chunks(2)
        ents = QuestionEntities(["e1"])
        ents.input_tokens, ents.output_tokens = 30, 8
        graph = MagicMock(triples=[("a", "b", "c")])
        with (
            patch("app.rag.strategies.graph.load_graph", return_value=graph),
            patch(
                "app.rag.strategies.graph.extract_question_entities",
                new=AsyncMock(return_value=ents),
            ),
            patch("app.rag.strategies.graph.neighbors", return_value=(set(), set())),
            patch("app.rag.retrieve.dense_search", new=AsyncMock(return_value=chunks)),
            patch("app.rag.strategies.graph._fetch_chunks_by_id", new=AsyncMock(return_value=[])),
            patch("app.rag.strategies.graph.rerank_async", new=AsyncMock(return_value=chunks)),
            patch("app.rag.strategies.graph.describe_subgraph", return_value="G"),
            patch(
                "app.rag.strategies.graph.generate_with_usage",
                new=AsyncMock(
                    return_value={"text": "a", "input_tokens": 200, "output_tokens": 50}
                ),
            ),
        ):
            result = await GraphRAG().run("q", top_k=2, model="claude-sonnet-5", prompt_version="default")

        assert result.input_tokens == 230
        assert result.output_tokens == 58
        assert result.extra["retrieved_ids"] == [c.id for c in chunks]
        assert "G" in result.extra["context_text"]


# --------------------------------------------------------------------------
# Agentic loop outcomes
# --------------------------------------------------------------------------


def _loop_out(text, stop_reason, error, iterations=2):
    return {
        "text": text,
        "trace": [],
        "input_tokens": 10,
        "output_tokens": 5,
        "iterations": iterations,
        "stop_reason": stop_reason,
        "error": error,
    }


@pytest.mark.asyncio
class TestAgenticOutcomes:
    """Provider-error text over 50 chars used to be returned as an answer at 0.8."""

    @pytest.mark.parametrize("stop_reason", ["provider_error", "max_iters"])
    async def test_a_failed_loop_is_a_refusal(self, stop_reason):
        from app.rag.strategies.agentic import AgenticRAG

        long_diag = "The agent stopped early after a provider error (APIConnectionError). " * 2
        with patch(
            "app.rag.strategies.agentic.tool_use_loop",
            new=AsyncMock(return_value=_loop_out(long_diag, stop_reason, True)),
        ):
            result = await AgenticRAG().run("q", top_k=8, model="claude-sonnet-5", prompt_version="default")

        assert result.refusal is True
        assert result.confidence == 0.0
        assert result.answer != long_diag
        assert result.extra["stop_reason"] == stop_reason

    async def test_finish_is_a_terminal_tool(self):
        from app.rag.strategies.agentic import AgenticRAG

        loop = AsyncMock(return_value=_loop_out("", "end_turn", False))
        with patch("app.rag.strategies.agentic.tool_use_loop", new=loop):
            await AgenticRAG().run("q", top_k=8, model="claude-sonnet-5", prompt_version="default")
        assert loop.await_args.kwargs["terminal_tools"] == {"finish"}

    async def test_search_top_k_is_capped(self):
        from app.rag.strategies.agentic import AgenticRAG

        captured = {}

        async def fake_loop(**kwargs):
            captured["search"] = kwargs["tool_handlers"]["search"]
            return _loop_out("", "end_turn", False)

        search = AsyncMock(return_value=RetrievalResult(chunks=[]))
        with (
            patch("app.rag.strategies.agentic.tool_use_loop", new=fake_loop),
            patch("app.rag.strategies.agentic.hybrid_search", new=search),
        ):
            await AgenticRAG().run("q", top_k=8, model="claude-sonnet-5", prompt_version="default")
            await captured["search"]({"query": "x", "top_k": 5000})
            await captured["search"]({"query": "x", "top_k": "junk"})

        assert [c.kwargs["final_k"] for c in search.await_args_list] == [12, 5]

    async def test_finish_answer_carries_the_context_it_read(self):
        from app.rag.strategies.agentic import AgenticRAG

        async def fake_loop(**kwargs):
            handlers = kwargs["tool_handlers"]
            await handlers["search"]({"query": "x"})
            await handlers["finish"]({"answer": "A", "citations": ["c1"]})
            return _loop_out("", "terminal_tool", False)

        hit = {"chunk_id": "c1", "doc_id": "d", "text": "chunk text", "filename": "f.md"}
        with (
            patch("app.rag.strategies.agentic.tool_use_loop", new=fake_loop),
            patch("app.rag.retrieve.embed_query_async", new=AsyncMock(return_value=[0.0])),
            patch("app.rag.retrieve.search", new=AsyncMock(return_value=[MagicMock(payload=hit)])),
        ):
            result = await AgenticRAG().run("q", top_k=8, model="claude-sonnet-5", prompt_version="default")

        assert result.answer == "A" and result.refusal is False
        assert result.extra["retrieved_ids"] == ["c1"]
        assert result.extra["retrieved_docs"] == ["f.md"]
        assert "chunk text" in result.extra["context_text"]


# --------------------------------------------------------------------------
# Error hygiene, trace ids, store errors
# --------------------------------------------------------------------------


@pytest.mark.asyncio
class TestErrorsAndTraces:
    async def test_provider_error_text_is_not_returned(self):
        from app.rag.generate import PROVIDER_UNAVAILABLE_MESSAGE, answer_question

        strat = MagicMock()
        strat.run = AsyncMock(side_effect=ProviderError("upstream body: secret-token-123"))
        with (
            patch("app.rag.generate.store_count", new=AsyncMock(return_value=5)),
            patch("app.rag.generate.get_strategy", return_value=strat),
        ):
            ans = await answer_question("q")

        assert "secret-token-123" not in ans.answer
        assert ans.answer == PROVIDER_UNAVAILABLE_MESSAGE

    async def test_a_missing_key_message_is_still_shown(self):
        from app.rag.generate import answer_question

        strat = MagicMock()
        strat.run = AsyncMock(side_effect=MissingKeyError("ANTHROPIC_API_KEY is not configured."))
        with (
            patch("app.rag.generate.store_count", new=AsyncMock(return_value=5)),
            patch("app.rag.generate.get_strategy", return_value=strat),
        ):
            ans = await answer_question("q")
        assert "ANTHROPIC_API_KEY" in ans.answer

    async def test_answer_carries_a_retrievable_trace_id(self):
        from app.rag.generate import answer_question
        from app.tracing import get_trace

        strat = MagicMock()
        strat.run = AsyncMock(
            return_value=StrategyResult(answer="a", sources=[Source(chunk_id="c", quote="q")])
        )
        with (
            patch("app.rag.generate.store_count", new=AsyncMock(return_value=5)),
            patch("app.rag.generate.get_strategy", return_value=strat),
        ):
            ans = await answer_question("q")

        assert ans.trace_id and get_trace(ans.trace_id) is not None

    async def test_compare_rows_carry_a_trace_id(self):
        from app.rag.generate import run_strategy_raw
        from app.tracing import get_trace

        strat = MagicMock()
        strat.run = AsyncMock(
            return_value=StrategyResult(answer="a", sources=[Source(chunk_id="c", quote="q")])
        )
        with (
            patch("app.rag.generate.store_count", new=AsyncMock(return_value=5)),
            patch("app.rag.generate.get_strategy", return_value=strat),
        ):
            result, err = await run_strategy_raw("q", strategy="classic")

        assert err is None and result is not None
        assert get_trace(result.trace_id) is not None

    async def test_store_unavailable_propagates(self):
        from app.rag.generate import answer_question

        class StoreUnavailable(RuntimeError):
            pass

        with (
            patch("app.rag.generate.store_count", new=AsyncMock(side_effect=StoreUnavailable())),
            pytest.raises(StoreUnavailable),
        ):
            await answer_question("q")


def test_http_provider_error_hides_the_exception_text():
    from fastapi.testclient import TestClient

    from app.main import app

    with patch(
        "app.api.ask.answer_question",
        new=AsyncMock(side_effect=ProviderError("upstream body: secret-token-123")),
    ):
        r = TestClient(app, raise_server_exceptions=False).post("/ask", json={"question": "q"})

    assert r.status_code == 503
    assert "secret-token-123" not in r.text


def test_trace_endpoint_serves_a_returned_trace_id(client):
    from app.tracing import start_trace

    with start_trace(name="t", inputs={}) as t:
        pass
    assert client.get(f"/traces/{t.id}").status_code == 200


# --------------------------------------------------------------------------
# Rerank cooldown
# --------------------------------------------------------------------------


class TestRerankCooldown:
    """A cross-encoder load failure used to be cached for the process lifetime."""

    def test_retries_after_the_cooldown(self, monkeypatch):
        import sys

        from app.rag import rerank

        monkeypatch.setattr(rerank, "_cross_encoder_model", None)
        monkeypatch.setattr(rerank, "_cross_encoder_failed_at", None)
        fake_module = MagicMock()
        fake_module.CrossEncoder.side_effect = OSError("download interrupted")
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

        assert rerank._load_cross_encoder() is None
        assert rerank._load_cross_encoder() is None  # cooling down: not retried
        assert fake_module.CrossEncoder.call_count == 1

        monkeypatch.setattr(
            rerank,
            "_cross_encoder_failed_at",
            rerank._cross_encoder_failed_at - rerank.CROSS_ENCODER_RETRY_SECONDS - 1,
        )
        fake_module.CrossEncoder.side_effect = None
        fake_module.CrossEncoder.return_value = "model"
        assert rerank._load_cross_encoder() == "model"
        assert fake_module.CrossEncoder.call_count == 2
        monkeypatch.setattr(rerank, "_cross_encoder_model", None)


# --------------------------------------------------------------------------
# Eval methodology
# --------------------------------------------------------------------------


@pytest.mark.asyncio
class TestJudgeInputs:
    async def test_judge_sees_full_context_ideal_answer_and_temperature_zero(self):
        from app.eval.judge import judge

        long_context = [f"chunk {i} " + "x" * 400 for i in range(8)]
        reply = {
            "text": json.dumps(
                {
                    "faithfulness": 1,
                    "answer_relevance": 1,
                    "context_precision": 1,
                    "context_recall": 1,
                    "answer_correctness": 0.25,
                }
            ),
            "input_tokens": 1,
            "output_tokens": 1,
        }
        gen = AsyncMock(return_value=reply)
        with patch("app.eval.judge.generate_with_usage", new=gen):
            scores = await judge("q", "a", long_context, ideal_answer="THE IDEAL")

        prompt = gen.await_args.kwargs["prompt"]
        assert "chunk 7" in prompt  # the old judge saw only the first five
        assert "THE IDEAL" in prompt
        assert gen.await_args.kwargs["temperature"] == 0.0
        assert scores["answer_correctness"] == 0.25

    async def test_correctness_is_none_without_an_ideal_answer(self):
        from app.eval.judge import judge

        reply = {
            "text": json.dumps(
                {"faithfulness": 1, "answer_relevance": 1, "context_precision": 1, "context_recall": 1}
            ),
            "input_tokens": 1,
            "output_tokens": 1,
        }
        gen = AsyncMock(return_value=reply)
        with patch("app.eval.judge.generate_with_usage", new=gen):
            scores = await judge("q", "a", ["c"])

        assert scores["answer_correctness"] is None
        assert "answer_correctness" not in gen.await_args.kwargs["prompt"]

    async def test_score_example_passes_generator_context_and_ideal(self):
        from app.eval.metrics import score_example

        judge_mock = AsyncMock(return_value=None)
        with patch("app.eval.metrics.judge", new=judge_mock):
            await score_example(
                {"question": "q", "ideal_answer": "ideal"},
                {"answer": "a", "sources": [{"quote": "short preview"}]},
                "FULL CONTEXT",
            )
        args = judge_mock.await_args
        assert args.args[2] == ["FULL CONTEXT"]
        assert args.kwargs["ideal_answer"] == "ideal"


@pytest.fixture
def one_example(tmp_path, monkeypatch):
    monkeypatch.setattr("app.eval.metrics.GOLDEN_DIR", tmp_path)
    monkeypatch.setattr("app.eval.metrics.RUNS_DIR", tmp_path / "runs")
    (tmp_path / "d.jsonl").write_text(
        '{"question": "one", "ideal_answer": "i", "expected_sources": ["b.md"]}\n',
        encoding="utf-8",
    )
    return tmp_path


def _detailed(docs=("b.md",), model="claude-sonnet-5"):
    ans = Answer(
        question="one",
        answer="a",
        sources=[Source(chunk_id="c0", quote="q", document="a.md")],
        confidence=0.9,
        provider="anthropic",
        model=model,
    )
    result = StrategyResult(
        answer="a",
        sources=ans.sources,
        extra={
            "retrieved_ids": ["c0", "c1"],
            "retrieved_docs": ["a.md", *docs],
            "context_text": "CTX",
        },
    )
    return ans, result


@pytest.mark.asyncio
class TestEvalRuns:
    async def test_retrieval_is_scored_on_the_full_context(self, one_example):
        """Cited sources alone (a.md) would miss b.md, which was in the context."""
        from app.eval.metrics import run_evaluation

        with (
            patch("app.eval.metrics.answer_question_detailed", new=AsyncMock(return_value=_detailed())),
            patch("app.eval.metrics.judge", new=AsyncMock(return_value=None)),
        ):
            result = await run_evaluation(dataset="d")

        ex = result["per_example"][0]
        assert ex["retrieval"]["hit"] is True
        assert ex["retrieval"]["mrr"] == 0.5
        assert ex["retrieved_ids"] == ["c0", "c1"]

    async def test_run_records_the_model_actually_used(self, one_example):
        from app.eval.metrics import run_evaluation

        with (
            patch(
                "app.eval.metrics.answer_question_detailed",
                new=AsyncMock(return_value=_detailed(model="claude-sonnet-5")),
            ),
            patch("app.eval.metrics.judge", new=AsyncMock(return_value=None)),
        ):
            result = await run_evaluation(dataset="d")  # no model override

        assert result["model"] == "claude-sonnet-5"
        assert result["requested_model"] is None
        assert result["provider"] == "anthropic"

    async def test_run_files_never_collide(self, one_example):
        from app.eval.metrics import run_evaluation

        with (
            patch("app.eval.metrics.answer_question_detailed", new=AsyncMock(return_value=_detailed())),
            patch("app.eval.metrics.judge", new=AsyncMock(return_value=None)),
        ):
            await run_evaluation(dataset="d")
            await run_evaluation(dataset="d")

        names = [p.name for p in (one_example / "runs").glob("*.json")]
        assert len(names) == 2
        assert all("classic" in n and "claude-sonnet-5" in n for n in names)

    async def test_aggregate_handles_missing_correctness(self):
        from app.eval.metrics import aggregate

        base = {"faithfulness": 1, "answer_relevance": 1, "context_precision": 1, "context_recall": 1}
        agg = aggregate([{**base, "answer_correctness": 0.5}, {**base, "answer_correctness": None}])
        assert agg.answer_correctness == 0.5
        assert aggregate([{**base, "answer_correctness": None}]).answer_correctness is None


def test_regressions_do_not_compare_across_judges(tmp_path, monkeypatch):
    from app.eval.regression import load_regressions

    monkeypatch.setattr("app.eval.regression.RUNS_DIR", tmp_path)
    for name, judge_model, faith in (("20260101T000000Z-a", "j1", 0.9), ("20260102T000000Z-b", "j2", 0.3)):
        (tmp_path / f"{name}.json").write_text(
            json.dumps(
                {
                    "dataset": "d",
                    "strategy": "classic",
                    "model": "m",
                    "judge_model": judge_model,
                    "aggregate": {"faithfulness": faith},
                }
            ),
            encoding="utf-8",
        )
    assert load_regressions() == []


class TestDataDir:
    def test_data_dir_setting_wins(self, tmp_path):
        assert Settings(_env_file=None, data_dir=str(tmp_path)).data_path == tmp_path

    def test_defaults_to_the_repo_data_directory(self):
        repo_data = Path(__file__).resolve().parents[2] / "data"
        assert Settings(_env_file=None).data_path == repo_data


# --------------------------------------------------------------------------
# Auth: atomic rate limiting and cost weighting
# --------------------------------------------------------------------------


@pytest.fixture
def sqlite_auth(tmp_path, monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(f"sqlite:///{tmp_path / 'auth.db'}")
    auth.Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(auth, "_engine", engine)
    monkeypatch.setattr(
        auth, "_SessionLocal", sessionmaker(autocommit=False, autoflush=False, bind=engine)
    )
    yield engine
    auth.Base.metadata.drop_all(bind=engine)


def _request(path="/ask"):
    req = MagicMock()
    req.url.path = path
    return req


def _principal(settings, monkeypatch, rpm):
    monkeypatch.setattr(settings, "require_api_key", True)
    key = auth.create_api_key("ci", requests_per_minute=rpm)
    return key, auth.require_api_key(_request(), authorization=f"Bearer {key}")


class TestRateLimitUnits:
    def test_charge_reprices_the_usage_row(self, sqlite_auth, settings, monkeypatch):
        _, principal = _principal(settings, monkeypatch, rpm=10)
        auth.charge(principal, 3)
        with auth.session_scope() as db:
            assert db.get(auth.APIKeyUsage, principal["usage_id"]).units == 3

    def test_weighted_requests_exhaust_the_budget(self, sqlite_auth, settings, monkeypatch):
        key, principal = _principal(settings, monkeypatch, rpm=4)
        auth.charge(principal, 3)  # e.g. one agentic run
        auth.require_api_key(_request(), authorization=f"Bearer {key}")  # 4th unit
        with pytest.raises(HTTPException) as exc:
            auth.require_api_key(_request(), authorization=f"Bearer {key}")
        assert exc.value.status_code == 429

    def test_charge_rejects_when_it_would_exceed(self, sqlite_auth, settings, monkeypatch):
        key, _ = _principal(settings, monkeypatch, rpm=5)
        second = auth.require_api_key(_request(), authorization=f"Bearer {key}")
        with pytest.raises(HTTPException) as exc:
            auth.charge(second, 5)
        assert exc.value.status_code == 429

    def test_an_oversized_request_is_admitted_on_an_empty_window(
        self, sqlite_auth, settings, monkeypatch
    ):
        """A 34-example eval must not be impossible on a 10/min key."""
        key, principal = _principal(settings, monkeypatch, rpm=10)
        auth.charge(principal, 34)
        with pytest.raises(HTTPException):
            auth.require_api_key(_request(), authorization=f"Bearer {key}")

    def test_charge_is_a_noop_when_anonymous(self):
        auth.charge({"id": None, "usage_id": None}, 50)

    def test_strategy_units(self):
        assert auth.strategy_units("agentic") == 3
        assert auth.strategy_units("classic") == 1


class TestAtomicity:
    def test_the_key_row_is_locked_for_the_check(self):
        """Check-then-insert raced; the key row is now locked FOR UPDATE."""
        from sqlalchemy.dialects import postgresql

        captured = {}
        db = MagicMock()

        def execute(stmt):
            captured["sql"] = str(stmt.compile(dialect=postgresql.dialect()))
            raise RuntimeError("stop")

        db.execute.side_effect = execute
        with pytest.raises(RuntimeError):
            auth._authenticate(db, "sk_x", lock=True)
        assert "FOR UPDATE" in captured["sql"]

    def test_dependencies_do_not_block_the_event_loop(self):
        import inspect

        assert not inspect.iscoroutinefunction(auth.require_api_key)

    def test_admin_alias(self):
        assert auth.require_admin is auth.require_admin_key


def test_legacy_usage_table_gains_the_units_column(tmp_path):
    from sqlalchemy import create_engine, inspect, text

    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE api_key_usage (id INTEGER PRIMARY KEY, api_key_id INTEGER, "
                "timestamp DATETIME, endpoint VARCHAR(256), tokens_input INTEGER, "
                "tokens_output INTEGER, model VARCHAR(256))"
            )
        )
    auth._add_missing_columns(engine)
    assert "units" in {c["name"] for c in inspect(engine).get_columns("api_key_usage")}


def test_compare_strategies_is_charged_per_strategy(client):
    charged = MagicMock()
    with (
        patch("app.api.compare.charge", new=charged),
        patch("app.api.compare.run_strategy_raw", new=AsyncMock(return_value=(None, "x"))),
    ):
        r = client.post(
            "/compare/strategies", json={"question": "q", "strategies": ["classic", "agentic"]}
        )
    assert r.status_code == 200
    assert charged.call_args.args[1] == 4


def test_eval_run_is_charged_per_example(client):
    charged = MagicMock()
    with (
        patch("app.api.eval_routes.charge", new=charged),
        patch("app.api.eval_routes.load_dataset", return_value=[{}, {}, {}]),
        patch("app.api.eval_routes.run_evaluation", new=AsyncMock(return_value={"cost": {}})),
    ):
        r = client.post("/eval/run", json={"dataset": "d", "strategy": "agentic"})
    assert r.status_code == 200
    assert charged.call_args.args[1] == 9
