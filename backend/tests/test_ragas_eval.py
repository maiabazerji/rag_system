"""RAGAS scoring: metric mapping, isolation, skips, and the --scorer wiring.

No test calls a model: metrics are fakes injected into ``score_with_ragas``,
and the harness tests patch ``score_with_ragas`` itself.
"""
import builtins
import importlib
import json
import sys
from unittest.mock import AsyncMock, patch

import pytest

from app.eval import metrics, ragas_eval
from app.eval.judge import JudgeOutcome
from app.eval.metrics import run_evaluation
from app.eval.ragas_eval import (
    RAGAS_METRICS,
    REFERENCE_METRICS,
    RagasOutcome,
    score_with_ragas,
)
from app.eval.regression import _config_key
from app.rag.strategies.base import StrategyResult
from app.schemas import Answer, Source


class FakeResult:
    def __init__(self, value):
        self.value = value


class FakeMetric:
    """Records its call and returns a fixed value (or raises)."""

    def __init__(self, value=0.8, error: Exception | None = None):
        self.value, self.error, self.calls = value, error, []

    async def ascore(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return FakeResult(self.value)


def fakes(**overrides) -> dict[str, FakeMetric]:
    out = {name: FakeMetric(0.5 + i / 10) for i, name in enumerate(RAGAS_METRICS)}
    out.update(overrides)
    return out


@pytest.mark.asyncio
class TestScoreWithRagas:
    async def test_scores_every_metric_under_ragas_names(self):
        m = fakes()
        out = await score_with_ragas("q", "an answer", ["c1", "c2"], reference="ref", metrics=m)
        assert set(out.scores) == set(RAGAS_METRICS)
        assert out.scores["ragas_faithfulness"] == 0.5
        assert not out.errors and not out.skipped
        # Each metric gets the arguments its RAGAS class expects.
        assert m["ragas_faithfulness"].calls[0] == {
            "user_input": "q", "response": "an answer", "retrieved_contexts": ["c1", "c2"],
        }
        assert m["ragas_answer_relevancy"].calls[0] == {"user_input": "q", "response": "an answer"}
        assert m["ragas_context_recall"].calls[0]["reference"] == "ref"

    async def test_reference_metrics_skipped_without_ideal_answer(self):
        m = fakes()
        out = await score_with_ragas("q", "a", ["c"], reference=None, metrics=m)
        assert set(out.scores) == {"ragas_faithfulness", "ragas_answer_relevancy"}
        assert set(out.skipped) == REFERENCE_METRICS
        assert all(not m[n].calls for n in REFERENCE_METRICS)

    async def test_one_failing_metric_does_not_drop_the_others(self):
        m = fakes(ragas_answer_relevancy=FakeMetric(error=RuntimeError("rate limited")))
        out = await score_with_ragas("q", "a", ["c"], reference="r", metrics=m)
        assert "ragas_answer_relevancy" not in out.scores
        assert out.errors["ragas_answer_relevancy"] == "RuntimeError: rate limited"
        assert len(out.scores) == len(RAGAS_METRICS) - 1
        assert out.error is None  # something was measured

    async def test_everything_failing_is_reported_not_raised(self):
        err = RuntimeError("credit balance is too low")
        m = {n: FakeMetric(error=err) for n in RAGAS_METRICS}
        out = await score_with_ragas("q", "a", ["c"], reference="r", metrics=m)
        assert out.scores == {}
        assert out.error_type == "RuntimeError"
        assert "credit balance" in out.error

    async def test_nan_is_not_a_score(self):
        m = fakes(ragas_faithfulness=FakeMetric(float("nan")))
        out = await score_with_ragas("q", "a", ["c"], metrics=m)
        assert "ragas_faithfulness" not in out.scores
        assert out.errors["ragas_faithfulness"] == "no numeric score returned"

    async def test_nothing_to_score_without_contexts_or_answer(self):
        m = fakes()
        no_ctx = await score_with_ragas("q", "a", ["", "  "], metrics=m)
        assert no_ctx.scores == {} and set(no_ctx.skipped.values()) == {"no retrieved contexts"}
        no_answer = await score_with_ragas("q", "  ", ["c"], metrics=m)
        assert set(no_answer.skipped.values()) == {"no answer"}
        assert all(not f.calls for f in m.values())

    async def test_missing_ragas_is_a_clean_error(self, monkeypatch):
        monkeypatch.setattr(ragas_eval, "ragas_available", lambda: False)
        out = await score_with_ragas("q", "a", ["c"])
        assert out.error_type == "RagasNotInstalled"
        assert "backend[ragas]" in out.error


def test_module_imports_without_ragas(monkeypatch):
    """The harness imports this module unconditionally, so it must not need RAGAS."""
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "ragas" or name.startswith("ragas."):
            raise ImportError("no ragas")
        return real_import(name, *args, **kwargs)

    for mod in [m for m in sys.modules if m == "ragas" or m.startswith("ragas.")]:
        monkeypatch.delitem(sys.modules, mod)
    monkeypatch.setattr(builtins, "__import__", blocked)
    reloaded = importlib.reload(ragas_eval)
    assert reloaded.RAGAS_METRICS == RAGAS_METRICS
    monkeypatch.setattr(builtins, "__import__", real_import)
    importlib.reload(ragas_eval)


# --- the harness ---------------------------------------------------------------

DATASET = [
    {"id": "a", "question": "qa", "ideal_answer": "A", "relevant_doc_ids": ["a"]},
    {"id": "b", "question": "qb", "relevant_doc_ids": ["b"]},
    {"id": "c", "question": "qc", "relevant_doc_ids": ["c"]},
    {"id": "e", "question": "qe", "expected_behavior": "refuse"},
]


def _pipeline(question, **kwargs):
    ans = Answer(
        question=question,
        answer=f"answer to {question}",
        sources=[Source(chunk_id=f"{question}:0", quote="q", document=f"{question}.md")],
        confidence=0.8,
        refusal=False,
        provider="anthropic",
        model="gen-model",
        latency_ms=100,
        input_tokens=90,
        output_tokens=10,
    )
    result = StrategyResult(
        answer=ans.answer,
        sources=ans.sources,
        refusal=False,
        extra={
            "retrieved_ids": [f"{question}:0"],
            "retrieved_docs": [f"{question}.md"],
            "context_texts": [f"chunk one for {question}", f"chunk two for {question}"],
            "context_text": f"context for {question}",
        },
    )
    return ans, result


def _judge(expected, got, context_text=None):
    return JudgeOutcome(
        scores={"faithfulness": 1.0, "answer_relevance": 1.0, "context_precision": 1.0,
                "context_recall": 1.0, "answer_correctness": None},
        reasoning={}, attempts=1, input_tokens=10, output_tokens=2, judge_model="judge-model",
    )


async def _ragas(question, answer, contexts, reference=None, **kwargs):
    if question == "qc":  # RAGAS fails entirely on this one
        return RagasOutcome(error_type="BadRequestError", error="credit balance is too low",
                            errors={"ragas_faithfulness": "BadRequestError: credit"})
    scores = {"ragas_faithfulness": 0.9, "ragas_answer_relevancy": 0.7}
    if reference:
        scores["ragas_answer_correctness"] = 0.6
    assert contexts == [f"chunk one for {question}", f"chunk two for {question}"]
    return RagasOutcome(scores=scores, ragas_version="0.4.3")


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(metrics, "GOLDEN_DIR", tmp_path)
    (tmp_path / "rg.jsonl").write_text(
        "\n".join(json.dumps(r) for r in DATASET) + "\n", encoding="utf-8"
    )
    judge_mock = AsyncMock(side_effect=_judge)
    ragas_mock = AsyncMock(side_effect=_ragas)

    async def go(**kwargs):
        with (
            patch("app.eval.metrics.answer_question_detailed", new=AsyncMock(side_effect=_pipeline)),
            patch("app.eval.metrics.score_example_detailed", new=judge_mock),
            patch("app.eval.metrics.score_with_ragas", new=ragas_mock),
        ):
            return await run_evaluation(dataset="rg", **kwargs)

    go.judge, go.ragas = judge_mock, ragas_mock
    return go


def _row(result, rid):
    return next(r for r in result["per_example"] if r["id"] == rid)


@pytest.mark.asyncio
class TestHarness:
    async def test_default_scorer_never_calls_ragas(self, run):
        r = await run()
        assert r["scorer"] == "judge" and r["ragas_version"] is None
        assert run.ragas.await_count == 0
        assert not any(m.startswith("ragas_") for m in r["aggregates"])

    async def test_ragas_only(self, run):
        r = await run(scorer="ragas")
        assert run.judge.await_count == 0
        assert r["judged"] is False and r["judge_model"] is None
        assert r["scorer"] == "ragas"
        # a, b scored by RAGAS; c failed in RAGAS; e is a deterministic refusal check.
        assert [_row(r, x)["status"] for x in "abce"] == [
            "scored", "scored", "judge_failed", "scored",
        ]
        failure = next(f for f in r["failures"] if f["id"] == "c")
        assert failure["stage"] == "ragas" and "credit" in failure["error"]
        agg = r["aggregates"]
        assert agg["ragas_faithfulness"] == {"mean": 0.9, "std": 0.0, "n": 2}
        assert agg["ragas_answer_correctness"]["n"] == 1  # only "a" has an ideal answer
        assert "faithfulness" not in agg  # no judge dimension invented
        assert _row(r, "a")["ragas"]["scores"]["ragas_answer_correctness"] == 0.6

    async def test_both_scores_the_same_answers_twice(self, run):
        r = await run(scorer="both")
        assert run.judge.await_count == 3 and run.ragas.await_count == 3
        agg = r["aggregates"]
        assert agg["faithfulness"]["n"] == 3
        assert agg["ragas_faithfulness"]["n"] == 2
        # The judge decides the status; RAGAS's failure on "c" stays on the row.
        assert _row(r, "c")["status"] == "scored"
        assert _row(r, "c")["ragas"]["error_type"] == "BadRequestError"
        order = list(agg)
        assert order.index("faithfulness") < order.index("ragas_faithfulness") < order.index(
            "latency_ms"
        )

    async def test_unknown_scorer_is_rejected(self, run):
        with pytest.raises(ValueError, match="scorer"):
            await run(scorer="magic")


def test_scorer_joins_the_regression_key_only_when_not_judge():
    base = {"dataset": "d", "strategy": "classic", "config_hash": "h"}
    assert _config_key(base) == _config_key({**base, "scorer": "judge"})
    assert len(_config_key({**base, "scorer": "ragas"})) == len(_config_key(base)) + 1
    assert _config_key({**base, "scorer": "ragas"}) != _config_key({**base, "scorer": "both"})


@pytest.mark.asyncio
class TestSamplingSafeClient:
    async def _call(self, model):
        seen = {}

        class Messages:
            async def create(self, **kwargs):
                seen.update(kwargs)
                return "ok"

        class Client:
            messages = Messages()

        client = ragas_eval._sampling_safe_client(Client())
        assert await client.messages.create(model=model, temperature=0.01, max_tokens=5) == "ok"
        return seen

    async def test_dropped_for_models_that_reject_sampling(self):
        seen = await self._call("claude-opus-5-5")
        assert "temperature" not in seen and "extra_body" not in seen
        assert seen["max_tokens"] == 5

    async def test_sent_as_extra_body_for_models_that_accept_it(self):
        seen = await self._call("claude-haiku-4-5")
        assert "temperature" not in seen
        assert seen["extra_body"] == {"temperature": 0.01}
