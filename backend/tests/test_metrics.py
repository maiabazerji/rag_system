"""Tests for the evaluation harness.

The central property here is the failure policy: a judge that cannot score an
answer must produce no score at all, never a plausible midpoint. On an
evaluation platform, a fabricated 0.5 is worse than a missing number.
"""
import json
from unittest.mock import AsyncMock, patch

import pytest

from app.eval.judge import _extract_json, judge
from app.eval.metrics import aggregate, load_dataset, run_evaluation, score_example
from app.eval.regression import load_regressions
from app.schemas import Answer, EvalScore, Source

SCORES = {
    "faithfulness": 0.9,
    "answer_relevance": 0.8,
    "context_precision": 0.7,
    "context_recall": 0.6,
}


def _fake_answer(text="a"):
    """A real Answer: _run_example reads its attributes, not just model_dump()."""
    return Answer(
        question="q",
        answer=text,
        sources=[Source(chunk_id="none", quote="")],
        confidence=0.9,
        model="claude-sonnet-5",
        latency_ms=1000,
        input_tokens=500,
        output_tokens=100,
    )


def _judge_reply(payload: str, tokens=(10, 10)):
    return {"text": payload, "input_tokens": tokens[0], "output_tokens": tokens[1]}


class TestExtractJson:
    def test_plain_object(self):
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_object(self):
        assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_object_wrapped_in_prose(self):
        assert _extract_json('Sure!\n{"a": 1}\nHope that helps.') == {"a": 1}

    def test_rejects_non_json(self):
        with pytest.raises(ValueError):
            _extract_json("I am not JSON at all.")

    def test_rejects_a_bare_array(self):
        with pytest.raises(ValueError, match="expected an object"):
            _extract_json("[1, 2, 3]")


@pytest.mark.asyncio
class TestJudge:
    async def test_returns_scores_on_a_well_formed_reply(self):
        reply = _judge_reply(json.dumps({**SCORES, "reasoning": "looks right"}))
        with patch(
            "app.eval.judge.generate_with_usage", new=AsyncMock(return_value=reply)
        ):
            result = await judge("q", "a", ["ctx"])

        assert result is not None
        assert result["faithfulness"] == 0.9
        assert result["reasoning"] == "looks right"

    async def test_clamps_out_of_range_scores(self):
        reply = _judge_reply(
            json.dumps({**SCORES, "faithfulness": 1.7, "context_recall": -0.4})
        )
        with patch(
            "app.eval.judge.generate_with_usage", new=AsyncMock(return_value=reply)
        ):
            result = await judge("q", "a", ["ctx"])

        assert result["faithfulness"] == 1.0
        assert result["context_recall"] == 0.0

    async def test_returns_none_when_the_provider_fails(self):
        """The regression this guards: a failed judge must not become a 0.5."""
        with patch(
            "app.eval.judge.generate_with_usage",
            new=AsyncMock(side_effect=RuntimeError("API down")),
        ):
            assert await judge("q", "a", ["ctx"]) is None

    async def test_returns_none_on_unparseable_output(self):
        with patch(
            "app.eval.judge.generate_with_usage",
            new=AsyncMock(return_value=_judge_reply("the answer was pretty good")),
        ):
            assert await judge("q", "a", ["ctx"]) is None

    async def test_returns_none_when_a_dimension_is_missing(self):
        partial = {k: v for k, v in SCORES.items() if k != "context_recall"}
        with patch(
            "app.eval.judge.generate_with_usage",
            new=AsyncMock(return_value=_judge_reply(json.dumps(partial))),
        ):
            assert await judge("q", "a", ["ctx"]) is None

    async def test_returns_none_when_a_score_is_not_numeric(self):
        bad = {**SCORES, "faithfulness": "very high"}
        with patch(
            "app.eval.judge.generate_with_usage",
            new=AsyncMock(return_value=_judge_reply(json.dumps(bad))),
        ):
            assert await judge("q", "a", ["ctx"]) is None


@pytest.mark.asyncio
class TestScoreExample:
    async def test_builds_an_evalscore(self):
        with patch("app.eval.metrics.judge", new=AsyncMock(return_value=dict(SCORES))):
            score = await score_example(
                {"question": "q"}, {"answer": "a", "sources": [{"quote": "ctx"}]}
            )
        assert isinstance(score, EvalScore)
        assert score.faithfulness == 0.9

    async def test_propagates_an_unscorable_example_as_none(self):
        with patch("app.eval.metrics.judge", new=AsyncMock(return_value=None)):
            assert await score_example({"question": "q"}, {"answer": "a"}) is None


class TestAggregate:
    def test_averages_each_dimension(self):
        agg = aggregate([SCORES, dict.fromkeys(SCORES, 0.5)])
        assert agg.faithfulness == pytest.approx(0.7)
        assert agg.context_recall == pytest.approx(0.55)

    def test_returns_none_for_an_empty_list(self):
        assert aggregate([]) is None

    def test_single_score_round_trips(self):
        assert aggregate([SCORES]).faithfulness == 0.9


class TestLoadDataset:
    def test_missing_dataset_is_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.eval.metrics.GOLDEN_DIR", tmp_path)
        assert load_dataset("nope") == []

    def test_blank_lines_are_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.eval.metrics.GOLDEN_DIR", tmp_path)
        (tmp_path / "d.jsonl").write_text(
            '{"question": "a"}\n\n{"question": "b"}\n', encoding="utf-8"
        )
        assert len(load_dataset("d")) == 2

    def test_malformed_line_names_the_line_number(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.eval.metrics.GOLDEN_DIR", tmp_path)
        (tmp_path / "d.jsonl").write_text('{"question": "a"}\n{oops\n', encoding="utf-8")
        with pytest.raises(ValueError, match="line 2"):
            load_dataset("d")

    def test_the_shipped_golden_dataset_parses(self):
        examples = load_dataset("golden_v1")
        assert examples, "golden_v1 should ship with examples"
        assert all("question" in ex for ex in examples)


@pytest.mark.asyncio
class TestRunEvaluation:
    async def test_unscored_examples_are_counted_and_excluded(
        self, tmp_path, monkeypatch
    ):
        """Two examples, one unscorable: the aggregate must reflect only the scored one."""
        monkeypatch.setattr("app.eval.metrics.GOLDEN_DIR", tmp_path)
        monkeypatch.setattr("app.eval.metrics.RUNS_DIR", tmp_path / "runs")
        (tmp_path / "d.jsonl").write_text(
            '{"question": "one"}\n{"question": "two"}\n', encoding="utf-8"
        )

        answer = AsyncMock(return_value=_fake_answer())

        with (
            patch("app.eval.metrics.answer_question", new=answer),
            patch(
                "app.eval.metrics.score_example",
                new=AsyncMock(side_effect=[EvalScore(**SCORES), None]),
            ),
        ):
            result = await run_evaluation(dataset="d")

        assert result["n"] == 2
        assert result["n_scored"] == 1
        assert result["n_unscored"] == 1
        assert result["aggregate"]["faithfulness"] == 0.9
        assert [p["score"] for p in result["per_example"]].count(None) == 1

    async def test_aggregate_is_none_when_nothing_scored(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.eval.metrics.GOLDEN_DIR", tmp_path)
        monkeypatch.setattr("app.eval.metrics.RUNS_DIR", tmp_path / "runs")
        (tmp_path / "d.jsonl").write_text('{"question": "one"}\n', encoding="utf-8")

        answer = AsyncMock(return_value=_fake_answer())

        with (
            patch("app.eval.metrics.answer_question", new=answer),
            patch("app.eval.metrics.score_example", new=AsyncMock(return_value=None)),
        ):
            result = await run_evaluation(dataset="d")

        assert result["aggregate"] is None
        assert result["n_unscored"] == 1

    async def test_missing_dataset_reports_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.eval.metrics.GOLDEN_DIR", tmp_path)
        result = await run_evaluation(dataset="does-not-exist")
        assert result["n"] == 0
        assert "error" in result


class TestRegressions:
    """Runs are only comparable when their whole configuration matches."""

    @staticmethod
    def _write(dir_path, name, *, model, faithfulness, dataset="golden_v1"):
        (dir_path / f"{name}.json").write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "provider": "anthropic",
                    "model": model,
                    "prompt_version": "default",
                    "aggregate": {**SCORES, "faithfulness": faithfulness},
                    "created_at": "2026-01-01T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )

    def test_detects_a_drop_within_one_configuration(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.eval.regression.RUNS_DIR", tmp_path)
        self._write(tmp_path, "20260101T000000Z-a", model="m1", faithfulness=0.9)
        self._write(tmp_path, "20260102T000000Z-b", model="m1", faithfulness=0.5)

        found = load_regressions()
        assert len(found) == 1
        assert found[0]["metric"] == "faithfulness"
        assert found[0]["delta"] == pytest.approx(-0.4)

    def test_ignores_drops_between_different_models(self, tmp_path, monkeypatch):
        """The old detector reported this as a regression. It is not one."""
        monkeypatch.setattr("app.eval.regression.RUNS_DIR", tmp_path)
        self._write(tmp_path, "20260101T000000Z-a", model="haiku", faithfulness=0.9)
        self._write(tmp_path, "20260102T000000Z-b", model="opus", faithfulness=0.4)

        assert load_regressions() == []

    def test_ignores_drops_between_different_datasets(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.eval.regression.RUNS_DIR", tmp_path)
        self._write(tmp_path, "20260101T000000Z-a", model="m1", faithfulness=0.9)
        self._write(
            tmp_path, "20260102T000000Z-b", model="m1", faithfulness=0.4, dataset="v2"
        )

        assert load_regressions() == []

    def test_small_drops_stay_under_the_threshold(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.eval.regression.RUNS_DIR", tmp_path)
        self._write(tmp_path, "20260101T000000Z-a", model="m1", faithfulness=0.90)
        self._write(tmp_path, "20260102T000000Z-b", model="m1", faithfulness=0.88)

        assert load_regressions(threshold=0.05) == []
        assert len(load_regressions(threshold=0.01)) == 1

    def test_runs_without_an_aggregate_are_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.eval.regression.RUNS_DIR", tmp_path)
        self._write(tmp_path, "20260101T000000Z-a", model="m1", faithfulness=0.9)
        (tmp_path / "20260102T000000Z-b.json").write_text(
            json.dumps({"dataset": "golden_v1", "model": "m1", "aggregate": None}),
            encoding="utf-8",
        )

        assert load_regressions() == []

    def test_unreadable_files_do_not_break_the_listing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.eval.regression.RUNS_DIR", tmp_path)
        self._write(tmp_path, "20260101T000000Z-a", model="m1", faithfulness=0.9)
        (tmp_path / "20260102T000000Z-corrupt.json").write_text("{not json", encoding="utf-8")

        assert load_regressions() == []
