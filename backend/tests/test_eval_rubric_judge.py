"""The versioned rubric and the judge's structured-output contract.

The judge's provider is mocked throughout: these tests pin what the judge does
with a reply (validate, retry once with a repair message, or fail with the
error recorded), never what a real model says.
"""
import json
from unittest.mock import AsyncMock, patch

import pytest

from app.eval import rubric
from app.eval.judge import (
    INVALID_OUTPUT,
    JudgeVerdict,
    _build_prompt,
    judge_detailed,
    validate_verdict,
)

GOOD = {
    "faithfulness": {"score": 1.0, "reasoning": "all claims in [0]"},
    "answer_relevance": {"score": 0.75, "reasoning": "answers, with padding"},
    "context_precision": {"score": 0.5, "reasoning": "half relevant"},
    "context_recall": {"score": 0.25, "reasoning": "little present"},
}
WITH_REF = {**GOOD, "answer_correctness": {"score": 0.0, "reasoning": "contradicts"}}


def _reply(payload, tokens=(100, 20)):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return {"text": text, "input_tokens": tokens[0], "output_tokens": tokens[1]}


class TestRubric:
    def test_fingerprint_is_pinned_to_the_version(self):
        """Edit the rubric text -> bump RUBRIC_VERSION and update this pin."""
        assert rubric.RUBRIC_VERSION == "2026-09.v2"
        assert rubric.rubric_fingerprint() == "4827ab327e75bb8d"

    def test_every_dimension_describes_every_anchor(self):
        for dim in rubric.DIMENSIONS_SPEC:
            assert sorted(dim.anchors) == list(rubric.ANCHORS), dim.name
            assert all(text.strip() for text in dim.anchors.values())
            assert dim.definition.strip()

    def test_correctness_needs_a_reference(self):
        assert "answer_correctness" not in {d.name for d in rubric.dimensions_for(False)}
        assert "answer_correctness" in {d.name for d in rubric.dimensions_for(True)}

    def test_prompt_embeds_the_rubric_and_json_shape(self):
        prompt = _build_prompt("q", "a", ["ctx"], "ref")
        assert rubric.render_rubric(True) in prompt
        assert rubric.RUBRIC_VERSION in prompt
        assert '"answer_correctness": {"score": <0-1>, "reasoning": "<why>"}' in prompt
        for anchor in ("0.00 =", "0.25 =", "0.50 =", "0.75 =", "1.00 ="):
            assert anchor in prompt

    def test_prompt_without_reference_omits_correctness(self):
        prompt = _build_prompt("q", "a", ["ctx"], None)
        assert "answer_correctness" not in prompt
        assert "Reference Answer" not in prompt


class TestValidateVerdict:
    def test_accepts_a_complete_verdict(self):
        verdict = validate_verdict(json.dumps(GOOD), has_reference=False)
        assert isinstance(verdict, JudgeVerdict)
        assert verdict.scores()["answer_relevance"] == 0.75
        assert verdict.scores()["answer_correctness"] is None
        assert verdict.reasoning()["context_recall"] == "little present"

    def test_integers_are_scores(self):
        payload = {**GOOD, "faithfulness": {"score": 1, "reasoning": "r"}}
        assert validate_verdict(json.dumps(payload), False).faithfulness.score == 1.0

    @pytest.mark.parametrize(
        "bad, fragment",
        [
            ({**GOOD, "faithfulness": {"score": 1.2, "reasoning": "r"}}, "faithfulness.score"),
            ({**GOOD, "faithfulness": {"score": -0.1, "reasoning": "r"}}, "faithfulness.score"),
            ({**GOOD, "faithfulness": {"score": "0.9", "reasoning": "r"}}, "faithfulness.score"),
            ({**GOOD, "faithfulness": {"score": True, "reasoning": "r"}}, "faithfulness.score"),
            ({**GOOD, "faithfulness": {"score": 0.9, "reasoning": ""}}, "faithfulness.reasoning"),
            ({**GOOD, "faithfulness": {"score": 0.9}}, "faithfulness.reasoning"),
            ({**GOOD, "faithfulness": 0.9}, "faithfulness"),
            ({k: v for k, v in GOOD.items() if k != "context_recall"}, "context_recall"),
        ],
    )
    def test_rejects_invalid_verdicts_naming_the_field(self, bad, fragment):
        with pytest.raises(ValueError, match=fragment.replace(".", r"\.")):
            validate_verdict(json.dumps(bad), has_reference=False)

    def test_correctness_required_with_a_reference(self):
        with pytest.raises(ValueError, match="answer_correctness"):
            validate_verdict(json.dumps(GOOD), has_reference=True)

    def test_correctness_dropped_without_a_reference(self):
        verdict = validate_verdict(json.dumps(WITH_REF), has_reference=False)
        assert verdict.answer_correctness is None

    def test_non_json_is_rejected(self):
        with pytest.raises(ValueError):
            validate_verdict("pretty good answer", has_reference=False)


@pytest.mark.asyncio
class TestJudgeRetryAndFailure:
    async def test_valid_first_reply(self):
        gen = AsyncMock(return_value=_reply(WITH_REF))
        with patch("app.eval.judge.generate_with_usage", new=gen):
            outcome = await judge_detailed("q", "a", ["c"], ideal_answer="ref", model="judge-x")

        assert outcome.ok
        assert outcome.attempts == 1
        assert outcome.scores["answer_correctness"] == 0.0
        assert outcome.reasoning["faithfulness"] == "all claims in [0]"
        assert outcome.judge_model == "judge-x"
        assert outcome.rubric_version == rubric.RUBRIC_VERSION
        assert (outcome.input_tokens, outcome.output_tokens) == (100, 20)
        assert gen.await_args.kwargs["temperature"] == 0.0

    async def test_one_repair_retry_then_success(self):
        bad = {**GOOD, "faithfulness": {"score": 7, "reasoning": "out of range"}}
        gen = AsyncMock(side_effect=[_reply(bad), _reply(GOOD)])
        with patch("app.eval.judge.generate_with_usage", new=gen):
            outcome = await judge_detailed("q", "a", ["c"])

        assert outcome.ok
        assert outcome.attempts == 2
        assert (outcome.input_tokens, outcome.output_tokens) == (200, 40)
        repair = gen.await_args_list[1].kwargs["prompt"]
        assert "could not be used" in repair
        assert "faithfulness.score" in repair  # the validation error is quoted back
        assert '"score": 7' in repair  # and so is the invalid reply

    async def test_two_invalid_replies_fail_without_a_score(self):
        """Never a clamped, defaulted or midpoint number."""
        gen = AsyncMock(return_value=_reply("I'd say about 0.5 overall"))
        with patch("app.eval.judge.generate_with_usage", new=gen):
            outcome = await judge_detailed("q", "a", ["c"])

        assert not outcome.ok
        assert outcome.scores is None
        assert outcome.attempts == 2
        assert gen.await_count == 2  # exactly one retry
        assert outcome.error_type == INVALID_OUTPUT
        assert "JSON" in outcome.error

    async def test_provider_error_fails_immediately(self):
        gen = AsyncMock(side_effect=RuntimeError("API down"))
        with patch("app.eval.judge.generate_with_usage", new=gen):
            outcome = await judge_detailed("q", "a", ["c"])

        assert outcome.scores is None
        assert outcome.error_type == "RuntimeError"
        assert outcome.error == "API down"
        assert outcome.attempts == 1

    async def test_provider_error_on_the_retry_is_recorded(self):
        gen = AsyncMock(side_effect=[_reply("nope"), TimeoutError("slow")])
        with patch("app.eval.judge.generate_with_usage", new=gen):
            outcome = await judge_detailed("q", "a", ["c"])

        assert outcome.scores is None
        assert outcome.error_type == "TimeoutError"
        assert outcome.attempts == 2

    async def test_outcome_serializes(self):
        gen = AsyncMock(return_value=_reply(GOOD))
        with patch("app.eval.judge.generate_with_usage", new=gen):
            outcome = await judge_detailed("q", "a", ["c"])
        assert json.loads(json.dumps(outcome.to_dict()))["scores"]["faithfulness"] == 1.0
