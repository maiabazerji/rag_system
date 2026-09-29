"""LLM-as-judge scoring for RAG answers.

Scores an answer along five dimensions using Claude as an impartial judge:

1. Faithfulness: are claims grounded in retrieved context (no hallucination)?
2. Answer Relevance: does the answer address the user's question?
3. Context Precision: are the retrieved chunks relevant and well ranked?
4. Context Recall: is all the information needed to answer present in context?
5. Answer Correctness: does the answer agree with the golden ideal answer?
   Only scored when the example has one; otherwise it is ``None``.

The rubric -- definitions and 0/0.25/0.5/0.75/1 anchors for each dimension --
lives in :mod:`app.eval.rubric` and is embedded verbatim in the prompt. Its
version is recorded with every judgement.

The judge sees the full context the generator actually used (not the short
source quotes shown in the UI), and runs at temperature 0 where the model
accepts it, so repeated runs of the same answer score the same.

Output contract: one JSON object with, per dimension, ``{"score": float in
[0, 1], "reasoning": non-empty string}``. The reply is validated with
:class:`JudgeVerdict`. The provider module exposes no forced-tool or
structured-output call for a single completion, so the contract is enforced
by validation rather than by the API: an invalid reply gets exactly one retry
carrying a repair message that quotes the validation error.

Failure policy: when the judge cannot produce a valid verdict, the outcome is
a failure with its error recorded -- never a clamped, defaulted or midpoint
score. A broken judge must never be indistinguishable from a mediocre answer;
callers exclude failed examples from aggregates and report them.

Reference:
    Rubric inspired by RAGAS (RAG Assessment) metrics.
    https://github.com/explodinggradients/ragas
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.config import settings
from app.eval.rubric import RUBRIC_VERSION, dimensions_for, render_rubric
from app.rag.providers.anthropic_provider import generate_with_usage

logger = logging.getLogger(__name__)

# Dimensions the judge must always score.
REQUIRED_DIMENSIONS = (
    "faithfulness",
    "answer_relevance",
    "context_precision",
    "context_recall",
)
# Scored only when the golden example carries an ideal answer.
OPTIONAL_DIMENSIONS = ("answer_correctness",)
DIMENSIONS = REQUIRED_DIMENSIONS + OPTIONAL_DIMENSIONS

# Upper bound on context characters sent to the judge. Generous enough for the
# full reranked context of every strategy; it only guards against runaway input.
MAX_JUDGE_CONTEXT_CHARS = 60_000

# How much of an invalid reply is quoted back in the repair message.
_MAX_ECHO_CHARS = 4_000

#: Error type recorded when the reply is still invalid after the retry.
INVALID_OUTPUT = "JudgeOutputInvalid"

JUDGE_SYSTEM = (
    "You are an impartial evaluator of retrieval-augmented answers. "
    "You reply with a single JSON object and nothing else -- no prose, no code fences."
)

JUDGE_PROMPT = """Score the answer below on {n_dimensions} dimensions.

Languages: the question, the retrieved context, the reference answer and the generated
answer may be in different languages (for example a French question over English
documents). Judge meaning, not wording or language: a claim translated faithfully from
the context is grounded, and a reference in another language still counts. The one place
language matters is Answer Relevance.

{rubric}

Question: {question}

Retrieved Context (exactly what the answering system was given):
{context}
{ideal_block}
Generated Answer:
{answer}

Respond with exactly this JSON shape. Every listed key is required, every score is a
number from 0 to 1, and every reasoning is one or two sentences justifying that score:
{json_shape}"""

REPAIR_PROMPT = """{original}

---
Your previous reply could not be used: {error}

Previous reply:
{previous}

Reply again with only the corrected JSON object in exactly the required shape."""

Score = Annotated[float, Field(ge=0.0, le=1.0, strict=True)]


class DimensionJudgement(BaseModel):
    """One dimension's score and the judge's justification for it."""

    model_config = ConfigDict(extra="forbid")

    score: Score
    reasoning: str = Field(min_length=1)


class JudgeVerdict(BaseModel):
    """A complete, validated judge reply.

    ``answer_correctness`` must be present exactly when the example has a
    reference answer; :func:`validate_verdict` enforces that.
    """

    model_config = ConfigDict(extra="ignore")

    faithfulness: DimensionJudgement
    answer_relevance: DimensionJudgement
    context_precision: DimensionJudgement
    context_recall: DimensionJudgement
    answer_correctness: DimensionJudgement | None = None

    def scores(self) -> dict[str, float | None]:
        """Flat ``{dimension: score}``; ``answer_correctness`` may be ``None``."""
        return {
            dim: (j.score if (j := getattr(self, dim)) is not None else None)
            for dim in DIMENSIONS
        }

    def reasoning(self) -> dict[str, str]:
        """``{dimension: reasoning}`` for the dimensions that were scored."""
        return {
            dim: j.reasoning for dim in DIMENSIONS if (j := getattr(self, dim)) is not None
        }


@dataclass
class JudgeOutcome:
    """Everything one judge invocation produced, success or failure.

    Attributes:
        scores: ``{dimension: score}`` on success, ``None`` on failure.
        reasoning: Per-dimension justification on success.
        error_type: Failure class (an exception name, or
            :data:`INVALID_OUTPUT`); ``None`` on success.
        error: Failure message; ``None`` on success.
        attempts: Model calls made (1, or 2 when a repair retry ran).
        input_tokens: Judge input tokens across all attempts.
        output_tokens: Judge output tokens across all attempts.
        judge_model: Model that judged.
        rubric_version: Rubric the scores are relative to.
    """

    scores: dict[str, float | None] | None = None
    reasoning: dict[str, str] = field(default_factory=dict)
    error_type: str | None = None
    error: str | None = None
    attempts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    judge_model: str | None = None
    rubric_version: str = RUBRIC_VERSION

    @property
    def ok(self) -> bool:
        return self.scores is not None

    @classmethod
    def failed(cls, error_type: str, error: str, **kwargs: Any) -> JudgeOutcome:
        """Build a failure outcome (convenient in tests and callers)."""
        return cls(scores=None, error_type=error_type, error=error, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_json(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a model response.

    Tolerates code fences and leading/trailing prose, both of which models emit
    occasionally even when told not to.

    Raises:
        ValueError: If no JSON object can be parsed from the text.
    """
    candidate = text.strip()

    fenced = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        # Fall back to the outermost {...} span.
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object found in judge response") from None
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as e:
            raise ValueError(f"judge response is not valid JSON: {e}") from e

    if not isinstance(parsed, dict):
        raise ValueError(f"judge returned {type(parsed).__name__}, expected an object")
    return parsed


def _format_validation_error(err: ValidationError) -> str:
    parts = []
    for e in err.errors():
        loc = ".".join(str(p) for p in e["loc"]) or "(root)"
        parts.append(f"{loc}: {e['msg']}")
    return "; ".join(parts)


def validate_verdict(text: str, has_reference: bool) -> JudgeVerdict:
    """Parse and validate a raw judge reply.

    Args:
        text: The model's reply.
        has_reference: Whether ``answer_correctness`` is required (and allowed).

    Returns:
        The validated verdict.

    Raises:
        ValueError: With a message naming every problem; suitable for quoting
            back to the model in a repair request.
    """
    parsed = _extract_json(text)
    try:
        verdict = JudgeVerdict.model_validate(parsed)
    except ValidationError as e:
        raise ValueError(_format_validation_error(e)) from e
    if has_reference and verdict.answer_correctness is None:
        raise ValueError("answer_correctness: required because a reference answer was given")
    if not has_reference and verdict.answer_correctness is not None:
        # Scored without a reference it would be invented; drop it rather than fail.
        verdict.answer_correctness = None
    return verdict


def _build_prompt(
    question: str, answer: str, context: list[str], ideal_answer: str | None
) -> str:
    """Render the judge prompt, with the correctness dimension when possible."""
    context_str = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(context))
    if len(context_str) > MAX_JUDGE_CONTEXT_CHARS:
        context_str = context_str[:MAX_JUDGE_CONTEXT_CHARS] + "\n[... context truncated ...]"

    has_reference = bool(ideal_answer)
    dims = [d.name for d in dimensions_for(has_reference)]
    shape = ",\n".join(
        f'  "{d}": {{"score": <0-1>, "reasoning": "<why>"}}' for d in dims
    )
    return JUDGE_PROMPT.format(
        n_dimensions=len(dims),
        rubric=render_rubric(has_reference),
        question=question,
        context=context_str or "(no context)",
        ideal_block=f"\nReference Answer:\n{ideal_answer}\n" if ideal_answer else "",
        answer=answer,
        json_shape="{\n" + shape + "\n}",
    )


async def _call(prompt: str, model: str, outcome: JudgeOutcome) -> str:
    """One judge model call; accumulates attempts and token usage."""
    outcome.attempts += 1
    result = await generate_with_usage(
        model=model,
        prompt=prompt,
        system=JUDGE_SYSTEM,
        max_tokens=1500,
        temperature=0.0,
    )
    outcome.input_tokens += int(result.get("input_tokens") or 0)
    outcome.output_tokens += int(result.get("output_tokens") or 0)
    return str(result.get("text") or "")


async def judge_detailed(
    question: str,
    answer: str,
    context: list[str],
    ideal_answer: str | None = None,
    *,
    model: str | None = None,
) -> JudgeOutcome:
    """Score a RAG answer and report exactly what happened.

    Makes one call; if the reply fails validation, makes one more with a
    repair message quoting the error. A provider error on either call ends the
    attempt (the provider layer already retries transient faults).

    Args:
        question: The question that was answered.
        answer: The answer text produced by the RAG system.
        context: The context the generator was given, in full.
        ideal_answer: The golden reference answer, if the example has one.
            Enables the ``answer_correctness`` dimension.
        model: Judge model; defaults to ``settings.judge_model``.

    Returns:
        A :class:`JudgeOutcome`. Never raises; never fabricates a score.
    """
    judge_model = model or settings.judge_model
    outcome = JudgeOutcome(judge_model=judge_model)
    has_reference = bool(ideal_answer)
    prompt = _build_prompt(question, answer, context, ideal_answer)

    current_prompt = prompt
    for attempt in (1, 2):
        try:
            text = await _call(current_prompt, judge_model, outcome)
        except Exception as e:
            logger.warning(
                "Judge call failed (%s: %s); example will be excluded from aggregates.",
                type(e).__name__,
                e,
            )
            outcome.error_type = type(e).__name__
            outcome.error = str(e) or type(e).__name__
            return outcome

        try:
            verdict = validate_verdict(text, has_reference)
        except ValueError as e:
            if attempt == 1:
                logger.info("Judge reply invalid (%s); retrying once with a repair message.", e)
                current_prompt = REPAIR_PROMPT.format(
                    original=prompt, error=e, previous=text[:_MAX_ECHO_CHARS] or "(empty)"
                )
                continue
            logger.warning(
                "Judge reply still invalid after repair (%s); example will be excluded "
                "from aggregates.",
                e,
            )
            outcome.error_type = INVALID_OUTPUT
            outcome.error = str(e)
            return outcome

        outcome.scores = verdict.scores()
        outcome.reasoning = verdict.reasoning()
        return outcome

    raise AssertionError("unreachable")  # pragma: no cover


async def judge(
    question: str,
    answer: str,
    context: list[str],
    ideal_answer: str | None = None,
) -> dict[str, Any] | None:
    """Score a RAG answer; the compact form of :func:`judge_detailed`.

    Returns:
        Dict with a float in [0, 1] for each required dimension,
        ``answer_correctness`` (a float, or ``None`` without an ideal answer),
        a combined ``reasoning`` string and ``reasoning_by_dimension`` -- or
        ``None`` if the judge failed. Callers must treat ``None`` as "not
        measured", never as a score.

    Example:
        >>> scores = await judge("What is AI?", "AI is...", ["AI stands for..."])
        >>> if scores is None:
        ...     print("judge unavailable -- example excluded from aggregate")
    """
    outcome = await judge_detailed(question, answer, context, ideal_answer)
    if outcome.scores is None:
        return None
    scores: dict[str, Any] = dict(outcome.scores)
    scores["reasoning"] = " ".join(f"[{d}] {r}" for d, r in outcome.reasoning.items())
    scores["reasoning_by_dimension"] = dict(outcome.reasoning)
    return scores
