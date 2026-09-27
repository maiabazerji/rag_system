"""LLM-as-judge scoring for RAG answers.

Scores an answer along five dimensions using Claude as an impartial judge:

1. Faithfulness: are claims grounded in retrieved context (no hallucination)?
2. Answer Relevance: does the answer address the user's question?
3. Context Precision: are the retrieved chunks relevant and well ranked?
4. Context Recall: is all the information needed to answer present in context?
5. Answer Correctness: does the answer agree with the golden ideal answer?
   Only scored when the example has one; otherwise it is ``None``.

The judge sees the full context the generator actually used (not the short
source quotes shown in the UI), and runs at temperature 0 where the model
accepts it, so repeated runs of the same answer score the same.

Failure policy: when the judge cannot produce a score, this module returns
``None`` rather than a plausible-looking number. A broken judge must never be
indistinguishable from a mediocre answer -- callers exclude unscored examples
from aggregates and report how many failed.

Reference:
    Rubric inspired by RAGAS (RAG Assessment) metrics.
    https://github.com/explodinggradients/ragas
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.config import settings
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

**Faithfulness (0-1):** What fraction of claims in the answer are grounded in the retrieved context?
- 1.0 = all claims supported by context
- 0.5 = some claims supported, some unsupported
- 0.0 = no claims grounded or hallucinated

**Answer Relevance (0-1):** Does the answer address the user's question?
- 1.0 = directly and completely answers the question
- 0.5 = partially addresses or tangential
- 0.0 = off-topic or irrelevant
- The answer is expected in the language of the question. An answer in another language
  scores at most 0.5 here, however good its content.

**Context Precision (0-1):** Are the retrieved chunks relevant and well-ranked?
- 1.0 = all chunks are highly relevant
- 0.5 = mixed relevance
- 0.0 = chunks are irrelevant

**Context Recall (0-1):** Is all needed information present in the context?
- 1.0 = everything needed to answer is in context
- 0.5 = some needed info is missing
- 0.0 = critical info missing
{correctness_rubric}
Question: {question}

Retrieved Context (exactly what the answering system was given):
{context}
{ideal_block}
Generated Answer:
{answer}

Respond with exactly this JSON shape:
{json_shape}"""

CORRECTNESS_RUBRIC = """
**Answer Correctness (0-1):** Does the answer agree with the reference answer?
- 1.0 = states the same facts as the reference, nothing contradicting it
- 0.5 = partially correct or missing key facts from the reference
- 0.0 = contradicts the reference or is wrong
"""


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


def _clamp_score(raw: Any, dimension: str) -> float:
    """Coerce a judge score to a float in [0, 1].

    Raises:
        ValueError: If the value is missing or not numeric.
    """
    if raw is None:
        raise ValueError(f"judge omitted the '{dimension}' score")
    try:
        value = float(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(f"'{dimension}' score is not numeric: {raw!r}") from e
    return max(0.0, min(1.0, value))


def _build_prompt(
    question: str, answer: str, context: list[str], ideal_answer: str | None
) -> str:
    """Render the judge prompt, with the correctness dimension when possible."""
    context_str = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(context))
    if len(context_str) > MAX_JUDGE_CONTEXT_CHARS:
        context_str = context_str[:MAX_JUDGE_CONTEXT_CHARS] + "\n[... context truncated ...]"

    dims = list(REQUIRED_DIMENSIONS)
    if ideal_answer:
        dims += list(OPTIONAL_DIMENSIONS)
    shape = ", ".join(f'"{d}": <0-1>' for d in dims)
    return JUDGE_PROMPT.format(
        n_dimensions=len(dims),
        correctness_rubric=CORRECTNESS_RUBRIC if ideal_answer else "",
        question=question,
        context=context_str or "(no context)",
        ideal_block=f"\nReference Answer:\n{ideal_answer}\n" if ideal_answer else "",
        answer=answer,
        json_shape="{" + shape + ', "reasoning": "brief justification"}',
    )


async def judge(
    question: str,
    answer: str,
    context: list[str],
    ideal_answer: str | None = None,
) -> dict[str, Any] | None:
    """Score a RAG answer using Claude.

    Args:
        question: The question that was answered.
        answer: The answer text produced by the RAG system.
        context: The context the generator was given, in full.
        ideal_answer: The golden reference answer, if the example has one.
            Enables the ``answer_correctness`` dimension.

    Returns:
        Dict with a float in [0, 1] for each required dimension,
        ``answer_correctness`` (a float, or ``None`` without an ideal answer)
        and a ``reasoning`` string -- or ``None`` if the judge call failed,
        its response could not be parsed, or a score was missing. Callers must
        treat ``None`` as "not measured", never as a score.

    Raises:
        No exceptions. Failures are logged and reported as ``None``.

    Example:
        >>> scores = await judge("What is AI?", "AI is...", ["AI stands for..."])
        >>> if scores is None:
        ...     print("judge unavailable -- example excluded from aggregate")
    """
    prompt = _build_prompt(question, answer, context, ideal_answer)

    try:
        result = await generate_with_usage(
            model=settings.judge_model,
            prompt=prompt,
            system=JUDGE_SYSTEM,
            max_tokens=1024,
            temperature=0.0,
        )
    except Exception as e:
        logger.warning(
            "Judge call failed (%s: %s); example will be excluded from aggregates.",
            type(e).__name__,
            e,
        )
        return None

    try:
        parsed = _extract_json(result["text"])
        scores: dict[str, Any] = {
            dim: _clamp_score(parsed.get(dim), dim) for dim in REQUIRED_DIMENSIONS
        }
        for dim in OPTIONAL_DIMENSIONS:
            scores[dim] = _clamp_score(parsed.get(dim), dim) if ideal_answer else None
    except ValueError as e:
        logger.warning(
            "Judge response unusable (%s); example will be excluded from aggregates.", e
        )
        return None

    scores["reasoning"] = str(parsed.get("reasoning", ""))
    return scores
