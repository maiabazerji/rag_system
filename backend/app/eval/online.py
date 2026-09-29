"""Online evaluation of a single strategy comparison.

``POST /compare/strategies`` with ``evaluate=true`` scores each strategy's row
with the same judge and retrieval metrics the offline harness uses
(:mod:`app.eval.metrics`), so a score seen on the Compare page means what it
means in a recorded run:

* the LLM judge (:func:`app.eval.judge.judge_detailed`) scores faithfulness,
  answer relevance and context precision/recall, plus answer correctness when a
  reference answer is known. A judge failure leaves every score ``None`` and
  records the error -- nothing is defaulted or invented;
* deterministic retrieval metrics (:func:`app.eval.retrieval.ranked_retrieval_metrics`)
  run over the documents behind the strategy's ranked context whenever the
  relevant documents are known.

The reference comes from the request (``reference``) or from a golden dataset
example (``golden``), looked up with :func:`app.eval.metrics.load_dataset`.
"""
from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.eval.dataset import (
    difficulty_of,
    example_id,
    expected_behavior_of,
    ideal_answer_of,
    question_type_of,
)
from app.eval.judge import judge_detailed
from app.eval.metrics import _REFUSAL_STATUSES, load_dataset
from app.eval.retrieval import (
    DEFAULT_K_VALUES,
    ranked_retrieval_metrics,
    relevant_documents,
)
from app.schemas import CompareReference, GoldenRef, StrategyComparison, StrategyEvaluation


class GoldenNotFound(LookupError):
    """The requested golden dataset or example does not exist."""


@dataclass(frozen=True)
class Reference:
    """What a comparison is scored against. Every part is optional."""

    ideal_answer: str | None = None
    relevant_docs: list[str] | None = None
    expect_refusal: bool = False

    @classmethod
    def from_request(cls, ref: CompareReference | None) -> Reference:
        if ref is None:
            return cls()
        docs = [d for d in (ref.relevant_doc_ids or []) if d.strip()] or None
        answer = ref.ideal_answer if ref.ideal_answer and ref.ideal_answer.strip() else None
        return cls(ideal_answer=answer, relevant_docs=docs)

    @classmethod
    def from_example(cls, example: Mapping[str, Any]) -> Reference:
        return cls(
            ideal_answer=ideal_answer_of(example),
            relevant_docs=relevant_documents(example),
            expect_refusal=expected_behavior_of(example) == "refuse",
        )


def _examples(dataset: str) -> list[dict]:
    examples = load_dataset(dataset)
    if not examples:
        raise GoldenNotFound(f"No golden dataset '{dataset}'")
    return examples


def golden_example(dataset: str, ex_id: str) -> dict:
    """One example of a golden dataset, by its stable id.

    Raises:
        GoldenNotFound: If the dataset or the example does not exist.
        ValueError: If the dataset file is malformed.
    """
    for i, ex in enumerate(_examples(dataset)):
        if example_id(ex, i) == ex_id:
            return ex
    raise GoldenNotFound(f"No example '{ex_id}' in golden dataset '{dataset}'")


def golden_questions(dataset: str) -> list[dict[str, Any]]:
    """The questions of a golden dataset for a picker -- never the answers.

    Raises:
        GoldenNotFound: If the dataset does not exist.
    """
    return [
        {
            "id": example_id(ex, i),
            "question": str(ex.get("question", "")),
            "question_type": question_type_of(ex),
            "difficulty": difficulty_of(ex),
        }
        for i, ex in enumerate(_examples(dataset))
    ]


def resolve_reference(
    reference: CompareReference | None, golden: GoldenRef | None
) -> Reference:
    """The reference a comparison request asks to be scored against.

    Raises:
        GoldenNotFound: If ``golden`` names a missing dataset or example.
    """
    if golden is not None:
        return Reference.from_example(golden_example(golden.dataset, golden.id))
    return Reference.from_request(reference)


def _doc_of_chunk(chunk_id: str) -> str:
    """Chunk ids are ``<doc_id>:<index>``; the document is the part before the colon."""
    return chunk_id.split(":", 1)[0]


def ranked_documents(row: StrategyComparison) -> list[str]:
    """Documents behind the strategy's context, best first, deduplicated.

    Prefers ``extra['retrieved_docs']`` (filenames, aligned with
    ``extra['retrieved_ids']``); falls back to the document part of each
    retrieved chunk id, then to the cited sources.
    """
    extra = row.extra or {}
    docs: Sequence[Any] | None = extra.get("retrieved_docs") or None
    ids: Sequence[Any] = extra.get("retrieved_ids") or []
    candidates: list[str] = []
    if docs:
        for i, d in enumerate(docs):
            if d:
                candidates.append(str(d))
            elif i < len(ids) and ids[i]:
                candidates.append(_doc_of_chunk(str(ids[i])))
    elif ids:
        candidates = [_doc_of_chunk(str(c)) for c in ids if c]
    else:
        candidates = [
            s.document or _doc_of_chunk(s.chunk_id)
            for s in row.sources
            if s.chunk_id != "none"
        ]
    return list(dict.fromkeys(candidates))


def _judge_context(row: StrategyComparison) -> list[str]:
    context = (row.extra or {}).get("context_text")
    if isinstance(context, str) and context:
        return [context]
    return [s.quote for s in row.sources if s.quote]


def _declined(row: StrategyComparison) -> bool:
    status = (row.status or "").lower()
    return row.refusal or status in _REFUSAL_STATUSES or status == "insufficient_context"


async def evaluate_row(
    row: StrategyComparison,
    reference: Reference,
    *,
    failed: bool = False,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> StrategyEvaluation:
    """Score one comparison row. Never raises and never fabricates a score.

    Args:
        row: The strategy's comparison row.
        reference: What to score against.
        failed: The strategy produced no answer (its row is an error message).
        k_values: Cutoffs for the ranked retrieval metrics.
    """
    base: dict[str, Any] = {
        "relevant_docs": reference.relevant_docs,
        "has_reference_answer": bool(reference.ideal_answer),
    }
    if failed:
        return StrategyEvaluation(
            status="not_evaluated",
            error_type="GenerationFailed",
            error="The strategy produced no answer to score.",
            **base,
        )

    docs = ranked_documents(row)
    base["retrieved_docs"] = docs

    if reference.expect_refusal:
        # Unanswerable by design: scored like the offline harness, without the
        # judge or retrieval metrics.
        return StrategyEvaluation(
            status="refusal_checked",
            correct_refusal=1.0 if _declined(row) else 0.0,
            **base,
        )

    retrieval = ranked_retrieval_metrics(reference.relevant_docs, docs, k_values)
    outcome = await judge_detailed(
        row.question, row.answer, _judge_context(row), ideal_answer=reference.ideal_answer
    )
    return StrategyEvaluation(
        status="scored" if outcome.ok else "judge_failed",
        scores=dict(outcome.scores) if outcome.scores is not None else None,
        reasoning=dict(outcome.reasoning),
        retrieval=retrieval,
        judge_model=outcome.judge_model,
        rubric_version=outcome.rubric_version,
        judge_attempts=outcome.attempts,
        judge_input_tokens=outcome.input_tokens,
        judge_output_tokens=outcome.output_tokens,
        error_type=outcome.error_type,
        error=outcome.error,
        **base,
    )


async def evaluate_rows(
    rows: Sequence[StrategyComparison], failed: Sequence[bool], reference: Reference
) -> list[StrategyEvaluation]:
    """Score every row concurrently (judge calls are independent)."""
    return list(
        await asyncio.gather(
            *(evaluate_row(r, reference, failed=f) for r, f in zip(rows, failed, strict=True))
        )
    )


def evaluation_units(n_strategies: int, reference: Reference) -> int:
    """Rate-limit units evaluation adds: one judge call per strategy.

    A golden example that expects a refusal is scored without the judge.
    """
    return 0 if reference.expect_refusal else n_strategies
