"""Evaluation harness: run a golden dataset through the RAG pipeline and score it.

Scoring uses the LLM judge in :mod:`app.eval.judge`. Examples the judge could
not score are recorded with ``score: null`` and excluded from the aggregate --
the run result reports how many, so a partially-failed run can never be mistaken
for a complete one.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections import Counter
from datetime import UTC, datetime

from app.config import settings
from app.eval.judge import DIMENSIONS, judge
from app.eval.retrieval import (
    aggregate_retrieval,
    retrieved_documents,
    score_retrieval_documents,
)
from app.rag.generate import answer_question_detailed
from app.rag.strategies.base import StrategyResult
from app.schemas import EvalScore, Strategy
from app.tracing.wandb_tracer import log_eval_table, log_metrics, wandb_run

logger = logging.getLogger(__name__)

# Resolved from settings (DATA_DIR) rather than from this file's location, which
# points somewhere unrelated when the package is installed in a container.
GOLDEN_DIR = settings.data_path / "golden"
RUNS_DIR = settings.data_path / "eval_runs"


def load_dataset(name: str) -> list[dict]:
    """Load a golden dataset from ``data/golden/<name>.jsonl``.

    Args:
        name: Dataset name without the ``.jsonl`` suffix.

    Returns:
        List of example dicts. Empty if the file does not exist.

    Raises:
        ValueError: If a line is present but is not valid JSON.
    """
    path = GOLDEN_DIR / f"{name}.jsonl"
    if not path.exists():
        return []

    examples: list[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            examples.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path.name} line {lineno} is not valid JSON: {e}") from e
    return examples


async def score_example(
    expected: dict, got: dict, context_text: str | None = None
) -> EvalScore | None:
    """Score one answer with the LLM judge.

    Args:
        expected: Golden example, expected to carry a ``question`` key and
            optionally an ``ideal_answer``.
        got: Serialized :class:`~app.schemas.Answer` for that question.
        context_text: The full context the generator used. Falls back to the
            answer's source quotes (truncated previews) when unavailable.

    Returns:
        An :class:`EvalScore`, or ``None`` when the judge could not score it.
    """
    question = expected.get("question", "")
    answer = got.get("answer", "")
    if context_text:
        context = [context_text]
    else:
        context = [s.get("quote", "") for s in got.get("sources", []) if s.get("quote")]

    scores = await judge(question, answer, context, ideal_answer=expected.get("ideal_answer"))
    if scores is None:
        return None
    return EvalScore.model_validate({dim: scores.get(dim) for dim in DIMENSIONS})


def _retrieved_docs(result: StrategyResult | None, sources) -> list[str]:
    """Documents behind the context a strategy used, best-ranked first.

    Every strategy records ``retrieved_docs`` for the full context it gave the
    generator, so all strategies are scored on the same basis. Only when that
    is missing (a strategy that never ran) do the cited sources stand in.
    """
    docs = (result.extra.get("retrieved_docs") if result is not None else None) or None
    if docs is None:
        return retrieved_documents(sources)
    ordered: dict[str, None] = {}
    for d in docs:
        if d:
            ordered.setdefault(d, None)
    return list(ordered)


async def _run_example(
    ex: dict,
    strategy: str,
    provider: str | None,
    model: str | None,
    prompt_version: str | None,
    semaphore: asyncio.Semaphore,
    judge_answers: bool = True,
) -> dict:
    """Answer one golden example, then score its retrieval and its answer."""
    async with semaphore:
        ans, result = await answer_question_detailed(
            question=ex["question"],
            strategy=strategy,
            provider=provider,
            model=model,
            prompt_version=prompt_version,
        )
        extra = result.extra if result is not None else {}
        # Retrieval is scored first and without a model call, so it is recorded
        # even when the judge is unavailable.
        retrieval = score_retrieval_documents(
            ex.get("expected_sources"), _retrieved_docs(result, ans.sources)
        )
        score = (
            await score_example(ex, ans.model_dump(), extra.get("context_text"))
            if judge_answers
            else None
        )

    return {
        "question": ex["question"],
        "answer": ans.answer,
        "ideal_answer": ex.get("ideal_answer"),
        "provider": ans.provider,
        "model": ans.model,
        "retrieved_ids": list(extra.get("retrieved_ids") or []),
        "refusal": ans.refusal,
        "latency_ms": ans.latency_ms,
        "input_tokens": ans.input_tokens,
        "output_tokens": ans.output_tokens,
        "score": score.model_dump() if score is not None else None,
        "retrieval": retrieval.model_dump() if retrieval is not None else None,
    }


async def run_evaluation(
    dataset: str = "golden_v1",
    strategy: Strategy = "classic",
    provider: str | None = None,
    model: str | None = None,
    prompt_version: str | None = None,
    judge_answers: bool = True,
) -> dict:
    """Run a golden dataset end to end and aggregate the scores.

    Examples run concurrently, bounded by ``settings.eval_concurrency``, so a
    hundred-question dataset does not take a hundred sequential round trips.

    Args:
        dataset: Golden dataset name (without ``.jsonl``).
        strategy: RAG strategy to evaluate. Run once per strategy to compare
            them; runs are only ever compared against others of the same
            strategy.
        provider: Optional provider override passed to the RAG pipeline.
        model: Optional model override.
        prompt_version: Optional prompt template version.
        judge_answers: When False, skip the LLM judge entirely. Retrieval is
            still scored (it needs no model call), which roughly halves the cost
            of a run at the price of the answer-quality dimensions.

    Returns:
        Dict with the run configuration, ``n`` examples attempted, ``n_scored``
        and ``n_unscored`` counts, the judge ``aggregate`` over scored examples
        only (``None`` if nothing scored), the deterministic
        ``retrieval_aggregate``, a ``cost`` block, and ``per_example`` detail.
    """
    examples = load_dataset(dataset)
    if not examples:
        return {
            "dataset": dataset,
            "strategy": strategy,
            "provider": provider,
            "model": model,
            "prompt_version": prompt_version,
            "n": 0,
            "n_scored": 0,
            "n_unscored": 0,
            "aggregate": None,
            "retrieval_aggregate": None,
            "per_example": [],
            "error": f"No examples found in dataset '{dataset}'",
        }

    run_name = (
        f"eval-{dataset}-{strategy}-{provider or 'default'}-{model or 'default'}"
        f"-{prompt_version or 'default'}"
    )
    config = {
        "dataset": dataset,
        "strategy": strategy,
        "provider": provider,
        "model": model,
        "prompt_version": prompt_version,
        "n_examples": len(examples),
    }

    semaphore = asyncio.Semaphore(settings.eval_concurrency)

    with wandb_run(name=run_name, config=config, tags=["eval", dataset]) as run:
        per_example = list(
            await asyncio.gather(
                *(
                    _run_example(
                        ex,
                        strategy,
                        provider,
                        model,
                        prompt_version,
                        semaphore,
                        judge_answers,
                    )
                    for ex in examples
                )
            )
        )

        # Record what actually ran, not what was requested: a run with no model
        # override used the configured default, and grouping it under `None`
        # would compare it against runs of some other default.
        resolved_model = _most_common(p["model"] for p in per_example) or model
        resolved_provider = _most_common(p["provider"] for p in per_example) or provider

        scored = [p["score"] for p in per_example if p["score"] is not None]
        agg = aggregate(scored)
        retrieval_scored = [
            p["retrieval"] for p in per_example if p["retrieval"] is not None
        ]
        retrieval_agg = aggregate_retrieval(retrieval_scored)

        if judge_answers and len(scored) < len(per_example):
            logger.warning(
                "Eval run '%s': %d of %d examples could not be scored and are "
                "excluded from the aggregate.",
                run_name,
                len(per_example) - len(scored),
                len(per_example),
            )

        if run is not None:
            try:
                log_eval_table(
                    run,
                    [
                        {
                            "question": p["question"],
                            "answer": p["answer"],
                            "ideal_answer": p["ideal_answer"],
                            "strategy": strategy,
                            "scored": p["score"] is not None,
                            "latency_ms": p["latency_ms"],
                            **(p["score"] or {}),
                            **{
                                f"retrieval_{k}": v
                                for k, v in (p["retrieval"] or {}).items()
                                if k in ("precision", "recall", "mrr", "hit")
                            },
                        }
                        for p in per_example
                    ],
                )
            except Exception as e:
                logger.warning(f"Failed to log eval table to W&B: {e}")
            try:
                metrics = dict(agg.model_dump()) if agg is not None else {}
                if retrieval_agg:
                    metrics.update(
                        {f"retrieval_{k}": v for k, v in retrieval_agg.items()}
                    )
                if metrics:
                    log_metrics(run, metrics)
            except Exception as e:
                logger.warning(f"Failed to log metrics to W&B: {e}")

    result = {
        "dataset": dataset,
        "strategy": strategy,
        "provider": resolved_provider,
        "model": resolved_model,
        "requested_provider": provider,
        "requested_model": model,
        "prompt_version": prompt_version,
        "judge_model": settings.judge_model if judge_answers else None,
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "n": len(examples),
        "judged": judge_answers,
        "n_scored": len(scored),
        "n_unscored": (len(examples) - len(scored)) if judge_answers else 0,
        "n_refused": sum(1 for p in per_example if p["refusal"]),
        "aggregate": agg.model_dump() if agg else None,
        "retrieval_aggregate": retrieval_agg,
        "cost": {
            "mean_latency_ms": (
                round(sum(p["latency_ms"] for p in per_example) / len(per_example))
                if per_example
                else 0
            ),
            "total_input_tokens": sum(p["input_tokens"] for p in per_example),
            "total_output_tokens": sum(p["output_tokens"] for p in per_example),
        },
        "per_example": per_example,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    _persist_run(result)
    return result


def _most_common(values) -> str | None:
    """The most frequent non-empty value, or None."""
    counts = Counter(v for v in values if v)
    return counts.most_common(1)[0][0] if counts else None


def _safe_filename(s: str | None) -> str:
    return "".join(
        c if c.isalnum() or c in "._-" else "_" for c in (s or "default")
    )


def _persist_run(result: dict) -> None:
    """Write a run to disk under a name unique even for concurrent runs.

    The name leads with a timestamp (so a sorted listing is chronological) and
    ends with a short random suffix: two runs of different strategies or models
    started in the same second used to overwrite each other.
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    name = (
        f"{stamp}-{_safe_filename(result['dataset'])}"
        f"-{_safe_filename(result.get('strategy'))}"
        f"-{_safe_filename(result.get('model'))}"
        f"-{_safe_filename(result['prompt_version'])}"
        f"-{uuid.uuid4().hex[:8]}.json"
    )
    (RUNS_DIR / name).write_text(json.dumps(result, indent=2), encoding="utf-8")


def aggregate(scores: list[dict]) -> EvalScore | None:
    """Average scored examples across every dimension.

    Args:
        scores: Score dicts from successfully judged examples. Callers must
            filter out unscored (``None``) examples first.

    Returns:
        Mean :class:`EvalScore`, or ``None`` if no examples were scored. An
        optional dimension (``answer_correctness``) is averaged over only the
        examples that have it, and is ``None`` when none do.
    """
    if not scores:
        return None
    means: dict[str, float | None] = {}
    for dim in DIMENSIONS:
        values = [s[dim] for s in scores if s.get(dim) is not None]
        means[dim] = sum(values) / len(values) if values else None
    return EvalScore.model_validate(means)
