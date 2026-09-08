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
from datetime import UTC, datetime
from pathlib import Path

from app.config import settings
from app.eval.judge import DIMENSIONS, judge
from app.eval.retrieval import aggregate_retrieval, score_retrieval
from app.rag.generate import answer_question
from app.schemas import EvalScore, Strategy
from app.tracing.wandb_tracer import log_eval_table, log_metrics, wandb_run

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
GOLDEN_DIR = ROOT / "data" / "golden"
RUNS_DIR = ROOT / "data" / "eval_runs"


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


async def score_example(expected: dict, got: dict) -> EvalScore | None:
    """Score one answer with the LLM judge.

    Args:
        expected: Golden example, expected to carry a ``question`` key.
        got: Serialized :class:`~app.schemas.Answer` for that question.

    Returns:
        An :class:`EvalScore`, or ``None`` when the judge could not score it.
    """
    question = expected.get("question", "")
    answer = got.get("answer", "")
    context = [s.get("quote", "") for s in got.get("sources", []) if s.get("quote")]

    scores = await judge(question, answer, context)
    if scores is None:
        return None
    return EvalScore(**{dim: scores[dim] for dim in DIMENSIONS})


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
        ans = await answer_question(
            question=ex["question"],
            strategy=strategy,
            provider=provider,
            model=model,
            prompt_version=prompt_version,
        )
        # Retrieval is scored first and without a model call, so it is recorded
        # even when the judge is unavailable.
        retrieval = score_retrieval(ex.get("expected_sources"), ans.sources)
        score = await score_example(ex, ans.model_dump()) if judge_answers else None

    return {
        "question": ex["question"],
        "answer": ans.answer,
        "ideal_answer": ex.get("ideal_answer"),
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
        "provider": provider,
        "model": model,
        "prompt_version": prompt_version,
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


def _safe_filename(s: str | None) -> str:
    return (s or "default").replace("/", "_").replace(":", "-")


def _persist_run(result: dict) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    name = (
        f"{stamp}-{_safe_filename(result['dataset'])}"
        f"-{_safe_filename(result.get('strategy'))}"
        f"-{_safe_filename(result['prompt_version'])}.json"
    )
    (RUNS_DIR / name).write_text(json.dumps(result, indent=2), encoding="utf-8")


def aggregate(scores: list[dict]) -> EvalScore | None:
    """Average scored examples across all four dimensions.

    Args:
        scores: Score dicts from successfully judged examples. Callers must
            filter out unscored (``None``) examples first.

    Returns:
        Mean :class:`EvalScore`, or ``None`` if no examples were scored.
    """
    if not scores:
        return None
    n = len(scores)
    return EvalScore(**{dim: sum(s[dim] for s in scores) / n for dim in DIMENSIONS})
