from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from ragas.metrics import (
    answer_relevance as ragas_answer_relevance,
    context_precision as ragas_context_precision,
    context_recall as ragas_context_recall,
    faithfulness as ragas_faithfulness,
)

from app.rag.generate import answer_question
from app.schemas import EvalScore
from app.tracing.wandb_tracer import log_eval_table, log_metrics, wandb_run

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
GOLDEN_DIR = ROOT / "data" / "golden"
RUNS_DIR = ROOT / "data" / "eval_runs"


def load_dataset(name: str) -> list[dict]:
    path = GOLDEN_DIR / f"{name}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def run_evaluation(
    dataset: str = "golden_v1",
    provider: str | None = None,
    model: str | None = None,
    prompt_version: str | None = None,
) -> dict:
    examples = load_dataset(dataset)
    if not examples:
        return {
            "dataset": dataset,
            "provider": provider,
            "model": model,
            "prompt_version": prompt_version,
            "n": 0,
            "aggregate": None,
            "per_example": [],
            "error": f"No examples found in dataset '{dataset}'",
        }

    per_example: list[dict] = []

    run_name = f"eval-{dataset}-{provider or 'default'}-{model or 'default'}-{prompt_version or 'default'}"
    config = {
        "dataset": dataset,
        "provider": provider,
        "model": model,
        "prompt_version": prompt_version,
        "n_examples": len(examples),
    }

    with wandb_run(name=run_name, config=config, tags=["eval", dataset]) as run:
        for ex in examples:
            ans = await answer_question(
                question=ex["question"],
                provider=provider,
                model=model,
                prompt_version=prompt_version,
            )
            got = ans.model_dump()
            score = await score_example(ex, got)
            per_example.append({
                "question": ex["question"],
                "answer": got.get("answer"),
                "ideal_answer": ex.get("ideal_answer"),
                "score": score.model_dump(),
            })

        agg = aggregate([p["score"] for p in per_example])

        if run is not None:
            try:
                log_eval_table(run, [
                    {
                        "question": p["question"],
                        "answer": p["answer"],
                        "ideal_answer": p["ideal_answer"],
                        **p["score"],
                    }
                    for p in per_example
                ])
            except Exception as e:
                print(f"Warning: Failed to log eval table to W&B: {e}")
            try:
                if agg is not None:
                    log_metrics(run, agg.model_dump())
            except Exception as e:
                print(f"Warning: Failed to log metrics to W&B: {e}")

    result = {
        "dataset": dataset,
        "provider": provider,
        "model": model,
        "prompt_version": prompt_version,
        "n": len(examples),
        "aggregate": agg.model_dump() if agg else None,
        "per_example": per_example,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _persist_run(result)
    return result


def _safe_filename(s: str | None) -> str:
    return (s or "default").replace("/", "_").replace(":", "-")


def _persist_run(result: dict) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"{stamp}-{_safe_filename(result['dataset'])}-{_safe_filename(result['prompt_version'])}.json"
    (RUNS_DIR / name).write_text(json.dumps(result, indent=2), encoding="utf-8")


async def score_example(expected: dict, got: dict) -> EvalScore:
    """Score using LLM judge."""
    from app.eval.judge import judge

    question = expected.get("question", "")
    answer = got.get("answer", "")
    sources = got.get("sources", [])
    context = [s.get("quote", "") for s in sources if s.get("quote")]

    scores = await judge(question, answer, context)
    return EvalScore(
        faithfulness=scores.get("faithfulness", 0.0),
        answer_relevance=scores.get("answer_relevance", 0.0),
        context_precision=scores.get("context_precision", 0.0),
        context_recall=scores.get("context_recall", 0.0),
    )


async def faithfulness_score(answer: str, context: list[str]) -> float:
    """Score faithfulness using RAGAS metric.

    Measures what fraction of claims in the answer are grounded in the context.

    Args:
        answer: The generated answer
        context: List of context chunks

    Returns:
        Faithfulness score (0-1)
    """
    try:
        if not context or not answer:
            return 0.0

        from ragas import Sample

        sample = Sample(
            question="",  # RAGAS faithfulness doesn't require question
            answer=answer,
            contexts=context,
        )

        # Run the metric with configured LLM (Anthropic)
        score = await ragas_faithfulness.ascore(sample=sample)
        return max(0.0, min(1.0, float(score)))
    except Exception as e:
        logger.warning(f"RAGAS faithfulness error: {e}")
        return 0.5


async def answer_relevance_score(question: str, answer: str) -> float:
    """Score answer relevance using RAGAS metric.

    Measures how well the answer addresses the user's question.

    Args:
        question: The user's question
        answer: The generated answer

    Returns:
        Answer relevance score (0-1)
    """
    try:
        if not question or not answer:
            return 0.0

        from ragas import Sample

        sample = Sample(
            question=question,
            answer=answer,
            contexts=[],  # Answer relevance doesn't require context
        )

        # Run the metric
        score = await ragas_answer_relevance.ascore(sample=sample)
        return max(0.0, min(1.0, float(score)))
    except Exception as e:
        logger.warning(f"RAGAS answer_relevance error: {e}")
        return 0.5


async def context_recall_score(expected: str | list[str], retrieved: list[str]) -> float:
    """Score context recall using RAGAS metric.

    Measures what percentage of ground truth information is in retrieved context.

    Args:
        expected: Ground truth answer or list of expected information
        retrieved: List of retrieved context chunks

    Returns:
        Context recall score (0-1)
    """
    try:
        if not retrieved or not expected:
            return 0.0

        from ragas import Sample

        expected_str = expected if isinstance(expected, str) else " ".join(expected)
        sample = Sample(
            question="",  # RAGAS context_recall doesn't require question
            answer=expected_str,
            contexts=retrieved,
        )

        # Run the metric
        score = await ragas_context_recall.ascore(sample=sample)
        return max(0.0, min(1.0, float(score)))
    except Exception as e:
        logger.warning(f"RAGAS context_recall error: {e}")
        return 0.5


async def context_precision_score(question: str, context: list[str], answer: str) -> float:
    """Score context precision using RAGAS metric.

    Measures what percentage of retrieved context is relevant to answering the question.

    Args:
        question: The user's question
        context: List of retrieved context chunks
        answer: The generated answer (used to infer relevance)

    Returns:
        Context precision score (0-1)
    """
    try:
        if not context or not question:
            return 0.0

        from ragas import Sample

        sample = Sample(
            question=question,
            answer=answer or "",
            contexts=context,
        )

        # Run the metric
        score = await ragas_context_precision.ascore(sample=sample)
        return max(0.0, min(1.0, float(score)))
    except Exception as e:
        logger.warning(f"RAGAS context_precision error: {e}")
        return 0.5


def aggregate(scores: list[dict]) -> EvalScore | None:
    if not scores:
        return None
    n = len(scores)
    return EvalScore(
        faithfulness=sum(s["faithfulness"] for s in scores) / n,
        answer_relevance=sum(s["answer_relevance"] for s in scores) / n,
        context_precision=sum(s["context_precision"] for s in scores) / n,
        context_recall=sum(s["context_recall"] for s in scores) / n,
    )
