from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from app.rag.generate import answer_question
from app.schemas import EvalScore
from app.tracing.wandb_tracer import log_eval_table, log_metrics, wandb_run

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
            score = score_example(ex, got)
            per_example.append({
                "question": ex["question"],
                "answer": got.get("answer"),
                "ideal_answer": ex.get("ideal_answer"),
                "score": score.model_dump(),
            })

        agg = aggregate([p["score"] for p in per_example])

        if run is not None:
            log_eval_table(run, [
                {
                    "question": p["question"],
                    "answer": p["answer"],
                    "ideal_answer": p["ideal_answer"],
                    **p["score"],
                }
                for p in per_example
            ])
            if agg is not None:
                log_metrics(run, agg.model_dump())

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


def _persist_run(result: dict) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = lambda s: (s or "default").replace("/", "_").replace(":", "-")
    name = f"{stamp}-{safe(result['dataset'])}-{safe(result['prompt_version'])}.json"
    (RUNS_DIR / name).write_text(json.dumps(result, indent=2), encoding="utf-8")


def score_example(expected: dict, got: dict) -> EvalScore:
    """Placeholder — wire RAGAS + judge here."""
    return EvalScore(
        faithfulness=0.0,
        answer_relevance=0.0,
        context_precision=0.0,
        context_recall=0.0,
    )


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
