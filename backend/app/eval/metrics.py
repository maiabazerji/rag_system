"""Evaluation harness: run a golden dataset through the RAG pipeline and score it.

Scoring uses the LLM judge in :mod:`app.eval.judge` and the deterministic
retrieval metrics in :mod:`app.eval.retrieval`.

Run accounting. Every example ends in exactly one status:

``scored``
    Generation succeeded and every applicable metric was measured: the judge
    returned a valid verdict or, for an example whose ``expected_behavior`` is
    ``refuse``, the deterministic refusal check ran (such examples are not
    judged and not retrieval-scored; ``correct_refusal`` is 1 when the system
    declined and 0 -- a hallucination -- when it answered).
``judge_failed``
    Generation succeeded; the judge errored or its reply stayed invalid after
    one repair retry. The error is recorded; no score is invented.
``generation_failed``
    The pipeline raised, or returned an error refusal without running a
    strategy (no documents, provider error, ...). Not judged, not
    retrieval-scored.
``not_judged``
    Generation succeeded and the judge was disabled (``--no-judge``).

Aggregates (``mean``/``std``/``n`` per metric) are computed only over the
examples where that metric was measured: judge dimensions over ``scored``
rows, retrieval metrics over rows that have relevance labels and a successful
generation, latency and tokens over successful generations. A partially
failed run therefore can never be mistaken for a complete one: the counts and
the ``failures`` list say exactly what is missing.

Retrieval is scored on every strategy's ranked ``extra["retrieved_docs"]``
(the documents behind ``extra["retrieved_ids"]``) at explicit cutoffs
(``k_values``). The cutoffs put strategies on the same footing: the agentic
strategy's list is everything it read, far longer than the reranked top-k of
the others, and only an @K metric compares the two fairly.

After a run is saved it is compared against its baseline (a pinned baseline
for its configuration, else the previous comparable run) and the regression
report is saved next to it (see :mod:`app.eval.regression`).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import statistics
import uuid
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.access import AccessScope
from app.config import settings
from app.eval import regression
from app.eval.dataset import (
    difficulty_of,
    example_id,
    expected_behavior_of,
    ideal_answer_of,
    question_type_of,
)
from app.eval.judge import DIMENSIONS, JudgeOutcome, judge, judge_detailed
from app.eval.retrieval import (
    DEFAULT_K_VALUES,
    aggregate_retrieval,
    ranked_metric_names,
    retrieved_documents,
    score_example_retrieval,
)
from app.eval.rubric import RUBRIC_VERSION, rubric_fingerprint
from app.rag.generate import answer_question_detailed
from app.rag.strategies.base import StrategyResult
from app.schemas import EvalScore, Strategy
from app.tracing.wandb_tracer import log_eval_table, log_metrics, wandb_run

logger = logging.getLogger(__name__)

#: Version of the saved run JSON layout. 1 = the pre-accounting format.
RUN_SCHEMA_VERSION = 2

# Resolved from settings (DATA_DIR) rather than from this file's location, which
# points somewhere unrelated when the package is installed in a container.
GOLDEN_DIR = settings.data_path / "golden"
RUNS_DIR = settings.data_path / "eval_runs"

STATUS_SCORED = "scored"
STATUS_JUDGE_FAILED = "judge_failed"
STATUS_GENERATION_FAILED = "generation_failed"
STATUS_NOT_JUDGED = "not_judged"

#: Cost metrics reported per question and aggregated like any other metric.
COST_METRICS = ("latency_ms", "total_tokens")

#: Deterministic answer-behaviour metrics (see :func:`_row_metrics`).
BEHAVIOUR_METRICS = ("correct_refusal", "false_refusal")

_REFUSAL_STATUSES = frozenset(
    {"refused", "refusal", "abstained", "insufficient_evidence", "not_answerable"}
)

# Settings that change what retrieval returns. Matched by prefix so knobs added
# later (hybrid weights, fusion constants, ...) join the config hash on their own.
_RETRIEVAL_SETTING_PREFIXES = (
    "retrieval_",
    "rerank_",
    "hybrid_",
    "bm25_",
    "fusion_",
    "rrf_",
    "chunk",  # chunk_size_tokens, chunk_overlap_tokens, chunking_* ...
    "embedding_model",
    "reranker_model",
)


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


def _judge_context(got: dict, context_text: str | None) -> list[str]:
    if context_text:
        return [context_text]
    return [s.get("quote", "") for s in got.get("sources", []) if s.get("quote")]


async def score_example(
    expected: dict, got: dict, context_text: str | None = None
) -> EvalScore | None:
    """Score one answer with the LLM judge.

    Args:
        expected: Golden example, expected to carry a ``question`` key and
            optionally an ``ideal_answer`` (or ``expected_answer``).
        got: Serialized :class:`~app.schemas.Answer` for that question.
        context_text: The full context the generator used. Falls back to the
            answer's source quotes (truncated previews) when unavailable.

    Returns:
        An :class:`EvalScore`, or ``None`` when the judge could not score it.
        Use :func:`score_example_detailed` to learn why.
    """
    scores = await judge(
        expected.get("question", ""),
        got.get("answer", ""),
        _judge_context(got, context_text),
        ideal_answer=ideal_answer_of(expected),
    )
    if scores is None:
        return None
    return EvalScore.model_validate({dim: scores.get(dim) for dim in DIMENSIONS})


async def score_example_detailed(
    expected: dict, got: dict, context_text: str | None = None
) -> JudgeOutcome:
    """Like :func:`score_example`, but returns the full :class:`JudgeOutcome`
    (scores or error, reasoning, attempts and judge token usage)."""
    return await judge_detailed(
        expected.get("question", ""),
        got.get("answer", ""),
        _judge_context(got, context_text),
        ideal_answer=ideal_answer_of(expected),
    )


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


def _retrieval_mode(extra: dict[str, Any]) -> str | None:
    """The retrieval mode a strategy reported.

    Strategies report it inside their retrieval diagnostics
    (``extra["retrieval"]["mode"]``); a top-level ``retrieval_mode`` key is
    honoured too.
    """
    mode = extra.get("retrieval_mode")
    if mode:
        return str(mode)
    diagnostics = extra.get("retrieval")
    if isinstance(diagnostics, dict) and diagnostics.get("mode"):
        return str(diagnostics["mode"])
    return None


def _grounding_fields(ans: Any) -> dict[str, Any]:
    """Grounding annotations, when the Answer model carries them.

    Read defensively: the fields are optional on :class:`~app.schemas.Answer`
    and absent from older versions of it.
    """
    out: dict[str, Any] = {}
    for attr in ("grounded", "status"):
        value = getattr(ans, attr, None)
        if value is not None:
            out[attr] = value
    claims = getattr(ans, "claims", None)
    if isinstance(claims, list):
        out["n_claims"] = len(claims)
    return out


def _is_refusal(ans: Any) -> bool:
    """Whether the answer declines to answer.

    The pipeline's ``refusal`` flag, or an abstaining grounding ``status``
    when the Answer model carries one.
    """
    status = getattr(ans, "status", None)
    return bool(ans.refusal) or (
        isinstance(status, str) and status.lower() in _REFUSAL_STATUSES
    )


def _failure(row: dict, stage: str, error_type: str, error: str) -> dict:
    return {
        "id": row["id"],
        "question": row["question"],
        "stage": stage,
        "error_type": error_type,
        "error": error,
    }


async def _run_example(
    ex: dict,
    index: int,
    strategy: str,
    provider: str | None,
    model: str | None,
    prompt_version: str | None,
    semaphore: asyncio.Semaphore,
    judge_answers: bool = True,
    access: AccessScope | None = None,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> dict:
    """Answer one golden example, then score its retrieval and its answer.

    Never raises: a pipeline exception becomes a ``generation_failed`` row.
    """
    row: dict[str, Any] = {
        "id": example_id(ex, index),
        "question": ex.get("question", ""),
        "question_type": question_type_of(ex),
        "difficulty": difficulty_of(ex),
        "expected_behavior": expected_behavior_of(ex),
        "strategy": strategy,
        "status": None,
        "answer": None,
        "ideal_answer": ideal_answer_of(ex),
        "provider": provider,
        "model": model,
        "retrieved_ids": [],
        "refusal": False,
        "refused": False,
        "latency_ms": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "score": None,
        "retrieval": None,
        "judge": None,
        "error": None,
        "metrics": {},
        "doc_ids": [],
    }

    async with semaphore:
        try:
            ans, result = await answer_question_detailed(
                question=row["question"],
                strategy=strategy,
                provider=provider,
                model=model,
                prompt_version=prompt_version,
                access=access,
            )
        except Exception as e:
            logger.warning(
                "Eval example %s: generation raised %s: %s", row["id"], type(e).__name__, e
            )
            row["status"] = STATUS_GENERATION_FAILED
            row["error"] = _failure(row, "generation", type(e).__name__, str(e))
            return row

        extra = result.extra if result is not None else {}
        docs = _retrieved_docs(result, ans.sources)
        row.update(
            {
                "answer": ans.answer,
                "provider": ans.provider,
                "model": ans.model,
                "retrieved_ids": list(extra.get("retrieved_ids") or []),
                "retrieval_mode": _retrieval_mode(extra),
                "refusal": ans.refusal,
                "latency_ms": ans.latency_ms,
                "input_tokens": ans.input_tokens,
                "output_tokens": ans.output_tokens,
                "total_tokens": ans.input_tokens + ans.output_tokens,
                # Documents behind the answer, so an erasure request can find this row.
                "doc_ids": sorted(
                    {src.chunk_id.split(":", 1)[0] for src in ans.sources if ":" in src.chunk_id}
                ),
                **_grounding_fields(ans),
            }
        )

        # A refusal with no strategy result is the pipeline reporting an error
        # (no documents, provider failure, bad prompt version), not a model
        # declining to answer: there is nothing meaningful to score.
        if result is None and ans.refusal:
            row["status"] = STATUS_GENERATION_FAILED
            row["error"] = _failure(row, "generation", "PipelineRefusal", ans.answer[:300])
            return row

        row["refused"] = _is_refusal(ans)
        if row["expected_behavior"] == "refuse":
            # Unanswerable by design: declining is the only correct behaviour,
            # and any answer is a hallucination. Scored deterministically; no
            # judge (its rubric assumes an answerable question) and no
            # retrieval metrics (there is nothing relevant to retrieve).
            row["status"] = STATUS_SCORED
            row["metrics"] = _row_metrics(row)
            return row

        # Retrieval is scored first and without a model call, so it is recorded
        # even when the judge is unavailable.
        row["retrieval"] = score_example_retrieval(ex, docs, k_values)

        if not judge_answers:
            row["status"] = STATUS_NOT_JUDGED
        else:
            outcome = await score_example_detailed(
                ex, ans.model_dump(), extra.get("context_text")
            )
            row["judge"] = {
                "model": outcome.judge_model,
                "rubric_version": outcome.rubric_version,
                "attempts": outcome.attempts,
                "input_tokens": outcome.input_tokens,
                "output_tokens": outcome.output_tokens,
                "reasoning": outcome.reasoning,
            }
            if outcome.scores is not None:
                row["status"] = STATUS_SCORED
                row["score"] = EvalScore.model_validate(outcome.scores).model_dump()
            else:
                row["status"] = STATUS_JUDGE_FAILED
                row["error"] = _failure(
                    row, "judge", outcome.error_type or "JudgeError", outcome.error or ""
                )

    row["metrics"] = _row_metrics(row)
    return row


def _row_metrics(row: dict) -> dict[str, float]:
    """Every measured metric of one row, flat. Unmeasured metrics are absent."""
    out: dict[str, float] = {}
    for dim, value in (row.get("score") or {}).items():
        if value is not None:
            out[dim] = float(value)
    retrieval = row.get("retrieval") or {}
    for key, value in retrieval.items():
        if key == "mrr" or "@" in key:
            out[key] = float(value)
    if row.get("status") != STATUS_GENERATION_FAILED:
        refused = 1.0 if row.get("refused") else 0.0
        if row.get("expected_behavior") == "refuse":
            out["correct_refusal"] = refused
        else:
            out["false_refusal"] = refused
        out["latency_ms"] = float(row.get("latency_ms") or 0)
        out["total_tokens"] = float(row.get("total_tokens") or 0)
    return out


def summarize(values: Iterable[float]) -> dict[str, float | int | None]:
    """``{"mean", "std", "n"}`` of the values.

    ``std`` is the sample standard deviation (``n - 1`` denominator), and
    ``None`` when ``n < 2``; ``mean`` is ``None`` when ``n == 0``.
    """
    vals = [float(v) for v in values]
    n = len(vals)
    return {
        "mean": statistics.fmean(vals) if n else None,
        "std": statistics.stdev(vals) if n >= 2 else None,
        "n": n,
    }


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolation percentile (``q`` in [0, 100]); ``None`` if empty."""
    vals = sorted(float(v) for v in values)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def metric_order(k_values: Sequence[int] = DEFAULT_K_VALUES) -> list[str]:
    """Report order: judge dimensions, retrieval metrics, cost."""
    return [*DIMENSIONS, *ranked_metric_names(k_values), *BEHAVIOUR_METRICS, *COST_METRICS]


def aggregate_rows(
    rows: Sequence[dict], k_values: Sequence[int] = DEFAULT_K_VALUES
) -> dict[str, dict[str, float | int | None]]:
    """Per-metric ``mean``/``std``/``n`` over the rows that measured it.

    A metric no row measured is omitted rather than reported as zero.
    """
    out: dict[str, dict[str, float | int | None]] = {}
    for name in metric_order(k_values):
        values = [r["metrics"][name] for r in rows if name in (r.get("metrics") or {})]
        if values:
            out[name] = summarize(values)
    return out


def _counts(rows: Sequence[dict]) -> dict[str, int]:
    status = Counter(r.get("status") for r in rows)
    return {
        "n_examples": len(rows),
        "n_scored": status[STATUS_SCORED],
        "n_judge_failed": status[STATUS_JUDGE_FAILED],
        "n_generation_failed": status[STATUS_GENERATION_FAILED],
        "n_not_judged": status[STATUS_NOT_JUDGED],
        "n_retrieval_scored": sum(1 for r in rows if r.get("retrieval") is not None),
    }


def breakdown(
    rows: Sequence[dict], key: str, k_values: Sequence[int] = DEFAULT_K_VALUES
) -> dict[str, dict[str, Any]]:
    """Counts and aggregates per value of ``row[key]`` (e.g. ``question_type``).

    Rows whose value is ``None`` are grouped under ``"unspecified"``.
    """
    groups: dict[str, list[dict]] = {}
    for r in rows:
        value = r.get(key)
        groups.setdefault(str(value) if value is not None else "unspecified", []).append(r)
    return {
        name: {**_counts(members), "aggregates": aggregate_rows(members, k_values)}
        for name, members in sorted(groups.items())
    }


def retrieval_config(mode: str | None = None) -> dict[str, Any]:
    """Settings that determine retrieval output, for the run's config hash.

    Args:
        mode: Retrieval mode the strategies reported, if any; otherwise
            ``settings.retrieval_mode`` when it exists, else ``"dense"``.
    """
    dumped = settings.model_dump()
    config: dict[str, Any] = {
        k: v
        for k, v in sorted(dumped.items())
        if k.startswith(_RETRIEVAL_SETTING_PREFIXES) and isinstance(v, str | int | float | bool)
    }
    config["mode"] = mode or getattr(settings, "retrieval_mode", None) or "dense"
    return config


def config_hash(config: dict[str, Any]) -> str:
    """Short, order-independent hash of a configuration dict."""
    blob = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


async def run_evaluation(
    dataset: str = "golden_v1",
    strategy: Strategy = "classic",
    provider: str | None = None,
    model: str | None = None,
    prompt_version: str | None = None,
    judge_answers: bool = True,
    access: AccessScope | None = None,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
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
        access: Retrieve only documents this scope may read. The API passes
            the caller's; ``None`` (scripts) is unrestricted.
        k_values: Rank cutoffs for the @K retrieval metrics.

    Returns:
        The run record (also saved under ``RUNS_DIR``). Key fields:
        configuration (``strategy``, ``model``, ``judge_model``,
        ``rubric_version``, ``retrieval_config``, ``config_hash``), counts
        (``n_examples``, ``n_scored``, ``n_unscored``, ``n_judge_failed``,
        ``n_generation_failed``), ``failures``, ``aggregates`` (mean/std/n per
        metric), ``by_question_type`` and ``by_difficulty`` breakdowns,
        ``cost``, and ``per_example`` rows. The legacy ``aggregate`` (judge
        means) and ``retrieval_aggregate`` blocks are kept for older clients.
    """
    k_values = tuple(sorted(set(k_values)))
    examples = load_dataset(dataset)
    if not examples:
        return {
            "schema_version": RUN_SCHEMA_VERSION,
            "dataset": dataset,
            "strategy": strategy,
            "provider": provider,
            "model": model,
            "prompt_version": prompt_version,
            "n": 0,
            "n_examples": 0,
            "n_scored": 0,
            "n_unscored": 0,
            "n_judge_failed": 0,
            "n_generation_failed": 0,
            "failures": [],
            "aggregate": None,
            "aggregates": {},
            "retrieval_aggregate": None,
            "per_example": [],
            "error": f"No examples found in dataset '{dataset}'",
        }

    run_name = (
        f"eval-{dataset}-{strategy}-{provider or 'default'}-{model or 'default'}"
        f"-{prompt_version or 'default'}"
    )
    wandb_config = {
        "dataset": dataset,
        "strategy": strategy,
        "provider": provider,
        "model": model,
        "prompt_version": prompt_version,
        "n_examples": len(examples),
        "rubric_version": RUBRIC_VERSION,
    }

    semaphore = asyncio.Semaphore(settings.eval_concurrency)

    with wandb_run(name=run_name, config=wandb_config, tags=["eval", dataset]) as run:
        per_example = list(
            await asyncio.gather(
                *(
                    _run_example(
                        ex,
                        i,
                        strategy,
                        provider,
                        model,
                        prompt_version,
                        semaphore,
                        judge_answers,
                        access,
                        k_values,
                    )
                    for i, ex in enumerate(examples)
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
        retrieval_agg = aggregate_retrieval(
            [p["retrieval"] for p in per_example if p["retrieval"] is not None]
        )
        counts = _counts(per_example)
        failures = [p["error"] for p in per_example if p.get("error")]

        if failures:
            logger.warning(
                "Eval run '%s': %d of %d examples failed (%d generation, %d judge) and "
                "are excluded from the aggregates.",
                run_name,
                len(failures),
                len(per_example),
                counts["n_generation_failed"],
                counts["n_judge_failed"],
            )

        if run is not None:
            _log_to_wandb(run, per_example, strategy, agg, retrieval_agg)

    generated = [p for p in per_example if p["status"] != STATUS_GENERATION_FAILED]
    latencies = [p["latency_ms"] for p in generated]
    mode = _most_common(p.get("retrieval_mode") for p in per_example)
    r_config = retrieval_config(mode)
    judge_rows = [p["judge"] for p in per_example if p.get("judge")]

    result = {
        "schema_version": RUN_SCHEMA_VERSION,
        "dataset": dataset,
        "strategy": strategy,
        "provider": resolved_provider,
        "model": resolved_model,
        "requested_provider": provider,
        "requested_model": model,
        "prompt_version": prompt_version,
        "judge_model": settings.judge_model if judge_answers else None,
        "rubric_version": RUBRIC_VERSION if judge_answers else None,
        "rubric_fingerprint": rubric_fingerprint() if judge_answers else None,
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "retrieval_mode": r_config["mode"],
        "retrieval_config": r_config,
        "config_hash": config_hash(r_config),
        "k_values": list(k_values),
        "judged": judge_answers,
        "n": len(examples),
        **counts,
        "n_unscored": len(examples) - counts["n_scored"],
        "n_retrieval_unlabeled": len(generated) - counts["n_retrieval_scored"],
        "n_refused": sum(1 for p in per_example if p["refusal"]),
        "failures": failures,
        "aggregates": aggregate_rows(per_example, k_values),
        "by_question_type": breakdown(per_example, "question_type", k_values),
        "by_difficulty": breakdown(per_example, "difficulty", k_values),
        "aggregate": agg.model_dump() if agg else None,
        "retrieval_aggregate": retrieval_agg,
        "cost": {
            "mean_latency_ms": round(statistics.fmean(latencies)) if latencies else 0,
            "p50_latency_ms": percentile(latencies, 50),
            "p95_latency_ms": percentile(latencies, 95),
            "total_input_tokens": sum(p["input_tokens"] for p in per_example),
            "total_output_tokens": sum(p["output_tokens"] for p in per_example),
            "mean_tokens_per_question": (
                statistics.fmean(p["total_tokens"] for p in generated) if generated else None
            ),
            "judge_input_tokens": sum(j["input_tokens"] for j in judge_rows),
            "judge_output_tokens": sum(j["output_tokens"] for j in judge_rows),
            "judge_retries": sum(max(0, j["attempts"] - 1) for j in judge_rows),
        },
        "per_example": per_example,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    path = _persist_run(result)
    result["id"] = path.stem
    result["regression"] = _regression_on_save(result, path)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _regression_on_save(result: dict, path: Path) -> dict[str, Any]:
    """Compare a just-saved run with its baseline; save the report next to it.

    Never fails the run: a missing or malformed thresholds file is recorded as
    an error in the returned summary.
    """
    try:
        report = regression.regression_report_for(result)
        json_path, md_path = regression.save_report(report, path.stem, path.parent)
    except (OSError, ValueError) as e:
        logger.warning("Regression check for run %s failed: %s", path.stem, e)
        return {"status": "ERROR", "error": f"{type(e).__name__}: {e}"}
    return regression.report_summary(report, json_path, md_path)


def _log_to_wandb(
    run: Any,
    per_example: list[dict],
    strategy: str,
    agg: EvalScore | None,
    retrieval_agg: dict | None,
) -> None:
    try:
        log_eval_table(
            run,
            [
                {
                    "question": p["question"],
                    "answer": p["answer"],
                    "ideal_answer": p["ideal_answer"],
                    "strategy": strategy,
                    "status": p["status"],
                    "question_type": p["question_type"],
                    "scored": p["score"] is not None,
                    "latency_ms": p["latency_ms"],
                    **(p["score"] or {}),
                    **{
                        f"retrieval_{k}": v
                        for k, v in (p["retrieval"] or {}).items()
                        if k in ("precision", "recall", "mrr", "hit") or "@" in k
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
            metrics.update({f"retrieval_{k}": v for k, v in retrieval_agg.items()})
        if metrics:
            log_metrics(run, metrics)
    except Exception as e:
        logger.warning(f"Failed to log metrics to W&B: {e}")


def _most_common(values) -> str | None:
    """The most frequent non-empty value, or None."""
    counts = Counter(v for v in values if v)
    return counts.most_common(1)[0][0] if counts else None


def _safe_filename(s: str | None) -> str:
    return "".join(
        c if c.isalnum() or c in "._-" else "_" for c in (s or "default")
    )


def _persist_run(result: dict) -> Path:
    """Write a run to disk under a name unique even for concurrent runs.

    The name leads with a microsecond timestamp (so a sorted listing is
    chronological, which baseline selection relies on) and ends with a short
    random suffix: two runs of different strategies or models started in the
    same second used to overwrite each other.

    Returns:
        The path written.
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    name = (
        f"{stamp}-{_safe_filename(result['dataset'])}"
        f"-{_safe_filename(result.get('strategy'))}"
        f"-{_safe_filename(result.get('model'))}"
        f"-{_safe_filename(result['prompt_version'])}"
        f"-{uuid.uuid4().hex[:8]}.json"
    )
    path = RUNS_DIR / name
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path


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
