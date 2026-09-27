"""Run a customer's own questions through candidate strategies and score the result.

Questions run sequentially per strategy, like /compare/strategies: the agentic
strategy issues many model calls and running strategies concurrently trips
provider rate limits.

Quality signals, when an ``ideal_answer`` is supplied:

- ``answer_f1``: token-overlap F1 between the answer and the ideal answer.
  Deterministic and free; a rough proxy for "said the same thing".
- ``judge_score``: mean of faithfulness and answer relevance from
  ``app.eval.judge.judge``. That judge is reference-free (it scores the answer
  against the question and retrieved context, not against the ideal answer),
  so it complements the F1 rather than replacing it. Judge tokens are not
  returned by ``judge()`` and so are not counted in the scorecard.
"""
from __future__ import annotations

import re
import statistics
from collections import Counter

from app.advisor.schemas import (
    ProjectProfile,
    StrategyName,
    StrategyScorecard,
    ValidateRequest,
    ValidateResponse,
    ValidateRow,
)
from app.advisor.scoring import score_strategies
from app.eval.judge import judge
from app.logging_config import get_structured_logger
from app.rag.generate import run_strategy_raw

logger = get_structured_logger(__name__)

_TOKEN = re.compile(r"\w+", re.UNICODE)


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token F1 between two strings, case-insensitive."""
    pred = _TOKEN.findall(prediction.lower())
    ref = _TOKEN.findall(reference.lower())
    if not pred or not ref:
        return 0.0
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    precision = common / len(pred)
    recall = common / len(ref)
    return round(2 * precision * recall / (precision + recall), 3)


def default_strategies(profile: ProjectProfile | None) -> list[StrategyName]:
    """Top two recommended strategies for ``profile`` (neutral profile if None)."""
    ranked = score_strategies(profile or ProjectProfile())
    return [r.strategy for r in ranked[:2]]


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 3) if values else None


def build_scorecard(strategy: StrategyName, rows: list[ValidateRow]) -> StrategyScorecard:
    """Aggregate one strategy's rows."""
    n = len(rows)
    ok = [r for r in rows if r.error is None]
    refusals = sum(1 for r in rows if r.refusal)
    total_in = sum(r.input_tokens for r in rows)
    total_out = sum(r.output_tokens for r in rows)
    judged = [r.judge_score for r in rows if r.judge_score is not None]
    f1s = [r.answer_f1 for r in rows if r.answer_f1 is not None]
    return StrategyScorecard(
        strategy=strategy,
        questions=n,
        errors=n - len(ok),
        refusals=refusals,
        refusal_rate=round(refusals / n, 3) if n else 0.0,
        avg_latency_ms=round(statistics.fmean([r.latency_ms for r in ok]), 1) if ok else 0.0,
        max_latency_ms=max((r.latency_ms for r in ok), default=0),
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        avg_tokens_per_question=round((total_in + total_out) / n, 1) if n else 0.0,
        judged=len(judged),
        avg_judge_score=_mean(judged),
        avg_answer_f1=_mean(f1s),
    )


def pick_winner(cards: list[StrategyScorecard]) -> tuple[StrategyName | None, str]:
    """Choose the measured winner and explain the rule that picked it.

    Quality signal, in order of preference: judge score, answer F1, then
    "did not refuse" (1 - refusal rate). Ties break on refusal rate, then
    tokens per question, then latency.
    """
    usable = [c for c in cards if c.errors < c.questions]
    if not usable:
        return None, "Every run failed; check that documents are ingested and the provider is up."

    if all(c.avg_judge_score is not None for c in usable):
        signal, label = (lambda c: c.avg_judge_score or 0.0), "judge score"
    elif all(c.avg_answer_f1 is not None for c in usable):
        signal, label = (lambda c: c.avg_answer_f1 or 0.0), "answer F1 vs ideal answers"
    else:
        signal, label = (lambda c: 1.0 - c.refusal_rate), "answer rate (no ideal answers given)"

    best = sorted(
        usable,
        key=lambda c: (-signal(c), c.refusal_rate, c.avg_tokens_per_question, c.avg_latency_ms),
    )[0]
    reason = (
        f"{best.strategy} has the best {label} ({signal(best):.2f}), "
        f"refusal rate {best.refusal_rate:.0%}, "
        f"{best.avg_tokens_per_question:.0f} tokens and {best.avg_latency_ms:.0f} ms per question."
    )
    return best.strategy, reason


async def run_validation(
    req: ValidateRequest,
) -> tuple[ValidateResponse, int, int]:
    """Run every question through every strategy.

    Returns:
        The response plus the total input and output tokens spent by strategies.
    """
    strategies = req.strategies or default_strategies(req.profile)
    rows: list[ValidateRow] = []
    total_in = total_out = 0

    for name in strategies:
        for q in req.questions:
            try:
                result, err = await run_strategy_raw(q.question, strategy=name, model=req.model)
            except Exception as e:
                logger.exception(
                    f"Advisor validation: {name} raised {type(e).__name__}: {e}",
                    extra_fields={"error_type": type(e).__name__, "strategy": name},
                )
                result, err = None, "This strategy failed. Check the backend logs."

            if err or result is None:
                rows.append(
                    ValidateRow(
                        question=q.question,
                        strategy=name,
                        answer=err or "No result returned.",
                        refusal=True,
                        error=err or "No result returned.",
                        latency_ms=0,
                        input_tokens=0,
                        output_tokens=0,
                    )
                )
                continue

            total_in += result.input_tokens
            total_out += result.output_tokens

            f1: float | None = None
            judge_score: float | None = None
            if q.ideal_answer:
                f1 = 0.0 if result.refusal else token_f1(result.answer, q.ideal_answer)
                if req.judge and not result.refusal:
                    scores = await judge(
                        q.question, result.answer, [s.quote for s in result.sources if s.quote]
                    )
                    if scores is not None:
                        judge_score = round(
                            (scores["faithfulness"] + scores["answer_relevance"]) / 2, 3
                        )
                elif req.judge and result.refusal:
                    judge_score = 0.0

            rows.append(
                ValidateRow(
                    question=q.question,
                    strategy=name,
                    answer=result.answer[:1500],
                    refusal=result.refusal,
                    latency_ms=result.latency_ms,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    answer_f1=f1,
                    judge_score=judge_score,
                )
            )

    cards = [build_scorecard(s, [r for r in rows if r.strategy == s]) for s in strategies]
    winner, reason = pick_winner(cards)
    return (
        ValidateResponse(
            strategies=strategies,
            scorecard=cards,
            measured_winner=winner,
            winner_reason=reason,
            rows=rows,
        ),
        total_in,
        total_out,
    )
