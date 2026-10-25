"""Follow-up questions and confidence for an advisor recommendation.

The description rarely states every fact the scorer uses, so the extracted
profile fills gaps with guesses. Asking about every gap would be tedious, so
this module asks only about the facts that matter: for each fact the caller has
not confirmed, it re-scores the strategies under every plausible answer and
keeps the question when an answer would change which strategy comes first (or,
failing that, when it moves the scores the most).

Everything is a pure function of the profile, like the scorer: no model call,
so follow-ups cost nothing and are reproducible. Question and option wording
lives in the frontend; the API returns stable ids and the override each option
stands for.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.advisor.profile import apply_overrides
from app.advisor.schemas import (
    AdviseOverrides,
    Confidence,
    FollowUpOption,
    FollowUpQuestion,
    Level,
    ProjectProfile,
    StrategyRecommendation,
)
from app.advisor.scoring import score_strategies

MAX_QUESTIONS = 5

# Score gap between the first two strategies, in points out of 100.
CLEAR_MARGIN = 15
THIN_MARGIN = 5


@dataclass(frozen=True)
class _Candidate:
    """A fact worth confirming, with the answers a client would pick from."""

    field: str
    options: tuple[tuple[str, dict[str, Any]], ...]
    # Asked even when it cannot change the ranking, because it changes the
    # deployment advice (where data may go).
    always_relevant: bool = False


CANDIDATES: tuple[_Candidate, ...] = (
    _Candidate(
        "question_mix",
        (
            ("facts", {"question_mix": {"single_fact": 80, "relational_multi_hop": 10,
                                        "exploratory_multi_step": 10}}),
            ("connect", {"question_mix": {"single_fact": 20, "relational_multi_hop": 70,
                                          "exploratory_multi_step": 10}}),
            ("research", {"question_mix": {"single_fact": 20, "relational_multi_hop": 20,
                                           "exploratory_multi_step": 60}}),
            ("mixed", {"question_mix": {"single_fact": 40, "relational_multi_hop": 30,
                                        "exploratory_multi_step": 30}}),
        ),
    ),
    _Candidate(
        "corpus_size_docs",
        (
            ("lt100", {"corpus_size_docs": 50}),
            ("lt1k", {"corpus_size_docs": 500}),
            ("lt10k", {"corpus_size_docs": 5_000}),
            ("lt100k", {"corpus_size_docs": 50_000}),
            ("more", {"corpus_size_docs": 200_000}),
        ),
    ),
    _Candidate(
        "entity_richness",
        (
            ("high", {"entity_richness": "high"}),
            ("medium", {"entity_richness": "medium"}),
            ("low", {"entity_richness": "low"}),
        ),
    ),
    _Candidate(
        "latency_budget_ms",
        (
            ("instant", {"latency_budget_ms": 2_000}),
            ("fast", {"latency_budget_ms": 5_000}),
            ("patient", {"latency_budget_ms": 15_000}),
            ("no_rush", {"latency_budget_ms": 60_000}),
        ),
    ),
    _Candidate(
        "cost_sensitivity",
        (
            ("low", {"cost_sensitivity": "low"}),
            ("medium", {"cost_sensitivity": "medium"}),
            ("high", {"cost_sensitivity": "high"}),
        ),
    ),
    _Candidate(
        "data_freshness",
        (
            ("static", {"data_freshness": "static"}),
            ("monthly", {"data_freshness": "monthly"}),
            ("weekly", {"data_freshness": "weekly"}),
            ("daily", {"data_freshness": "daily"}),
            ("realtime", {"data_freshness": "realtime"}),
        ),
    ),
    _Candidate(
        "compliance",
        (
            ("anywhere", {"compliance": []}),
            ("eu", {"compliance": ["EU only", "GDPR"]}),
            ("on_prem", {"compliance": ["on-prem"]}),
        ),
        always_relevant=True,
    ),
)


def _rescore(profile: ProjectProfile, override: dict[str, Any]) -> list[StrategyRecommendation]:
    """Rank the strategies as if the caller had given ``override``."""
    return score_strategies(apply_overrides(profile, AdviseOverrides(**override)))


def _current_option(candidate: _Candidate, profile: ProjectProfile) -> str | None:
    """The option the extracted profile is closest to, or None when it says nothing."""
    if candidate.field == "corpus_size_docs":
        if profile.corpus_size_docs is None:
            return None
        n = profile.corpus_size_docs
        return "lt100" if n < 100 else "lt1k" if n < 1_000 else "lt10k" if n < 10_000 \
            else "lt100k" if n < 100_000 else "more"
    if candidate.field == "latency_budget_ms":
        ms = profile.latency_budget_ms
        if ms is None:
            return None
        return "instant" if ms <= 2_000 else "fast" if ms <= 5_000 else "patient" \
            if ms < 30_000 else "no_rush"
    if candidate.field == "compliance":
        if not profile.compliance and not profile.data_residency:
            return None
        joined = " ".join([*profile.compliance, profile.data_residency or ""]).lower()
        if any(h in joined for h in ("prem", "self-host", "air-gap", "sur site")):
            return "on_prem"
        return "eu" if joined.strip() else "anywhere"
    if candidate.field == "question_mix":
        m = profile.question_mix
        shares = {"facts": m.single_fact, "connect": m.relational_multi_hop,
                  "research": m.exploratory_multi_step}
        top = max(shares, key=lambda k: shares[k])
        return top if shares[top] >= 50 else "mixed"
    value = getattr(profile, candidate.field)
    return str(value) if value is not None else None


def follow_up_questions(
    profile: ProjectProfile, ranked: list[StrategyRecommendation]
) -> list[FollowUpQuestion]:
    """Up to ``MAX_QUESTIONS`` questions whose answer could change the advice.

    Facts the caller already confirmed (``profile.overridden_fields``) are never
    asked again. Questions that can flip the top strategy come first, then the
    ones that move the scores the most, then deployment questions.
    """
    confirmed = set(profile.overridden_fields)
    current_top = ranked[0].strategy
    scored: list[tuple[int, int, FollowUpQuestion]] = []

    for cand in CANDIDATES:
        if cand.field in confirmed:
            continue
        tops: set[str] = set()
        spread = 0
        options: list[FollowUpOption] = []
        for option_id, override in cand.options:
            rescored = _rescore(profile, override)
            tops.add(rescored[0].strategy)
            by_name = {r.strategy: r.score for r in rescored}
            spread = max(spread, max(abs(by_name[r.strategy] - r.score) for r in ranked))
            options.append(
                FollowUpOption(id=option_id, overrides=override,
                               top_strategy=rescored[0].strategy)
            )
        changes_top = len(tops) > 1 or (tops and current_top not in tops)
        if not changes_top and spread < 5 and not cand.always_relevant:
            continue
        question = FollowUpQuestion(
            id=cand.field,
            options=options,
            current=_current_option(cand, profile),
            changes_recommendation=bool(changes_top),
        )
        priority = 0 if changes_top else 1 if not cand.always_relevant else 2
        scored.append((priority, -spread, question))

    scored.sort(key=lambda item: (item[0], item[1]))
    # When no answer can change the top pick and its lead is already clear, a
    # question that only nudges scores would not change the advice: skip it.
    settled = (
        not any(p == 0 for p, _, _ in scored)
        and len(ranked) > 1
        and ranked[0].score - ranked[1].score >= CLEAR_MARGIN
    )
    # Deployment questions keep a slot: they never move the ranking, but data
    # location decides where the whole stack may run.
    ranking = [q for p, _, q in scored if p == 0 or (p == 1 and not settled)]
    deployment = [q for p, _, q in scored if p == 2]
    return ranking[: MAX_QUESTIONS - len(deployment)] + deployment


def confidence(
    ranked: list[StrategyRecommendation], questions: list[FollowUpQuestion]
) -> Confidence:
    """How settled the top pick is, from the score margin and open questions."""
    margin = ranked[0].score - ranked[1].score if len(ranked) > 1 else 100
    open_flips = sum(1 for q in questions if q.changes_recommendation)
    level: Level
    if margin >= CLEAR_MARGIN and open_flips == 0:
        level = "high"
    elif margin < THIN_MARGIN or open_flips >= 2:
        level = "low"
    else:
        level = "medium"
    return Confidence(level=level, margin=margin, open_questions=open_flips)
