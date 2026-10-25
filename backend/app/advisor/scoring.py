"""Deterministic, explainable scoring of the three RAG strategies for a project.

Everything here is a pure function of a ``ProjectProfile``: no I/O, no model
calls, no randomness. Each score is a base value plus named contributions, and
every contribution that moves the score also emits a sentence in ``reasons`` so
a reader can see exactly why a strategy ranked where it did.

The weights encode two measurements:

- The current pipeline on ``golden_fr_business_v1`` (2026-10-25, 13 French
  business documents, 43 chunks, 40 questions, claude-sonnet-5, judge rubric
  2026-09.v2, graph built). Agentic had the best answer correctness (0.87 vs
  0.72 Classic, 0.75 Graph) and won every multi-hop, comparison, ambiguous and
  adversarial slice; single-fact questions were a tie (0.95 vs 0.97). It used
  ~1.6x Classic's tokens. Classic refused unanswerable questions best (0.80 vs
  0.60). Graph matched Classic's retrieval and refused 17% of answerable
  questions. Latency there was dominated by the CPU cross-encoder, so it says
  little about other hardware and is not used to move scores.
- The previous pipeline's baseline (README, "Historical baseline": 41
  documents, 107 chunks, 34 questions), where Classic and Graph scored
  identically because dense top-k already covered a small corpus. That is why
  Graph is only favoured on large, entity-rich corpora.

One small corpus is thin evidence: the ranking is a prior, and
``/advise/validate`` measures the client's own data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.advisor.messages import msg, pct
from app.advisor.schemas import (
    HybridRoute,
    HybridRouting,
    ProjectProfile,
    StrategyName,
    StrategyRecommendation,
    SuggestedConfig,
)
from app.config import settings
from app.rag.embed import is_multilingual_model

STRATEGIES: tuple[StrategyName, ...] = ("classic", "graph", "agentic")

# Measured on golden_fr_business_v1 (tokens per question, p50 latency); classic = 1.0.
# Latency includes the CPU cross-encoder, which Classic and Graph run and Agentic does not.
BASELINE_RELATIVE_TOKENS: dict[StrategyName, float] = {
    "classic": 1.0,
    "graph": round(6_732 / 6_336, 2),  # 1.06
    "agentic": round(10_367 / 6_336, 2),  # 1.64
}
BASELINE_RELATIVE_LATENCY: dict[StrategyName, float] = {
    "classic": 1.0,
    "graph": round(27_423 / 27_251, 2),  # 1.01
    "agentic": round(9_207 / 27_251, 2),  # 0.34
}

SIZE_ORDINAL = {"tiny": 0, "small": 1, "medium": 2, "large": 3, "xlarge": 4}
LEVEL_ORDINAL = {"low": 0, "medium": 1, "high": 2}

TIGHT_LATENCY_MS = 5_000
VERY_TIGHT_LATENCY_MS = 2_000
GENEROUS_LATENCY_MS = 30_000

MULTILINGUAL_EMBEDDING = "BAAI/bge-m3"
MULTILINGUAL_EMBEDDING_ALT = "intfloat/multilingual-e5-large"
MULTILINGUAL_RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
SMALL_MODEL = "claude-haiku-4-5"

SHORT_DOC_HINTS = ("faq", "ticket", "chat", "email", "message", "q&a", "snippet")
LONG_DOC_HINTS = ("contract", "legal", "manual", "report", "paper", "book", "regulation", "spec")

EU_PATTERN = re.compile(
    r"\b(eu|ue|europe\w*|gdpr|rgpd|union europ\w*|france|germany|allemagne|schrems)\b"
)
ONPREM_HINTS = ("on-prem", "on prem", "onprem", "premise", "self-host", "air-gap", "airgap",
                "sovereign", "souverain", "sur site")


@dataclass
class _Score:
    """Accumulates a score and the sentences that justify it."""

    value: float
    reasons: list[str] = field(default_factory=list)
    tradeoffs: list[str] = field(default_factory=list)

    def add(self, delta: float, reason: str | None = None) -> None:
        self.value += delta
        if reason:
            self.reasons.append(reason)

    def cap(self, ceiling: float, reason: str) -> None:
        if self.value > ceiling:
            self.value = ceiling
            self.reasons.append(reason)

    @property
    def clamped(self) -> int:
        return int(round(max(0.0, min(100.0, self.value))))


# --- small profile predicates (public so tests and the router can reuse them) ---


def mix_fractions(profile: ProjectProfile) -> tuple[float, float, float]:
    """Return (single_fact, relational, exploratory) as fractions of 1."""
    m = profile.question_mix
    return m.single_fact / 100, m.relational_multi_hop / 100, m.exploratory_multi_step / 100


def is_tight_latency(profile: ProjectProfile) -> bool:
    return profile.latency_budget_ms is not None and profile.latency_budget_ms <= TIGHT_LATENCY_MS


def is_generous_latency(profile: ProjectProfile) -> bool:
    return (
        profile.latency_budget_ms is not None and profile.latency_budget_ms >= GENEROUS_LATENCY_MS
    )


def is_non_english(profile: ProjectProfile) -> bool:
    return any(not lang.lower().startswith("en") for lang in profile.languages)


def needs_eu_residency(profile: ProjectProfile) -> bool:
    haystack = " ".join([*profile.compliance, profile.data_residency or ""]).lower()
    return bool(EU_PATTERN.search(haystack))


def needs_on_prem(profile: ProjectProfile) -> bool:
    haystack = " ".join([*profile.compliance, profile.data_residency or ""]).lower()
    return any(h in haystack for h in ONPREM_HINTS)


def graph_viable(profile: ProjectProfile) -> bool:
    """Graph can differentiate only on a big enough, entity-bearing corpus."""
    return SIZE_ORDINAL[profile.corpus_size] >= 2 and profile.entity_richness != "low"


def agentic_viable(profile: ProjectProfile) -> bool:
    """Agentic is affordable only without a tight latency or cost constraint."""
    return not is_tight_latency(profile) and profile.cost_sensitivity != "high"


# --- per-strategy scores ------------------------------------------------------


def score_classic(profile: ProjectProfile, lang: str = "en") -> _Score:
    sf, rel, exp = mix_fractions(profile)
    s = _Score(60, reasons=[msg("classic.baseline", lang)])
    s.add(25 * sf, msg("classic.single_fact", lang, pct=pct(sf, lang)) if sf >= 0.4 else None)
    if rel > 0:
        s.add(-15 * rel, msg("classic.relational", lang) if rel >= 0.4 else None)
    if exp > 0:
        s.add(-15 * exp, msg("classic.exploratory", lang) if exp >= 0.4 else None)
    if is_tight_latency(profile):
        s.add(10, msg("classic.tight_latency", lang, ms=profile.latency_budget_ms))
    if profile.cost_sensitivity == "high":
        s.add(8, msg("classic.high_cost", lang))
    if SIZE_ORDINAL[profile.corpus_size] <= 1:
        s.add(5, msg("classic.small_corpus", lang))
    if profile.data_freshness in ("daily", "realtime"):
        s.add(4, msg("classic.fresh", lang))

    s.tradeoffs += [
        msg("classic.tradeoff.single_pass", lang),
        msg("classic.tradeoff.rerank", lang),
    ]
    return s


def score_graph(profile: ProjectProfile, lang: str = "en") -> _Score:
    sf, rel, exp = mix_fractions(profile)
    s = _Score(45)
    s.add(40 * rel, msg("graph.relational", lang, pct=pct(rel, lang)) if rel >= 0.3 else None)
    size = profile.corpus_size
    size_label = msg(f"size.{size}", lang)
    size_delta = {"tiny": -10, "small": -10, "medium": 5, "large": 12, "xlarge": 15}[size]
    s.add(
        size_delta,
        msg("graph.size_large", lang, size=size_label) if size_delta > 0
        else msg("graph.size_small", lang, size=size_label),
    )
    ent_delta = {"low": -12, "medium": 3, "high": 12}[profile.entity_richness]
    s.add(
        ent_delta,
        msg("graph.entities_rich", lang) if ent_delta > 3 else
        (msg("graph.entities_sparse", lang) if ent_delta < 0 else None),
    )
    if profile.data_freshness == "daily":
        s.add(-6, msg("graph.daily", lang))
    elif profile.data_freshness == "realtime":
        s.add(-12, msg("graph.realtime", lang))
    if profile.cost_sensitivity == "high":
        s.add(-5, msg("graph.high_cost", lang))

    s.tradeoffs += [
        msg("graph.tradeoff.extraction", lang, model=settings.graph_extraction_model),
        msg("graph.tradeoff.top_k", lang),
        msg("graph.tradeoff.quality", lang),
    ]
    return s


def score_agentic(profile: ProjectProfile, lang: str = "en") -> _Score:
    sf, rel, exp = mix_fractions(profile)
    s = _Score(30)
    s.add(50 * exp, msg("agentic.exploratory", lang, pct=pct(exp, lang)) if exp >= 0.3 else None)
    s.add(25 * rel, msg("agentic.relational", lang) if rel >= 0.3 else None)
    if sf > 0:
        s.add(-15 * sf, msg("agentic.single_fact", lang) if sf >= 0.5 else None)
    # Scaled to the measured ~1.6x tokens (was -8 / -20 at the old ~1.9x).
    cost_delta = {"low": 5, "medium": -5, "high": -15}[profile.cost_sensitivity]
    s.add(
        cost_delta,
        msg("agentic.low_cost", lang) if cost_delta > 0
        else msg("agentic.cost", lang, level=msg(f"level.{profile.cost_sensitivity}", lang)),
    )
    if is_tight_latency(profile):
        s.add(-20, msg("agentic.tight_latency", lang, ms=profile.latency_budget_ms))
    elif is_generous_latency(profile):
        s.add(5, msg("agentic.generous_latency", lang))
    if not agentic_viable(profile):
        s.cap(40, msg("agentic.capped", lang))

    s.tradeoffs += [
        msg("agentic.tradeoff.tokens", lang),
        msg("agentic.tradeoff.latency", lang, iters=settings.agentic_max_iters),
        msg("agentic.tradeoff.precision", lang),
    ]
    return s


# --- suggested configuration --------------------------------------------------


def _chunk_size(profile: ProjectProfile) -> int:
    types = " ".join(profile.document_types).lower()
    if any(h in types for h in SHORT_DOC_HINTS) and not any(h in types for h in LONG_DOC_HINTS):
        return 300
    if any(h in types for h in LONG_DOC_HINTS):
        return 800
    return 600


def suggested_config(
    strategy: StrategyName, profile: ProjectProfile, lang: str = "en"
) -> SuggestedConfig:
    """Starting configuration for ``strategy`` on this project, in repo settings terms."""
    chunk = _chunk_size(profile)
    overlap = int(round(chunk * 0.13))
    multilingual = is_non_english(profile)
    embedding = MULTILINGUAL_EMBEDDING if multilingual else settings.embedding_model
    reranker = MULTILINGUAL_RERANKER if multilingual else settings.reranker_model
    size = SIZE_ORDINAL[profile.corpus_size]
    rerank_top_k = 5 if profile.cost_sensitivity == "high" else 8
    generator = settings.generator_model
    notes: list[str] = []

    if multilingual:
        notes.append(msg(
            "config.multilingual", lang,
            embedding=MULTILINGUAL_EMBEDDING, embedding_alt=MULTILINGUAL_EMBEDDING_ALT,
        ))
    sf, _, _ = mix_fractions(profile)
    if profile.cost_sensitivity == "high" and sf >= 0.6 and strategy == "classic":
        generator = SMALL_MODEL
        notes.append(msg("config.small_model", lang, model=SMALL_MODEL))

    if strategy == "classic":
        rerank = not (
            profile.latency_budget_ms is not None
            and profile.latency_budget_ms <= VERY_TIGHT_LATENCY_MS
        )
        if not rerank:
            notes.append(msg("config.rerank_off", lang))
        return SuggestedConfig(
            retrieval_top_k=[20, 30, 50, 50, 50][size],
            rerank_top_k=rerank_top_k,
            rerank=rerank,
            chunk_size_tokens=chunk,
            chunk_overlap_tokens=overlap,
            embedding_model=embedding,
            models={"generator": generator, "reranker": reranker},
            expected_relative_cost=BASELINE_RELATIVE_TOKENS["classic"],
            expected_relative_latency=BASELINE_RELATIVE_LATENCY["classic"],
            notes=notes,
        )

    if strategy == "graph":
        notes.append(msg("config.graph_top_k", lang))
        notes.append(msg("config.graph_build", lang))
        return SuggestedConfig(
            retrieval_top_k=15 if size >= 3 else 10,
            rerank_top_k=rerank_top_k,
            rerank=True,
            chunk_size_tokens=chunk,
            chunk_overlap_tokens=overlap,
            embedding_model=embedding,
            models={
                "generator": generator,
                "graph_extraction": settings.graph_extraction_model,
                "reranker": reranker,
            },
            expected_relative_cost=BASELINE_RELATIVE_TOKENS["graph"],
            expected_relative_latency=BASELINE_RELATIVE_LATENCY["graph"],
            notes=[*notes, msg("config.graph_cost", lang)],
        )

    notes.append(msg("config.agentic_iters", lang, iters=settings.agentic_max_iters))
    return SuggestedConfig(
        retrieval_top_k=5,
        rerank_top_k=rerank_top_k,
        rerank=False,
        chunk_size_tokens=chunk,
        chunk_overlap_tokens=overlap,
        embedding_model=embedding,
        models={"agent": settings.agentic_model},
        expected_relative_cost=BASELINE_RELATIVE_TOKENS["agentic"],
        expected_relative_latency=BASELINE_RELATIVE_LATENCY["agentic"],
        notes=[*notes, msg("config.agentic_latency", lang)],
    )


# --- ranking, routing and notes ----------------------------------------------


def score_strategies(profile: ProjectProfile, lang: str = "en") -> list[StrategyRecommendation]:
    """Score and rank classic, graph and agentic for ``profile``.

    Ties break in the order classic, graph, agentic: cheapest and most
    predictable first. ``lang`` only changes the wording, never the scores.
    """
    scores = {
        "classic": score_classic(profile, lang),
        "graph": score_graph(profile, lang),
        "agentic": score_agentic(profile, lang),
    }
    classic_value = scores["classic"].value
    if SIZE_ORDINAL[profile.corpus_size] <= 1:
        scores["graph"].cap(classic_value - 5, msg("graph.capped_small", lang))

    order = sorted(
        STRATEGIES, key=lambda n: (-scores[n].clamped, STRATEGIES.index(n))
    )
    return [
        StrategyRecommendation(
            strategy=name,
            rank=i + 1,
            score=scores[name].clamped,
            reasons=scores[name].reasons,
            tradeoffs=scores[name].tradeoffs,
            suggested_config=suggested_config(name, profile, lang),
        )
        for i, name in enumerate(order)
    ]


def hybrid_routing(profile: ProjectProfile, lang: str = "en") -> HybridRouting:
    """Suggest routing by question type when the question mix is split."""
    sf, rel, exp = mix_fractions(profile)
    shares = {"single_fact": sf, "relational_multi_hop": rel, "exploratory_multi_step": exp}
    split = max(shares.values()) < 0.7 and sum(1 for v in shares.values() if v >= 0.25) >= 2

    targets: dict[str, StrategyName] = {
        "single_fact": "classic",
        "relational_multi_hop": "graph" if graph_viable(profile) else "classic",
        "exploratory_multi_step": "agentic" if agentic_viable(profile) else "classic",
    }
    routes = [
        HybridRoute(question_type=qt, strategy=targets[qt], share_pct=round(v * 100, 1))  # type: ignore[arg-type]
        for qt, v in shares.items()
        if v > 0
    ]
    distinct = {r.strategy for r in routes if r.share_pct >= 20}

    if not split:
        return HybridRouting(recommended=False, rationale=msg("hybrid.single", lang), routes=routes)
    if len(distinct) < 2:
        return HybridRouting(
            recommended=False, rationale=msg("hybrid.collapsed", lang), routes=routes
        )
    return HybridRouting(
        recommended=True, rationale=msg("hybrid.recommended", lang), routes=routes
    )


def compliance_notes(profile: ProjectProfile, lang: str = "en") -> list[str]:
    """Generic deployment notes for residency, compliance and language needs."""
    notes: list[str] = []
    if needs_eu_residency(profile):
        notes += [
            msg("notes.eu_hosting", lang),
            msg("notes.eu_claude", lang),
            msg("notes.eu_embeddings", lang),
        ]
    if any("gdpr" in c.lower() or "rgpd" in c.lower() for c in profile.compliance):
        notes += [msg("notes.gdpr", lang), msg("notes.gdpr_tracing", lang)]
    if any("sensitive" in c.lower() for c in profile.compliance):
        notes.append(msg("notes.sensitive", lang))
    if needs_on_prem(profile):
        notes.append(msg("notes.on_prem", lang))
    if is_non_english(profile):
        langs = ", ".join(profile.languages)
        if is_multilingual_model(settings.embedding_model):
            notes.append(msg(
                "notes.languages_multilingual", lang,
                langs=langs, embedding=settings.embedding_model, better=MULTILINGUAL_EMBEDDING,
            ))
        else:
            notes.append(msg(
                "notes.languages_english_only", lang,
                langs=langs,
                embedding=settings.embedding_model,
                reranker=settings.reranker_model,
                multi=MULTILINGUAL_EMBEDDING,
                multi_alt=MULTILINGUAL_EMBEDDING_ALT,
                multi_reranker=MULTILINGUAL_RERANKER,
            ))
    if notes:
        notes.append(msg("notes.disclaimer", lang))
    return notes
