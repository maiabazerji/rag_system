"""Deterministic, explainable scoring of the three RAG strategies for a project.

Everything here is a pure function of a ``ProjectProfile``: no I/O, no model
calls, no randomness. Each score is a base value plus named contributions, and
every contribution that moves the score also emits a sentence in ``reasons`` so
a reader can see exactly why a strategy ranked where it did.

The weights encode what the bundled baseline measured (README, "Baseline on
the bundled corpus", 41 documents / 107 chunks / 34 questions):

- Classic and Graph scored identically on retrieval. With ``RETRIEVAL_TOP_K=50``
  dense search already returns nearly half of a small corpus, so the graph walk
  has nothing left to add. Graph only earns its keep on large, entity-rich
  corpora with relational questions, and with a smaller ``RETRIEVAL_TOP_K``.
- Agentic used ~1.9x the tokens of Classic and refused 6 of 34 questions. It
  won retrieval precision, so it is justified only for ambiguous, multi-step
  research where the budget (cost and latency) is generous.
- Classic had the best recall, MRR and answer relevance with zero refusals and
  the lowest token count: it is the default for single-fact lookups, tight
  latency and cost-sensitive projects.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

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

# Measured on the bundled corpus (README baseline); classic = 1.0.
BASELINE_RELATIVE_TOKENS: dict[StrategyName, float] = {
    "classic": 1.0,
    "graph": round(397_431 / 389_251, 2),  # 1.02
    "agentic": round(727_101 / 389_251, 2),  # 1.87
}
BASELINE_RELATIVE_LATENCY: dict[StrategyName, float] = {
    "classic": 1.0,
    "graph": round(15_204 / 21_675, 2),  # 0.70
    "agentic": round(17_976 / 21_675, 2),  # 0.83
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


def score_classic(profile: ProjectProfile) -> _Score:
    sf, rel, exp = mix_fractions(profile)
    s = _Score(
        60,
        reasons=[
            "Strongest strategy on the bundled baseline: best recall, MRR and answer "
            "relevance, zero refusals and the lowest token count."
        ],
    )
    s.add(25 * sf, f"{sf:.0%} of questions are single-fact lookups, which dense retrieval "
                   "plus a cross-encoder rerank answers well." if sf >= 0.4 else None)
    if rel > 0:
        s.add(-15 * rel, "Relational questions spread across documents can be missed by "
                         "top-k similarity alone." if rel >= 0.4 else None)
    if exp > 0:
        s.add(-15 * exp, "Open-ended research questions get one retrieval pass, with no "
                         "chance to refine the query." if exp >= 0.4 else None)
    if is_tight_latency(profile):
        s.add(10, f"A {profile.latency_budget_ms} ms budget favours the single, predictable "
                  "retrieve-rerank-generate path.")
    if profile.cost_sensitivity == "high":
        s.add(8, "High cost sensitivity: one generation call per question, fewest tokens.")
    if SIZE_ORDINAL[profile.corpus_size] <= 1:
        s.add(5, "On a corpus this size dense search already covers most of it; the "
                 "baseline shows Graph adds nothing over Classic there.")
    if profile.data_freshness in ("daily", "realtime"):
        s.add(4, "Frequent updates only need re-embedding of changed documents; no graph "
                 "re-extraction.")

    s.tradeoffs += [
        "Single retrieval pass: questions that need facts from several documents "
        "may miss the second hop.",
        "The CPU cross-encoder rerank is the largest per-query latency cost; tune "
        "RETRIEVAL_TOP_K down or disable rerank for very tight budgets.",
    ]
    return s


def score_graph(profile: ProjectProfile) -> _Score:
    sf, rel, exp = mix_fractions(profile)
    s = _Score(45)
    s.add(40 * rel, f"{rel:.0%} of questions are relational/multi-hop, where walking "
                    "entity links surfaces chunks similarity search misses."
          if rel >= 0.3 else None)
    size = profile.corpus_size
    size_delta = {"tiny": -10, "small": -10, "medium": 5, "large": 12, "xlarge": 15}[size]
    s.add(
        size_delta,
        f"A {size} corpus is large enough that dense top-k no longer covers it, so the "
        "graph walk can add chunks." if size_delta > 0 else
        f"A {size} corpus is small: in the baseline (107 chunks) Graph scored exactly "
        "like Classic because dense search already returned half the corpus.",
    )
    ent_delta = {"low": -12, "medium": 3, "high": 12}[profile.entity_richness]
    s.add(
        ent_delta,
        "Documents are rich in named entities (people, companies, products, clauses), "
        "which is what the graph is built from." if ent_delta > 3 else
        ("Few structured entities: the extracted graph would be sparse." if ent_delta < 0
         else None),
    )
    if profile.data_freshness == "daily":
        s.add(-6, "Daily updates require re-running entity extraction on changed documents.")
    elif profile.data_freshness == "realtime":
        s.add(-12, "Real-time updates fight the offline graph-extraction step.")
    if profile.cost_sensitivity == "high":
        s.add(-5, "Graph extraction is an extra LLM pass over the whole corpus at ingest.")

    s.tradeoffs += [
        f"Offline entity extraction over every document ({settings.graph_extraction_model}) "
        "adds ingest cost and must be re-run when documents change.",
        "Only differentiates from Classic with a smaller RETRIEVAL_TOP_K; at the default "
        "of 50 it behaved identically on the baseline.",
        "Answer quality depends on entity-extraction quality; the walk is limited to one hop.",
    ]
    return s


def score_agentic(profile: ProjectProfile) -> _Score:
    sf, rel, exp = mix_fractions(profile)
    s = _Score(30)
    s.add(50 * exp, f"{exp:.0%} of questions are ambiguous, multi-step research, where "
                    "letting the model search, read and re-query pays off."
          if exp >= 0.3 else None)
    s.add(10 * rel)
    if sf > 0:
        s.add(-15 * sf, "Mostly single-fact questions: an agent loop spends ~2x tokens "
                        "for answers Classic already gets right." if sf >= 0.5 else None)
    cost_delta = {"low": 5, "medium": -8, "high": -20}[profile.cost_sensitivity]
    s.add(
        cost_delta,
        "Low cost sensitivity leaves room for the ~1.9x token spend." if cost_delta > 0
        else "Agentic used 1.9x the tokens of Classic on the baseline, which cuts against "
             f"{profile.cost_sensitivity} cost sensitivity.",
    )
    if is_tight_latency(profile):
        s.add(-20, f"Several sequential model round-trips do not fit a "
                   f"{profile.latency_budget_ms} ms budget reliably.")
    elif is_generous_latency(profile):
        s.add(5, "A generous latency budget absorbs multiple tool-use round-trips.")
    if not agentic_viable(profile):
        s.cap(40, "Capped: agentic is only justified with a generous cost and latency budget.")

    s.tradeoffs += [
        "~1.9x the tokens of Classic and 6/34 refusals on the baseline; refusals score "
        "near zero on faithfulness and relevance.",
        f"Latency varies with the number of tool calls (up to AGENTIC_MAX_ITERS="
        f"{settings.agentic_max_iters}).",
        "Wins retrieval and context precision: it fetches less, and what it fetches is "
        "more on-topic.",
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


def suggested_config(strategy: StrategyName, profile: ProjectProfile) -> SuggestedConfig:
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
        notes.append(
            f"Non-English content: use a multilingual embedding model ({MULTILINGUAL_EMBEDDING} "
            f"or {MULTILINGUAL_EMBEDDING_ALT}) and a multilingual cross-encoder, then re-ingest "
            "(the vector dimension changes)."
        )
    sf, _, _ = mix_fractions(profile)
    if profile.cost_sensitivity == "high" and sf >= 0.6 and strategy == "classic":
        generator = SMALL_MODEL
        notes.append(
            f"Mostly single-fact questions under high cost sensitivity: try {SMALL_MODEL} as "
            "the generator and confirm on /advise/validate that quality holds."
        )

    if strategy == "classic":
        rerank = not (
            profile.latency_budget_ms is not None
            and profile.latency_budget_ms <= VERY_TIGHT_LATENCY_MS
        )
        if not rerank:
            notes.append(
                "Rerank off: the CPU cross-encoder is the largest per-query latency cost. "
                "Re-enable it if accuracy on validation drops."
            )
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
        notes.append(
            "Keep RETRIEVAL_TOP_K small so the graph walk contributes chunks dense search "
            "missed; at 50 the baseline showed no difference from Classic."
        )
        notes.append("Run POST /graph/build after ingest, and again after document updates.")
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
            notes=[*notes, "Relative cost excludes the one-off graph extraction at ingest."],
        )

    notes.append(
        f"Keep AGENTIC_MAX_ITERS low ({settings.agentic_max_iters} today) and watch the "
        "refusal rate on /advise/validate; refusals were the baseline's main failure."
    )
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
        notes=[*notes, "Latency is from the bundled baseline and grows with each tool call."],
    )


# --- ranking, routing and notes ----------------------------------------------


def score_strategies(profile: ProjectProfile) -> list[StrategyRecommendation]:
    """Score and rank classic, graph and agentic for ``profile``.

    Ties break in the order classic, graph, agentic: cheapest and most
    predictable first.
    """
    scores = {
        "classic": score_classic(profile),
        "graph": score_graph(profile),
        "agentic": score_agentic(profile),
    }
    classic_value = scores["classic"].value
    if SIZE_ORDINAL[profile.corpus_size] <= 1:
        scores["graph"].cap(
            classic_value - 5,
            "Capped below Classic: on small corpora Graph RAG is Classic with extra steps.",
        )

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
            suggested_config=suggested_config(name, profile),
        )
        for i, name in enumerate(order)
    ]


def hybrid_routing(profile: ProjectProfile) -> HybridRouting:
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
        return HybridRouting(
            recommended=False,
            rationale="One question type dominates; a single strategy is simpler to run and "
            "evaluate.",
            routes=routes,
        )
    if len(distinct) < 2:
        return HybridRouting(
            recommended=False,
            rationale="The question mix is split, but the constraints (corpus size, entities, "
            "cost, latency) send every type to the same strategy.",
            routes=routes,
        )
    return HybridRouting(
        recommended=True,
        rationale="The question mix is split. Classify each question (a cheap classifier call "
        "or keyword rules) and route it: single-fact to Classic, and the other types to the "
        "strategy that fits them. Validate each route on its own slice of questions.",
        routes=routes,
    )


def compliance_notes(profile: ProjectProfile) -> list[str]:
    """Generic deployment notes for residency, compliance and language needs."""
    notes: list[str] = []
    if needs_eu_residency(profile):
        notes += [
            "EU residency: self-host Qdrant (vectors) and Postgres (API keys and usage) in an "
            "EU region or on your own infrastructure; both are open source and run from this "
            "repo's Docker setup.",
            "Claude is also offered through cloud providers (for example Amazon Bedrock and "
            "Google Cloud Vertex AI) that let you pick EU regions. Check the provider's current "
            "regional availability and data-processing terms for the model you need, and point "
            "the generator at that deployment.",
            "Embeddings and reranking run locally (sentence-transformers), so document text is "
            "not sent to a third party to be embedded; question and context text is sent to "
            "the LLM provider at answer time.",
        ]
    if any("gdpr" in c.lower() or "rgpd" in c.lower() for c in profile.compliance):
        notes += [
            "GDPR: you remain responsible for a lawful basis, a data-processing agreement with "
            "each processor, retention limits and erasure. Make sure deleting a document also "
            "removes its chunks from Qdrant and its triples from the entity graph.",
            "Tracing and logging (Langfuse, W&B, structured logs) can capture question and "
            "answer text: disable them or self-host them in the same region for personal data.",
        ]
    if needs_on_prem(profile):
        notes.append(
            "On-premises: the vector store, database, embeddings and reranker all run on your "
            "hardware. Claude is a hosted API, so a strict air-gap needs a local generator "
            "(GENERATOR_PROVIDER=local with Ollama); expect different quality and re-run "
            "/advise/validate."
        )
    if is_non_english(profile):
        langs = ", ".join(profile.languages)
        if is_multilingual_model(settings.embedding_model):
            notes.append(
                f"Languages ({langs}): the configured {settings.embedding_model} embedding "
                "model is multilingual, so questions and documents can be in different "
                f"languages. {MULTILINGUAL_EMBEDDING} ranks better at a higher CPU cost; "
                "switching models requires a re-ingest."
            )
        else:
            notes.append(
                f"Languages ({langs}): the configured {settings.embedding_model} embedding "
                f"model and {settings.reranker_model} reranker are English-only. Switch "
                f"EMBEDDING_MODEL to {MULTILINGUAL_EMBEDDING} or {MULTILINGUAL_EMBEDDING_ALT} "
                "and RERANKER_MODEL to a multilingual cross-encoder such as "
                f"{MULTILINGUAL_RERANKER}, then re-ingest."
            )
    if notes:
        notes.append(
            "These notes are general engineering guidance, not legal advice or a statement "
            "of certification; confirm requirements with your compliance team."
        )
    return notes
