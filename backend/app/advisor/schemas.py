"""Request and response models for the strategy advisor (/advise)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

StrategyName = Literal["classic", "graph", "agentic"]
Level = Literal["low", "medium", "high"]
CorpusSize = Literal["tiny", "small", "medium", "large", "xlarge"]
Freshness = Literal["static", "monthly", "weekly", "daily", "realtime"]

CORPUS_BUCKETS: tuple[tuple[int, CorpusSize], ...] = (
    (100, "tiny"),
    (1_000, "small"),
    (10_000, "medium"),
    (100_000, "large"),
)


def corpus_bucket(docs: int) -> CorpusSize:
    """Map a document count to a corpus-size bucket.

    tiny < 100 <= small < 1k <= medium < 10k <= large < 100k <= xlarge.
    """
    for limit, bucket in CORPUS_BUCKETS:
        if docs < limit:
            return bucket
    return "xlarge"


def _normalize_level(value: object) -> object:
    """Accept ``med``/``mid`` as aliases for ``medium``."""
    if isinstance(value, str):
        v = value.strip().lower()
        return "medium" if v in {"med", "mid", "moderate"} else v
    return value


class QuestionMix(BaseModel):
    """Share of each question type, in percent. Normalized to sum to 100."""

    single_fact: float = Field(default=60.0, ge=0)
    relational_multi_hop: float = Field(default=25.0, ge=0)
    exploratory_multi_step: float = Field(default=15.0, ge=0)

    @model_validator(mode="after")
    def _normalize(self) -> QuestionMix:
        total = self.single_fact + self.relational_multi_hop + self.exploratory_multi_step
        if total <= 0:
            self.single_fact, self.relational_multi_hop, self.exploratory_multi_step = (
                60.0,
                25.0,
                15.0,
            )
            return self
        scale = 100.0 / total
        self.single_fact = round(self.single_fact * scale, 1)
        self.relational_multi_hop = round(self.relational_multi_hop * scale, 1)
        self.exploratory_multi_step = round(
            100.0 - self.single_fact - self.relational_multi_hop, 1
        )
        return self


class AdviseOverrides(BaseModel):
    """Structured facts the caller already knows. Each one beats extraction."""

    corpus_size_docs: int | None = Field(default=None, ge=0, le=100_000_000)
    languages: list[str] | None = Field(default=None, max_length=20)
    latency_budget_ms: int | None = Field(default=None, ge=100, le=600_000)
    cost_sensitivity: Level | None = None
    data_freshness: Freshness | None = None
    question_examples: list[str] | None = Field(default=None, max_length=50)
    compliance: list[str] | None = Field(
        default=None,
        max_length=20,
        description='Residency/compliance needs, e.g. "EU only", "GDPR", "on-prem".',
    )
    entity_richness: Level | None = None
    question_mix: QuestionMix | None = None

    @field_validator("cost_sensitivity", "entity_richness", mode="before")
    @classmethod
    def _levels(cls, v: object) -> object:
        return _normalize_level(v)


class AdviseRequest(BaseModel):
    """Body of POST /advise."""

    description: str = Field(
        ...,
        min_length=1,
        max_length=20_000,
        description="Free-text project description, in any language.",
    )
    overrides: AdviseOverrides = Field(default_factory=AdviseOverrides)


class ProjectProfile(BaseModel):
    """Normalized view of the project that the scorer consumes."""

    corpus_size: CorpusSize = "small"
    corpus_size_docs: int | None = None
    document_types: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=lambda: ["en"])
    question_mix: QuestionMix = Field(default_factory=QuestionMix)
    latency_budget_ms: int | None = None
    cost_sensitivity: Level = "medium"
    data_freshness: Freshness = "weekly"
    compliance: list[str] = Field(default_factory=list)
    data_residency: str | None = None
    entity_richness: Level = "medium"
    example_questions: list[str] = Field(default_factory=list)
    source: Literal["llm", "heuristic"] = "heuristic"
    overridden_fields: list[str] = Field(default_factory=list)

    @field_validator("cost_sensitivity", "entity_richness", mode="before")
    @classmethod
    def _levels(cls, v: object) -> object:
        return _normalize_level(v)


class SuggestedConfig(BaseModel):
    """Starting configuration for a strategy, mapped onto this repo's settings."""

    retrieval_top_k: int
    rerank_top_k: int
    rerank: bool
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    embedding_model: str
    models: dict[str, str]
    expected_relative_cost: float = Field(description="Tokens per question vs classic = 1.0.")
    expected_relative_latency: float = Field(description="Wall-clock vs classic = 1.0.")
    notes: list[str] = Field(default_factory=list)


class StrategyRecommendation(BaseModel):
    strategy: StrategyName
    rank: int
    score: int = Field(ge=0, le=100)
    reasons: list[str]
    tradeoffs: list[str]
    suggested_config: SuggestedConfig


class HybridRoute(BaseModel):
    question_type: Literal["single_fact", "relational_multi_hop", "exploratory_multi_step"]
    strategy: StrategyName
    share_pct: float


class HybridRouting(BaseModel):
    recommended: bool
    rationale: str
    routes: list[HybridRoute] = Field(default_factory=list)


class NextStep(BaseModel):
    summary: str
    validate_endpoint: str = "/advise/validate"
    suggested_strategies: list[StrategyName]
    example_payload: dict


class AdviseResponse(BaseModel):
    profile: ProjectProfile
    recommendations: list[StrategyRecommendation]
    top_strategy: StrategyName
    hybrid_routing: HybridRouting
    compliance_notes: list[str]
    next_step: NextStep


# --- validation -------------------------------------------------------------

MAX_VALIDATE_QUESTIONS = 20
MAX_VALIDATE_STRATEGIES = 3


class ValidateQuestion(BaseModel):
    question: str = Field(..., min_length=1, max_length=2_000)
    ideal_answer: str | None = Field(default=None, max_length=5_000)


class ValidateRequest(BaseModel):
    """Body of POST /advise/validate."""

    questions: list[ValidateQuestion] = Field(
        ..., min_length=1, max_length=MAX_VALIDATE_QUESTIONS
    )
    strategies: list[StrategyName] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_VALIDATE_STRATEGIES,
        description="Defaults to the top two strategies for `profile` (or a neutral profile).",
    )
    profile: ProjectProfile | None = Field(
        default=None, description="Profile from /advise, used to pick default strategies."
    )
    model: str | None = None
    judge: bool = Field(
        default=True,
        description="Run the LLM judge on questions that carry an ideal_answer.",
    )

    @field_validator("strategies")
    @classmethod
    def _dedupe(cls, v: list[StrategyName] | None) -> list[StrategyName] | None:
        if v is None:
            return None
        return list(dict.fromkeys(v))


class ValidateRow(BaseModel):
    question: str
    strategy: StrategyName
    answer: str
    refusal: bool
    error: str | None = None
    latency_ms: int
    input_tokens: int
    output_tokens: int
    answer_f1: float | None = None
    judge_score: float | None = None


class StrategyScorecard(BaseModel):
    strategy: StrategyName
    questions: int
    errors: int
    refusals: int
    refusal_rate: float
    avg_latency_ms: float
    max_latency_ms: int
    total_input_tokens: int
    total_output_tokens: int
    avg_tokens_per_question: float
    judged: int
    avg_judge_score: float | None = None
    avg_answer_f1: float | None = None


class ValidateResponse(BaseModel):
    strategies: list[StrategyName]
    scorecard: list[StrategyScorecard]
    measured_winner: StrategyName | None
    winner_reason: str
    rows: list[ValidateRow]
