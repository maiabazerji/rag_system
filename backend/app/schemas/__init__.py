"""Pydantic schemas for EvalRAG API requests and responses.

These models define the contract between frontend and backend, enforce validation,
and provide structured documentation for the API.
"""

from typing import Literal, get_args

from pydantic import BaseModel, Field, field_validator, model_validator

from app.config import settings
from app.prompts.loader import prompt_version_exists

Strategy = Literal["classic", "graph", "agentic"]
Provider = Literal["anthropic", "openai", "local"]


def effective_provider(provider: str | None, strategy: str) -> str:
    """The provider a request will actually run on.

    Graph and agentic need Anthropic features, so they always run there.
    """
    if strategy in ("graph", "agentic"):
        return "anthropic"
    return provider or settings.generator_provider


def _check_model(model: str | None, provider: str) -> None:
    """Reject a model override that is not allowed for the provider.

    Raises:
        ValueError: If ``model`` is set and not in the provider's allowlist.
    """
    if model is None:
        return
    allowed = settings.allowed_models(provider)
    if model not in allowed:
        raise ValueError(
            f"model '{model}' is not allowed for provider '{provider}'. "
            f"Allowed: {', '.join(sorted(allowed))}"
        )


class _RunOptionsMixin(BaseModel):
    """Shared validation for prompt_version and model/provider overrides.

    Model IDs are free text on the wire, so they are checked against an
    allowlist (see ``Settings.allowed_models``): an arbitrary string would
    otherwise reach the provider, and cost accounting, verbatim.
    """

    @field_validator("prompt_version", check_fields=False)
    @classmethod
    def _prompt_version_exists(cls, v: str | None) -> str | None:
        if v is not None and not prompt_version_exists(v):
            raise ValueError(f"unknown prompt_version '{v}'")
        return v

    @model_validator(mode="after")
    def _model_is_allowed(self):
        model = getattr(self, "model", None)
        if hasattr(self, "strategies"):
            # /compare/strategies runs every strategy on Anthropic.
            _check_model(model, "anthropic")
        else:
            _check_model(
                model,
                effective_provider(
                    getattr(self, "provider", None), getattr(self, "strategy", "classic")
                ),
            )
        return self


class _QuestionMixin(BaseModel):
    """Shared validation: a question must contain more than whitespace."""

    @field_validator("question", check_fields=False)
    @classmethod
    def _question_is_not_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("question cannot be blank")
        return stripped


class Source(BaseModel):
    """A source chunk used to ground an answer.

    Attributes:
        chunk_id: Unique identifier for the retrieved chunk.
        quote: The exact text from the chunk that supports the answer.
        score: Retrieval score (0-1) indicating relevance confidence.
               None if score is not applicable for this strategy.

    Example:
        {
            "chunk_id": "doc_123_chunk_5",
            "quote": "Claude is an AI assistant made by Anthropic.",
            "document": "about_claude.md",
            "score": 0.92
        }
    """

    chunk_id: str = Field(description="Unique chunk identifier")
    quote: str = Field(description="Exact text from the chunk supporting the answer")
    document: str | None = Field(
        default=None,
        description="Filename this chunk came from. Used to score retrieval "
        "against a golden example's expected_sources, and to show a readable "
        "citation instead of a chunk hash.",
    )
    score: float | None = Field(
        default=None, ge=0, le=1, description="Relevance score (0-1)"
    )
    # Citation metadata (additive; every field is optional so older clients and
    # payloads without these fields keep working).
    handle: str | None = Field(
        default=None,
        description="Citation handle the answer text uses for this source, e.g. 'S1'. "
        "Inline markers like [S1] in the answer refer to it.",
    )
    document_id: str | None = Field(default=None, description="Parent document id")
    title: str | None = Field(default=None, description="Document title, when known")
    section: str | None = Field(default=None, description="Section or heading, when known")
    page: int | None = Field(default=None, description="First page of the chunk, when known")
    relevance_score: float | None = Field(
        default=None,
        description="Raw retrieval score of this chunk: the rerank score when "
        "available, else the fused or dense score. Not bounded to 0-1.",
    )


class RetrievedChunkDiagnostics(BaseModel):
    """Where one final context chunk came from in the retrieval pipeline."""

    chunk_id: str
    doc_id: str
    dense_rank: int | None = Field(
        default=None, description="1-based rank in the dense list; null if absent"
    )
    sparse_rank: int | None = Field(
        default=None, description="1-based rank in the BM25 list; null if absent"
    )
    fused_score: float | None = Field(
        default=None, description="Reciprocal Rank Fusion score"
    )
    rerank_score: float | None = Field(
        default=None, description="Reranker score; null when not reranked"
    )


class RetrievalStageCounts(BaseModel):
    """Candidates coming out of each retrieval stage."""

    dense: int = 0
    sparse: int = 0
    fused: int = 0
    reranked: int = 0
    final: int = 0


class RetrievalStageLatency(BaseModel):
    """Wall time of each retrieval stage, in milliseconds.

    Dense and sparse run concurrently, so they overlap rather than add up.
    """

    preprocess: float = 0.0
    dense: float = 0.0
    sparse: float = 0.0
    fusion: float = 0.0
    rerank: float = 0.0


class RetrievalDiagnostics(BaseModel):
    """How the context behind an answer was retrieved. Never contains chunk text."""

    mode: Literal["dense", "sparse", "hybrid"]
    reranker: Literal["cross-encoder", "bm25-fallback", "none"] = "none"
    query_truncated: bool = False
    counts: RetrievalStageCounts = Field(default_factory=RetrievalStageCounts)
    latency_ms: RetrievalStageLatency = Field(default_factory=RetrievalStageLatency)
    degraded: list[str] = Field(
        default_factory=list,
        description="Stages that failed and were skipped, e.g. ['sparse']",
    )
    chunks: list[RetrievedChunkDiagnostics] = Field(
        default_factory=list, description="The final context chunks, in rank order"
    )


GroundingStatus = Literal["answered", "partial", "insufficient_context"]


class Claim(BaseModel):
    """One factual statement of an answer and the sources that back it."""

    text: str = Field(description="The claim, as stated in the answer")
    citations: list[str] = Field(
        default_factory=list, description="Validated citation handles, e.g. ['S1', 'S3']"
    )
    chunk_ids: list[str] = Field(
        default_factory=list, description="Chunk ids behind `citations`, in the same order"
    )
    supported: bool = Field(
        default=True, description="Whether the model marked the claim as supported by the context"
    )


class _GroundingFields(BaseModel):
    """Grounding and citation fields shared by Answer and StrategyComparison."""

    grounded: bool = Field(
        default=False,
        description="True when status is 'answered', every claim cites at least one "
        "retrieved chunk, and no citation pointed outside the context.",
    )
    status: GroundingStatus | None = Field(
        default=None,
        description="'answered', 'partial' or 'insufficient_context'. None when no "
        "answer was generated (e.g. a provider error).",
    )
    claims: list[Claim] = Field(default_factory=list, description="Claims with their citations")
    invalid_citations: list[str] = Field(
        default_factory=list,
        description="Citation handles the model used that were not in its context; "
        "they were removed from the answer text.",
    )
    citation_count: int = Field(
        default=0, ge=0, description="Number of distinct valid sources cited"
    )
    unsupported_notes: str | None = Field(
        default=None,
        description="What the model said the context does not cover, if anything.",
    )


class Answer(_GroundingFields):
    """A generated answer with sources and confidence.

    Attributes:
        question: The question that was asked.
        answer: The generated answer text.
        sources: List of chunks grounding the answer.
        confidence: Model's confidence in the answer (0-1).
        refusal: Whether the model refused to answer.
        provider: The LLM provider used (anthropic, openai, local).
        model: The specific model ID used for generation.

    Example:
        {
            "question": "Who built Claude?",
            "answer": "Claude was built by Anthropic, an AI safety company.",
            "sources": [
                {
                    "chunk_id": "doc_1_chunk_0",
                    "quote": "Claude is an AI assistant made by Anthropic.",
                    "score": 0.95
                }
            ],
            "confidence": 0.89,
            "refusal": false,
            "provider": "anthropic",
            "model": "claude-sonnet-5"
        }
    """

    question: str = Field(description="The question asked")
    answer: str = Field(description="The generated answer text")
    sources: list[Source] = Field(
        min_length=1,
        description="Chunks the answer cites (validated), in order of first citation",
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="Evidence score (0-1) derived from citation coverage, grounding "
        "status and the relevance of cited chunks; not a calibrated probability",
    )
    refusal: bool = Field(
        default=False, description="Whether the model refused to answer"
    )
    provider: str | None = Field(
        default=None, description="LLM provider (anthropic, openai, local)"
    )
    model: str | None = Field(
        default=None, description="Specific model ID used (e.g. claude-sonnet-5)"
    )
    latency_ms: int = Field(
        default=0, ge=0, description="End-to-end time to produce this answer (ms)"
    )
    input_tokens: int = Field(
        default=0, ge=0, description="Tokens sent to the model"
    )
    output_tokens: int = Field(
        default=0, ge=0, description="Tokens generated by the model"
    )
    trace_id: str | None = Field(
        default=None, description="Request trace id; fetch it from /traces/{trace_id}"
    )
    retrieval: RetrievalDiagnostics | None = Field(
        default=None, description="Retrieval pipeline diagnostics, when the strategy has them"
    )


class Chunk(BaseModel):
    """A document chunk in the vector store.

    Attributes:
        id: Unique identifier for this chunk.
        doc_id: ID of the source document.
        text: The chunk text content.
        tokens: Approximate token count.
        section: Optional section/heading within the document.
        metadata: Additional metadata (source, page_number, etc.).

    Example:
        {
            "id": "doc_123_chunk_5",
            "doc_id": "doc_123",
            "text": "Claude is an AI assistant made by Anthropic...",
            "tokens": 150,
            "section": "About",
            "metadata": {
                "source": "blog.md",
                "page_number": 2
            }
        }
    """

    id: str = Field(description="Unique chunk identifier")
    doc_id: str = Field(description="Parent document ID")
    text: str = Field(description="Chunk content text")
    tokens: int = Field(ge=0, description="Approximate token count")
    section: str | None = Field(
        default=None, description="Section/heading within document"
    )
    metadata: dict = Field(
        default_factory=dict, description="Additional metadata (source, page, etc.)"
    )


class AskRequest(_QuestionMixin, _RunOptionsMixin):
    """Request to ask a question and get an answer.

    Attributes:
        question: The question to answer.
        top_k: Number of chunks to retrieve (1-100, default 8).
        provider: Override default LLM provider (anthropic, openai, local).
        model: Override default model ID.
        prompt_version: Optional prompt variant identifier.
        strategy: RAG strategy to use (classic, graph, agentic).

    Example:
        {
            "question": "How does Claude handle security?",
            "top_k": 10,
            "strategy": "classic",
            "model": "claude-sonnet-5"
        }
    """

    question: str = Field(
        min_length=1, max_length=4000, description="Question to answer"
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Chunks to keep after reranking. Defaults to RERANK_TOP_K.",
    )
    provider: Provider | None = Field(default=None, description="Override LLM provider")
    model: str | None = Field(
        default=None, max_length=128, description="Override model ID"
    )
    prompt_version: str | None = Field(
        default=None,
        max_length=64,
        pattern=r"^[A-Za-z0-9._-]+$",
        description="Optional prompt variant",
    )
    strategy: Strategy = Field(
        default="classic", description="RAG strategy: classic, graph, or agentic"
    )


class CompareVariant(_RunOptionsMixin):
    """One model/provider configuration in a comparison.

    Example:
        {"provider": "anthropic", "model": "claude-sonnet-5", "strategy": "classic"}
    """

    provider: Provider | None = Field(default=None, description="LLM provider")
    model: str | None = Field(default=None, max_length=128, description="Model ID")
    prompt_version: str | None = Field(
        default=None,
        max_length=64,
        pattern=r"^[A-Za-z0-9._-]+$",
        description="Prompt template version",
    )
    strategy: Strategy = Field(default="classic", description="RAG strategy")


class CompareRequest(_QuestionMixin):
    """Request to compare multiple model/provider combinations.

    Attributes:
        question: The question to compare on.
        variants: List of configurations to compare
                 (provider, model, prompt_version, etc.).

    Example:
        {
            "question": "What is RAG?",
            "variants": [
                {"provider": "anthropic", "model": "claude-sonnet-5"},
                {"provider": "openai", "model": "gpt-4o-mini"}
            ]
        }
    """

    question: str = Field(
        min_length=1, max_length=4000, description="Question to compare on"
    )
    variants: list[CompareVariant] = Field(
        min_length=1,
        max_length=6,
        description="Model/provider configurations to compare",
    )


_NAME_PATTERN = r"^[A-Za-z0-9_-][A-Za-z0-9._-]*$"


class CompareReference(BaseModel):
    """A caller-supplied reference for scoring a strategy comparison."""

    ideal_answer: str | None = Field(
        default=None, max_length=8000, description="Reference answer"
    )
    relevant_doc_ids: list[str] | None = Field(
        default=None,
        max_length=50,
        description="Documents relevant to the question (filenames or ids)",
    )


class GoldenRef(BaseModel):
    """Points at one example of a golden dataset."""

    dataset: str = Field(min_length=1, max_length=64, pattern=_NAME_PATTERN)
    id: str = Field(min_length=1, max_length=128)


class CompareStrategiesRequest(_QuestionMixin, _RunOptionsMixin):
    """Request to compare RAG strategies on the same question.

    Attributes:
        question: The question to run through all strategies.
        strategies: List of strategies to compare (classic, graph, agentic).
        model: Optional model override for all strategies.

    Example:
        {
            "question": "How are embeddings used in RAG?",
            "strategies": ["classic", "graph", "agentic"],
            "model": "claude-sonnet-5"
        }
    """

    question: str = Field(
        min_length=1, max_length=4000, description="Question to compare strategies on"
    )
    strategies: list[Strategy] = Field(
        default_factory=lambda: list(get_args(Strategy)),
        min_length=1,
        max_length=3,
        description="Strategies to compare: classic, graph, agentic",
    )
    model: str | None = Field(
        default=None, max_length=128, description="Optional model override"
    )
    evaluate: bool = Field(
        default=False,
        description="Also score each strategy's answer with the LLM judge and, when "
        "relevant documents are known, with deterministic retrieval metrics. "
        "Costs one extra rate-limit unit per strategy.",
    )
    reference: CompareReference | None = Field(
        default=None,
        description="Reference to score against: an ideal answer (enables "
        "answer_correctness) and/or relevant documents (enables recall@K, MRR, nDCG).",
    )
    golden: GoldenRef | None = Field(
        default=None,
        description="Take the reference from a golden dataset example instead. "
        "Mutually exclusive with `reference`.",
    )

    @model_validator(mode="after")
    def _one_reference(self):
        if self.reference is not None and self.golden is not None:
            raise ValueError("pass either `reference` or `golden`, not both")
        return self


class EvalRunRequest(_RunOptionsMixin):
    """Request to run evaluation on a dataset.

    Attributes:
        dataset: Golden dataset to evaluate against (e.g., golden_v1).
        provider: Override default LLM provider.
        model: Override default model ID.
        prompt_version: Optional prompt variant identifier.

    Example:
        {
            "dataset": "golden_v1",
            "strategy": "graph",
            "model": "claude-sonnet-5"
        }
    """

    dataset: str = Field(
        default="golden_v1",
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9._-]+$",
        description="Golden dataset name",
    )
    strategy: Strategy = Field(
        default="classic",
        description="RAG strategy to evaluate. Run once per strategy to compare them.",
    )
    provider: Provider | None = Field(
        default=None, description="Override LLM provider"
    )
    model: str | None = Field(
        default=None, max_length=128, description="Override model ID"
    )
    prompt_version: str | None = Field(
        default=None,
        max_length=64,
        pattern=r"^[A-Za-z0-9._-]+$",
        description="Optional prompt variant",
    )


class EvalScore(BaseModel):
    """Evaluation scores for an answer.

    Attributes:
        faithfulness: How much the answer sticks to retrieved context (0-1).
        answer_relevance: How relevant the answer is to the question (0-1).
        context_precision: Fraction of retrieved context that supports the answer (0-1).
        context_recall: Fraction of relevant context that was retrieved (0-1).

    Example:
        {
            "faithfulness": 0.95,
            "answer_relevance": 0.88,
            "context_precision": 0.92,
            "context_recall": 0.85
        }
    """

    faithfulness: float = Field(
        description="How faithful to retrieved context (0-1)"
    )
    answer_relevance: float = Field(
        description="Relevance of answer to question (0-1)"
    )
    context_precision: float = Field(
        description="Precision of retrieved context (0-1)"
    )
    context_recall: float = Field(description="Recall of relevant context (0-1)")
    answer_correctness: float | None = Field(
        default=None,
        description="Agreement with the golden ideal answer (0-1); None when the "
        "example has no ideal answer",
    )


class RetrievalScore(BaseModel):
    """How well retrieval found the documents a golden example expects.

    Computed by comparing the documents behind the retrieved context (the
    strategy's ranked ``retrieved_docs``; the answer's cited sources only when a
    strategy recorded none) against the example's relevance labels
    (``relevant_doc_ids`` or ``expected_sources``). Entirely deterministic -- no
    model call -- so these numbers are available even when the LLM judge is
    unavailable. These are unranked, whole-list figures; the eval harness adds
    ranked metrics at explicit cutoffs (``recall@5``, ``ndcg@10``, ...), which
    are the ones to compare across strategies (see ``app.eval.retrieval``).

    Attributes:
        precision: Fraction of retrieved documents that are relevant (0-1).
        recall: Fraction of relevant documents that were retrieved (0-1).
        hit: Whether at least one relevant document was retrieved.
        mrr: Reciprocal rank of the first relevant document (1.0 if it ranked
            first, 0.5 if second, 0.0 if absent).
        retrieved: Documents behind the retrieved context, best-ranked first.
        expected: Documents the golden example labels relevant.

    Example:
        {
            "precision": 0.5, "recall": 1.0, "hit": true, "mrr": 1.0,
            "retrieved": ["rag_overview.md", "chunking_strategies.md"],
            "expected": ["rag_overview.md"]
        }
    """

    precision: float = Field(ge=0, le=1, description="Retrieved documents that are relevant")
    recall: float = Field(ge=0, le=1, description="Relevant documents that were retrieved")
    hit: bool = Field(description="At least one relevant document was retrieved")
    mrr: float = Field(
        ge=0, le=1, description="Reciprocal rank of the first relevant document"
    )
    retrieved: list[str] = Field(
        default_factory=list, description="Documents behind the retrieved context, best first"
    )
    expected: list[str] = Field(
        default_factory=list, description="Documents the example labels relevant"
    )


class HumanRatingRequest(BaseModel):
    """Request to rate an answer manually.

    Attributes:
        question: The question that was answered.
        answer: The answer text.
        rating: Human rating (1-5 stars).
        comment: Optional comment explaining the rating.
        provider: LLM provider used.
        model: Model ID used.
        prompt_version: Prompt variant used.

    Example:
        {
            "question": "What is machine learning?",
            "answer": "Machine learning is a subset of AI...",
            "rating": 5,
            "comment": "Accurate and comprehensive.",
            "model": "claude-sonnet-5"
        }
    """

    question: str = Field(min_length=1, max_length=4000, description="The question")
    answer: str = Field(min_length=1, max_length=20000, description="The answer text")
    rating: int = Field(ge=1, le=5, description="Rating (1-5 stars)")
    comment: str | None = Field(
        default=None, max_length=2000, description="Optional comment on the rating"
    )
    provider: str | None = Field(default=None, description="LLM provider")
    model: str | None = Field(default=None, description="Model ID")
    prompt_version: str | None = Field(default=None, description="Prompt variant")


class HumanRating(HumanRatingRequest):
    """A recorded human rating with metadata.

    Extends HumanRatingRequest with storage metadata.

    Attributes:
        id: Unique rating identifier.
        created_at: ISO 8601 timestamp when rating was created.

    Example:
        {
            "question": "What is machine learning?",
            "answer": "Machine learning is a subset of AI...",
            "rating": 5,
            "comment": "Accurate and comprehensive.",
            "model": "claude-sonnet-5",
            "id": "rating_12345",
            "created_at": "2024-08-07T14:30:00Z"
        }
    """

    id: str = Field(description="Unique rating ID")
    created_at: str = Field(description="ISO 8601 creation timestamp")


class StrategyComparison(_GroundingFields):
    """Result of running one strategy on a question.

    Used internally to compare strategies side-by-side.

    Attributes:
        strategy: Strategy name (classic, graph, agentic).
        question: The question answered.
        answer: The generated answer.
        sources: Retrieved sources supporting the answer.
        confidence: Model confidence (0-1).
        refusal: Whether the model refused.
        latency_ms: Response time in milliseconds.
        input_tokens: Tokens sent to the model.
        output_tokens: Tokens received from the model.
        iterations: For agentic strategy, tool calls made.
        trace: Structured trace events for debugging.
        extra: Additional metadata.

    Example:
        {
            "strategy": "classic",
            "question": "What is RAG?",
            "answer": "RAG combines retrieval and generation...",
            "sources": [...],
            "confidence": 0.92,
            "refusal": false,
            "latency_ms": 1240,
            "input_tokens": 450,
            "output_tokens": 125,
            "iterations": 1,
            "trace": [...],
            "extra": {}
        }
    """

    strategy: str = Field(description="Strategy name (classic, graph, agentic)")
    question: str = Field(description="The question")
    answer: str = Field(description="The answer")
    sources: list[Source] = Field(description="Cited sources (validated)")
    confidence: float = Field(ge=0, le=1, description="Evidence score (0-1), see Answer")
    refusal: bool = Field(default=False, description="Whether model refused")
    latency_ms: int = Field(ge=0, description="Response time (ms)")
    input_tokens: int = Field(ge=0, description="Tokens sent to model")
    output_tokens: int = Field(ge=0, description="Tokens from model")
    iterations: int = Field(ge=0, description="Tool calls (agentic only)")
    trace: list[dict] = Field(
        default_factory=list, description="Structured trace events"
    )
    extra: dict = Field(default_factory=dict, description="Additional metadata")
    trace_id: str | None = Field(
        default=None, description="Request trace id; fetch it from /traces/{trace_id}"
    )
    retrieval: RetrievalDiagnostics | None = Field(
        default=None, description="Retrieval pipeline diagnostics, when the strategy has them"
    )
    evaluation: "StrategyEvaluation | None" = Field(
        default=None,
        description="Judge and retrieval scores; only with /compare/strategies evaluate=true",
    )


class StrategyEvaluation(BaseModel):
    """Scores for one strategy's row of a comparison (``evaluate=true``).

    Judge scores are ``None`` when the judge failed -- never a default -- and
    ``error`` says why. Retrieval metrics are present only when the relevant
    documents are known.
    """

    status: Literal["scored", "judge_failed", "refusal_checked", "not_evaluated"] = Field(
        description="'scored': judge verdict recorded; 'judge_failed': see error; "
        "'refusal_checked': golden example expects a refusal, scored "
        "deterministically; 'not_evaluated': the strategy produced no answer."
    )
    scores: dict[str, float | None] | None = Field(
        default=None,
        description="Judge scores: faithfulness, answer_relevance, context_precision, "
        "context_recall, answer_correctness (null without a reference answer)",
    )
    reasoning: dict[str, str] = Field(default_factory=dict, description="Judge reasoning")
    retrieval: dict[str, float] | None = Field(
        default=None, description="Ranked retrieval metrics: mrr, recall@K, ndcg@K, ..."
    )
    retrieved_docs: list[str] = Field(
        default_factory=list, description="Documents behind the context, best first"
    )
    relevant_docs: list[str] | None = Field(
        default=None, description="Documents the reference labels relevant"
    )
    has_reference_answer: bool = False
    correct_refusal: float | None = Field(
        default=None, description="1 if the strategy declined an unanswerable example"
    )
    judge_model: str | None = None
    rubric_version: str | None = None
    judge_attempts: int = 0
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0
    error_type: str | None = None
    error: str | None = None


StrategyComparison.model_rebuild()
