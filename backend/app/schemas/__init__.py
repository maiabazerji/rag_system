from pydantic import BaseModel, Field


class Source(BaseModel):
    chunk_id: str
    quote: str
    score: float | None = None


class Answer(BaseModel):
    question: str
    answer: str
    sources: list[Source] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    refusal: bool = False
    provider: str | None = None
    model: str | None = None


class Chunk(BaseModel):
    id: str
    doc_id: str
    text: str
    tokens: int
    section: str | None = None
    metadata: dict = Field(default_factory=dict)


class AskRequest(BaseModel):
    question: str
    top_k: int = 8
    provider: str | None = None
    model: str | None = None
    prompt_version: str | None = None


class CompareRequest(BaseModel):
    question: str
    variants: list[dict]  # [{provider, model, prompt_version}, ...]


class EvalRunRequest(BaseModel):
    dataset: str = "golden_v1"
    provider: str | None = None
    model: str | None = None
    prompt_version: str | None = None


class EvalScore(BaseModel):
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float


class HumanRatingRequest(BaseModel):
    question: str
    answer: str
    rating: int = Field(ge=1, le=5)
    comment: str | None = None
    provider: str | None = None
    model: str | None = None
    prompt_version: str | None = None


class HumanRating(HumanRatingRequest):
    id: str
    created_at: str
