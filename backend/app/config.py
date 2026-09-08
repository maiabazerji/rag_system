import logging
from urllib.parse import urlparse

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Anthropic models this project has been exercised against. Selecting a model
# outside this set is a warning, not an error -- new models ship faster than
# this list is updated. Retired IDs are deliberately absent.
VALID_ANTHROPIC_MODELS = {
    "claude-fable-5-1",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
}

# OpenAI models
VALID_OPENAI_MODELS = {
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
}

# Valid generator providers
VALID_PROVIDERS = {"anthropic", "openai", "local"}

# Valid RAG strategies
VALID_STRATEGIES = {"classic", "graph", "agentic"}

# Wandb modes
VALID_WANDB_MODES = {"online", "offline", "disabled"}


def _validate_url(value: str, name: str, *, schemes: tuple[str, ...] | None = None) -> str:
    """Validate that a string is a usable URL.

    Args:
        value: The URL string.
        name: Setting name, used in error messages.
        schemes: If given, the scheme must be one of these.

    Raises:
        ValueError: If the URL is empty, malformed, or uses a disallowed scheme.
    """
    if not value:
        raise ValueError(f"{name} cannot be empty")
    parsed = urlparse(value)
    if not parsed.scheme:
        raise ValueError(f"{name} must include a scheme (e.g. http://)")
    if schemes and parsed.scheme not in schemes:
        raise ValueError(f"{name} must use one of {schemes}, got '{parsed.scheme}'")
    if not parsed.netloc:
        raise ValueError(f"{name} must include a host")
    return value


def _warn_unknown_model(value: str, name: str, known: set[str]) -> str:
    """Warn (but do not fail) when a model ID is outside the known set."""
    if not value:
        raise ValueError(f"{name} cannot be empty")
    if value not in known:
        logger.warning(
            "%s is set to '%s', which is not in the known-good list for this "
            "project (%s). This is fine for a newly released model; double-check "
            "the spelling if calls start failing with a 404.",
            name,
            value,
            ", ".join(sorted(known)),
        )
    return value


class Settings(BaseSettings):
    """EvalRAG configuration.

    All settings can be overridden by environment variables or a `.env` file.
    Field validation runs at construction time; cross-field runtime checks live
    in :meth:`validate_startup`, which the app calls from its lifespan handler
    so that importing this module never requires a configured environment.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM Provider API Keys
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key for Claude models. Required to generate answers.",
    )
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key. Optional, used if generator_provider=openai.",
    )

    # Vector Database (Qdrant)
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant vector database URL (http://host:port).",
    )
    qdrant_collection: str = Field(
        default="evalrag",
        min_length=1,
        description="Qdrant collection name for storing document embeddings.",
    )

    # SQL Database (Postgres) -- backs API keys and usage accounting
    postgres_url: str = Field(
        default="postgresql+psycopg://evalrag:pw@localhost:5432/evalrag",
        description="PostgreSQL connection string.",
    )

    # Tracing Backend (Langfuse)
    langfuse_host: str = Field(
        default="http://localhost:3100",
        description="Langfuse tracing backend URL.",
    )
    langfuse_public_key: str = Field(default="", description="Langfuse public key. Optional.")
    langfuse_secret_key: str = Field(default="", description="Langfuse secret key. Optional.")

    # Weights & Biases (W&B) - Eval dashboards
    wandb_api_key: str = Field(default="", description="W&B API key. Optional.")
    wandb_project: str = Field(default="evalrag", description="W&B project name.")
    wandb_mode: str = Field(
        default="online", description="W&B mode: online, offline, or disabled."
    )

    # Embeddings
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        min_length=1,
        description="HuggingFace embedding model ID.",
    )
    reranker_model: str = Field(
        # An English 6-layer cross-encoder. The multilingual 12-layer mMARCO
        # model this replaced took ~14.5s to rerank 50 candidates on CPU
        # against ~7.1s here, with no measurable ranking benefit on an English
        # corpus, and reranking is the single largest cost in a query.
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        min_length=1,
        description="Cross-encoder model used to rerank retrieved chunks.",
    )

    # Generator Configuration
    generator_provider: str = Field(
        default="anthropic",
        description="LLM provider for generation: anthropic, openai, or local.",
    )
    generator_model: str = Field(
        default="claude-sonnet-5",
        description="Claude model for answer generation.",
    )
    openai_generator_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model for generation if generator_provider=openai.",
    )

    # Ollama (Local LLM)
    ollama_host: str = Field(
        default="http://localhost:11434", description="Ollama API host."
    )
    ollama_model: str = Field(
        default="llama3.1:8b", min_length=1, description="Ollama model name."
    )

    # Evaluation Models
    judge_model: str = Field(
        default="claude-opus-5", description="Claude model for LLM-as-judge evaluation."
    )
    graph_extraction_model: str = Field(
        default="claude-haiku-4-5",
        description="Claude model for knowledge graph extraction.",
    )
    agentic_model: str = Field(
        default="claude-sonnet-5",
        description="Claude model for the agentic retrieval strategy.",
    )

    # RAG Hyperparameters
    chunk_size_tokens: int = Field(
        default=600, ge=100, le=2000, description="Document chunk size in words."
    )
    chunk_overlap_tokens: int = Field(
        default=80, ge=0, le=500, description="Overlap between consecutive chunks."
    )
    retrieval_top_k: int = Field(
        default=50, ge=1, le=100, description="Candidates to retrieve before reranking."
    )
    rerank_top_k: int = Field(
        default=8, ge=1, le=50, description="Chunks to keep after reranking."
    )
    max_answer_tokens: int = Field(
        default=1024, ge=64, le=8192, description="Max tokens in a generated answer."
    )

    # Agentic Strategy
    agentic_max_iters: int = Field(
        default=15, ge=1, le=50, description="Maximum iterations for the agentic loop."
    )

    # Graph RAG
    graph_data_dir: str = Field(
        default="data/graph",
        min_length=1,
        description="Directory for knowledge graph data. Use an absolute path in Docker.",
    )

    # Evaluation
    eval_concurrency: int = Field(
        default=4, ge=1, le=32, description="Golden examples evaluated in parallel."
    )

    # Ingestion limits
    max_upload_mb: int = Field(
        default=25, ge=1, le=500, description="Largest accepted upload, in megabytes."
    )

    # Provider call behaviour
    provider_timeout_seconds: float = Field(
        default=60.0, gt=0, le=600, description="Per-request timeout for LLM calls."
    )
    provider_max_retries: int = Field(
        default=3, ge=0, le=10, description="Retries for transient LLM failures."
    )

    # Authentication
    require_api_key: bool = Field(
        default=False,
        description=(
            "Require an 'Authorization: Bearer <key>' header on mutating and "
            "LLM-spending endpoints. Off by default so a fresh clone runs; turn "
            "on before exposing the API beyond localhost."
        ),
    )
    admin_key: str = Field(
        default="",
        description="Secret for /admin endpoints. They return 503 while unset.",
    )

    # CORS
    cors_origins: str = Field(
        default="http://localhost:5173",
        description="Comma-separated list of allowed CORS origins.",
    )

    @property
    def cors_origin_list(self) -> list[str]:
        """CORS origins as a cleaned list, ignoring stray whitespace and blanks."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        """Upload ceiling in bytes."""
        return self.max_upload_mb * 1024 * 1024

    @field_validator("qdrant_url")
    @classmethod
    def _v_qdrant(cls, v: str) -> str:
        return _validate_url(v, "QDRANT_URL", schemes=("http", "https"))

    @field_validator("langfuse_host")
    @classmethod
    def _v_langfuse(cls, v: str) -> str:
        return _validate_url(v, "LANGFUSE_HOST", schemes=("http", "https"))

    @field_validator("ollama_host")
    @classmethod
    def _v_ollama(cls, v: str) -> str:
        return _validate_url(v, "OLLAMA_HOST", schemes=("http", "https"))

    @field_validator("postgres_url")
    @classmethod
    def _v_postgres(cls, v: str) -> str:
        if not v:
            raise ValueError("POSTGRES_URL cannot be empty")
        if not v.startswith(("postgresql://", "postgresql+psycopg://")):
            raise ValueError(
                "POSTGRES_URL must start with 'postgresql://' or 'postgresql+psycopg://'"
            )
        if not urlparse(v).netloc:
            raise ValueError("POSTGRES_URL must include a host")
        return v

    @field_validator("generator_provider")
    @classmethod
    def _v_provider(cls, v: str) -> str:
        if v.lower() not in VALID_PROVIDERS:
            raise ValueError(
                f"GENERATOR_PROVIDER must be one of {sorted(VALID_PROVIDERS)}, got '{v}'"
            )
        return v.lower()

    @field_validator("generator_model", "judge_model", "graph_extraction_model", "agentic_model")
    @classmethod
    def _v_anthropic_model(cls, v: str, info) -> str:
        return _warn_unknown_model(v, info.field_name.upper(), VALID_ANTHROPIC_MODELS)

    @field_validator("openai_generator_model")
    @classmethod
    def _v_openai_model(cls, v: str) -> str:
        return _warn_unknown_model(v, "OPENAI_GENERATOR_MODEL", VALID_OPENAI_MODELS)

    @field_validator("wandb_mode")
    @classmethod
    def _v_wandb_mode(cls, v: str) -> str:
        if v.lower() not in VALID_WANDB_MODES:
            raise ValueError(
                f"WANDB_MODE must be one of {sorted(VALID_WANDB_MODES)}, got '{v}'"
            )
        return v.lower()

    @field_validator("chunk_overlap_tokens")
    @classmethod
    def _v_overlap(cls, v: int, info) -> int:
        size = info.data.get("chunk_size_tokens")
        if size is not None and v >= size:
            raise ValueError(
                f"CHUNK_OVERLAP_TOKENS ({v}) must be smaller than "
                f"CHUNK_SIZE_TOKENS ({size}); otherwise chunking never advances."
            )
        return v

    def validate_startup(self) -> None:
        """Check cross-field configuration consistency when the server boots.

        Called from the app lifespan, not at import, so that tests, linters and
        `--help` do not require a configured environment.

        Raises:
            ValueError: If the selected generator provider has no credentials.
        """
        errors: list[str] = []
        warnings: list[str] = []

        if self.generator_provider == "anthropic" and not self.anthropic_api_key:
            errors.append(
                "GENERATOR_PROVIDER is 'anthropic' but ANTHROPIC_API_KEY is not set."
            )
        elif self.generator_provider == "openai" and not self.openai_api_key:
            errors.append(
                "GENERATOR_PROVIDER is 'openai' but OPENAI_API_KEY is not set."
            )
        elif self.generator_provider == "local":
            logger.info("Local generator (Ollama) at %s", self.ollama_host)

        if not self.anthropic_api_key:
            warnings.append(
                "ANTHROPIC_API_KEY is not set. Evaluation (LLM judge), graph "
                "extraction and the agentic strategy will be unavailable."
            )

        if self.langfuse_public_key and not self.langfuse_secret_key:
            warnings.append(
                "LANGFUSE_PUBLIC_KEY is set but LANGFUSE_SECRET_KEY is missing; "
                "traces will not be recorded."
            )

        if self.require_api_key and not self.admin_key:
            warnings.append(
                "REQUIRE_API_KEY is on but ADMIN_KEY is unset, so there is no way "
                "to mint a key. Set ADMIN_KEY and use scripts/setup_auth.py."
            )

        if not self.require_api_key:
            warnings.append(
                "REQUIRE_API_KEY is off: /ask, /compare, /ingest and /eval accept "
                "unauthenticated requests. Fine for local use; turn it on before "
                "exposing this beyond localhost."
            )

        for w in warnings:
            logger.warning("Config warning: %s", w)

        if errors:
            raise ValueError(
                "Configuration validation failed:\n  " + "\n  ".join(errors)
            )

        logger.info("Configuration validation passed")


settings = Settings()
