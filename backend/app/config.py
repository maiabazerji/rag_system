import logging
from urllib.parse import urlparse

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Anthropic Claude models that support tool use and are production-ready
VALID_ANTHROPIC_MODELS = {
    "claude-opus-4-7",
    "claude-opus-4-1-20250805",
    "claude-sonnet-4-6",
    "claude-sonnet-4-20250514",
    "claude-haiku-4-5-20251001",
}

# OpenAI models
VALID_OPENAI_MODELS = {
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
}

# Ollama models (user-installable, can't validate exhaustively)
VALID_OLLAMA_MODELS = {"llama3.1:8b", "llama2:13b", "mistral:latest"}

# Valid generator providers
VALID_PROVIDERS = {"anthropic", "openai", "local"}

# Valid RAG strategies
VALID_STRATEGIES = {"classic", "graph", "agentic"}

# Wandb modes
VALID_WANDB_MODES = {"online", "offline", "disabled"}


class Settings(BaseSettings):
    """EvalRAG configuration with comprehensive validation.

    All settings can be overridden via environment variables (.env file).
    URL and model fields are validated on startup to catch configuration errors early.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM Provider API Keys
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key for Claude models. Required for production use.",
    )
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key for GPT models. Optional, used if generator_provider=openai.",
    )
    cohere_api_key: str = Field(
        default="",
        description="Cohere API key. Optional, for future embedding/reranking support.",
    )

    # Vector Database (Qdrant)
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant vector database URL (http://host:port). Must be a valid URL.",
    )
    qdrant_collection: str = Field(
        default="evalrag",
        min_length=1,
        description="Qdrant collection name for storing document embeddings.",
    )

    # SQL Database (Postgres)
    postgres_url: str = Field(
        default="postgresql+psycopg://evalrag:pw@localhost:5432/evalrag",
        description="PostgreSQL connection string. Must be a valid psycopg URL.",
    )

    # Cache (Redis)
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL. Used for caching and queues.",
    )

    # Tracing Backend (Langfuse)
    langfuse_host: str = Field(
        default="http://localhost:3000",
        description="Langfuse tracing backend URL. Must be a valid URL.",
    )
    langfuse_public_key: str = Field(
        default="",
        description="Langfuse public key for trace authentication. Optional.",
    )
    langfuse_secret_key: str = Field(
        default="",
        description="Langfuse secret key for trace authentication. Optional.",
    )

    # Weights & Biases (W&B) - Eval dashboards
    wandb_api_key: str = Field(
        default="",
        description="Weights & Biases API key. Optional, for eval-run dashboards.",
    )
    wandb_project: str = Field(
        default="evalrag",
        description="W&B project name for storing eval runs.",
    )
    wandb_mode: str = Field(
        default="online",
        description="W&B mode: online, offline, or disabled.",
    )

    # Embeddings
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="HuggingFace embedding model ID. Must be a valid model identifier.",
    )

    # Generator Configuration
    generator_provider: str = Field(
        default="anthropic",
        description="LLM provider for generation: anthropic, openai, or local.",
    )
    generator_model: str = Field(
        default="claude-sonnet-4-6",
        description="Claude model for answer generation. Must be a valid Anthropic model.",
    )
    openai_generator_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model for generation if generator_provider=openai.",
    )

    # Ollama (Local LLM)
    ollama_host: str = Field(
        default="http://localhost:11434",
        description="Ollama API host. Used if generator_provider=local.",
    )
    ollama_model: str = Field(
        default="llama3.1:8b",
        description="Ollama model name for local generation.",
    )

    # Evaluation Models
    judge_model: str = Field(
        default="claude-opus-4-7",
        description="Claude model for LLM-as-judge evaluation. Must be a valid Anthropic model.",
    )
    graph_extraction_model: str = Field(
        default="claude-haiku-4-5-20251001",
        description="Claude model for knowledge graph extraction. Must be a valid Anthropic model.",
    )
    agentic_model: str = Field(
        default="claude-sonnet-4-6",
        description="Claude model for agentic retrieval strategy. Must be a valid Anthropic model.",
    )

    # RAG Hyperparameters
    chunk_size_tokens: int = Field(
        default=600,
        ge=100,
        le=2000,
        description="Document chunk size in tokens. Valid range: 100–2000.",
    )
    chunk_overlap_tokens: int = Field(
        default=80,
        ge=0,
        le=500,
        description="Overlap between consecutive chunks in tokens. Valid range: 0–500.",
    )
    retrieval_top_k: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Number of documents to retrieve. Valid range: 1–100.",
    )
    rerank_top_k: int = Field(
        default=8,
        ge=1,
        le=50,
        description="Number of documents to keep after reranking. Valid range: 1–50.",
    )

    # Agentic Strategy
    agentic_max_iters: int = Field(
        default=15,
        ge=1,
        le=50,
        description="Maximum iterations for agentic loop. Valid range: 1–50.",
    )

    # Graph RAG
    graph_data_dir: str = Field(
        default="data/graph",
        description="Directory to store knowledge graph data.",
    )

    # CORS
    cors_origins: str = Field(
        default="http://localhost:5173",
        description="Comma-separated list of allowed CORS origins.",
    )

    @field_validator("qdrant_url")
    @classmethod
    def validate_qdrant_url(cls, v: str) -> str:
        """Validate Qdrant URL format.

        Args:
            v: URL string to validate

        Raises:
            ValueError: If URL is invalid
        """
        if not v:
            raise ValueError("QDRANT_URL cannot be empty")
        try:
            parsed = urlparse(v)
            if not parsed.scheme:
                raise ValueError("QDRANT_URL must include a scheme (http/https)")
            if not parsed.netloc:
                raise ValueError("QDRANT_URL must include a host")
        except Exception as e:
            raise ValueError(f"QDRANT_URL is not a valid URL: {e}")
        return v

    @field_validator("postgres_url")
    @classmethod
    def validate_postgres_url(cls, v: str) -> str:
        """Validate PostgreSQL connection URL format.

        Args:
            v: Connection string to validate

        Raises:
            ValueError: If URL is invalid or not a psycopg URL
        """
        if not v:
            raise ValueError("POSTGRES_URL cannot be empty")
        if not v.startswith(("postgresql://", "postgresql+psycopg://")):
            raise ValueError(
                "POSTGRES_URL must start with 'postgresql://' or 'postgresql+psycopg://'"
            )
        try:
            parsed = urlparse(v)
            if not parsed.netloc:
                raise ValueError("POSTGRES_URL must include a host")
        except Exception as e:
            raise ValueError(f"POSTGRES_URL is not a valid connection string: {e}")
        return v

    @field_validator("redis_url")
    @classmethod
    def validate_redis_url(cls, v: str) -> str:
        """Validate Redis URL format.

        Args:
            v: URL string to validate

        Raises:
            ValueError: If URL is invalid
        """
        if not v:
            raise ValueError("REDIS_URL cannot be empty")
        if not v.startswith("redis://"):
            raise ValueError("REDIS_URL must start with 'redis://'")
        try:
            parsed = urlparse(v)
            if not parsed.netloc:
                raise ValueError("REDIS_URL must include a host")
        except Exception as e:
            raise ValueError(f"REDIS_URL is not a valid URL: {e}")
        return v

    @field_validator("langfuse_host")
    @classmethod
    def validate_langfuse_host(cls, v: str) -> str:
        """Validate Langfuse host URL format.

        Args:
            v: URL string to validate

        Raises:
            ValueError: If URL is invalid
        """
        if not v:
            raise ValueError("LANGFUSE_HOST cannot be empty")
        try:
            parsed = urlparse(v)
            if not parsed.scheme:
                raise ValueError("LANGFUSE_HOST must include a scheme (http/https)")
            if not parsed.netloc:
                raise ValueError("LANGFUSE_HOST must include a host")
        except Exception as e:
            raise ValueError(f"LANGFUSE_HOST is not a valid URL: {e}")
        return v

    @field_validator("generator_provider")
    @classmethod
    def validate_generator_provider(cls, v: str) -> str:
        """Validate generator provider is supported.

        Args:
            v: Provider name to validate

        Raises:
            ValueError: If provider is not in the valid list
        """
        if v.lower() not in VALID_PROVIDERS:
            raise ValueError(
                f"GENERATOR_PROVIDER must be one of {VALID_PROVIDERS}, got '{v}'"
            )
        return v.lower()

    @field_validator("generator_model")
    @classmethod
    def validate_generator_model(cls, v: str) -> str:
        """Validate generator model is recognized.

        Args:
            v: Model ID to validate

        Raises:
            ValueError: If model is not recognized
        """
        if not v:
            raise ValueError("GENERATOR_MODEL cannot be empty")
        if v not in VALID_ANTHROPIC_MODELS:
            logger.warning(
                f"GENERATOR_MODEL '{v}' is not in the list of known models. "
                f"Known Anthropic models: {VALID_ANTHROPIC_MODELS}"
            )
        return v

    @field_validator("judge_model")
    @classmethod
    def validate_judge_model(cls, v: str) -> str:
        """Validate judge model is recognized.

        Args:
            v: Model ID to validate

        Raises:
            ValueError: If model is not recognized
        """
        if not v:
            raise ValueError("JUDGE_MODEL cannot be empty")
        if v not in VALID_ANTHROPIC_MODELS:
            logger.warning(
                f"JUDGE_MODEL '{v}' is not in the list of known models. "
                f"Known Anthropic models: {VALID_ANTHROPIC_MODELS}"
            )
        return v

    @field_validator("graph_extraction_model")
    @classmethod
    def validate_graph_extraction_model(cls, v: str) -> str:
        """Validate graph extraction model is recognized.

        Args:
            v: Model ID to validate

        Raises:
            ValueError: If model is not recognized
        """
        if not v:
            raise ValueError("GRAPH_EXTRACTION_MODEL cannot be empty")
        if v not in VALID_ANTHROPIC_MODELS:
            logger.warning(
                f"GRAPH_EXTRACTION_MODEL '{v}' is not in the list of known models. "
                f"Known Anthropic models: {VALID_ANTHROPIC_MODELS}"
            )
        return v

    @field_validator("agentic_model")
    @classmethod
    def validate_agentic_model(cls, v: str) -> str:
        """Validate agentic model is recognized.

        Args:
            v: Model ID to validate

        Raises:
            ValueError: If model is not recognized
        """
        if not v:
            raise ValueError("AGENTIC_MODEL cannot be empty")
        if v not in VALID_ANTHROPIC_MODELS:
            logger.warning(
                f"AGENTIC_MODEL '{v}' is not in the list of known models. "
                f"Known Anthropic models: {VALID_ANTHROPIC_MODELS}"
            )
        return v

    @field_validator("openai_generator_model")
    @classmethod
    def validate_openai_generator_model(cls, v: str) -> str:
        """Validate OpenAI generator model is recognized.

        Args:
            v: Model ID to validate

        Raises:
            ValueError: If model is not recognized
        """
        if not v:
            raise ValueError("OPENAI_GENERATOR_MODEL cannot be empty")
        if v not in VALID_OPENAI_MODELS:
            logger.warning(
                f"OPENAI_GENERATOR_MODEL '{v}' is not in the list of known models. "
                f"Known OpenAI models: {VALID_OPENAI_MODELS}"
            )
        return v

    @field_validator("wandb_mode")
    @classmethod
    def validate_wandb_mode(cls, v: str) -> str:
        """Validate W&B mode is recognized.

        Args:
            v: Mode string to validate

        Raises:
            ValueError: If mode is not valid
        """
        if v.lower() not in VALID_WANDB_MODES:
            raise ValueError(
                f"WANDB_MODE must be one of {VALID_WANDB_MODES}, got '{v}'"
            )
        return v.lower()

    @field_validator("embedding_model")
    @classmethod
    def validate_embedding_model(cls, v: str) -> str:
        """Validate embedding model ID format.

        Args:
            v: Model ID to validate

        Raises:
            ValueError: If model ID is empty
        """
        if not v:
            raise ValueError("EMBEDDING_MODEL cannot be empty")
        # Allow any non-empty string as there are many valid HuggingFace models
        return v

    def validate_startup(self) -> None:
        """Validate required configuration on startup.

        This method is called after Pydantic field validation to check runtime
        configuration consistency (e.g., provider vs. API key mismatch).

        Raises:
            ValueError: If critical configuration requirements are not met.
        """
        warnings = []
        errors = []

        # Anthropic key is required as primary generator for evaluation
        if not self.anthropic_api_key:
            errors.append(
                "ANTHROPIC_API_KEY is not set. Required for evaluation "
                "(judge models, graph extraction) and as fallback generator."
            )

        # Warn if selected provider is not configured
        if self.generator_provider == "anthropic":
            if not self.anthropic_api_key:
                errors.append(
                    f"Generator provider is '{self.generator_provider}' but "
                    "ANTHROPIC_API_KEY is not set"
                )
        elif self.generator_provider == "openai":
            if not self.openai_api_key:
                errors.append(
                    f"Generator provider is '{self.generator_provider}' but "
                    "OPENAI_API_KEY is not set"
                )
        elif self.generator_provider == "local":
            logger.info("Local generator (Ollama) is configured")

        # Warn about optional tracing/monitoring
        if self.langfuse_public_key and not self.langfuse_secret_key:
            warnings.append(
                "LANGFUSE_PUBLIC_KEY is set but LANGFUSE_SECRET_KEY is missing. "
                "Traces will not be recorded."
            )

        if self.wandb_api_key and self.wandb_mode == "offline":
            logger.info("W&B is configured in offline mode")

        # Log warnings
        for warning in warnings:
            logger.warning(f"Config warning: {warning}")

        # Raise if critical errors exist
        if errors:
            error_msg = "Configuration validation failed:\n  " + "\n  ".join(errors)
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("Configuration validation passed")


settings = Settings()
settings.validate_startup()
