import logging
from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Anthropic models this project has been exercised against. Configuring a model
# outside this set is a warning, not an error -- new models ship faster than
# this list is updated -- but per-request `model` overrides are restricted to
# this set plus the configured models and EXTRA_ALLOWED_MODELS. Retired IDs are
# deliberately absent.
VALID_ANTHROPIC_MODELS = {
    "claude-fable-5-1",
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-haiku-4-5-20251001",
    "claude-haiku-4-5",  # alias of the snapshot above
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

# Repository root when running from a checkout (backend/app/config.py -> repo).
# In a container the package may be installed elsewhere, which is why the data
# directory is a setting rather than derived from this path alone.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# What ingestion does with personal data it detects (see app.privacy.pii)
VALID_PII_MODES = {"off", "mask", "reject"}

# Telemetry modes: where traces may be sent. See docs/monitoring.md.
VALID_TELEMETRY_MODES = {"off", "self_hosted", "cloud"}


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
    qdrant_api_key: str | None = Field(
        default=None,
        description="Qdrant API key. Must match QDRANT__SERVICE__API_KEY on the server.",
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
    wandb_enabled: bool = Field(
        default=False,
        description="Opt in to W&B eval dashboards. Also gated by TELEMETRY_MODE.",
    )
    wandb_base_url: str = Field(
        default="",
        description="Self-hosted W&B server URL. Empty means the W&B cloud.",
    )

    # Telemetry policy (see app/tracing/policy.py and docs/monitoring.md)
    telemetry_mode: str = Field(
        default="off",
        description="Where traces may go: off, self_hosted, or cloud.",
    )
    telemetry_allowed_hosts: str = Field(
        default="localhost,langfuse,127.0.0.1",
        description="Comma-separated exporter hosts permitted in self_hosted mode.",
    )
    telemetry_cloud_opt_in: bool = Field(
        default=False,
        description="Explicit acknowledgement required before cloud mode exports anything.",
    )
    telemetry_include_content: bool = Field(
        default=True,
        description=(
            "Export question, answer and context text (after redaction). When false, "
            "only timings, token counts and model names leave the process."
        ),
    )
    telemetry_queue_size: int = Field(
        default=1000,
        ge=10,
        le=100_000,
        description="Traces buffered for export before new ones are dropped.",
    )

    # Prometheus metrics
    metrics_enabled: bool = Field(
        default=False,
        description="Serve GET /metrics (to localhost, or with the admin key).",
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
        default="claude-opus-5-5", description="Claude model for LLM-as-judge evaluation."
    )
    graph_extraction_model: str = Field(
        default="claude-haiku-4-5-20251001",
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

    # Per-request model overrides
    extra_allowed_models: str = Field(
        default="",
        description=(
            "Comma-separated model IDs accepted as per-request overrides in "
            "addition to the built-in known-good list and the configured models."
        ),
    )

    # Evaluation
    data_dir: str = Field(
        default="",
        description=(
            "Directory holding golden/ datasets and eval_runs/. Defaults to the "
            "repository's data/ directory; set an absolute path in Docker."
        ),
    )
    eval_concurrency: int = Field(
        default=4, ge=1, le=32, description="Golden examples evaluated in parallel."
    )

    # Ingestion limits
    max_upload_mb: int = Field(
        default=25, ge=1, le=500, description="Largest accepted upload, in megabytes."
    )
    max_uncompressed_mb: int = Field(
        default=200,
        ge=1,
        le=4096,
        description=(
            "Largest total uncompressed size of a zip-based document (docx, odt, "
            "pptx, ...), in megabytes. Guards against zip bombs."
        ),
    )

    # OCR for scanned PDFs
    ocr_enabled: bool = Field(
        default=True,
        description=(
            "OCR PDF pages that carry (almost) no text layer. Needs the tesseract "
            "and poppler binaries; without them OCR is skipped with a warning."
        ),
    )
    ocr_languages: str = Field(
        default="fra+eng",
        min_length=1,
        description="Tesseract language packs to OCR with, joined by '+'.",
    )
    ocr_max_pages: int = Field(
        default=50, ge=1, le=2000, description="Most pages OCR'd per document."
    )
    ocr_min_chars_per_page: int = Field(
        default=50,
        ge=0,
        le=10000,
        description="A PDF page with fewer extracted characters than this is OCR'd.",
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

    # Privacy (GDPR). See docs/gdpr/README.md.
    pii_mode_ingest: str = Field(
        default="mask",
        description=(
            "What ingestion does with detected personal data: 'off' (index as "
            "is), 'mask' (replace with [EMAIL], [NIR], ... placeholders) or "
            "'reject' (refuse the document with a 422)."
        ),
    )
    pii_redact_logs: bool = Field(
        default=True, description="Mask detected personal data in log lines."
    )
    pii_redact_traces: bool = Field(
        default=True,
        description="Mask detected personal data in stored and exported traces.",
    )
    retention_traces_days: int = Field(
        default=7, ge=0, description="Days to keep request traces. 0 keeps them forever."
    )
    retention_eval_runs_days: int = Field(
        default=365, ge=0, description="Days to keep eval run files. 0 keeps them forever."
    )
    retention_audit_days: int = Field(
        default=365, ge=0, description="Days to keep audit events. 0 keeps them forever."
    )

    # Access control: tenants, document groups and audit
    default_tenant: str = Field(
        default="default",
        min_length=1,
        max_length=128,
        description=(
            "Tenant for principals that carry none (API keys without one, OIDC "
            "users when OIDC_TENANT_CLAIM is unset, local mode) and for chunks "
            "indexed before tenants existed."
        ),
    )
    acl_default_public: bool = Field(
        default=False,
        description=(
            "Label uploads that name no groups as 'public' instead of with the "
            "uploader's own groups."
        ),
    )
    acl_legacy_public: bool = Field(
        default=True,
        description=(
            "Treat chunks indexed before per-document ACLs existed (no "
            "acl_groups payload) as 'public'. Turn off to hide them until "
            "they are re-ingested with groups."
        ),
    )
    audit_store_questions: bool = Field(
        default=False,
        description=(
            "Store question text in the audit log. Off by default for privacy: "
            "only a SHA-256 hash of the question is kept."
        ),
    )

    # SSO via OpenID Connect (Microsoft Entra ID, ProConnect, Keycloak, ...)
    oidc_issuer: str = Field(
        default="",
        description=(
            "OIDC issuer URL. When set (and REQUIRE_API_KEY is on), a Bearer "
            "JWT from this issuer is accepted alongside API keys."
        ),
    )
    oidc_audience: str = Field(
        default="", description="Expected 'aud' claim (the API's client id / app id URI)."
    )
    oidc_groups_claim: str = Field(
        default="groups",
        min_length=1,
        description=(
            "Claim holding the user's groups. Dotted paths reach nested claims, "
            "e.g. 'realm_access.roles' for Keycloak or 'roles' for Entra app roles."
        ),
    )
    oidc_admin_group: str = Field(
        default="", description="Group whose members are admins. Empty: no OIDC admins."
    )
    oidc_tenant_claim: str = Field(
        default="",
        description=(
            "Claim naming the user's tenant (e.g. 'tid' for Entra). Empty: every "
            "OIDC user belongs to DEFAULT_TENANT."
        ),
    )
    oidc_algorithms: str = Field(
        default="RS256,PS256,ES256",
        description="Comma-separated asymmetric JWS algorithms accepted on OIDC tokens.",
    )
    oidc_jwks_ttl_seconds: int = Field(
        default=3600, ge=60, le=86400, description="How long fetched signing keys are cached."
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
    def data_path(self) -> Path:
        """Resolved data directory (see ``data_dir``).

        Falls back to ``<repo>/data`` when it exists, else ``./data`` relative
        to the working directory -- never to a path computed from the install
        location, which in a container points somewhere unrelated.
        """
        if self.data_dir:
            return Path(self.data_dir).expanduser()
        repo_data = _REPO_ROOT / "data"
        if repo_data.is_dir():
            return repo_data
        return Path("data").resolve()

    def allowed_models(self, provider: str) -> set[str]:
        """Model IDs a request may select for ``provider``.

        The known-good list for the provider, plus every model this deployment
        is configured to use, plus ``EXTRA_ALLOWED_MODELS``.
        """
        extra = {m.strip() for m in self.extra_allowed_models.split(",") if m.strip()}
        if provider == "openai":
            return VALID_OPENAI_MODELS | {self.openai_generator_model} | extra
        if provider == "local":
            return {self.ollama_model} | extra
        return (
            VALID_ANTHROPIC_MODELS
            | {
                self.generator_model,
                self.agentic_model,
                self.judge_model,
                self.graph_extraction_model,
            }
            | extra
        )

    @property
    def telemetry_allowed_host_list(self) -> list[str]:
        """Allowed exporter hosts, lower-cased, ignoring blanks."""
        return [
            h.strip().lower() for h in self.telemetry_allowed_hosts.split(",") if h.strip()
        ]

    @property
    def max_upload_bytes(self) -> int:
        """Upload ceiling in bytes."""
        return self.max_upload_mb * 1024 * 1024

    @property
    def max_uncompressed_bytes(self) -> int:
        """Uncompressed ceiling for zip-based documents, in bytes."""
        return self.max_uncompressed_mb * 1024 * 1024

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

    @field_validator("oidc_issuer")
    @classmethod
    def _v_oidc_issuer(cls, v: str) -> str:
        # Kept verbatim (trailing slash included): `iss` must match it exactly.
        v = v.strip()
        return _validate_url(v, "OIDC_ISSUER", schemes=("https", "http")) if v else v

    @property
    def oidc_algorithm_list(self) -> list[str]:
        """Accepted OIDC signing algorithms. Symmetric (HS*) and 'none' never are."""
        return [
            a.strip()
            for a in self.oidc_algorithms.split(",")
            if a.strip() and a.strip()[:2] in ("RS", "PS", "ES", "Ed")
        ]

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

    @field_validator("pii_mode_ingest")
    @classmethod
    def _v_pii_mode(cls, v: str) -> str:
        if v.lower() not in VALID_PII_MODES:
            raise ValueError(
                f"PII_MODE_INGEST must be one of {sorted(VALID_PII_MODES)}, got '{v}'"
            )
        return v.lower()

    @field_validator("telemetry_mode")
    @classmethod
    def _v_telemetry_mode(cls, v: str) -> str:
        mode = v.strip().lower().replace("-", "_")
        if mode not in VALID_TELEMETRY_MODES:
            raise ValueError(
                f"TELEMETRY_MODE must be one of {sorted(VALID_TELEMETRY_MODES)}, got '{v}'"
            )
        return mode

    @field_validator("wandb_base_url")
    @classmethod
    def _v_wandb_base_url(cls, v: str) -> str:
        return _validate_url(v, "WANDB_BASE_URL", schemes=("http", "https")) if v else v

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

        if self.oidc_issuer and not self.oidc_audience:
            errors.append(
                "OIDC_ISSUER is set but OIDC_AUDIENCE is not; without it any token "
                "the issuer mints for any application would be accepted."
            )
        if self.oidc_issuer and not self.require_api_key:
            warnings.append(
                "OIDC_ISSUER is set but REQUIRE_API_KEY is off, so tokens are "
                "never checked and every caller is the local principal."
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
