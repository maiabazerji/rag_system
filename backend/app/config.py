from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    openai_api_key: str = ""
    cohere_api_key: str = ""

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "evalrag"

    postgres_url: str = "postgresql+psycopg://evalrag:pw@localhost:5432/evalrag"
    redis_url: str = "redis://localhost:6379/0"

    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    wandb_api_key: str = ""
    wandb_project: str = "evalrag"
    wandb_mode: str = "online"

    embedding_model: str = "BAAI/bge-small-en-v1.5"

    generator_provider: str = "local"
    generator_model: str = "claude-sonnet-4-6"
    openai_generator_model: str = "gpt-4o-mini"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    judge_model: str = "claude-opus-4-6"

    chunk_size_tokens: int = 600
    chunk_overlap_tokens: int = 80
    retrieval_top_k: int = 50
    rerank_top_k: int = 8


settings = Settings()
