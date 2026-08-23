"""
Centralized Configuration
Uses pydantic-settings for validated environment variables.
"""

from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Load .env into os.environ so LangChain/LangSmith SDK can read tracing config
load_dotenv()

class Settings(BaseSettings):

    # # LLM Configuration
    # openai_api_key: str

    primary_model: str = "openrouter/google/gemini-2.0-flash-exp:free"
    # Must differ from primary_model, otherwise with_fallbacks retries the same failing model
    fallback_model: str = "openrouter/meta-llama/llama-3.3-70b-instruct:free"
    embedding_model: str = "text-embedding-3-small"
    openrouter_api_key: str = Field(default="", validation_alias="OPEN_ROUTER_API_KEY")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1", validation_alias="OPEN_ROUTER_BASE_URL")

    # Optional shared API key. When set, mutating endpoints require header "X-API-Key".
    api_key: str = Field(default="", validation_alias="API_KEY")

    # LangSmith
    langchain_tracing_v2: bool = Field(default=True, validation_alias="LANGSMITH_TRACING")
    langchain_api_key: str = Field(default="", validation_alias="LANGSMITH_API_KEY")
    langchain_project: str = Field(default="production-api", validation_alias="LANGSMITH_PROJECT")
    langchain_endpoint: str = Field(default="", validation_alias="LANGSMITH_ENDPOINT")


    # Application
    app_env: str = "development"
    log_level: str = "INFO"
    rate_limit: str = "20/minute"
    cache_ttl_seconds: int = 300
    max_retries: int = 3
    # Draw the mermaid graph PNG on startup (slow; needs network). Off by default.
    debug_draw_graph: bool = Field(default=False, validation_alias="DEBUG_DRAW_GRAPH")

    # Cache backend: "auto" (redis in production, memory in dev), "memory", or "redis"
    cache_backend: str = Field(default="auto", validation_alias="CACHE_BACKEND")
    redis_url: str = Field(default="redis://localhost:6379/0", validation_alias="REDIS_URL")

    # Database (PostgreSQL in production; also used by PGVector when enabled)
    database_url: str = Field(default="sqlite:///./orders.db", validation_alias="DATABASE_URL")
    db_echo_logs: bool = Field(default=False, validation_alias="DB_ECHO_LOGS")

    # Vector store: opt-in PGVector for embeddings storage (independent of the main DB choice)
    use_pgvector: bool = Field(default=False, validation_alias="USE_PGVECTOR")
    pgvector_collection: str = Field(default="policy_docs", validation_alias="PGVECTOR_COLLECTION")

    # RAG / Ingestion
    dataset_dir: str = "./dataset-docs"
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection: str = "policy_docs"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_top_k: int = 10
    rerank_top_k: int = 5
    max_retrieval_retries: int = 1
    ingest_on_startup: bool = True

    # Checkpoints
    checkpoint_storage: str = Field(default="sqlite", validation_alias="CHECKPOINT_STORAGE")
    checkpoint_path: str = Field(default="./checkpoints.db", validation_alias="CHECKPOINT_PATH")
    checkpoint_table: str = Field(default="checkpoints", validation_alias="CHECKPOINT_TABLE")

    model_config = {
        "env_file": ".env",
        "extra": "ignore",
        "populate_by_name": True,
    }

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

@lru_cache
def get_settings() -> Settings:
    """Cached settings instance - loaded once, reused everywhere."""
    return Settings()
