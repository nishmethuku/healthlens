"""Centralized runtime configuration. Single source of truth for env-derived settings."""
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: str = Field(default="development", alias="ENVIRONMENT")

    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    # Groq retired llama-3.3-70b-versatile in Aug 2026; gpt-oss-120b is its
    # production replacement. Override with GROQ_MODEL if Groq changes it again.
    groq_model: str = Field(default="openai/gpt-oss-120b", alias="GROQ_MODEL")
    ncbi_api_key: str | None = Field(default=None, alias="NCBI_API_KEY")

    # Fail-closed: if unset, every request is rejected rather than silently
    # allowed through. Callers authenticate with header X-API-Key.
    healthlens_api_key: str | None = Field(default=None, alias="HEALTHLENS_API_KEY")

    # Comma-separated origins, e.g. "https://healthlens.app,https://www.healthlens.app"
    cors_origins: str = Field(default="http://localhost:5173", alias="CORS_ORIGINS")

    rate_limit_per_minute: int = Field(default=20, alias="RATE_LIMIT_PER_MINUTE")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    embedding_model_id: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL_ID"
    )
    # Some embedding models expect an instruction prefix, e.g. e5 uses
    # "query: " / "passage: " and bge uses a query instruction. Leave empty for MiniLM.
    embedding_query_prefix: str = Field(default="", alias="EMBEDDING_QUERY_PREFIX")
    embedding_doc_prefix: str = Field(default="", alias="EMBEDDING_DOC_PREFIX")
    reranker_model_id: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL_ID"
    )

    # How the BM25 and dense scores are combined: "weighted" (min-max blend) or "rrf"
    # (reciprocal rank fusion). Compare both with eval/eval_harness.py before changing.
    fusion_method: str = Field(default="weighted", alias="FUSION_METHOD")
    dense_weight: float = Field(default=0.5, alias="DENSE_WEIGHT")
    # Candidates taken from fusion before the cross-encoder reranks them down to top_k.
    rerank_pool_size: int = Field(default=25, alias="RERANK_POOL_SIZE")

    # Keyword-only retrieval with no embedding model or reranker. Fits in the
    # 512 MB of a free host; answer quality is a bit lower than the full pipeline.
    lite_mode: bool = Field(default=False, alias="LITE_MODE")

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
