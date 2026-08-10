"""Centralized runtime configuration. Single source of truth for env-derived settings."""
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: str = Field(default="development", alias="ENVIRONMENT")

    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    ncbi_api_key: str | None = Field(default=None, alias="NCBI_API_KEY")

    # Comma-separated origins, e.g. "https://healthlens.app,https://www.healthlens.app"
    cors_origins: str = Field(default="http://localhost:5173", alias="CORS_ORIGINS")

    rate_limit_per_minute: int = Field(default=20, alias="RATE_LIMIT_PER_MINUTE")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # NeuML/pubmedbert-base-embeddings was evaluated as a domain-specific swap
    # (2026-08-10) but reverted: on an 8k-doc coverage subset, its dense leg
    # still returned 0.0 recall (same failure pattern as MiniLM) and dragged
    # hybrid fusion below plain BM25 alone. MiniLM + cross-encoder reranking
    # is the only combination with proven full-200k-corpus improvement so far
    # (hybrid_rerank 0.12 vs hybrid 0.092 — see eval/results_rerank.json).
    embedding_model_id: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL_ID"
    )
    reranker_model_id: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL_ID"
    )

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
