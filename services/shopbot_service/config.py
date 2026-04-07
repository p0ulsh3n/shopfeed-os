"""
ShopBot Configuration
====================
Centralized config via Pydantic Settings — reads from env vars or .env file.
All secrets via env (never hardcoded).
"""
from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ShopBotSettings(BaseSettings):
    """
    Production-ready settings with full env-var support.
    Override any value via env or .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="SHOPBOT_",
        case_sensitive=False,
    )

    # ─────────────────────────── APP ─────────────────────────────
    app_name: str = "ShopFeed ShopBot"
    environment: Literal["development", "staging", "production"] = "production"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8070
    api_workers: int = 4

    # ─────────────────────────── AUTH ────────────────────────────
    # Internal secret for service-to-service calls
    internal_api_key: str = Field(..., description="Internal API key (required)")

    # ─────────────────────────── vLLM ────────────────────────────
    # vLLM OpenAI-compatible server URL (self-hosted)
    vllm_base_url: str = "http://vllm-server:8000/v1"
    vllm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
    vllm_timeout: int = 60           # seconds
    vllm_max_tokens: int = 1024
    vllm_temperature: float = 0.3    # Low temp → more factual, fewer hallucinations
    vllm_top_p: float = 0.9

    # Automatic Prefix Caching — enabled server-side in vLLM;
    # here we control the system prompt structure to maximize cache hits
    vllm_enable_streaming: bool = True

    # ──────────────────────── EMBEDDINGS ─────────────────────────
    # Best multilingual embedding model for e-commerce (MTEB 2026)
    # multilingual-e5-large-instruct: 560M, 1024-dim, supports FR/AR/EN natively
    embedding_model: str = "intfloat/multilingual-e5-large-instruct"
    embedding_dim: int = 1024
    embedding_batch_size: int = 64
    embedding_device: str = "cuda"   # "cpu" for dev

    # Embedding Quantization (2026 best practice):
    # - "binary": 32× memory reduction, minimal quality loss (>95% retained)
    # - "int8":   4× memory reduction, <1% quality loss
    # - "float32": no quantization (dev/debug only)
    embedding_quantization: Literal["binary", "int8", "float32"] = "int8"
    embedding_rescoring: bool = True  # Rescore with float32 after binary retrieval

    # ──────────────────────── DATABASE ───────────────────────────
    database_url: PostgresDsn = Field(
        default="postgresql+asyncpg://shopbot:shopbot@postgres:5432/shopfeed",
        description="PostgreSQL DSN with pgvector extension",
    )

    # pgvector HNSW index parameters (2026 standard)
    hnsw_m: int = 16              # Number of connections per node
    hnsw_ef_construction: int = 64  # Build-time quality (higher = better index)
    hnsw_ef_search: int = 40       # Query-time quality

    # ─────────────────────── HYBRID SEARCH ───────────────────────
    # Retrieval candidates before RRF fusion
    retrieval_top_k_dense: int = 50
    retrieval_top_k_sparse: int = 50
    # Final results after RRF + reranking
    retrieval_final_top_k: int = 8
    # BM25 parameters
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    # RRF constant (60 is the de-facto 2026 standard from Cormack & Clarke)
    rrf_k: int = 60

    # ─────────────────────── CATALOG SYNC ────────────────────────
    # PostgreSQL LISTEN/NOTIFY channel for real-time catalog updates
    catalog_notify_channel: str = "shopbot_catalog_updates"
    sync_batch_size: int = 32       # Products per sync batch
    sync_max_retries: int = 3

    # ──────────────────────── CACHE (Redis) ──────────────────────
    redis_url: str = "redis://redis:6379/2"
    cache_ttl_seconds: int = 300    # 5 min cache for repeated queries
    cache_enabled: bool = True

    # ──────────────────────── SHOP CONTEXT ───────────────────────
    # Max catalog products to include in RAG context
    max_context_products: int = 8
    # Max conversation turns to keep in memory
    max_history_turns: int = 10
    # System prompt language detection
    auto_detect_language: bool = True
    default_language: str = "fr"    # ShopFeed is primarily French-speaking

    @field_validator("embedding_dim")
    @classmethod
    def validate_embedding_dim(cls, v: int) -> int:
        supported = [384, 768, 1024]
        if v not in supported:
            raise ValueError(f"embedding_dim must be one of {supported}")
        return v


@lru_cache(maxsize=1)
def get_settings() -> ShopBotSettings:
    """
    Cached singleton — call this everywhere to avoid re-parsing env.
    Usage:
        from services.shopbot_service.config import get_settings
        settings = get_settings()
    """
    return ShopBotSettings()
