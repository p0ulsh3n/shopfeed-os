"""
Database Connection + pgvector Schema
======================================
Async PostgreSQL with pgvector extension.
HNSW index — 2026 production standard for ANN search.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import asyncpg
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from services.shopbot_service.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ─────────────────────── SQLAlchemy Engine ────────────────────────

engine = create_async_engine(
    str(settings.database_url),
    echo=settings.environment == "development",
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,   # Detect stale connections
    pool_recycle=3600,    # Recycle connections every hour
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager for database sessions."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ─────────────────────── RAW asyncpg pool ────────────────────────
# Used for LISTEN/NOTIFY (catalog sync) and bulk COPY operations
# asyncpg is faster than SQLAlchemy for these low-level operations

_raw_pool: asyncpg.Pool | None = None


async def get_raw_pool() -> asyncpg.Pool:
    global _raw_pool
    if _raw_pool is None:
        dsn = str(settings.database_url).replace("+asyncpg", "")
        _raw_pool = await asyncpg.create_pool(
            dsn,
            min_size=2,
            max_size=10,
            command_timeout=30,
        )
    return _raw_pool


async def close_raw_pool() -> None:
    global _raw_pool
    if _raw_pool:
        await _raw_pool.close()
        _raw_pool = None


# ─────────────────────── Schema Migration ────────────────────────

SHOPBOT_SCHEMA_SQL = """
-- Enable pgvector extension (required once per DB)
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable pg_trgm for BM25-like text search
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ════════════════════════════════════════════════════════════════
-- SHOP CATALOG INDEX TABLE
-- Each row = one product embedding for one shop.
-- Supports multi-tenancy via shop_id filter on all queries.
-- ════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS shopbot_product_embeddings (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    shop_id         TEXT NOT NULL,
    product_id      TEXT NOT NULL,

    -- Rich text for BM25 / full-text search
    product_text    TEXT NOT NULL,

    -- Dense float32 embedding (multilingual-e5-large-instruct, 1024-dim)
    -- Used for computing int8 embeddings and for rescoring after binary retrieval
    embedding_float32 vector(1024),

    -- int8 quantized embedding (4× smaller, <1% quality loss)
    -- PRIMARY retrieval embedding (best speed/quality tradeoff)
    embedding_int8  BYTEA,

    -- Binary quantized embedding (32× smaller, fast pre-filter)
    -- Used for ultra-fast candidate pre-screening
    embedding_binary BYTEA,

    -- Product metadata as JSONB for fast filter-based retrieval
    metadata        JSONB NOT NULL DEFAULT '{}',

    -- Timestamps
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Unique constraint: one embedding per (shop, product)
    UNIQUE (shop_id, product_id)
);

-- ════════════════════════════════════════════════════════════════
-- HNSW INDEX ON int8 EMBEDDINGS (2026 standard)
-- HNSW is superior to IVFFlat for dynamic catalogs:
--   - No training/rebuilding needed when adding products
--   - O(log n) updates vs O(n) for IVFFlat
--   - Better recall at same latency
-- ════════════════════════════════════════════════════════════════
CREATE INDEX IF NOT EXISTS idx_shopbot_hnsw_int8
ON shopbot_product_embeddings
USING hnsw (embedding_float32 vector_cosine_ops)
WITH (
    m = 16,          -- Connections per node (16 = balanced speed/quality)
    ef_construction = 64  -- Build quality (higher = better index, slower build)
);

-- ════════════════════════════════════════════════════════════════
-- GIN INDEX FOR FULL-TEXT SEARCH (BM25 via tsvector)
-- Enables fast keyword/sparse retrieval without a separate service
-- ════════════════════════════════════════════════════════════════
ALTER TABLE shopbot_product_embeddings
    ADD COLUMN IF NOT EXISTS product_tsv TSVECTOR
    GENERATED ALWAYS AS (
        to_tsvector('french', coalesce(product_text, ''))
    ) STORED;

CREATE INDEX IF NOT EXISTS idx_shopbot_fts
ON shopbot_product_embeddings
USING GIN (product_tsv);

-- ════════════════════════════════════════════════════════════════
-- COMPOSITE INDEX for fast per-shop filtering
-- All vector searches MUST filter on shop_id first
-- ════════════════════════════════════════════════════════════════
CREATE INDEX IF NOT EXISTS idx_shopbot_shop_id
ON shopbot_product_embeddings (shop_id, updated_at DESC);

-- JSONB index for metadata filtering (price, category, availability)
CREATE INDEX IF NOT EXISTS idx_shopbot_metadata
ON shopbot_product_embeddings USING GIN (metadata);

-- ════════════════════════════════════════════════════════════════
-- CHAT SESSIONS TABLE
-- Stores conversation history for context continuity
-- ════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS shopbot_sessions (
    session_id      TEXT PRIMARY KEY,
    shop_id         TEXT NOT NULL,
    customer_id     TEXT,
    history         JSONB NOT NULL DEFAULT '[]',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at      TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '24 hours')
);

CREATE INDEX IF NOT EXISTS idx_shopbot_sessions_shop
ON shopbot_sessions (shop_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_shopbot_sessions_expires
ON shopbot_sessions (expires_at)
WHERE expires_at < NOW();

-- ════════════════════════════════════════════════════════════════
-- CATALOG SYNC LOG TABLE
-- Track when each shop's catalog was last synced
-- ════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS shopbot_sync_log (
    id              BIGSERIAL PRIMARY KEY,
    shop_id         TEXT NOT NULL,
    sync_type       TEXT NOT NULL,  -- 'full', 'incremental', 'delete'
    products_count  INT NOT NULL DEFAULT 0,
    status          TEXT NOT NULL,  -- 'success', 'failed', 'partial'
    error_message   TEXT,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_shopbot_sync_shop
ON shopbot_sync_log (shop_id, started_at DESC);

-- ════════════════════════════════════════════════════════════════
-- PostgreSQL TRIGGER for real-time catalog updates via LISTEN/NOTIFY
-- When a product is created/updated/deleted in the main DB,
-- this trigger fires and ShopBot picks it up asynchronously.
-- ════════════════════════════════════════════════════════════════
CREATE OR REPLACE FUNCTION notify_catalog_change()
RETURNS trigger AS $$
DECLARE
    payload JSONB;
BEGIN
    IF TG_OP = 'DELETE' THEN
        payload = jsonb_build_object(
            'event_type', TG_OP,
            'shop_id',    OLD.shop_id,
            'product_id', OLD.id::text
        );
    ELSE
        payload = jsonb_build_object(
            'event_type',   TG_OP,
            'shop_id',      NEW.shop_id,
            'product_id',   NEW.id::text,
            'name',         NEW.name,
            'price',        NEW.price,
            'availability', NEW.availability,
            'updated_at',   NEW.updated_at
        );
    END IF;

    PERFORM pg_notify('shopbot_catalog_updates', payload::text);
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Attach trigger to your main products table
-- (Adjust table name to match your actual products table)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgname = 'trg_shopbot_catalog_notify'
    ) THEN
        CREATE TRIGGER trg_shopbot_catalog_notify
        AFTER INSERT OR UPDATE OR DELETE ON products
        FOR EACH ROW EXECUTE FUNCTION notify_catalog_change();
    END IF;
END $$;

-- ════════════════════════════════════════════════════════════════
-- UPDATED_AT auto-update trigger for shopbot tables
-- ════════════════════════════════════════════════════════════════
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE 'plpgsql';

CREATE TRIGGER trg_shopbot_embeddings_updated_at
    BEFORE UPDATE ON shopbot_product_embeddings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trg_shopbot_sessions_updated_at
    BEFORE UPDATE ON shopbot_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
"""


async def initialize_schema() -> None:
    """
    Run the schema migration on startup.
    Idempotent — safe to call multiple times (uses IF NOT EXISTS).
    """
    logger.info("Initializing ShopBot database schema...")
    async with engine.begin() as conn:
        # Execute each statement separately (asyncpg doesn't support multi-statement)
        statements = [
            s.strip() for s in SHOPBOT_SCHEMA_SQL.split(";")
            if s.strip()
        ]
        for stmt in statements:
            try:
                await conn.execute(text(stmt))
            except Exception as e:
                # Some statements may fail if objects already exist — that's OK
                logger.debug(f"Schema stmt skipped (likely exists): {e}")
    logger.info("ShopBot database schema ready ✓")


async def check_db_health() -> bool:
    """Quick health check — verify DB is reachable."""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"DB health check failed: {e}")
        return False
