-- ════════════════════════════════════════════════════════════════
-- ShopBot PostgreSQL Init Script
-- ════════════════════════════════════════════════════════════════
-- Auto-executed on first container start via docker-entrypoint-initdb.d/
-- Creates all required tables, extensions, and indexes.
-- ════════════════════════════════════════════════════════════════

-- ── Extensions ────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS vector;          -- pgvector
CREATE EXTENSION IF NOT EXISTS pg_trgm;         -- Trigram similarity
CREATE EXTENSION IF NOT EXISTS unaccent;        -- Accent-insensitive search

-- ── Custom text search config (French + unaccent) ─────────────────
-- Used for BM25/tsvector to handle accents in French product names
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_ts_config WHERE cfgname = 'shopbot_french'
    ) THEN
        CREATE TEXT SEARCH CONFIGURATION shopbot_french (COPY = french);
        ALTER TEXT SEARCH CONFIGURATION shopbot_french
            ALTER MAPPING FOR hword, hword_part, word
            WITH unaccent, french_stem;
    END IF;
END;
$$;

-- ════════════════════════════════════════════════════════════════
-- TABLE: shopbot_products (Vector store for RAG retrieval)
-- ════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS shopbot_products (
    -- Identity
    id              TEXT PRIMARY KEY,
    shop_id         TEXT NOT NULL,

    -- Embedding (multilingual-e5-large-instruct, 1024-dim)
    -- Stored as int8 (quantized) for 4× memory savings
    embedding       VECTOR(1024),

    -- BM25 sparse retrieval (pre-tokenized product text)
    bm25_text       TSVECTOR,

    -- Product metadata (JSONB — flexible schema, no ALTER TABLE needed)
    -- Contains: name, price, currency, description, category, tags,
    --           images[], availability, stock_quantity, attributes{}
    metadata        JSONB NOT NULL DEFAULT '{}',

    -- Sync tracking
    product_updated_at  TIMESTAMPTZ,
    indexed_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── HNSW vector index (2026 standard — faster than IVFFlat) ──────
-- m=16: good balance of memory/speed for catalogs <1M products
-- ef_construction=64: build quality (higher = better index, slower build)
CREATE INDEX IF NOT EXISTS idx_shopbot_products_embedding_hnsw
    ON shopbot_products
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ── BM25 full-text search index ───────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_shopbot_products_bm25
    ON shopbot_products
    USING GIN (bm25_text);

-- ── Composite index for shop-level filtering ──────────────────────
CREATE INDEX IF NOT EXISTS idx_shopbot_products_shop
    ON shopbot_products (shop_id, indexed_at DESC);

-- ── JSONB metadata index (for category/availability filters) ──────
CREATE INDEX IF NOT EXISTS idx_shopbot_products_metadata
    ON shopbot_products
    USING GIN (metadata jsonb_path_ops);

-- ── Updated_at auto-update trigger ────────────────────────────────
CREATE OR REPLACE FUNCTION update_shopbot_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_shopbot_products_updated ON shopbot_products;
CREATE TRIGGER trg_shopbot_products_updated
    BEFORE UPDATE ON shopbot_products
    FOR EACH ROW EXECUTE FUNCTION update_shopbot_updated_at();

-- ════════════════════════════════════════════════════════════════
-- TABLE: shopbot_sessions (Conversation history)
-- ════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS shopbot_sessions (
    session_id      TEXT PRIMARY KEY,
    shop_id         TEXT NOT NULL,
    history         JSONB NOT NULL DEFAULT '[]',  -- List of ChatMessage objects
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at      TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '24 hours')
);

CREATE INDEX IF NOT EXISTS idx_shopbot_sessions_shop
    ON shopbot_sessions (shop_id, expires_at DESC);

CREATE INDEX IF NOT EXISTS idx_shopbot_sessions_expires
    ON shopbot_sessions (expires_at);

DROP TRIGGER IF EXISTS trg_shopbot_sessions_updated ON shopbot_sessions;
CREATE TRIGGER trg_shopbot_sessions_updated
    BEFORE UPDATE ON shopbot_sessions
    FOR EACH ROW EXECUTE FUNCTION update_shopbot_updated_at();

-- ════════════════════════════════════════════════════════════════
-- LISTEN/NOTIFY: Real-time catalog synchronization
-- ════════════════════════════════════════════════════════════════
-- Fires when a product is created, updated, or deleted.
-- ShopBot listens on 'shopbot_catalog_updates' channel.
-- Payload format: {"event": "upsert|delete", "product_id": "...", "shop_id": "..."}

CREATE OR REPLACE FUNCTION notify_shopbot_catalog_change()
RETURNS TRIGGER AS $$
DECLARE
    payload JSONB;
BEGIN
    IF TG_OP = 'DELETE' THEN
        payload = jsonb_build_object(
            'event', 'delete',
            'product_id', OLD.id,
            'shop_id', OLD.shop_id
        );
        PERFORM pg_notify('shopbot_catalog_updates', payload::TEXT);
        RETURN OLD;
    ELSE
        payload = jsonb_build_object(
            'event', 'upsert',
            'product_id', NEW.id,
            'shop_id', NEW.shop_id,
            'updated_at', NOW()
        );
        PERFORM pg_notify('shopbot_catalog_updates', payload::TEXT);
        RETURN NEW;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Attach trigger to the main products table
-- NOTE: This assumes the main backend's 'products' table exists.
-- If the table doesn't exist yet, this trigger will be skipped gracefully.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'products'
    ) THEN
        DROP TRIGGER IF EXISTS trg_shopbot_catalog_notify ON products;
        CREATE TRIGGER trg_shopbot_catalog_notify
            AFTER INSERT OR UPDATE OR DELETE ON products
            FOR EACH ROW EXECUTE FUNCTION notify_shopbot_catalog_change();
        RAISE NOTICE 'ShopBot catalog LISTEN/NOTIFY trigger created on products table';
    ELSE
        RAISE NOTICE 'products table not found — catalog trigger will be created after backend migration';
    END IF;
END;
$$;

-- ════════════════════════════════════════════════════════════════
-- SESSION CLEANUP JOB (run periodically)
-- ════════════════════════════════════════════════════════════════
-- Called by a cron job or pg_cron extension to purge expired sessions.
-- Usage: SELECT cleanup_expired_shopbot_sessions();
CREATE OR REPLACE FUNCTION cleanup_expired_shopbot_sessions()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM shopbot_sessions WHERE expires_at < NOW();
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ════════════════════════════════════════════════════════════════
-- GRANTS (run as superuser, then restrict)
-- ════════════════════════════════════════════════════════════════
GRANT SELECT, INSERT, UPDATE, DELETE ON shopbot_products TO shopbot;
GRANT SELECT, INSERT, UPDATE, DELETE ON shopbot_sessions TO shopbot;

-- ── Done ──────────────────────────────────────────────────────────
DO $$ BEGIN
    RAISE NOTICE '✓ ShopBot schema initialized successfully';
END; $$;
