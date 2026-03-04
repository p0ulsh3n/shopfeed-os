-- ShopFeed OS — PostgreSQL Schema Migrations
-- All tables from Blueprint Sections 40-43

-- ──────────────────────────────────────────────────────────
-- Users
-- ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email           VARCHAR(255) UNIQUE,
    phone           VARCHAR(50),
    full_name       VARCHAR(200) DEFAULT '',
    avatar_url      TEXT,
    date_of_birth   DATE,
    gender          VARCHAR(20),
    city            VARCHAR(100),
    commune         VARCHAR(100),
    country         CHAR(2) DEFAULT 'CI',
    lat             DOUBLE PRECISION,
    lon             DOUBLE PRECISION,
    persona         VARCHAR(30) DEFAULT 'unknown',
    loyalty_points  INTEGER DEFAULT 0,
    is_verified     BOOLEAN DEFAULT FALSE,
    fcm_token       TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_country ON users(country);

-- ──────────────────────────────────────────────────────────
-- Vendors
-- ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS vendors (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             UUID NOT NULL REFERENCES users(id),
    shop_name           VARCHAR(100) NOT NULL,
    description         TEXT DEFAULT '',
    logo_url            TEXT,
    banner_url          TEXT,
    tier                VARCHAR(20) DEFAULT 'bronze',
    account_weight      REAL DEFAULT 1.0,
    cvr_30d             REAL DEFAULT 0.0,
    avg_rating          REAL DEFAULT 0.0,
    total_sales         INTEGER DEFAULT 0,
    on_time_delivery    REAL DEFAULT 1.0,
    publication_freq    REAL DEFAULT 0.0,
    is_verified         BOOLEAN DEFAULT FALSE,
    geo_zone_id         INTEGER,
    geo_commune         VARCHAR(100),
    geo_city            VARCHAR(100),
    geo_country         CHAR(2) DEFAULT 'CI',
    live_enabled        BOOLEAN DEFAULT FALSE,
    stripe_account_id   VARCHAR(100),
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_vendors_user ON vendors(user_id);
CREATE INDEX idx_vendors_tier ON vendors(tier);

-- ──────────────────────────────────────────────────────────
-- Products
-- ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS products (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vendor_id           UUID NOT NULL REFERENCES vendors(id),
    platform_sku        VARCHAR(50) UNIQUE,
    vendor_sku          VARCHAR(50),
    title               VARCHAR(200) NOT NULL,
    description_short   VARCHAR(280) DEFAULT '',
    description_full    TEXT DEFAULT '',
    category_id         INTEGER DEFAULT 0,
    subcategory_id      INTEGER,
    brand               VARCHAR(100),
    base_price          NUMERIC(12,2) NOT NULL,
    compare_at_price    NUMERIC(12,2),
    currency            CHAR(3) DEFAULT 'EUR',
    base_stock          INTEGER DEFAULT 0,
    has_variants        BOOLEAN DEFAULT FALSE,

    -- Media
    photos              JSONB DEFAULT '[]',
    video_url           TEXT,
    clip_embedding      VECTOR(512),         -- pgvector extension
    cv_score            REAL,
    ai_description      TEXT,

    -- Attributes
    attributes          JSONB DEFAULT '{}',
    tags                TEXT[] DEFAULT '{}',

    -- Shipping
    weight_g            INTEGER,
    dimensions_cm       JSONB,
    processing_days     INTEGER DEFAULT 3,
    shipping_config     JSONB DEFAULT '{}',
    return_policy       VARCHAR(30) DEFAULT 'free_30d',

    -- Status
    status              VARCHAR(30) DEFAULT 'draft',
    pool_level          VARCHAR(5) DEFAULT 'L1',
    freshness_boost_until TIMESTAMPTZ,

    -- Flash Sale
    flash_sale_active   BOOLEAN DEFAULT FALSE,
    flash_sale_price    NUMERIC(12,2),
    flash_sale_ends_at  TIMESTAMPTZ,

    created_at          TIMESTAMPTZ DEFAULT NOW(),
    published_at        TIMESTAMPTZ,
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_products_vendor ON products(vendor_id);
CREATE INDEX idx_products_category ON products(category_id);
CREATE INDEX idx_products_status ON products(status);
CREATE INDEX idx_products_pool ON products(pool_level);

-- ──────────────────────────────────────────────────────────
-- Product Variants
-- ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS product_variants (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    product_id      UUID NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    variant_sku     VARCHAR(50) NOT NULL,
    type1_value     VARCHAR(100),
    type2_value     VARCHAR(100),
    price           NUMERIC(12,2) NOT NULL,
    compare_at_price NUMERIC(12,2),
    stock           INTEGER DEFAULT 0,
    barcode         VARCHAR(50),
    image_url       TEXT,
    is_active       BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_variants_product ON product_variants(product_id);

-- ──────────────────────────────────────────────────────────
-- Product Content (feed items)
-- ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS product_content (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    product_id      UUID NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    vendor_id       UUID NOT NULL REFERENCES vendors(id),
    content_type    VARCHAR(20) DEFAULT 'photo',
    media_urls      JSONB DEFAULT '[]',
    pool_level      VARCHAR(5) DEFAULT 'L1',
    impressions     INTEGER DEFAULT 0,
    clicks          INTEGER DEFAULT 0,
    add_to_carts    INTEGER DEFAULT 0,
    purchases       INTEGER DEFAULT 0,
    buy_now_count   INTEGER DEFAULT 0,
    shares          INTEGER DEFAULT 0,
    saves           INTEGER DEFAULT 0,
    skip_rate       REAL DEFAULT 0.0,
    cvr             REAL DEFAULT 0.0,
    embedding       VECTOR(512),
    live_session_id UUID,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_content_product ON product_content(product_id);
CREATE INDEX idx_content_vendor ON product_content(vendor_id);
CREATE INDEX idx_content_pool ON product_content(pool_level);

-- ──────────────────────────────────────────────────────────
-- Orders
-- ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS orders (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    buyer_id        UUID NOT NULL REFERENCES users(id),
    vendor_id       UUID NOT NULL REFERENCES vendors(id),
    status          VARCHAR(20) DEFAULT 'pending',
    items           JSONB DEFAULT '[]',
    total_gmv       NUMERIC(12,2) DEFAULT 0.0,
    currency        CHAR(3) DEFAULT 'EUR',
    shipping_address JSONB DEFAULT '{}',
    payment_method  VARCHAR(30),
    payment_status  VARCHAR(20) DEFAULT 'pending',
    tracking_number VARCHAR(100),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    confirmed_at    TIMESTAMPTZ,
    shipped_at      TIMESTAMPTZ,
    delivered_at    TIMESTAMPTZ
);

CREATE INDEX idx_orders_buyer ON orders(buyer_id);
CREATE INDEX idx_orders_vendor ON orders(vendor_id);
CREATE INDEX idx_orders_status ON orders(status);

-- ──────────────────────────────────────────────────────────
-- Live Sessions
-- ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS live_sessions (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vendor_id           UUID NOT NULL REFERENCES vendors(id),
    title               VARCHAR(200) DEFAULT '',
    status              VARCHAR(20) DEFAULT 'scheduled',
    live_type           VARCHAR(20) DEFAULT 'instant',
    stream_key          VARCHAR(100),
    playback_url        TEXT,
    scheduled_at        TIMESTAMPTZ,
    started_at          TIMESTAMPTZ,
    ended_at            TIMESTAMPTZ,
    peak_concurrent     INTEGER DEFAULT 0,
    total_viewers       INTEGER DEFAULT 0,
    total_gmv           NUMERIC(12,2) DEFAULT 0.0,
    live_score          REAL DEFAULT 0.0,
    pool_level          VARCHAR(5) DEFAULT 'L1',
    flash_sale_active   BOOLEAN DEFAULT FALSE,
    flash_sale_ends_at  TIMESTAMPTZ,
    pinned_product_ids  UUID[] DEFAULT '{}',
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_live_vendor ON live_sessions(vendor_id);
CREATE INDEX idx_live_status ON live_sessions(status);

-- ──────────────────────────────────────────────────────────
-- Geo Zones (Section 43)
-- ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS geo_zones (
    id                  SERIAL PRIMARY KEY,
    country_code        CHAR(2) NOT NULL,
    country_name        VARCHAR(100) NOT NULL,
    region              VARCHAR(100),
    city                VARCHAR(100) NOT NULL,
    commune             VARCHAR(100) NOT NULL,
    aliases             TEXT[] DEFAULT '{}',
    center_lat          DOUBLE PRECISION DEFAULT 0.0,
    center_lon          DOUBLE PRECISION DEFAULT 0.0,
    adjacent_ids        INTEGER[] DEFAULT '{}',
    timezone            VARCHAR(50) DEFAULT 'Africa/Abidjan',
    UNIQUE(country_code, city, commune)
);

CREATE INDEX idx_geo_country ON geo_zones(country_code);
CREATE INDEX idx_geo_city ON geo_zones(city);

-- ──────────────────────────────────────────────────────────
-- Order Geo Classifications (Section 43)
-- ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS order_geo_classifications (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id            UUID NOT NULL REFERENCES orders(id),
    vendor_id           UUID NOT NULL REFERENCES vendors(id),
    buyer_country       CHAR(2) DEFAULT 'CI',
    buyer_city          VARCHAR(100) DEFAULT '',
    buyer_commune       VARCHAR(100) DEFAULT '',
    buyer_lat           DOUBLE PRECISION,
    buyer_lon           DOUBLE PRECISION,
    vendor_commune      VARCHAR(100) DEFAULT '',
    geo_level           VARCHAR(5) DEFAULT 'L4',
    geo_cluster_id      INTEGER,
    distance_km         REAL DEFAULT 0.0,
    classification_method VARCHAR(20) DEFAULT 'geo_hierarchy',
    confidence          REAL DEFAULT 1.0,
    is_outlier          BOOLEAN DEFAULT FALSE,
    shipping_suggested  VARCHAR(200) DEFAULT '',
    classified_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_geo_class_order ON order_geo_classifications(order_id);
CREATE INDEX idx_geo_class_vendor ON order_geo_classifications(vendor_id);
CREATE INDEX idx_geo_class_level ON order_geo_classifications(geo_level);

-- ──────────────────────────────────────────────────────────
-- User Profiles (ML Feature Store)
-- ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS user_profiles (
    user_id             UUID PRIMARY KEY REFERENCES users(id),
    category_prefs      JSONB DEFAULT '{}',
    price_ranges        JSONB DEFAULT '{}',
    intent_level        VARCHAR(20) DEFAULT 'low',
    active_categories   TEXT[] DEFAULT '{}',
    embedding           VECTOR(256),
    purchase_frequency  REAL DEFAULT 0.0,
    avg_order_value     REAL DEFAULT 0.0,
    top_vendors         UUID[] DEFAULT '{}',
    persona             VARCHAR(30) DEFAULT 'unknown',
    geo_cluster         INTEGER DEFAULT 0,
    last_active_at      TIMESTAMPTZ,
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ──────────────────────────────────────────────────────────
-- Follows
-- ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS follows (
    user_id     UUID NOT NULL REFERENCES users(id),
    vendor_id   UUID NOT NULL REFERENCES vendors(id),
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (user_id, vendor_id)
);

CREATE INDEX idx_follows_vendor ON follows(vendor_id);

-- ══════════════════════════════════════════════════════════
-- ML TABLES — Section 42 (14 tables total)
-- ══════════════════════════════════════════════════════════

-- ──────────────────────────────────────────────────────────
-- feed_videos — Upload & Video Pipeline (Section 42)
-- ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS feed_videos (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vendor_id               UUID NOT NULL REFERENCES vendors(id),
    product_ids             UUID[] DEFAULT '{}',

    upload_status           VARCHAR(20) DEFAULT 'uploading',
    -- uploading / uploaded / transcoding / moderation / approved / rejected / published / paused / deleted

    -- Media URLs
    raw_url                 TEXT,
    hls_url                 TEXT,
    mp4_720p_url            TEXT,
    mp4_360p_url            TEXT,
    thumbnail_url           TEXT,

    -- Media metadata
    duration_sec            REAL,
    resolution              VARCHAR(20),
    file_size_bytes         BIGINT,
    aspect_ratio            VARCHAR(20) DEFAULT 'vertical_916',

    -- ML embeddings
    audio_embedding         VECTOR(128),       -- VGGish (Section 16)
    visual_embeddings       JSONB DEFAULT '[]', -- [{frame_sec, embedding_512}] CLIP × 5 frames
    transcript              TEXT,               -- Whisper ASR
    detected_products       JSONB DEFAULT '[]', -- [{product_id, confidence, timestamp_sec}]

    -- Content
    caption                 TEXT DEFAULT '',
    hashtags                TEXT[] DEFAULT '{}',

    -- Moderation
    moderation_results      JSONB DEFAULT '{}', -- SightEngine: {nudity_score, phone_detected, ...}
    cv_score                REAL,               -- Quality score (Section 15)

    -- Scoring
    account_weight_at_post  REAL DEFAULT 1.0,
    initial_pool            VARCHAR(5) DEFAULT 'L1',
    current_pool            VARCHAR(5) DEFAULT 'L1',
    score_cvr               REAL DEFAULT 0.0,
    score_watch_time        REAL DEFAULT 0.0,
    score_engagement        REAL DEFAULT 0.0,

    -- Counters
    total_views             BIGINT DEFAULT 0,
    total_likes             INTEGER DEFAULT 0,
    total_shares            INTEGER DEFAULT 0,
    total_add_to_cart       INTEGER DEFAULT 0,
    total_purchases         INTEGER DEFAULT 0,
    gmv_attributed          NUMERIC(12,2) DEFAULT 0.0,

    -- Lifecycle
    published_at            TIMESTAMPTZ,
    expires_at              TIMESTAMPTZ,
    transcoding_job_id      TEXT,

    -- Advertising (Section 37)
    is_boosted              BOOLEAN DEFAULT FALSE,
    boost_budget_spent      NUMERIC(10,2) DEFAULT 0.0,

    created_at              TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_feed_videos_vendor ON feed_videos(vendor_id);
CREATE INDEX idx_feed_videos_status ON feed_videos(upload_status);
CREATE INDEX idx_feed_videos_pool ON feed_videos(current_pool);
CREATE INDEX idx_feed_videos_published ON feed_videos(published_at);
CREATE INDEX idx_feed_videos_boosted ON feed_videos(is_boosted) WHERE is_boosted = TRUE;

-- ──────────────────────────────────────────────────────────
-- ml_item_features — ML Feature Store per product/content
-- ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS ml_item_features (
    item_id                 UUID PRIMARY KEY,
    item_type               VARCHAR(20) DEFAULT 'product',  -- product / video / carousel / live

    visual_embedding        VECTOR(512),    -- CLIP/FashionSigLIP
    text_embedding          VECTOR(768),    -- sentence-transformers
    combined_embedding      VECTOR(256),    -- Two-Tower output (compressed for retrieval)

    price_norm              REAL,           -- log(price / avg_category)
    freshness               REAL,           -- exp(-age_hours / 168) [0,1]
    category_vec            JSONB,          -- one-hot + learned embedding
    attribute_vec           JSONB,          -- 228 iMaterialist attributes encoded

    vendor_embedding        VECTOR(64),     -- Vendor embedding (learned)

    cvr_7d                  REAL DEFAULT 0.0,    -- CVR rolling 7 days
    ctr_7d                  REAL DEFAULT 0.0,    -- CTR rolling 7 days
    watch_time_avg          REAL DEFAULT 0.0,    -- Watch time % (videos)
    seller_weight           REAL DEFAULT 1.0,    -- Account Weight snapshot
    stock_signal            REAL DEFAULT 1.0,    -- [0,1]

    computed_at             TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_ml_items_type ON ml_item_features(item_type);
CREATE INDEX idx_ml_items_computed ON ml_item_features(computed_at);

-- ──────────────────────────────────────────────────────────
-- ml_user_features — ML Profile per user (Section 42)
-- ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS ml_user_features (
    user_id                 UUID PRIMARY KEY REFERENCES users(id),

    user_embedding          VECTOR(256),    -- Two-Tower learned embedding
    interest_categories     JSONB DEFAULT '{}',  -- {category_id → score [0,1]}
    price_range_pref        JSONB DEFAULT '{"min":0,"max":100,"avg":50}',

    behavior_sequence       JSONB DEFAULT '[]',  -- Last 200 interactions [{item_id, action, ts}]
    long_term_sequence      JSONB DEFAULT '[]',  -- 30-day compressed (for SIM model)

    -- RFM Features
    rfm_recency             INTEGER DEFAULT 365,   -- Days since last purchase
    rfm_frequency           INTEGER DEFAULT 0,     -- Purchases in 90 days
    rfm_monetary            NUMERIC(10,2) DEFAULT 0.0,  -- GMV 90 days

    -- Context
    device_os               VARCHAR(10) DEFAULT 'android',
    country                 CHAR(2) DEFAULT 'CI',
    city                    VARCHAR(100) DEFAULT '',

    followed_vendors        UUID[] DEFAULT '{}',
    watchlist_categories    INTEGER[] DEFAULT '{}',

    updated_at              TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_ml_users_updated ON ml_user_features(updated_at);

-- ──────────────────────────────────────────────────────────
-- ml_interactions — Every user × item event (Section 42)
-- Source of truth for training data
-- ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS ml_interactions (
    id                      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id                 UUID NOT NULL REFERENCES users(id),
    item_id                 UUID NOT NULL,
    item_type               VARCHAR(20) DEFAULT 'product',
    action                  VARCHAR(30) NOT NULL,
    -- view / like / comment / share / add_to_cart / purchase / wishlist /
    -- skip / dwell / scroll_past / live_join / buy_now_click / live_buy
    action_weight           REAL DEFAULT 1.0,
    dwell_time_ms           INTEGER DEFAULT 0,
    watch_pct               REAL DEFAULT 0.0,
    source                  VARCHAR(20) DEFAULT 'feed',
    -- feed / search / marketplace / live / notif / direct / following
    position_in_feed        INTEGER DEFAULT 0,
    session_id              UUID,
    device_os               VARCHAR(10) DEFAULT 'android',
    country                 CHAR(2) DEFAULT 'CI',
    timestamp               TIMESTAMPTZ DEFAULT NOW()
) PARTITION BY RANGE (timestamp);

-- Monthly partitions (create dynamically in production)
CREATE TABLE ml_interactions_default PARTITION OF ml_interactions DEFAULT;

CREATE INDEX idx_ml_interactions_user ON ml_interactions(user_id);
CREATE INDEX idx_ml_interactions_item ON ml_interactions(item_id);
CREATE INDEX idx_ml_interactions_action ON ml_interactions(action);
CREATE INDEX idx_ml_interactions_session ON ml_interactions(session_id);
CREATE INDEX idx_ml_interactions_ts ON ml_interactions(timestamp);

-- ──────────────────────────────────────────────────────────
-- ml_training_samples — Nightly generated from ml_interactions
-- ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS ml_training_samples (
    id                      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_features_json      JSONB NOT NULL,
    item_features_json      JSONB NOT NULL,
    label                   SMALLINT NOT NULL,   -- 0 or 1 (CTR label)
    weight                  REAL DEFAULT 1.0,    -- Sample weight for imbalanced classes
    split                   VARCHAR(10) DEFAULT 'train',  -- train / val / test
    generated_at            TIMESTAMPTZ DEFAULT NOW()
);

-- ──────────────────────────────────────────────────────────
-- ml_model_registry — Model versioning (Section 42)
-- ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS ml_model_registry (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name              VARCHAR(100) NOT NULL,
    -- two_tower_retrieval / din_ranking / dien_sequential / bst_attention /
    -- deepfm_ranking / ple_mtl / geo_order_classifier
    version                 VARCHAR(20) NOT NULL,
    status                  VARCHAR(20) DEFAULT 'training',
    -- training / staging / production / retired
    s3_path                 TEXT,
    training_samples        INTEGER DEFAULT 0,
    metrics                 JSONB DEFAULT '{}',
    -- {auc_roc, recall@10, ndcg@10, precision@5}
    deployed_at             TIMESTAMPTZ,
    trained_at              TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_model_registry_name ON ml_model_registry(model_name);
CREATE INDEX idx_model_registry_status ON ml_model_registry(status);

-- Enforce: only 1 model per name in production
CREATE UNIQUE INDEX idx_model_registry_production
    ON ml_model_registry(model_name) WHERE status = 'production';

-- ──────────────────────────────────────────────────────────
-- ml_live_features — Real-time live session ML features
-- ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS ml_live_features (
    live_id                 UUID PRIMARY KEY REFERENCES live_sessions(id),
    concurrent_viewers      INTEGER DEFAULT 0,
    gmv_per_minute          REAL DEFAULT 0.0,
    live_score              REAL DEFAULT 0.0,
    peak_score              REAL DEFAULT 0.0,
    product_featured_ids    UUID[] DEFAULT '{}',
    buy_now_count_last_5min INTEGER DEFAULT 0,
    updated_at              TIMESTAMPTZ DEFAULT NOW()
);

-- ──────────────────────────────────────────────────────────
-- feature_store_daily — Daily snapshot for batch training
-- ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS feature_store_daily (
    date                    DATE NOT NULL,
    item_id                 UUID NOT NULL,
    -- All ml_item_features columns + daily metrics
    visual_embedding        VECTOR(512),
    text_embedding          VECTOR(768),
    price_norm              REAL,
    freshness               REAL,
    vendor_embedding        VECTOR(64),
    cvr_daily               REAL DEFAULT 0.0,
    ctr_daily               REAL DEFAULT 0.0,
    gmv_daily               NUMERIC(10,2) DEFAULT 0.0,
    views_daily             INTEGER DEFAULT 0,
    PRIMARY KEY (date, item_id)
);

-- ──────────────────────────────────────────────────────────
-- ml_feed_scores_cache — Cached ranking scores (TTL 15min)
-- Note: Primary store is Redis, this is the persistence layer
-- ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS ml_feed_scores_cache (
    user_id                 UUID NOT NULL,
    item_id                 UUID NOT NULL,
    score                   REAL NOT NULL,
    computed_at             TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (user_id, item_id)
);

-- ──────────────────────────────────────────────────────────
-- ml_ab_tests — A/B test configurations and results
-- ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS ml_ab_tests (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    test_name               VARCHAR(100) NOT NULL UNIQUE,
    variant_a_model         VARCHAR(100) NOT NULL,
    variant_b_model         VARCHAR(100) NOT NULL,
    user_split_pct          REAL DEFAULT 50.0,    -- % of users in variant A
    start_date              TIMESTAMPTZ DEFAULT NOW(),
    end_date                TIMESTAMPTZ,
    metrics_json            JSONB DEFAULT '{}',   -- {variant_a: {auc, cvr, gmv}, variant_b: {...}}
    winner                  VARCHAR(10),          -- 'a', 'b', or null
    status                  VARCHAR(20) DEFAULT 'running'  -- running / completed / cancelled
);

-- ──────────────────────────────────────────────────────────
-- ml_cold_start_queue — New items/vendors awaiting initialization
-- ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS ml_cold_start_queue (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    item_id                 UUID NOT NULL,
    item_type               VARCHAR(20) DEFAULT 'product',  -- product / vendor / video
    strategy                VARCHAR(30) DEFAULT 'category_match',
    -- credit / category_match / lookalike / content_scoring
    initial_pool            VARCHAR(5) DEFAULT 'L1',
    impressions_guaranteed  INTEGER DEFAULT 500,
    impressions_served      INTEGER DEFAULT 0,
    boosted_until           TIMESTAMPTZ,
    status                  VARCHAR(20) DEFAULT 'pending',  -- pending / active / completed
    created_at              TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_cold_start_status ON ml_cold_start_queue(status);
CREATE INDEX idx_cold_start_item ON ml_cold_start_queue(item_id);

-- ──────────────────────────────────────────────────────────
-- ml_content_pools — Pool cache L1-L4 per category
-- ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS ml_content_pools (
    id                      SERIAL PRIMARY KEY,
    pool_level              VARCHAR(5) NOT NULL,   -- L1, L2, L3, L4
    category_id             INTEGER NOT NULL,
    item_ids_json           JSONB DEFAULT '[]',    -- Ordered list of item UUIDs
    last_computed           TIMESTAMPTZ DEFAULT NOW(),
    ttl_seconds             INTEGER DEFAULT 300,   -- 5 min default
    UNIQUE (pool_level, category_id)
);

-- ──────────────────────────────────────────────────────────
-- ml_embedding_index — FAISS index registry
-- ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS ml_embedding_index (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    index_name              VARCHAR(100) NOT NULL,  -- e.g. 'two_tower_v3'
    index_path_s3           TEXT NOT NULL,
    vector_count            INTEGER DEFAULT 0,
    dim                     INTEGER DEFAULT 256,
    built_at                TIMESTAMPTZ DEFAULT NOW(),
    status                  VARCHAR(20) DEFAULT 'building'  -- building / active / retired
);

CREATE UNIQUE INDEX idx_embedding_index_active
    ON ml_embedding_index(index_name) WHERE status = 'active';
