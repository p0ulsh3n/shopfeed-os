-- ============================================================================
-- ShopFeed-OS ML Database Schema
-- Base: ml_db (PostgreSQL 16 + pgvector + PostGIS)
-- Migrations: Alembic
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;      -- pgvector
CREATE EXTENSION IF NOT EXISTS postgis;     -- pour geo_classifier backup

-- ============================================================================
-- 1. ml_item_features
-- Feature vectors pré-calculés par item (product, video, live)
-- Invalidés via Kafka shopfeed.product.events
-- ============================================================================

CREATE TABLE ml_item_features (
  item_id               UUID NOT NULL,
  item_type             VARCHAR(20) NOT NULL,   -- product|video|carousel|live
  visual_embedding      vector(512),            -- CLIP/FashionSigLIP (gelé)
  text_embedding        vector(768),            -- sentence-transformers (gelé)
  combined_embedding    vector(256),            -- Two-Tower item tower output (entraînable)
  price_norm            FLOAT,                  -- log(price / avg_category_price)
  freshness             FLOAT,                  -- exp(-age_h/168) [0,1] — recalculé /h
  category_vec          vector(20),             -- learned category embedding
  vendor_embedding      vector(64),             -- learned vendor embedding (entraînable)
  cvr_7d                FLOAT DEFAULT 0,        -- CVR rolling 7j — signal fort
  ctr_7d                FLOAT DEFAULT 0,
  watch_time_avg        FLOAT DEFAULT 0,
  seller_weight         FLOAT DEFAULT 1.0,      -- account_weight du vendeur
  stock_signal          FLOAT DEFAULT 1.0,      -- min(stock,100)/100
  pool_level            VARCHAR(5) DEFAULT 'L1',-- L1|L2|L3|L4|L5|L6
  computed_at           TIMESTAMP DEFAULT NOW(),
  PRIMARY KEY (item_id, item_type)
);
CREATE INDEX idx_ml_items_type ON ml_item_features(item_type);
CREATE INDEX idx_ml_items_computed ON ml_item_features(computed_at DESC);
-- HNSW pour retrieval ANN pgvector (backup FAISS)
CREATE INDEX idx_ml_items_combined ON ml_item_features
  USING hnsw (combined_embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- ============================================================================
-- 2. ml_user_features
-- Profil ML utilisateur — mis à jour via Kafka shopfeed.feed.events
-- ============================================================================

CREATE TABLE ml_user_features (
  user_id               UUID PRIMARY KEY,
  user_embedding        vector(256),            -- Two-Tower user tower output
  interest_categories   JSONB DEFAULT '{}',     -- {cat_id: score_float}
  price_range_pref      JSONB DEFAULT '{}',     -- {min, max, avg}
  behavior_sequence     JSONB DEFAULT '[]',     -- 200 dernières interactions pour DIN
  long_term_sequence    JSONB DEFAULT '[]',     -- 30j historique compressé pour SIM
  rfm_recency           INT DEFAULT 999,        -- jours depuis dernier achat
  rfm_frequency         INT DEFAULT 0,          -- achats 90j
  rfm_monetary          DECIMAL(10,2) DEFAULT 0,-- GMV 90j
  persona               VARCHAR(30) DEFAULT 'unknown',
  device_os             VARCHAR(20),
  country               VARCHAR(2),
  city                  VARCHAR(100),
  followed_vendors      UUID[] DEFAULT '{}',
  updated_at            TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_ml_user_country ON ml_user_features(country);
CREATE INDEX idx_ml_user_updated ON ml_user_features(updated_at DESC);

-- ============================================================================
-- 3. ml_interactions
-- Source de vérité training — toutes interactions annotées avec poids
-- Volume: ~50M lignes/mois — partitionné mensuellement
-- ============================================================================

CREATE TABLE ml_interactions (
  id                BIGSERIAL PRIMARY KEY,
  user_id           UUID NOT NULL,
  item_id           UUID NOT NULL,
  item_type         VARCHAR(20) NOT NULL,   -- product|video|carousel|live
  action            VARCHAR(30) NOT NULL,
  -- view|like|comment|share|add_to_cart|purchase|wishlist|skip|dwell|
  -- scroll_past|live_join|buy_now_click|live_buy|zoom|pause|watch_pct
  action_weight     FLOAT NOT NULL,         -- cf. TASK_WEIGHTS dans streaming_trainer
  dwell_time_ms     INT DEFAULT 0,
  watch_pct         FLOAT DEFAULT 0,
  source            VARCHAR(30),            -- feed|marketplace|live|search|ads
  position_in_feed  INT,
  session_id        UUID,
  device_os         VARCHAR(20),
  country           VARCHAR(2),
  timestamp         TIMESTAMP DEFAULT NOW()
) PARTITION BY RANGE (timestamp);

CREATE TABLE ml_interactions_2026_01 PARTITION OF ml_interactions
  FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE ml_interactions_2026_02 PARTITION OF ml_interactions
  FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
CREATE TABLE ml_interactions_2026_03 PARTITION OF ml_interactions
  FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');
-- Ajouter une partition par mois via cron ou Alembic

CREATE INDEX idx_ml_inter_user ON ml_interactions(user_id);
CREATE INDEX idx_ml_inter_item ON ml_interactions(item_id);
CREATE INDEX idx_ml_inter_ts ON ml_interactions(timestamp DESC);
CREATE INDEX idx_ml_inter_action ON ml_interactions(action);

-- ============================================================================
-- 4. ml_training_samples
-- Samples pré-construits pour batch training (batch nocturne)
-- ============================================================================

CREATE TABLE ml_training_samples (
  id                 UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_features_json JSONB NOT NULL,   -- sérialisé user_to_features()
  item_features_json JSONB NOT NULL,   -- sérialisé product_to_features()
  label              FLOAT NOT NULL,   -- 0.0 ou 1.0 (ou continu pour régression)
  task               VARCHAR(30),      -- buy_now|purchase|add_to_cart|save|watch_time
  weight             FLOAT DEFAULT 1.0,-- sample importance weighting
  split              VARCHAR(10) DEFAULT 'train',  -- train|val|test
  generated_at       TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_ml_samples_task ON ml_training_samples(task, split);
CREATE INDEX idx_ml_samples_generated ON ml_training_samples(generated_at DESC);

-- ============================================================================
-- 5. ml_model_registry
-- Registre des modèles entraînés — lié aux artifacts S3
-- ============================================================================

CREATE TABLE ml_model_registry (
  id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  model_name        VARCHAR(100) NOT NULL,
  -- two_tower|mtl_ple|din|dien|bst|deepfm|sim|geo_classifier
  version           VARCHAR(20) NOT NULL,    -- semver: 1.0.0
  status            VARCHAR(20) DEFAULT 'staging',
  -- training|staging|production|retired
  s3_path           TEXT NOT NULL,           -- s3://shopfeed-ml-models/...
  training_samples  INT,
  metrics           JSONB DEFAULT '{}',
  -- {auc_roc, recall_at_10, ndcg_at_10, precision_at_5, watch_time_mse}
  trained_at        TIMESTAMP,
  deployed_at       TIMESTAMP,
  retrained_at      TIMESTAMP,
  notes             TEXT,
  UNIQUE (model_name, version)
);
CREATE INDEX idx_model_reg_name ON ml_model_registry(model_name, status);
CREATE INDEX idx_model_reg_prod ON ml_model_registry(model_name)
  WHERE status = 'production';

-- ============================================================================
-- 6. ml_ab_tests
-- A/B tests de modèles — tracking du trafic et des métriques
-- ============================================================================

CREATE TABLE ml_ab_tests (
  id                 UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  test_name          VARCHAR(100) NOT NULL UNIQUE,
  description        TEXT,
  variant_a_model    VARCHAR(100),    -- model_name version actuelle
  variant_b_model    VARCHAR(100),    -- model_name challenger
  traffic_split_pct  FLOAT DEFAULT 50.0,  -- % trafic vers variant B
  status             VARCHAR(20) DEFAULT 'running', -- running|paused|concluded
  start_date         DATE,
  end_date           DATE,
  metrics_json       JSONB DEFAULT '{}',  -- résultats running par variante
  winner             VARCHAR(2),          -- A|B|NULL si en cours
  statistical_sig    BOOLEAN DEFAULT false, -- significant à p < 0.05
  created_at         TIMESTAMP DEFAULT NOW()
);

-- ============================================================================
-- 7. ml_content_pools
-- Cache des pools de trafic L1-L6 par catégorie
-- TTL 15min — recalculé si les métriques des items évoluent
-- ============================================================================

CREATE TABLE ml_content_pools (
  pool_level        VARCHAR(5) NOT NULL,   -- L1|L2|L3|L4|L5|L6
  category_id       INT NOT NULL,
  item_ids_json     JSONB NOT NULL,        -- liste ordonnée item_ids
  impression_min    INT,                   -- seuil min du pool
  impression_max    INT,                   -- seuil max du pool
  last_computed     TIMESTAMP DEFAULT NOW(),
  ttl_seconds       INT DEFAULT 900,       -- 15min
  PRIMARY KEY (pool_level, category_id)
);

-- ============================================================================
-- 8. ml_cold_start_queue
-- Queue des nouveaux items en phase cold-start (3 premiers contenus)
-- ============================================================================

CREATE TABLE ml_cold_start_queue (
  item_id         UUID NOT NULL,
  item_type       VARCHAR(20) NOT NULL,  -- product|video|vendor
  vendor_id       UUID,
  strategy        VARCHAR(50),  -- category_match|lookalike|content_pure
  initial_pool    VARCHAR(5) DEFAULT 'L1',
  impressions_guaranteed INT DEFAULT 500,  -- impressions garanties
  impressions_used INT DEFAULT 0,
  boosted_until   TIMESTAMP,
  created_at      TIMESTAMP DEFAULT NOW(),
  PRIMARY KEY (item_id)
);
CREATE INDEX idx_cold_start_vendor ON ml_cold_start_queue(vendor_id);
CREATE INDEX idx_cold_start_until ON ml_cold_start_queue(boosted_until);

-- ============================================================================
-- 9. ml_live_features
-- Features ML en temps réel pour les lives actifs
-- Mis à jour par le monolith streaming trainer toutes les 30s
-- ============================================================================

CREATE TABLE ml_live_features (
  live_id               UUID PRIMARY KEY,
  concurrent_viewers    INT DEFAULT 0,
  gmv_per_minute        FLOAT DEFAULT 0,
  buy_now_count_5min    INT DEFAULT 0,
  live_score            FLOAT DEFAULT 0,
  -- LiveScore = viewers×1 + peak×0.5 + purchase_rate×100 + gifts×2 + comments×10
  peak_score            FLOAT DEFAULT 0,
  explorer_score        FLOAT DEFAULT 0,
  product_featured_ids  UUID[] DEFAULT '{}',
  vendor_embedding      vector(64),
  updated_at            TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_live_features_score ON ml_live_features(live_score DESC);
CREATE INDEX idx_live_features_updated ON ml_live_features(updated_at DESC);

-- ============================================================================
-- 10. feature_store_daily
-- Snapshots journaliers des features pour analyse rétrospective et training
-- ============================================================================

CREATE TABLE feature_store_daily (
  date              DATE NOT NULL,
  item_id           UUID NOT NULL,
  item_type         VARCHAR(20) NOT NULL,
  cvr_daily         FLOAT DEFAULT 0,
  ctr_daily         FLOAT DEFAULT 0,
  gmv_daily         DECIMAL(10,2) DEFAULT 0,
  views_daily       INT DEFAULT 0,
  pool_level        VARCHAR(5),
  seller_weight     FLOAT DEFAULT 1.0,
  PRIMARY KEY (date, item_id)
) PARTITION BY RANGE (date);

CREATE TABLE feature_store_daily_2026_q1 PARTITION OF feature_store_daily
  FOR VALUES FROM ('2026-01-01') TO ('2026-04-01');
CREATE TABLE feature_store_daily_2026_q2 PARTITION OF feature_store_daily
  FOR VALUES FROM ('2026-04-01') TO ('2026-07-01');

-- ============================================================================
-- 11. ml_embedding_index
-- Métadonnées des index FAISS sauvegardés sur S3
-- ============================================================================

CREATE TABLE ml_embedding_index (
  index_name      VARCHAR(50) PRIMARY KEY,
  -- products_clip|users_embedding|feed_clip|...
  index_path_s3   TEXT NOT NULL,          -- s3://shopfeed-ml-models/faiss_indexes/...
  vector_count    INT NOT NULL DEFAULT 0,
  dim             INT NOT NULL,
  built_at        TIMESTAMP DEFAULT NOW(),
  status          VARCHAR(20) DEFAULT 'active'  -- building|active|outdated
);

-- ============================================================================
-- 12. ml_geo_zones
-- Zones géographiques pour geosort (backup PostGIS dans geosort_db)
-- ============================================================================

CREATE TABLE ml_geo_zones (
  id                  SERIAL PRIMARY KEY,
  country_code        VARCHAR(2) NOT NULL,
  country_name        VARCHAR(100),
  region              VARCHAR(100),
  city                VARCHAR(100),
  commune             VARCHAR(100),
  aliases             TEXT[] DEFAULT '{}',
  center_lat          FLOAT NOT NULL,
  center_lon          FLOAT NOT NULL,
  geo_polygon         GEOMETRY(POLYGON, 4326),
  adjacent_commune_ids INT[] DEFAULT '{}',
  country_phone_code  VARCHAR(10)
);
CREATE INDEX idx_ml_geo_country ON ml_geo_zones(country_code);
CREATE INDEX idx_ml_geo_city ON ml_geo_zones(city);
CREATE INDEX idx_ml_geo_polygon ON ml_geo_zones USING GIST(geo_polygon);
