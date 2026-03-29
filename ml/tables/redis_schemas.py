"""
Redis Schemas ML — Documentation complète des clés Redis pour shopfeed-os.
Namespace: Redis DB 1 (ML uniquement — séparé de DB 0 shop-backend)

TTL conventions:
  session_intent: 1800s  (30min = durée session max)
  user_features:  3600s  (1h — invalide après chaque interaction)
  item_features:  86400s (24h — invalide si price/stock/pool change)
  trending:       1800s  (30min — vite obsolète)
  live_features:  TTL dynamique (jusqu'à la fin du live)
"""

from __future__ import annotations
from dataclasses import dataclass

# ── Redis DB 1 — ML Namespaces ────────────────────────────────────────────────

ML_REDIS_DB = 1

@dataclass(frozen=True)
class RedisKey:
    """Générateur de clés Redis ML typées."""
    pattern: str
    ttl_seconds: int
    data_type: str  # string|hash|zset|list|json
    description: str

    def format(self, **kwargs) -> str:
        return self.pattern.format(**kwargs)


# ── Feature Store Keys ────────────────────────────────────────────────────────

USER_FEATURES = RedisKey(
    pattern="ml_user_features:{user_id}",
    ttl_seconds=3600,
    data_type="string",
    description=(
        "MessagePack bytes (float32 tensor) — user embedding 256d + metadata. "
        "Invalidé à chaque interaction via Kafka consumer."
    ),
)

ITEM_FEATURES = RedisKey(
    pattern="ml_item_features:{item_id}",
    ttl_seconds=86400,
    data_type="string",
    description=(
        "MessagePack bytes (float32 tensor) — item embedding 512d + metadata. "
        "Invalidé si price|stock|pool_level change (shopfeed.product.events)."
    ),
)

SESSION_INTENT = RedisKey(
    pattern="ml_session_intent:{session_id}",
    ttl_seconds=1800,
    data_type="string",
    description=(
        "MessagePack bytes float32 128d — vecteur d'intent session BST output. "
        "Mis à jour par POST /v1/session/intent-vector."
    ),
)

VENDOR_EMBEDDING = RedisKey(
    pattern="ml_vendor_emb:{vendor_id}",
    ttl_seconds=86400,
    data_type="string",
    description="MessagePack bytes float32 64d — vendor embedding entraînable.",
)

# ── Trending / Pools ──────────────────────────────────────────────────────────

TRENDING_ITEMS = RedisKey(
    pattern="ml_trending_items:{category_id}",
    ttl_seconds=1800,
    data_type="zset",
    description=(
        "Sorted set item_id→composite_score. "
        "Top items trendants dans la catégorie. TTL 30min."
    ),
)

POOL_ITEMS = RedisKey(
    pattern="ml_pool:{pool_level}:{category_id}",
    ttl_seconds=900,
    data_type="string",
    description="JSON list item_ids dans ce pool (L1-L6) par catégorie. TTL 15min.",
)

# ── Session ASR Index ─────────────────────────────────────────────────────────

ASR_INDEX = RedisKey(
    pattern="ml_asr_index:{session_id}",
    ttl_seconds=1800,
    data_type="hash",
    description=(
        "HSET {entity: score} — entités extraites par Whisper ASR des vidéos regardées. "
        "Utilisé pour Contextual Search S1 (Section 19 blueprint)."
    ),
)

COMMENT_ENTITIES = RedisKey(
    pattern="ml_comment_entities:{content_id}",
    ttl_seconds=300,
    data_type="zset",
    description="Sorted set entity→count extrait des commentaires (NLP). TTL 5min.",
)

# ── Live ML Features ──────────────────────────────────────────────────────────

LIVE_ML_FEATURES = RedisKey(
    pattern="ml_live_features:{live_id}",
    ttl_seconds=0,  # TTL dynamique jusqu'à live_ended event
    data_type="hash",
    description=(
        "HSET {viewers, gmv_per_min, buy_now_5min, live_score, explorer_score}. "
        "Mis à jour par monolith streaming trainer toutes les 30s."
    ),
)

EXPLORER_SCORE = RedisKey(
    pattern="live:explorer_score:{live_id}",
    ttl_seconds=60,
    data_type="string",
    description=(
        "Float ExplorerScore pour le tri /api/v1/live/explorer. "
        "Recalculé toutes les 60s depuis les métriques Redis live."
    ),
)

# ── Model Cache ───────────────────────────────────────────────────────────────

FEED_SCORES_CACHE = RedisKey(
    pattern="ml_feed_scores:{user_id}:{item_id}",
    ttl_seconds=900,
    data_type="string",
    description="JSON {score, mtl_scores} — cache de scoring 15min pour un user×item.",
)

LAST_TRAINING_TS = RedisKey(
    pattern="ml_last_training_ts",
    ttl_seconds=0,
    data_type="string",
    description="ISO timestamp du dernier batch training ou update monolith.",
)

# ── Fonctions helpers ─────────────────────────────────────────────────────────

def user_features_key(user_id: str) -> str:
    return f"ml_user_features:{user_id}"

def item_features_key(item_id: str) -> str:
    return f"ml_item_features:{item_id}"

def session_intent_key(session_id: str) -> str:
    return f"ml_session_intent:{session_id}"

def vendor_emb_key(vendor_id: str) -> str:
    return f"ml_vendor_emb:{vendor_id}"

def trending_key(category_id: int) -> str:
    return f"ml_trending_items:{category_id}"

def pool_key(pool_level: str, category_id: int) -> str:
    return f"ml_pool:{pool_level}:{category_id}"

def asr_index_key(session_id: str) -> str:
    return f"ml_asr_index:{session_id}"

def live_ml_key(live_id: str) -> str:
    return f"ml_live_features:{live_id}"

def feed_score_key(user_id: str, item_id: str) -> str:
    return f"ml_feed_scores:{user_id}:{item_id}"


# ── Summary table ─────────────────────────────────────────────────────────────

ALL_KEYS: list[RedisKey] = [
    USER_FEATURES,
    ITEM_FEATURES,
    SESSION_INTENT,
    VENDOR_EMBEDDING,
    TRENDING_ITEMS,
    POOL_ITEMS,
    ASR_INDEX,
    COMMENT_ENTITIES,
    LIVE_ML_FEATURES,
    EXPLORER_SCORE,
    FEED_SCORES_CACHE,
    LAST_TRAINING_TS,
]
