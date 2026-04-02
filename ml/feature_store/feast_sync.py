"""
Feast Feature Store Synchronization (archi-2026 §9.2)
======================================================
Bridges the custom shopfeed-os feature store with Feast for
online/offline feature consistency.

Architecture:
    Flink (real-time) → Feast Online Store (Redis) → Triton Inference
    Spark (batch)     → Feast Offline Store (S3)   → PyTorch Training

This eliminates training-serving skew — the #1 cause of silent
model degradation in production recommender systems.

Requires:
    pip install feast[redis]>=0.38
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from feast import FeatureStore, Entity, FeatureView, Field, FileSource
    from feast.types import Float32, String, Int64, UnixTimestamp
    from feast.infra.online_stores.redis import RedisOnlineStoreConfig
    HAS_FEAST = True
except ImportError:
    HAS_FEAST = False
    logger.warning("feast not installed — feature store sync disabled. pip install feast[redis]>=0.38")


# ── Feast Feature Definitions ─────────────────────────────────

FEATURE_REPO_PATH = os.getenv("FEAST_REPO_PATH", "feature_repo")


class FeastSync:
    """Synchronizes shopfeed-os features with Feast for production serving.

    This does NOT replace the existing feature store — it wraps it with
    Feast for online/offline consistency guarantees.
    """

    def __init__(self, repo_path: Optional[str] = None):
        self.repo_path = repo_path or FEATURE_REPO_PATH
        self._store: Optional[Any] = None

    @property
    def store(self) -> Optional[Any]:
        """Lazy-load Feast store."""
        if self._store is None and HAS_FEAST:
            try:
                self._store = FeatureStore(repo_path=self.repo_path)
            except Exception as e:
                logger.warning("Feast store init failed: %s", e)
        return self._store

    # ── Online Feature Retrieval ───────────────────────────────

    def get_online_features(
        self,
        entity_rows: list[dict[str, Any]],
        feature_refs: list[str],
    ) -> Optional[dict[str, list]]:
        """Retrieve features from the online store (Redis) for inference.

        Args:
            entity_rows: [{"user_id": "u123"}, {"user_id": "u456"}]
            feature_refs: ["user_features:engagement_rate", "user_features:avg_watch_pct"]

        Returns:
            Dict of feature_name → list of values, aligned with entity_rows.
        """
        if not self.store:
            return None

        try:
            response = self.store.get_online_features(
                features=feature_refs,
                entity_rows=entity_rows,
            )
            return response.to_dict()
        except Exception as e:
            logger.error("Feast online feature retrieval failed: %s", e)
            return None

    # ── Offline Feature Retrieval (for training) ───────────────

    def get_training_features(
        self,
        entity_df: Any,
        feature_refs: list[str],
    ) -> Optional[Any]:
        """Retrieve historical features for training (from S3 / offline store).

        Args:
            entity_df: pandas DataFrame with entity keys + event_timestamp
            feature_refs: list of feature references

        Returns:
            pandas DataFrame with features joined to entity_df.
        """
        if not self.store:
            return None

        try:
            training_df = self.store.get_historical_features(
                entity_df=entity_df,
                features=feature_refs,
            ).to_df()
            return training_df
        except Exception as e:
            logger.error("Feast historical feature retrieval failed: %s", e)
            return None

    # ── Feature Materialization ────────────────────────────────

    def materialize_online(self, end_date: Optional[str] = None) -> bool:
        """Materialize offline features to the online store (Redis).

        This is called by the Kubeflow pipeline after Spark batch
        feature computation to make features available for inference.
        """
        if not self.store:
            return False

        try:
            from datetime import datetime
            end = datetime.fromisoformat(end_date) if end_date else datetime.utcnow()
            start = end - timedelta(days=7)

            self.store.materialize(
                start_date=start,
                end_date=end,
            )
            logger.info("Feast materialization complete: %s → %s", start, end)
            return True
        except Exception as e:
            logger.error("Feast materialization failed: %s", e)
            return False

    # ── Push to Online Store (from Flink/streaming) ────────────

    def push_features(
        self,
        push_source_name: str,
        feature_values: dict[str, Any],
    ) -> bool:
        """Push real-time features to the online store.

        Called by streaming processors (Flink/Redpanda consumers)
        to update user features in real-time.

        Args:
            push_source_name: name of the push source in Feast
            feature_values: {"user_id": "u123", "likes_1h": 14, "engagement_rate": 0.12}
        """
        if not self.store:
            return False

        try:
            import pandas as pd
            df = pd.DataFrame([feature_values])
            self.store.push(push_source_name, df)
            return True
        except Exception as e:
            logger.error("Feast push failed: %s", e)
            return False


# ── Feature Definitions (Feast Feature Repo) ──────────────────

def generate_feature_repo(output_dir: str = "feature_repo") -> None:
    """Generate the Feast feature repository configuration.

    Creates the feature definitions that map to our existing feature store.
    Run this once to initialize the Feast repo, then `feast apply`.
    """
    repo_dir = Path(output_dir)
    repo_dir.mkdir(parents=True, exist_ok=True)

    # feature_store.yaml
    store_yaml = """project: shopfeed
provider: local
registry: data/registry.db
online_store:
  type: redis
  connection_string: ${REDIS_URL}
offline_store:
  type: file
entity_key_serialization_version: 2
"""
    (repo_dir / "feature_store.yaml").write_text(store_yaml)

    # Feature definitions
    features_py = '''"""ShopFeed Feature Definitions for Feast."""

from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, PushSource
from feast.types import Float32, Float64, Int64, String

# ── Entities ──────────────────────────────────────────────────

user = Entity(
    name="user_id",
    join_keys=["user_id"],
    description="ShopFeed user",
)

item = Entity(
    name="item_id",
    join_keys=["item_id"],
    description="Product or video item",
)

# ── User Features (computed by Spark batch, hourly) ───────────

user_features_source = FileSource(
    path="data/user_features.parquet",
    timestamp_field="event_timestamp",
)

user_features = FeatureView(
    name="user_features",
    entities=[user],
    ttl=timedelta(hours=6),
    schema=[
        Field(name="engagement_rate", dtype=Float32),
        Field(name="avg_watch_pct", dtype=Float32),
        Field(name="purchase_frequency", dtype=Float32),
        Field(name="rfm_recency", dtype=Float32),
        Field(name="rfm_frequency", dtype=Float32),
        Field(name="rfm_monetary", dtype=Float32),
        Field(name="session_count_7d", dtype=Int64),
        Field(name="category_diversity", dtype=Float32),
        Field(name="price_sensitivity", dtype=Float32),
        Field(name="vulnerability_score", dtype=Float32),
    ],
    source=user_features_source,
)

# ── Real-time User Features (pushed by Flink/Redpanda) ────────

user_realtime_push = PushSource(
    name="user_realtime_features",
    batch_source=user_features_source,
)

user_realtime = FeatureView(
    name="user_realtime",
    entities=[user],
    ttl=timedelta(minutes=30),
    schema=[
        Field(name="likes_1h", dtype=Int64),
        Field(name="views_1h", dtype=Int64),
        Field(name="cart_adds_1h", dtype=Int64),
        Field(name="active_session_duration_s", dtype=Float32),
        Field(name="scroll_velocity_avg", dtype=Float32),
        Field(name="micro_pause_count", dtype=Int64),
    ],
    source=user_realtime_push,
)

# ── Item Features (computed at ingestion + hourly refresh) ─────

item_features_source = FileSource(
    path="data/item_features.parquet",
    timestamp_field="event_timestamp",
)

item_features = FeatureView(
    name="item_features",
    entities=[item],
    ttl=timedelta(hours=2),
    schema=[
        Field(name="like_rate_1h", dtype=Float32),
        Field(name="completion_rate_1h", dtype=Float32),
        Field(name="share_rate_1h", dtype=Float32),
        Field(name="viral_score", dtype=Float32),
        Field(name="freshness_score", dtype=Float32),
        Field(name="cv_quality_score", dtype=Float32),
        Field(name="price_normalized", dtype=Float32),
    ],
    source=item_features_source,
)
'''
    (repo_dir / "features.py").write_text(features_py)
    logger.info("Feast feature repo generated at: %s", repo_dir)
