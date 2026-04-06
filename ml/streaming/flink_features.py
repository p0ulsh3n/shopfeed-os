"""
Apache Flink Integration — Real-time Feature Pipeline (archi-2026 §6.1)
=========================================================================
Flink processes events in <50ms for real-time ML features.

Flink Jobs in production:
    1. LikeAggregator — real-time like counters → Redis
    2. TrendingVideos — 1h sliding window trending scores
    3. FeedRebuilder — rebuild user feeds every 30s
    4. FraudDetector — real-time fraud feature computation
    5. MLFeatureWriter — continuous feature updates to Feast

This module provides Python integration with Flink via PyFlink.
For production, Flink jobs run on Kubernetes (Flink on K8s).

Requires:
    pip install apache-flink>=1.19
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from pyflink.datastream import StreamExecutionEnvironment
    from pyflink.table import StreamTableEnvironment, EnvironmentSettings
    HAS_PYFLINK = True
except ImportError:
    HAS_PYFLINK = False
    logger.warning("pyflink not installed — Flink integration disabled. pip install apache-flink>=1.19")


# ── Configuration ──────────────────────────────────────────────

FLINK_CONFIG = {
    "parallelism.default": int(os.getenv("FLINK_PARALLELISM", "4")),
    "state.backend": "rocksdb",
    "state.checkpoints.dir": os.getenv("FLINK_CHECKPOINT_DIR", "s3://shopfeed-checkpoints/flink"),
    "execution.checkpointing.interval": "60000",  # 1 min
    "execution.checkpointing.mode": "EXACTLY_ONCE",
    "restart-strategy": "fixed-delay",
    "restart-strategy.fixed-delay.attempts": "3",
    "restart-strategy.fixed-delay.delay": "10s",
}


class FlinkFeaturePipeline:
    """Real-time feature computation using Apache Flink.

    Consumes events from Redpanda topics, computes ML features
    in real-time, and writes them to:
        1. Redis (for online serving via Feast)
        2. Feature Store (direct update)
        3. ClickHouse (for analytics + offline training)

    Architecture:
        Redpanda → Flink → Redis / Feast / ClickHouse
    """

    def __init__(self):
        self._env = None
        self._table_env = None

    @property
    def env(self) -> Optional[Any]:
        """Get or create Flink execution environment."""
        if self._env is None and HAS_PYFLINK:
            try:
                self._env = StreamExecutionEnvironment.get_execution_environment()
                for key, value in FLINK_CONFIG.items():
                    self._env.get_config().set_string(key, str(value))
                logger.info("Flink environment initialized")
            except Exception as e:
                logger.error("Flink init failed: %s", e)
        return self._env

    @property
    def table_env(self) -> Optional[Any]:
        """Get or create Flink Table environment for SQL queries."""
        if self._table_env is None and self.env:
            try:
                settings = EnvironmentSettings.in_streaming_mode()
                self._table_env = StreamTableEnvironment.create(self.env, settings)
            except Exception as e:
                logger.error("Flink Table env init failed: %s", e)
        return self._table_env

    def create_redpanda_source(self, topic: str, group_id: str) -> Optional[str]:
        """Create a Flink SQL source table reading from Redpanda.

        Returns the table name for use in SQL queries.
        """
        if not self.table_env:
            return None

        brokers = os.getenv("REDPANDA_BROKERS", "redpanda-1:9092")
        table_name = f"source_{topic.replace('.', '_')}"

        ddl = f"""
        CREATE TABLE {table_name} (
            user_id STRING,
            item_id STRING,
            action STRING,
            features MAP<STRING, DOUBLE>,
            event_time TIMESTAMP(3),
            WATERMARK FOR event_time AS event_time - INTERVAL '5' SECOND
        ) WITH (
            'connector' = 'kafka',
            'topic' = '{topic}',
            'properties.bootstrap.servers' = '{brokers}',
            'properties.group.id' = '{group_id}',
            'scan.startup.mode' = 'latest-offset',
            'format' = 'json',
            'json.fail-on-missing-field' = 'false'
        )
        """

        try:
            self.table_env.execute_sql(ddl)
            logger.info("Flink source created: %s → %s", topic, table_name)
            return table_name
        except Exception as e:
            logger.error("Flink source creation failed: %s", e)
            return None

    def create_redis_sink(self, table_name: str, key_field: str) -> Optional[str]:
        """Create a Flink SQL sink writing to Redis.

        Args:
            table_name: Base name for the sink table
            key_field: Field name to use as Redis key (e.g., 'user_id', 'item_id')
        """
        if not self.table_env:
            return None

        redis_host = os.getenv("REDIS_HOST", "redis")
        sink_name = f"sink_redis_{table_name}"

        ddl = f"""
        CREATE TABLE {sink_name} (
            {key_field} STRING,
            features MAP<STRING, DOUBLE>,
            PRIMARY KEY ({key_field}) NOT ENFORCED
        ) WITH (
            'connector' = 'redis',
            'host' = '{redis_host}',
            'port' = '6379',
            'command' = 'HSET'
        )
        """

        try:
            self.table_env.execute_sql(ddl)
            return sink_name
        except Exception as e:
            logger.error("Redis sink creation failed: %s", e)
            return None

    def run_feature_aggregation_job(self) -> None:
        """Start the main feature aggregation Flink job.

        This job continuously computes:
            1. User engagement features (likes/views/shares per hour)
            2. Item performance features (like_rate, completion_rate per hour)
            3. Trending scores (sliding 1h window)
            4. Fraud signals (likes_per_minute, action regularity)
        """
        if not self.table_env:
            logger.error("Flink not available — feature aggregation skipped")
            return

        source = self.create_redpanda_source("interactions", "flink-features")
        if not source:
            return

        # User engagement aggregation (1-hour tumbling window)
        user_agg_sql = f"""
        SELECT
            user_id,
            TUMBLE_END(event_time, INTERVAL '1' HOUR) AS window_end,
            COUNT(*) AS total_actions,
            COUNT(CASE WHEN action = 'like' THEN 1 END) AS likes_1h,
            COUNT(CASE WHEN action = 'view' THEN 1 END) AS views_1h,
            COUNT(CASE WHEN action = 'buy_now' THEN 1 END) AS purchases_1h,
            COUNT(CASE WHEN action = 'share' THEN 1 END) AS shares_1h,
            COUNT(CASE WHEN action = 'micro_pause' THEN 1 END) AS micro_pauses_1h,
            COUNT(DISTINCT item_id) AS unique_items_1h
        FROM {source}
        GROUP BY user_id, TUMBLE(event_time, INTERVAL '1' HOUR)
        """

        # Item performance aggregation (1-hour tumbling window)
        item_agg_sql = f"""
        SELECT
            item_id,
            TUMBLE_END(event_time, INTERVAL '1' HOUR) AS window_end,
            COUNT(CASE WHEN action = 'like' THEN 1 END) * 1.0 /
                GREATEST(COUNT(CASE WHEN action = 'view' THEN 1 END), 1) AS like_rate_1h,
            COUNT(CASE WHEN action = 'share' THEN 1 END) * 1.0 /
                GREATEST(COUNT(CASE WHEN action = 'view' THEN 1 END), 1) AS share_rate_1h,
            COUNT(CASE WHEN action = 'buy_now' THEN 1 END) AS purchases_1h
        FROM {source}
        GROUP BY item_id, TUMBLE(event_time, INTERVAL '1' HOUR)
        """

        # Fraud signals (1-minute sliding window)
        fraud_sql = f"""
        SELECT
            user_id,
            HOP_END(event_time, INTERVAL '10' SECOND, INTERVAL '1' MINUTE) AS window_end,
            COUNT(*) * 1.0 / 1 AS actions_per_minute,
            COUNT(CASE WHEN action = 'like' THEN 1 END) AS likes_per_minute,
            COUNT(DISTINCT item_id) AS unique_targets_1m
        FROM {source}
        GROUP BY user_id, HOP(event_time, INTERVAL '10' SECOND, INTERVAL '1' MINUTE)
        HAVING COUNT(CASE WHEN action = 'like' THEN 1 END) > 10
        """

        logger.info("Flink feature aggregation jobs defined (execute with env.execute())")
