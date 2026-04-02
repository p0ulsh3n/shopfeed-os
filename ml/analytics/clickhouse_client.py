"""
ClickHouse Analytics Client (archi-2026 §7.4)
================================================
Column-oriented OLAP database for real-time analytics on billions of rows.

Use cases in shopfeed-os:
    - Dashboard admin (real-time metrics)
    - A/B test result aggregation
    - ML drift monitoring (feature distribution queries)
    - Trending content scoring
    - Fraud pattern analysis at scale

ClickHouse handles queries like "top 100 videos by region in the last
6 hours" on billions of rows in <2 seconds.

Requires:
    pip install clickhouse-connect>=0.7
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import clickhouse_connect
    HAS_CLICKHOUSE = True
except ImportError:
    HAS_CLICKHOUSE = False
    logger.warning("clickhouse-connect not installed. pip install clickhouse-connect>=0.7")


# ── Configuration ──────────────────────────────────────────────

CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST", "clickhouse")
CLICKHOUSE_PORT = int(os.getenv("CLICKHOUSE_PORT", "8123"))
CLICKHOUSE_DB = os.getenv("CLICKHOUSE_DB", "shopfeed")
CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "default")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "")


# ── Table Schemas ──────────────────────────────────────────────

TABLE_SCHEMAS = {
    "events": """
        CREATE TABLE IF NOT EXISTS events (
            event_id      UUID DEFAULT generateUUIDv4(),
            event_type    LowCardinality(String),
            user_id       String,
            item_id       String,
            creator_id    String,
            category_id   UInt32,
            country       LowCardinality(String),
            device_type   LowCardinality(String),
            session_id    String,
            watch_duration_pct  Float32 DEFAULT 0,
            metadata      String DEFAULT '{}',
            ts            DateTime64(3)
        )
        ENGINE = MergeTree()
        PARTITION BY toYYYYMM(ts)
        ORDER BY (event_type, ts, user_id)
        TTL ts + INTERVAL 2 YEAR
        SETTINGS index_granularity = 8192
    """,
    "ml_predictions": """
        CREATE TABLE IF NOT EXISTS ml_predictions (
            request_id    String,
            user_id       String,
            model_name    LowCardinality(String),
            model_version String,
            scores        Array(Float32),
            item_ids      Array(String),
            latency_ms    Float32,
            experiment    LowCardinality(String) DEFAULT '',
            ts            DateTime64(3)
        )
        ENGINE = MergeTree()
        PARTITION BY toYYYYMM(ts)
        ORDER BY (model_name, ts)
        TTL ts + INTERVAL 1 YEAR
    """,
    "fraud_events": """
        CREATE TABLE IF NOT EXISTS fraud_events (
            user_id       String,
            fraud_score   Float32,
            action_taken  LowCardinality(String),
            triggered_rules Array(String),
            features      String,
            ts            DateTime64(3)
        )
        ENGINE = MergeTree()
        PARTITION BY toYYYYMM(ts)
        ORDER BY (ts, user_id)
        TTL ts + INTERVAL 2 YEAR
    """,
}


class ClickHouseAnalytics:
    """Production analytics client for ClickHouse.

    Used for:
        1. Ingesting all events from Flink for long-term analytics
        2. ML monitoring queries (drift detection, A/B test results)
        3. Admin dashboard real-time metrics
        4. Fraud analysis at scale
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ):
        self.host = host or CLICKHOUSE_HOST
        self.port = port or CLICKHOUSE_PORT
        self._client = None

    @property
    def client(self) -> Optional[Any]:
        """Get or create ClickHouse client."""
        if self._client is None and HAS_CLICKHOUSE:
            try:
                self._client = clickhouse_connect.get_client(
                    host=self.host,
                    port=self.port,
                    database=CLICKHOUSE_DB,
                    username=CLICKHOUSE_USER,
                    password=CLICKHOUSE_PASSWORD,
                )
                logger.info("ClickHouse connected: %s:%s/%s", self.host, self.port, CLICKHOUSE_DB)
            except Exception as e:
                logger.error("ClickHouse connection failed: %s", e)
        return self._client

    def init_tables(self) -> bool:
        """Create all analytics tables if they don't exist."""
        if not self.client:
            return False

        try:
            for name, ddl in TABLE_SCHEMAS.items():
                self.client.command(ddl)
                logger.info("ClickHouse table ensured: %s", name)
            return True
        except Exception as e:
            logger.error("Table init failed: %s", e)
            return False

    def insert_events(self, events: list[dict]) -> int:
        """Batch insert events into ClickHouse.

        Called by Flink to persist all interaction events.
        """
        if not self.client or not events:
            return 0

        try:
            columns = list(events[0].keys())
            data = [[e.get(c) for c in columns] for e in events]
            self.client.insert("events", data, column_names=columns)
            return len(events)
        except Exception as e:
            logger.error("Event insert failed: %s", e)
            return 0

    def insert_predictions(self, predictions: list[dict]) -> int:
        """Log ML prediction results for monitoring."""
        if not self.client or not predictions:
            return 0

        try:
            columns = list(predictions[0].keys())
            data = [[p.get(c) for c in columns] for p in predictions]
            self.client.insert("ml_predictions", data, column_names=columns)
            return len(predictions)
        except Exception as e:
            logger.error("Prediction insert failed: %s", e)
            return 0

    def query(self, sql: str, params: Optional[dict] = None) -> Optional[Any]:
        """Execute an analytical query.

        Returns result as a list of dicts.
        """
        if not self.client:
            return None

        try:
            result = self.client.query(sql, parameters=params)
            return result.result_rows
        except Exception as e:
            logger.error("Query failed: %s", e)
            return None

    # ── Pre-built Analytics Queries ────────────────────────────

    def get_trending_items(self, hours: int = 6, country: str = "", limit: int = 100) -> list:
        """Top trending items by engagement in the last N hours."""
        country_filter = f"AND country = '{country}'" if country else ""
        sql = f"""
        SELECT
            item_id,
            countIf(event_type = 'like') AS likes,
            countIf(event_type = 'view') AS views,
            countIf(event_type = 'share') AS shares,
            countIf(event_type = 'buy_now') AS purchases,
            likes * 3 + shares * 5 + purchases * 10 AS engagement_score
        FROM events
        WHERE ts >= now() - INTERVAL {hours} HOUR
        {country_filter}
        GROUP BY item_id
        ORDER BY engagement_score DESC
        LIMIT {limit}
        """
        return self.query(sql) or []

    def get_model_latency_stats(self, model_name: str, hours: int = 24) -> Optional[dict]:
        """Get P50/P95/P99 latency for a model."""
        sql = f"""
        SELECT
            model_name,
            count() AS request_count,
            quantile(0.50)(latency_ms) AS p50_ms,
            quantile(0.95)(latency_ms) AS p95_ms,
            quantile(0.99)(latency_ms) AS p99_ms,
            avg(latency_ms) AS avg_ms
        FROM ml_predictions
        WHERE model_name = %(model)s
          AND ts >= now() - INTERVAL {hours} HOUR
        GROUP BY model_name
        """
        result = self.query(sql, {"model": model_name})
        if result and len(result) > 0:
            row = result[0]
            return {
                "model_name": row[0],
                "request_count": row[1],
                "p50_ms": float(row[2]),
                "p95_ms": float(row[3]),
                "p99_ms": float(row[4]),
                "avg_ms": float(row[5]),
            }
        return None

    def get_ab_test_results(self, experiment: str, hours: int = 24) -> list:
        """Compare engagement between experiment groups."""
        sql = f"""
        SELECT
            experiment,
            model_version,
            count() AS request_count,
            avg(latency_ms) AS avg_latency
        FROM ml_predictions
        WHERE experiment = %(exp)s
          AND ts >= now() - INTERVAL {hours} HOUR
        GROUP BY experiment, model_version
        """
        return self.query(sql, {"exp": experiment}) or []
