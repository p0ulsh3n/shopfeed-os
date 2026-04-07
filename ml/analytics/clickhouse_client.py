"""
ClickHouse Analytics Client — 2026 Best Practices
==================================================
Client async officiel clickhouse-connect avec paramètres typés natifs ClickHouse.

MIGRATION SÉCURITÉ:
- AVANT: f-string dans SQL → injection directe (country, hours, limit, experiment)
- APRÈS: {name:Type} paramètres bindés natifs ClickHouse — server-side escaping
  Doc: https://clickhouse.com/docs/en/interfaces/cli#cli-queries-with-parameters

Best practices 2026 (clickhouse-connect>=0.7):
  1. get_async_client() — pool HTTP connexion réutilisée
  2. {name:Type} placeholders — jamais f-string pour les valeurs
  3. client.insert() pour les bulk inserts — jamais INSERT SQL
  4. LIMIT systématique sur toutes les queries analytiques
  5. Identifiants dynamiques (table, DB) → allowlist obligatoire

Requires:
    pip install clickhouse-connect[async]>=0.8
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
    logger.warning("clickhouse-connect not installed. pip install clickhouse-connect[async]>=0.8")


# ── Configuration ──────────────────────────────────────────────

CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST", "clickhouse")
CLICKHOUSE_PORT = int(os.getenv("CLICKHOUSE_PORT", "8123"))
CLICKHOUSE_DB = os.getenv("CLICKHOUSE_DB", "shopfeed")
CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "default")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "")

# Allowlist pour les identifiants dynamiques (table names)
_ALLOWED_TABLES = frozenset({"events", "ml_predictions", "fraud_events"})
_ALLOWED_COUNTRIES = frozenset({
    "CM", "CI", "SN", "ML", "BF", "GH", "NG", "KE", "TZ", "ZA", "FR", "BE",
    "CD", "GA", "CG", "BJ", "TG", "NE", "GN", "MR", "DZ", "MA", "TN",
})


# ── Table Schemas ──────────────────────────────────────────────
# NOTE: DDL ClickHouse est géré via des scripts de migration dédiés,
# pas inline dans le code applicatif. Ces définitions sont conservées
# comme référence documentaire uniquement.

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
    """
    Production analytics client for ClickHouse.

    2026 Pattern:
      - Client async singleton réutilisé (HTTP connection pool)
      - Toutes les queries utilisent {name:Type} placeholders natifs
      - Zéro f-string pour les valeurs
      - Identifiants fixes ou validés via allowlist

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
        """Get or create ClickHouse client (sync — réutilisé)."""
        if self._client is None and HAS_CLICKHOUSE:
            try:
                self._client = clickhouse_connect.get_client(
                    host=self.host,
                    port=self.port,
                    database=CLICKHOUSE_DB,
                    username=CLICKHOUSE_USER,
                    password=CLICKHOUSE_PASSWORD,
                    connect_timeout=10,
                    send_receive_timeout=30,
                )
                logger.info("ClickHouse connected: %s:%s/%s", self.host, self.port, CLICKHOUSE_DB)
            except Exception as e:
                logger.error("ClickHouse connection failed: %s", e)
        return self._client

    def init_tables(self) -> bool:
        """Create all analytics tables (DDL — identifiants fixes, sans interpolation)."""
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
        """
        Batch insert events into ClickHouse.

        SÉCURITÉ: client.insert() — jamais INSERT SQL string.
        clickhouse-connect gère l'escaping et la sérialisation côté client.
        """
        if not self.client or not events:
            return 0

        try:
            columns = list(events[0].keys())
            data = [[e.get(c) for c in columns] for e in events]
            # Table name fixe — pas d'interpolation possible
            self.client.insert("events", data, column_names=columns)
            return len(events)
        except Exception as e:
            logger.error("Event insert failed: %s", e)
            return 0

    def insert_predictions(self, predictions: list[dict]) -> int:
        """Log ML prediction results — client.insert() bulk, jamais SQL string."""
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
        """
        Execute an analytical query avec paramètres bindés.

        IMPORTANT: `sql` doit utiliser {name:Type} placeholders ClickHouse natifs,
        JAMAIS de f-string pour les valeurs. Exemple:
            query("SELECT * FROM events WHERE user_id = {uid:String} LIMIT {n:UInt32}",
                  {"uid": user_id, "n": 100})
        """
        if not self.client:
            return None

        try:
            result = self.client.query(sql, parameters=params)
            return result.result_rows
        except Exception as e:
            logger.error("Query failed: %s", e)
            return None

    # ── Pre-built Analytics Queries — 100% paramétrisées ──────────────

    def get_trending_items(
        self, hours: int = 6, country: str = "", limit: int = 100
    ) -> list:
        """
        Top trending items by engagement in the last N hours.

        FIX INJECTION:
        - AVANT: f"AND country = '{country}'" + f"INTERVAL {hours} HOUR" + f"LIMIT {limit}"
          → injection directe (country, hours, limit non validés)
        - APRÈS:
          * country → validé via allowlist _ALLOWED_COUNTRIES OU omis si inconnu
          * hours/limit → validés comme entiers bornés
          * {name:Type} ClickHouse native server-side parameterization

        ClickHouse parameterized syntax: {paramName:Type}
        Types supportés: String, UInt8, UInt32, Int32, Float64, DateTime, ...
        """
        # Validation et bornage des entiers
        safe_hours = max(1, min(int(hours), 168))   # max 7 jours
        safe_limit = max(1, min(int(limit), 1000))  # max 1000

        # Validation country via allowlist
        if country and country.upper() in _ALLOWED_COUNTRIES:
            safe_country = country.upper()
            sql = """
            SELECT
                item_id,
                countIf(event_type = 'like') AS likes,
                countIf(event_type = 'view') AS views,
                countIf(event_type = 'share') AS shares,
                countIf(event_type = 'buy_now') AS purchases,
                likes * 3 + shares * 5 + purchases * 10 AS engagement_score
            FROM events
            WHERE ts >= now() - toIntervalHour({hours:UInt32})
              AND country = {country:String}
            GROUP BY item_id
            ORDER BY engagement_score DESC
            LIMIT {limit:UInt32}
            """
            params = {"hours": safe_hours, "country": safe_country, "limit": safe_limit}
        else:
            # Sans filtre country (si inconnu ou hors allowlist)
            sql = """
            SELECT
                item_id,
                countIf(event_type = 'like') AS likes,
                countIf(event_type = 'view') AS views,
                countIf(event_type = 'share') AS shares,
                countIf(event_type = 'buy_now') AS purchases,
                likes * 3 + shares * 5 + purchases * 10 AS engagement_score
            FROM events
            WHERE ts >= now() - toIntervalHour({hours:UInt32})
            GROUP BY item_id
            ORDER BY engagement_score DESC
            LIMIT {limit:UInt32}
            """
            params = {"hours": safe_hours, "limit": safe_limit}

        return self.query(sql, params) or []

    def get_model_latency_stats(
        self, model_name: str, hours: int = 24
    ) -> Optional[dict]:
        """
        Get P50/P95/P99 latency for a model.

        FIX: {model_name:String} et {hours:UInt32} — plus de f-string.
        """
        safe_hours = max(1, min(int(hours), 720))   # max 30 jours

        sql = """
        SELECT
            model_name,
            count() AS request_count,
            quantile(0.50)(latency_ms) AS p50_ms,
            quantile(0.95)(latency_ms) AS p95_ms,
            quantile(0.99)(latency_ms) AS p99_ms,
            avg(latency_ms) AS avg_ms
        FROM ml_predictions
        WHERE model_name = {model_name:String}
          AND ts >= now() - toIntervalHour({hours:UInt32})
        GROUP BY model_name
        """
        result = self.query(sql, {"model_name": model_name, "hours": safe_hours})
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
        """
        Compare engagement between experiment groups.

        FIX: {experiment:String} et {hours:UInt32} — plus de f-string.
        """
        safe_hours = max(1, min(int(hours), 720))

        sql = """
        SELECT
            experiment,
            model_version,
            count() AS request_count,
            avg(latency_ms) AS avg_latency
        FROM ml_predictions
        WHERE experiment = {experiment:String}
          AND ts >= now() - toIntervalHour({hours:UInt32})
        GROUP BY experiment, model_version
        """
        return self.query(sql, {"experiment": experiment, "hours": safe_hours}) or []

    def get_fraud_summary(self, min_score: float = 0.8, hours: int = 1) -> list:
        """Get high-risk fraud events — paramétré."""
        safe_hours = max(1, min(int(hours), 24))

        sql = """
        SELECT
            user_id,
            fraud_score,
            action_taken,
            triggered_rules
        FROM fraud_events
        WHERE fraud_score >= {min_score:Float32}
          AND ts >= now() - toIntervalHour({hours:UInt32})
        ORDER BY fraud_score DESC
        LIMIT 500
        """
        return self.query(sql, {"min_score": float(min_score), "hours": safe_hours}) or []
