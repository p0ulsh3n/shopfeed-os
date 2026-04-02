"""
ScyllaDB Interaction Store (archi-2026 §7.2)
===============================================
High-performance wide-column store for interaction persistence.

ScyllaDB replaces Cassandra:
    - C++ shard-per-core architecture (no JVM GC pauses)
    - 10× higher throughput than Cassandra
    - Used by Discord for trillions of messages

Use cases in shopfeed-os:
    - Persistent like/interaction history (after Redis writes)
    - Comment threads
    - DM messages
    - User feed history
    - Follow/follower relationships

Requires:
    pip install scyllapy>=1.3 (or cassandra-driver>=3.29 as fallback)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from cassandra.cluster import Cluster
    from cassandra.policies import DCAwareRoundRobinPolicy, TokenAwarePolicy
    from cassandra.query import BatchStatement, SimpleStatement
    HAS_CASSANDRA_DRIVER = True
except ImportError:
    HAS_CASSANDRA_DRIVER = False
    logger.warning("cassandra-driver not installed. pip install cassandra-driver>=3.29")


# ── Configuration ──────────────────────────────────────────────

SCYLLA_HOSTS = os.getenv("SCYLLA_HOSTS", "scylla-1,scylla-2,scylla-3").split(",")
SCYLLA_KEYSPACE = os.getenv("SCYLLA_KEYSPACE", "shopfeed")
SCYLLA_REPLICATION_FACTOR = int(os.getenv("SCYLLA_REPLICATION_FACTOR", "3"))


# ── Table Schemas ──────────────────────────────────────────────

KEYSPACE_DDL = f"""
CREATE KEYSPACE IF NOT EXISTS {SCYLLA_KEYSPACE}
WITH replication = {{
    'class': 'NetworkTopologyStrategy',
    'datacenter1': {SCYLLA_REPLICATION_FACTOR}
}}
"""

TABLE_SCHEMAS = {
    "interactions": """
        CREATE TABLE IF NOT EXISTS interactions (
            item_id     UUID,
            user_id     UUID,
            action      TEXT,
            created_at  TIMESTAMP,
            metadata    TEXT,
            PRIMARY KEY (item_id, created_at, user_id)
        ) WITH CLUSTERING ORDER BY (created_at DESC)
          AND compaction = {'class': 'TimeWindowCompactionStrategy',
                           'compaction_window_size': 1,
                           'compaction_window_unit': 'DAYS'}
          AND default_time_to_live = 0
    """,
    "user_feed_history": """
        CREATE TABLE IF NOT EXISTS user_feed_history (
            user_id     UUID,
            score       FLOAT,
            item_id     UUID,
            feed_type   TEXT,
            created_at  TIMESTAMP,
            PRIMARY KEY (user_id, created_at, item_id)
        ) WITH CLUSTERING ORDER BY (created_at DESC)
          AND default_time_to_live = 2592000
    """,
    "user_interactions": """
        CREATE TABLE IF NOT EXISTS user_interactions (
            user_id     UUID,
            action      TEXT,
            item_id     UUID,
            created_at  TIMESTAMP,
            PRIMARY KEY ((user_id, action), created_at, item_id)
        ) WITH CLUSTERING ORDER BY (created_at DESC)
    """,
    "follow_graph": """
        CREATE TABLE IF NOT EXISTS follow_graph (
            user_id     UUID,
            follower_id UUID,
            created_at  TIMESTAMP,
            PRIMARY KEY (user_id, follower_id)
        )
    """,
    "ml_training_events": """
        CREATE TABLE IF NOT EXISTS ml_training_events (
            partition_date DATE,
            event_time    TIMESTAMP,
            user_id       UUID,
            item_id       UUID,
            action        TEXT,
            features      TEXT,
            PRIMARY KEY (partition_date, event_time, user_id, item_id)
        ) WITH CLUSTERING ORDER BY (event_time DESC)
          AND default_time_to_live = 2592000
    """,
}


class ScyllaInteractionStore:
    """Production interaction persistence using ScyllaDB.

    Architecture:
        Redis (instant write <1ms) → Redpanda event → Flink consumer
        → ScyllaDB (persistent write, async, ~50ms after)

    The ML training pipeline reads from ScyllaDB for historical data
    that's too old for Redis but needed for batch training.
    """

    def __init__(
        self,
        hosts: Optional[list[str]] = None,
        keyspace: Optional[str] = None,
    ):
        self.hosts = hosts or SCYLLA_HOSTS
        self.keyspace = keyspace or SCYLLA_KEYSPACE
        self._cluster = None
        self._session = None

    @property
    def session(self) -> Optional[Any]:
        """Get or create ScyllaDB session."""
        if self._session is None and HAS_CASSANDRA_DRIVER:
            try:
                policy = TokenAwarePolicy(DCAwareRoundRobinPolicy())
                self._cluster = Cluster(
                    contact_points=self.hosts,
                    load_balancing_policy=policy,
                    protocol_version=4,
                )
                self._session = self._cluster.connect()
                logger.info("ScyllaDB connected: %s", self.hosts)
            except Exception as e:
                logger.error("ScyllaDB connection failed: %s", e)
        return self._session

    def init_schema(self) -> bool:
        """Initialize keyspace and all tables."""
        if not self.session:
            return False

        try:
            self.session.execute(KEYSPACE_DDL)
            self.session.set_keyspace(self.keyspace)

            for table_name, ddl in TABLE_SCHEMAS.items():
                self.session.execute(ddl)
                logger.info("ScyllaDB table ensured: %s.%s", self.keyspace, table_name)

            return True
        except Exception as e:
            logger.error("Schema init failed: %s", e)
            return False

    def persist_interaction(
        self,
        item_id: str,
        user_id: str,
        action: str,
        metadata: Optional[dict] = None,
    ) -> bool:
        """Persist a single interaction (called by Flink consumer)."""
        if not self.session:
            return False

        try:
            import uuid
            import json
            from datetime import datetime

            self.session.execute(
                f"INSERT INTO {self.keyspace}.interactions "
                f"(item_id, user_id, action, created_at, metadata) "
                f"VALUES (%s, %s, %s, %s, %s)",
                (
                    uuid.UUID(item_id),
                    uuid.UUID(user_id),
                    action,
                    datetime.utcnow(),
                    json.dumps(metadata or {}),
                ),
            )
            return True
        except Exception as e:
            logger.error("Persist interaction failed: %s", e)
            return False

    def batch_persist(self, interactions: list[dict]) -> int:
        """Batch persist interactions (higher throughput)."""
        if not self.session or not interactions:
            return 0

        try:
            import uuid
            import json
            from datetime import datetime

            batch = BatchStatement()
            stmt = self.session.prepare(
                f"INSERT INTO {self.keyspace}.interactions "
                f"(item_id, user_id, action, created_at, metadata) "
                f"VALUES (?, ?, ?, ?, ?)"
            )

            for interaction in interactions:
                batch.add(stmt, (
                    uuid.UUID(interaction["item_id"]),
                    uuid.UUID(interaction["user_id"]),
                    interaction["action"],
                    datetime.utcnow(),
                    json.dumps(interaction.get("metadata", {})),
                ))

            self.session.execute(batch)
            return len(interactions)
        except Exception as e:
            logger.error("Batch persist failed: %s", e)
            return 0

    def get_item_interactions(
        self,
        item_id: str,
        limit: int = 100,
    ) -> list[dict]:
        """Get recent interactions for an item (likes, views, etc.)."""
        if not self.session:
            return []

        try:
            import uuid
            rows = self.session.execute(
                f"SELECT user_id, action, created_at FROM {self.keyspace}.interactions "
                f"WHERE item_id = %s ORDER BY created_at DESC LIMIT %s",
                (uuid.UUID(item_id), limit),
            )
            return [
                {"user_id": str(r.user_id), "action": r.action, "created_at": r.created_at}
                for r in rows
            ]
        except Exception as e:
            logger.error("Get interactions failed: %s", e)
            return []

    def get_user_history(
        self,
        user_id: str,
        action: str = "view",
        limit: int = 100,
    ) -> list[str]:
        """Get user's interaction history (for ML training data)."""
        if not self.session:
            return []

        try:
            import uuid
            rows = self.session.execute(
                f"SELECT item_id FROM {self.keyspace}.user_interactions "
                f"WHERE user_id = %s AND action = %s "
                f"ORDER BY created_at DESC LIMIT %s",
                (uuid.UUID(user_id), action, limit),
            )
            return [str(r.item_id) for r in rows]
        except Exception as e:
            logger.error("Get user history failed: %s", e)
            return []

    def close(self) -> None:
        """Clean shutdown."""
        if self._cluster:
            self._cluster.shutdown()
            self._cluster = None
            self._session = None
