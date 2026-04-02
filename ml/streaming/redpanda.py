"""
Redpanda Event Streaming (archi-2026 §5 — replaces Kafka)
============================================================
Redpanda is a C++ Kafka-compatible streaming platform:
    - 3× lower latency vs Kafka (P99 ~1-2ms vs ~5-10ms)
    - 10× less RAM (400MB vs 4GB per broker)
    - No ZooKeeper dependency
    - 100% Kafka API compatible (same producers/consumers)

This module provides the production event bus for:
    - ML training data streaming (interactions → model updates)
    - Feature pipeline events (Flink/Spark consumers)
    - Real-time counters (likes, views, shares)
    - Fraud detection events
    - Video processing pipeline events

The Monolith streaming trainer (`ml/monolith/`) uses this instead of
raw aiokafka for production deployments.

Requires:
    pip install confluent-kafka>=2.3  (or aiokafka>=0.10 for async)
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

try:
    from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
    HAS_AIOKAFKA = True
except ImportError:
    HAS_AIOKAFKA = False

try:
    from confluent_kafka import Producer, Consumer, KafkaError
    HAS_CONFLUENT = True
except ImportError:
    HAS_CONFLUENT = False


# ── Configuration ──────────────────────────────────────────────

REDPANDA_BROKERS = os.getenv(
    "REDPANDA_BROKERS",
    os.getenv("KAFKA_BROKERS", "redpanda-1:9092,redpanda-2:9092,redpanda-3:9092"),
)

# Topic definitions — all topics with partition counts
TOPICS = {
    "user.events":          {"partitions": 12, "retention_ms": 604800000},      # 7 days
    "video.events":         {"partitions": 24, "retention_ms": 604800000},
    "interactions":         {"partitions": 48, "retention_ms": 604800000},      # Highest volume
    "notifications":        {"partitions": 12, "retention_ms": 259200000},      # 3 days
    "feed.updates":         {"partitions": 24, "retention_ms": 86400000},       # 1 day
    "ml.training.data":     {"partitions": 12, "retention_ms": 2592000000},     # 30 days
    "ml.features.realtime": {"partitions": 24, "retention_ms": 86400000},
    "analytics.raw":        {"partitions": 48, "retention_ms": 5184000000},     # 60 days
    "fraud.alerts":         {"partitions": 6, "retention_ms": 2592000000},
    "moderation.events":    {"partitions": 12, "retention_ms": 2592000000},
}


class RedpandaProducer:
    """Production event producer for Redpanda (Kafka-compatible).

    Supports both confluent-kafka (sync, higher throughput) and
    aiokafka (async, for use in asyncio services).
    """

    def __init__(
        self,
        brokers: Optional[str] = None,
        client_id: str = "shopfeed-ml",
        acks: str = "all",
        linger_ms: int = 5,
        batch_size: int = 1_000_000,
    ):
        self.brokers = brokers or REDPANDA_BROKERS
        self._producer = None
        self._async_producer = None
        self.client_id = client_id
        self.acks = acks
        self.linger_ms = linger_ms
        self.batch_size = batch_size

    def _get_sync_producer(self) -> Optional[Any]:
        """Get or create confluent-kafka producer."""
        if self._producer is not None:
            return self._producer

        if not HAS_CONFLUENT:
            return None

        self._producer = Producer({
            "bootstrap.servers": self.brokers,
            "client.id": self.client_id,
            "acks": self.acks,
            "linger.ms": self.linger_ms,
            "batch.size": self.batch_size,
            "compression.type": "lz4",
            "retries": 3,
            "retry.backoff.ms": 100,
        })
        return self._producer

    async def _get_async_producer(self) -> Optional[Any]:
        """Get or create aiokafka producer."""
        if self._async_producer is not None:
            return self._async_producer

        if not HAS_AIOKAFKA:
            return None

        self._async_producer = AIOKafkaProducer(
            bootstrap_servers=self.brokers,
            client_id=self.client_id,
            acks=self.acks,
            linger_ms=self.linger_ms,
            max_batch_size=self.batch_size,
            compression_type="lz4",
        )
        await self._async_producer.start()
        return self._async_producer

    def produce(self, topic: str, value: dict, key: Optional[str] = None) -> bool:
        """Produce a message synchronously (confluent-kafka)."""
        producer = self._get_sync_producer()
        if not producer:
            logger.warning("No sync producer available — message dropped")
            return False

        try:
            payload = json.dumps(value, default=str).encode("utf-8")
            key_bytes = key.encode("utf-8") if key else None
            producer.produce(topic, value=payload, key=key_bytes)
            producer.poll(0)  # Trigger delivery callbacks
            return True
        except Exception as e:
            logger.error("Produce failed: topic=%s error=%s", topic, e)
            return False

    async def produce_async(self, topic: str, value: dict, key: Optional[str] = None) -> bool:
        """Produce a message asynchronously (aiokafka)."""
        producer = await self._get_async_producer()
        if not producer:
            return self.produce(topic, value, key)  # Fallback to sync

        try:
            payload = json.dumps(value, default=str).encode("utf-8")
            key_bytes = key.encode("utf-8") if key else None
            await producer.send_and_wait(topic, value=payload, key=key_bytes)
            return True
        except Exception as e:
            logger.error("Async produce failed: topic=%s error=%s", topic, e)
            return False

    def flush(self, timeout: float = 10.0) -> None:
        """Flush pending messages."""
        if self._producer:
            self._producer.flush(timeout)

    async def close(self) -> None:
        """Clean shutdown."""
        if self._async_producer:
            await self._async_producer.stop()
        if self._producer:
            self._producer.flush(5)


class RedpandaConsumer:
    """Production event consumer for Redpanda.

    Used by:
        - Monolith streaming trainer (ml.training.data topic)
        - Flink feature writers (all topics)
        - Fraud detection (interactions topic)
    """

    def __init__(
        self,
        topics: list[str],
        group_id: str,
        brokers: Optional[str] = None,
        auto_offset_reset: str = "latest",
    ):
        self.topics = topics
        self.group_id = group_id
        self.brokers = brokers or REDPANDA_BROKERS
        self.auto_offset_reset = auto_offset_reset
        self._consumer = None

    async def start(self) -> bool:
        """Start the consumer."""
        if not HAS_AIOKAFKA:
            logger.error("aiokafka required for async consumer. pip install aiokafka>=0.10")
            return False

        try:
            self._consumer = AIOKafkaConsumer(
                *self.topics,
                bootstrap_servers=self.brokers,
                group_id=self.group_id,
                auto_offset_reset=self.auto_offset_reset,
                enable_auto_commit=True,
                auto_commit_interval_ms=1000,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            )
            await self._consumer.start()
            logger.info("Consumer started: topics=%s group=%s", self.topics, self.group_id)
            return True
        except Exception as e:
            logger.error("Consumer start failed: %s", e)
            return False

    async def consume(self, handler: Callable, batch_size: int = 100) -> None:
        """Consume messages and process with handler function.

        The handler receives a list of deserialized message dicts.
        This is the main event loop for streaming consumers.
        """
        if not self._consumer:
            return

        try:
            async for msg in self._consumer:
                try:
                    await handler(msg.value)
                except Exception as e:
                    logger.error("Handler error: topic=%s offset=%d error=%s",
                                 msg.topic, msg.offset, e)
        except Exception as e:
            logger.error("Consumer loop error: %s", e)
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the consumer."""
        if self._consumer:
            await self._consumer.stop()
            logger.info("Consumer stopped: group=%s", self.group_id)


# ── ML Training Data Producer ─────────────────────────────────

class MLTrainingDataProducer:
    """Specialized producer for ML training events.

    Publishes interaction events to the ml.training.data topic
    in the format expected by the Monolith streaming trainer.
    """

    def __init__(self):
        self._producer = RedpandaProducer(client_id="shopfeed-ml-trainer")

    async def publish_interaction(
        self,
        user_id: str,
        item_id: str,
        action: str,
        features: dict[str, Any],
    ) -> bool:
        """Publish an interaction event for online model training."""
        event = {
            "user_id": user_id,
            "item_id": item_id,
            "action": action,
            "features": features,
            "timestamp_ms": int(time.time() * 1000),
        }
        return await self._producer.produce_async(
            "ml.training.data", event, key=user_id,
        )

    async def publish_fraud_alert(
        self,
        user_id: str,
        fraud_score: float,
        action: str,
        details: dict,
    ) -> bool:
        """Publish a fraud detection alert."""
        event = {
            "user_id": user_id,
            "fraud_score": fraud_score,
            "action": action,
            "details": details,
            "timestamp_ms": int(time.time() * 1000),
        }
        return await self._producer.produce_async(
            "fraud.alerts", event, key=user_id,
        )
