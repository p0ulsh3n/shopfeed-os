"""Kafka producer factory — async with log fallback."""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")

_kafka_producer = None


async def get_kafka_producer():
    """Get or create async Kafka producer."""
    global _kafka_producer
    if _kafka_producer is not None:
        return _kafka_producer
    try:
        from aiokafka import AIOKafkaProducer
        _kafka_producer = AIOKafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v, default=str).encode(),
        )
        await _kafka_producer.start()
        logger.info("Kafka producer connected: %s", KAFKA_BOOTSTRAP)
        return _kafka_producer
    except Exception as exc:
        logger.warning("Kafka unavailable (%s), using log fallback", exc)
        _kafka_producer = LogKafkaProducer()
        return _kafka_producer


class LogKafkaProducer:
    """Fallback Kafka producer that logs events (dev/test)."""

    async def send_and_wait(self, topic: str, value: dict, key: bytes | None = None):
        logger.debug("KAFKA[%s] %s", topic, json.dumps(value, default=str)[:200])

    async def stop(self):
        pass


async def publish_event(topic: str, event_data: dict) -> None:
    """Publish an event to a Kafka topic (or log fallback)."""
    producer = await get_kafka_producer()
    await producer.send_and_wait(topic, event_data)
