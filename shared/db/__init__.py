"""Database connection factories — PostgreSQL, Redis, Kafka.

Each factory is lazy, async, and supports graceful fallback
when infrastructure is unavailable (dev/test mode).
"""

from .kafka import LogKafkaProducer, get_kafka_producer, publish_event
from .postgres import get_pg_pool
from .redis import InMemoryPipeline, InMemoryRedis, get_redis

__all__ = [
    # Redis
    "get_redis",
    "InMemoryRedis",
    "InMemoryPipeline",
    # Kafka
    "get_kafka_producer",
    "publish_event",
    "LogKafkaProducer",
    # PostgreSQL
    "get_pg_pool",
]
