"""
shared/db/__init__.py
Expose les points d'entrée principaux de la couche DB.
Usage : `from shared.db import get_db, engine`
"""
from shared.db.session import (
    AsyncSessionLocal,
    check_db_health,
    close_engine,
    engine,
    get_db,
    get_db_session,
)

# Kafka + Redis restent dans leurs modules séparés
from shared.db.kafka import get_kafka_producer
from shared.db.redis import get_redis_client

__all__ = [
    "engine",
    "AsyncSessionLocal",
    "get_db",
    "get_db_session",
    "check_db_health",
    "close_engine",
    "get_kafka_producer",
    "get_redis_client",
]
