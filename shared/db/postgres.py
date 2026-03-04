"""PostgreSQL connection factory — async with graceful fallback."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql://shopfeed:shopfeed@localhost:5432/shopfeed")

_pg_pool = None


async def get_pg_pool():
    """Get or create async PostgreSQL connection pool."""
    global _pg_pool
    if _pg_pool is not None:
        return _pg_pool
    try:
        import asyncpg
        _pg_pool = await asyncpg.create_pool(POSTGRES_DSN, min_size=2, max_size=20)
        logger.info("PostgreSQL pool created: %s", POSTGRES_DSN.split("@")[-1])
        return _pg_pool
    except Exception as exc:
        logger.warning("PostgreSQL unavailable (%s)", exc)
        return None
