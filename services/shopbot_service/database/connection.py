"""
Database Connection — ShopBot Service
======================================
SQLAlchemy 2.0 async engine + session factory.

MIGRATION:
- SUPPRIMÉ: SHOPBOT_SCHEMA_SQL (300 lignes de DDL inline)
  → Remplacé par les migrations Alembic (alembic upgrade head)
- GARDÉ: pool asyncpg brut UNIQUEMENT pour LISTEN/NOTIFY
  (SQLAlchemy 2.0 ne supporte pas nativement pg LISTEN/NOTIFY)
- GARDÉ: initialize_schema() → SUPPRIMÉ, géré par Alembic
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import asyncpg
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from services.shopbot_service.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ─────────────────────── SQLAlchemy Engine ────────────────────────

engine = create_async_engine(
    str(settings.database_url),
    echo=settings.environment == "development",
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager pour les sessions DB."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ─────────────────────── RAW asyncpg pool ────────────────────────
# JUSTIFICATION: SQLAlchemy 2.0 ne supporte pas pg LISTEN/NOTIFY.
# Ce pool est utilisé UNIQUEMENT par catalog_sync.py pour LISTEN/NOTIFY.
# TOUTE autre utilisation DB doit passer par AsyncSessionLocal.

_raw_pool: asyncpg.Pool | None = None


async def get_raw_pool() -> asyncpg.Pool:
    """Pool asyncpg brut — RÉSERVÉ pour LISTEN/NOTIFY uniquement."""
    global _raw_pool
    if _raw_pool is None:
        dsn = str(settings.database_url).replace("+asyncpg", "")
        _raw_pool = await asyncpg.create_pool(
            dsn,
            min_size=2,
            max_size=10,
            command_timeout=30,
        )
    return _raw_pool


async def close_raw_pool() -> None:
    global _raw_pool
    if _raw_pool:
        await _raw_pool.close()
        _raw_pool = None


# ─────────────────────── Health Check ────────────────────────────

async def check_db_health() -> bool:
    """Quick health check — vérifie que la DB est joignable."""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error("DB health check failed: %s", exc)
        return False


# ─────────────────────── NOTE: Migrations ────────────────────────
# Le schéma (tables, index, extensions pgvector) est géré par Alembic.
# Lancer: alembic upgrade head
#
# Modèles ORM correspondants:
#   - shopbot_product_embeddings → ShopbotProductEmbeddingORM (à créer)
#   - shopbot_sessions           → ShopbotSessionORM (à créer)
#   - shopbot_sync_log           → ShopbotSyncLogORM (à créer)
