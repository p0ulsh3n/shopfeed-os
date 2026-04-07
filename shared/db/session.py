"""
SQLAlchemy 2.0 — Async Session Factory
========================================
Fournit :
  - `engine` : AsyncEngine unique (connection pool)
  - `AsyncSessionLocal` : sessionmaker
  - `get_db_session()` : context manager + FastAPI Depends injection
  - `get_settings_dsn()` : DSN depuis env avec fallback sécurisé
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# DSN — postgresql+asyncpg://user:pass@host:port/db
# ─────────────────────────────────────────────────────────────
_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://shopfeed:shopfeed@localhost:5432/shopfeed",
)

# ─────────────────────────────────────────────────────────────
# Engine — pool partagé entre tous les services
# ─────────────────────────────────────────────────────────────
engine = create_async_engine(
    _DATABASE_URL,
    echo=os.getenv("SQL_ECHO", "false").lower() == "true",
    pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
    max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")),
    pool_pre_ping=True,    # Détecte les connexions mortes
    pool_recycle=3600,     # Recycle toutes les heures
)

# ─────────────────────────────────────────────────────────────
# Session Factory
# ─────────────────────────────────────────────────────────────
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Important pour async : pas de lazy loading post-commit
    autoflush=False,
)


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager async pour les sessions DB.

    Usage (service code) :
        async with get_db_session() as session:
            result = await repository.get_by_id(session, user_id)

    Gestion automatique :
      - commit si pas d'exception
      - rollback + re-raise si exception
      - close systématiquement
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI Dependency Injection — utiliser avec `Depends(get_db)`.

    Exemple :
        @app.get("/users/{id}")
        async def get_user(
            user_id: str,
            session: AsyncSession = Depends(get_db),
        ):
            return await user_repo.get_by_id(session, user_id)
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def check_db_health() -> bool:
    """Health check — vérifie que la DB est joignable."""
    from sqlalchemy import text
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error("DB health check failed: %s", exc)
        return False


async def close_engine() -> None:
    """Fermer proprement le pool (shutdown du service)."""
    await engine.dispose()
    logger.info("SQLAlchemy engine disposed")
