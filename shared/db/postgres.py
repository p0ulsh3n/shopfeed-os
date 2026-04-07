"""
shared/db/postgres.py
======================
MIGRATION complète : asyncpg brut → SQLAlchemy 2.0.

Ce module est désormais un simple re-export vers shared.db.session.
Il existait avant pour fournir un pool asyncpg directement.
Maintenant, tout passe par SQLAlchemy AsyncSessionLocal.

Compatibilité : les imports existants `from shared.db.postgres import get_db_pool`
sont redirigés via les fonctions de compatibilité ci-dessous.
"""
from __future__ import annotations

import logging

from shared.db.session import check_db_health, engine

logger = logging.getLogger(__name__)


async def get_db_pool():
    """
    DEPRECATED — Utiliser `from shared.db.session import get_db` à la place.
    Retourne None et log un warning pour guider la migration.
    """
    logger.warning(
        "get_db_pool() is deprecated — use 'from shared.db.session import get_db' "
        "with FastAPI Depends(get_db) or get_db_session() context manager instead."
    )
    return None


async def release_db_pool():
    """DEPRECATED — Le pool SQLAlchemy est géré automatiquement par l'engine."""
    pass


# Health check — proxied vers SQLAlchemy engine
health_check = check_db_health

__all__ = ["get_db_pool", "release_db_pool", "health_check"]
