"""
Circuit Breakers — H-11 FIX
============================

Intègre aiobreaker (v1.2.0+) pour protéger les appels inter-services
contre les pannes en cascade (cascading failures).

Pattern: Circuit Breaker (Nygard 2007)
  - CLOSED: Appels normaux. Si N erreurs consécutives → OPEN
  - OPEN: Rejette immédiatement sans appeler le service → fail fast
  - HALF-OPEN: Laisse passer 1 appel test → si succès → CLOSED, sinon → OPEN

Usage:
    from shared.resilience.circuit_breakers import ml_inference_breaker

    async def call_ml():
        try:
            # Utiliser l'API explicite (compatible toutes versions aiobreaker)
            result = await ml_inference_breaker.call_async(my_async_func, *args)
        except aiobreaker.CircuitBreakerError:
            # Circuit ouvert — utiliser le fallback
            return fallback_response()

NOTE H-11: Le décorateur @breaker sur une fonction async peut avoir des
incompatibilités avec certaines versions de aiobreaker. Préférer l'API
explicite breaker.call_async() qui est garantie stable.

Dépendances:
    pip install aiobreaker>=1.2.0
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _make_breaker(
    name: str,
    fail_max: int = 5,
    reset_timeout: float = 60.0,
):
    """Crée un circuit breaker aiobreaker avec les paramètres donnés.

    Args:
        name: Nom pour le logging et monitoring
        fail_max: Nombre d'échecs consécutifs avant d'ouvrir le circuit
        reset_timeout: Secondes avant de tenter HALF-OPEN

    Returns:
        CircuitBreaker instance ou None si aiobreaker non installé
    """
    try:
        import aiobreaker
        from datetime import timedelta

        breaker = aiobreaker.CircuitBreaker(
            fail_max=fail_max,
            reset_timeout=timedelta(seconds=reset_timeout),
        )
        logger.info(
            "CircuitBreaker '%s': fail_max=%d, reset_timeout=%.0fs",
            name, fail_max, reset_timeout,
        )
        return breaker

    except ImportError:
        logger.error(
            "aiobreaker package not found for '%s'. "
            "Install: pip install aiobreaker>=1.2.0 "
            "Circuit breakers are DISABLED — cascading failures not protected.",
            name,
        )
        return None


# ── Instances de circuit breakers par service ─────────────────────────────

# ML Inference Service: 5 erreurs → open 60s
# SLA critique (<80ms) — fail fast si le service ML est down
ml_inference_breaker = _make_breaker(
    name="ml_inference",
    fail_max=5,
    reset_timeout=60.0,
)

# Redis: 10 erreurs → open 30s (Redis est plus robuste, shorter timeout)
redis_breaker = _make_breaker(
    name="redis",
    fail_max=10,
    reset_timeout=30.0,
)

# User Service: 5 erreurs → open 45s
user_service_breaker = _make_breaker(
    name="user_service",
    fail_max=5,
    reset_timeout=45.0,
)

# Catalog Service (ScyllaDB/Postgres): 3 erreurs → open 60s
catalog_breaker = _make_breaker(
    name="catalog",
    fail_max=3,
    reset_timeout=60.0,
)


async def call_with_breaker(breaker, coro_func, *args, fallback=None, **kwargs):
    """Helper pour appeler une coroutine avec un circuit breaker.

    H-11: Utilise l'API explicite call_async() qui est stable sur
    toutes les versions de aiobreaker (contrairement à @decorator).

    Args:
        breaker: CircuitBreaker instance (de ce module)
        coro_func: Fonction async à appeler
        *args: Arguments pour coro_func
        fallback: Valeur de fallback si le circuit est ouvert ou None
        **kwargs: Keyword arguments pour coro_func

    Returns:
        Résultat de coro_func ou fallback

    Example:
        result = await call_with_breaker(
            ml_inference_breaker,
            http_client.get,
            "http://ml-service/v1/feed/rank",
            json=payload,
            fallback=[],
        )
    """
    if breaker is None:
        # aiobreaker non installé — appel direct sans protection
        return await coro_func(*args, **kwargs)

    try:
        import aiobreaker
        return await breaker.call_async(coro_func, *args, **kwargs)
    except aiobreaker.CircuitBreakerError as e:
        logger.warning(
            "Circuit breaker OPEN [%s]: %s — returning fallback",
            getattr(breaker, "name", "unknown"), e,
        )
        return fallback
    except Exception:
        raise
