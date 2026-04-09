"""
Redis-backed Sliding Window Rate Limiter — Section 10
=======================================================

H-03 FIX: Bug original — le VRAI bug n'était PAS pipe(ZCARD → ZADD) non-atomique,
mais pipeline(ZADD) + zrem() séparé HORS du pipeline.

Le code original faisait:
  1. pipeline: ZREMRANGEBYSCORE + ZCARD + ZADD + EXPIRE → exécuté atomiquement
  2. results[1] < max → ok
  3. results[1] >= max → zrem(redis_key, str(now))  ← RACE CONDITION ICI

Entre l'étape 1 et 3, un autre pod peut lire l'entrée ZADDée comme valide.
Le ZADD étant dans le pipeline, il est toujours exécuté même si on rejette.

SOLUTION: Script Lua Redis exécuté atomiquement (single-threaded Redis).
Le check + le ZADD conditionnel sont dans la même opération atomique.
Redéfinit aussi le window en millisecondes pour plus de précision.
"""

from __future__ import annotations

import logging
import time
import uuid

logger = logging.getLogger(__name__)


# H-03 FIX: Script Lua atomique — le check et le ZADD conditionnel sont
# dans une seule opération atomique sur le serveur Redis.
# Redis garantit qu'aucun autre script/commande ne s'exécute pendant ce script.
_LUA_SLIDING_WINDOW = """
local key = KEYS[1]
local window_ms = tonumber(ARGV[1])
local max_req   = tonumber(ARGV[2])
local now_ms    = tonumber(ARGV[3])
local req_id    = ARGV[4]
local window_start = now_ms - window_ms

-- Nettoyer les requêtes hors-fenêtre
redis.call('ZREMRANGEBYSCORE', key, '-inf', window_start)

-- Compter les requêtes dans la fenêtre actuelle
local current = redis.call('ZCARD', key)

-- Refuser si la limite est atteinte (les deux ops ci-dessus sont atomiques)
if current >= max_req then
    return {0, current, 0}
end

-- Accepter et enregistrer la requête
redis.call('ZADD', key, now_ms, req_id)
redis.call('PEXPIRE', key, window_ms + 1000)
return {1, current + 1, max_req - current - 1}
"""


class RateLimiter:
    """Redis sliding window rate limiter — Lua atomique + fallback mémoire.

    H-03 FIX: Remplace le bug ZADD-dans-pipeline + ZREM-séparé par un script
    Lua atomique. Check et ZADD conditionnel sont une seule opération.

    Args:
        redis_client: async Redis client (redis.asyncio). Si None, fallback
            in-memory (single-replica uniquement, loggue un warning).
        max_requests: nombre de requêtes autorisées par fenêtre
        window_seconds: durée de la fenêtre glissante en secondes
    """

    def __init__(
        self,
        redis_client=None,
        max_requests: int = 100,
        window_seconds: int = 60,
    ):
        self.redis = redis_client
        self.max_requests = max_requests
        self.window = window_seconds
        self._lua_script = None  # lazy-registered
        # In-memory fallback (seulement correct pour single-replica)
        self._buckets: dict[str, list[float]] = {}
        if redis_client is None:
            logger.warning(
                "RateLimiter: no Redis client provided — using in-memory fallback. "
                "This is NOT safe in multi-replica deployments (k8s). "
                "Inject a redis_client at startup."
            )

    def _get_lua_script(self):
        """Lazy-registration du script Lua (connexion Redis disponible)."""
        if self._lua_script is None:
            self._lua_script = self.redis.register_script(_LUA_SLIDING_WINDOW)
        return self._lua_script

    async def check(self, key: str) -> bool:
        """Retourne True si la requête est autorisée, False si rate limit dépassé."""
        if self.redis is not None:
            return await self._check_redis(key)
        return await self._check_memory(key)

    async def _check_redis(self, key: str) -> bool:
        """Script Lua atomique — aucune race condition possible."""
        now_ms = int(time.time() * 1000)  # millisecondes pour plus de précision
        req_id = f"{now_ms}:{uuid.uuid4().hex[:8]}"
        try:
            script = self._get_lua_script()
            result = await script(
                keys=[f"rate:{key}"],
                args=[self.window * 1000, self.max_requests, now_ms, req_id],
            )
            allowed: bool = result[0] == 1
            return allowed
        except Exception as e:
            logger.warning("RateLimiter Redis error (%s) — fail open", e)
            return True  # Fail open: ne pas bloquer les utilisateurs si Redis est down

    async def _check_memory(self, key: str) -> bool:
        """Fallback in-memory — correct uniquement pour single-replica."""
        now = time.time()
        bucket = self._buckets.setdefault(key, [])
        cutoff = now - self.window
        self._buckets[key] = [t for t in bucket if t > cutoff]

        if len(self._buckets[key]) >= self.max_requests:
            return False

        self._buckets[key].append(now)
        return True
