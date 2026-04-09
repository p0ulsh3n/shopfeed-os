"""
ML Inference API — Service Token Authentication
================================================

C-02 FIX: Le service ML inference exposait 20 endpoints sans authentification.
N'importe quel pod dans le cluster pouvait déclencher un réentraînement,
extraire des embeddings utilisateurs, ou manipuler les enchères publicitaires.

Ce module implémente un middleware de token inter-service injecté au niveau
FastAPI(dependencies=[...]) pour être appliqué à TOUS les endpoints.

C-02 + H-02b FIX: La comparaison utilise hmac.compare_digest() (constant-time)
au lieu de l'opérateur == (court-circuite à la première différence de byte,
vulnérable aux timing attacks).

IMPORTANT POUR LES TESTS UNITAIRES:
Le token est validé dans verify_service_token() uniquement — pas au niveau module.
Ainsi, les tests peuvent importer ce module sans définir ML_SERVICE_TOKEN.
"""

from __future__ import annotations

import hmac
import logging
import os

from fastapi import HTTPException, Request, status

logger = logging.getLogger(__name__)

# Chemins exemptés d'authentification (health checks)
_EXEMPT_PATHS = frozenset({"/v1/health", "/health"})


def _get_service_token() -> str:
    """Récupère le token depuis l'env. Retourne "" si absent (géré dans le middleware)."""
    return os.getenv("ML_SERVICE_TOKEN", "")


async def verify_service_token(request: Request) -> None:
    """Dependency FastAPI: vérifie le token inter-service sur tous les endpoints ML.

    C-02: Appliqué via FastAPI(dependencies=[Depends(verify_service_token)])
    C-02 + H-02b: Utilise hmac.compare_digest() — résistant aux timing attacks.

    Le token doit être passé dans le header X-Service-Token.
    Le health check est accessible sans token.

    Raises:
        HTTPException(401): si le token est absent ou invalide.
        HTTPException(503): si ML_SERVICE_TOKEN n'est pas configuré côté serveur
                            (ne pas révéler l'absence de config en 401).
    """
    # Health checks exemptés — Kubernetes liveness/readiness probes
    if request.url.path in _EXEMPT_PATHS:
        return

    expected = _get_service_token()
    if not expected:
        # Le serveur n'est pas configuré — erreur de déploiement
        logger.critical(
            "ML_SERVICE_TOKEN is not configured. "
            "Set this env var via K8s Secret. "
            "Generate with: openssl rand -hex 32"
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service temporarily unavailable",  # Ne pas révéler la cause
        )

    provided = request.headers.get("X-Service-Token", "")
    if not provided:
        logger.warning(
            "ML API access attempt without service token: path=%s ip=%s",
            request.url.path,
            request.client.host if request.client else "unknown",
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Service token required",
        )

    # H-02b FIX: Comparaison constant-time — résistant aux timing attacks.
    # L'opérateur == court-circuite à la première différence de byte,
    # permettant de deviner le token caractère par caractère via les temps de réponse.
    if not hmac.compare_digest(
        provided.encode("utf-8"),
        expected.encode("utf-8"),
    ):
        logger.warning(
            "Invalid ML service token: path=%s ip=%s",
            request.url.path,
            request.client.host if request.client else "unknown",
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Service token required",
        )
