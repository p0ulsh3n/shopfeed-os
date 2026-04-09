"""
Security Headers Middleware — H-10 FIX
=======================================

Intègre secweb (v1.30.10, Janvier 2026) pour injecter automatiquement
les headers de sécurité HTTP sur toutes les réponses.

Headers configurés selon OWASP 2026 et NIST guidelines:
  - Strict-Transport-Security (HSTS): Force HTTPS 1 an + subdomains + preload
  - Content-Security-Policy (CSP): Fine-grained content control
  - X-Content-Type-Options: nosniff — empêche le MIME sniffing
  - X-Frame-Options: DENY — remplacé par CSP frame-ancestors mais conservé
    pour compatibilité vieux browsers
  - Referrer-Policy: strict-origin-when-cross-origin
  - Permissions-Policy: désactive features non utilisées (caméra, micro, etc.)
  - Cross-Origin-Embedder-Policy / Cross-Origin-Opener-Policy

NOTE: X-XSS-Protection est intentionnellement ABSENT — déprécié en 2026,
peut créer des vulnérabilités XSS dans de vieux browsers (Chrome < 78).

Usage (dans routes.py ou main.py):
    from shared.security.headers import add_security_headers
    add_security_headers(app)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)


def add_security_headers(app: "FastAPI") -> None:
    """Ajoute le middleware secweb à l'application FastAPI.

    H-10 FIX: Les headers de sécurité HTTP n'étaient pas configurés.
    Un attaquant pouvait exploiter l'absence de HSTS pour faire du downgrade,
    l'absence de CSP pour injecter du JS, etc.

    Args:
        app: FastAPI application instance
    """
    try:
        from secweb import SecWeb  # pip install secweb>=1.30.10

        SecWeb(
            app=app,
            Option={
                # HSTS: Force HTTPS pendant 1 an, inclut sous-domaines, pré-chargement
                "hsts": {
                    "max-age": 31536000,
                    "includeSubDomains": True,
                    "preload": True,
                },
                # CSP: Politique stricte — ajuster selon les CDN utilisés
                "csp": {
                    "default-src": ["'self'"],
                    "script-src": ["'self'"],
                    "style-src": ["'self'", "'unsafe-inline'"],  # Ajuster si inline styles nécessaires
                    "img-src": ["'self'", "data:", "https:"],
                    "connect-src": ["'self'"],
                    "font-src": ["'self'"],
                    "object-src": ["'none'"],
                    "base-uri": ["'self'"],
                    "form-action": ["'self'"],
                    # frame-ancestors 'none' remplace X-Frame-Options: DENY en 2026
                    "frame-ancestors": ["'none'"],
                    "upgrade-insecure-requests": True,
                },
                # Empêche le MIME sniffing — critique pour les uploads de fichiers
                "xcto": True,
                # X-Frame-Options conservé pour compat vieux browsers (CSP frame-ancestors prime)
                "xfo": {"X-Frame-Options": "DENY"},
                # Limiter le Referrer aux mêmes origines pour les cross-origin requests
                "referrer": {"Referrer-Policy": "strict-origin-when-cross-origin"},
                # Désactiver les features du navigateur non utilisées par l'app
                "permissionsPolicy": {
                    "camera": "()",
                    "microphone": "()",
                    "geolocation": "()",
                    "payment": "()",
                    "usb": "()",
                    "interest-cohort": "()",  # Désactive FLoC/Topics API
                },
                # Isolation cross-origin pour SharedArrayBuffer / Spectre mitigations
                "coep": {"Cross-Origin-Embedder-Policy": "require-corp"},
                "coop": {"Cross-Origin-Opener-Policy": "same-origin"},
            },
        )
        logger.info("Security headers middleware loaded (secweb)")

    except ImportError:
        logger.error(
            "secweb package not found. "
            "Install it: pip install secweb>=1.30.10 "
            "Security headers (HSTS, CSP, etc.) are NOT active."
        )
