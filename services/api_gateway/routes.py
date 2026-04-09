"""
API Gateway — FastAPI App + Routes — Section 10
================================================

Auth notes:
- POST /api/v1/auth/register  → hashes password, persists via UserRepository
- POST /api/v1/auth/login     → vérifie via UserRepository.get_by_email()
- GET  /api/v1/me             → retourne le user depuis JWT (Depends(verify_token))

Migration: asyncpg brut conn.fetchrow() → UserRepository (SQLAlchemy 2.0)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from shared.db.session import get_db
from shared.repositories.user_repository import UserRepository
from shared.security.headers import add_security_headers  # H-10 FIX

from .auth import create_access_token, pwd_context, verify_token
from .rate_limiter import RateLimiter
from .schemas import LoginRequest, RegisterRequest, TokenData

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ShopFeed OS — API Gateway",
    version="1.0.0",
    description="Central entry point for the ShopFeed OS platform.",
)

# C-01 FIX: allow_origins=["*"] + allow_credentials=True est interdit par la spec CORS.
# Les navigateurs rejettent silencieusement les requêtes credentialed vers un wildcard.
# On charge les origines autorisées depuis une variable d'env pour chaque déploiement.
_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
ALLOWED_ORIGINS: list[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,          # Jamais ["*"] avec credentials
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-User-ID", "X-Session-ID"],
    expose_headers=["X-Request-ID"],
    max_age=600,
)

# H-10 FIX: Injecter HSTS, CSP, X-Frame-Options, Referrer-Policy, etc.
# Doit être appelé APRÈS add_middleware(CORSMiddleware).
add_security_headers(app)

rate_limiter = RateLimiter(redis_client=None, max_requests=100, window_seconds=60)

_user_repo = UserRepository()


@app.on_event("startup")
async def _inject_redis_into_rate_limiter() -> None:
    """Wire the shared Redis client into the rate limiter at startup."""
    redis = getattr(app.state, "redis", None)
    if redis is not None:
        rate_limiter.redis = redis
        logger.info("RateLimiter: Redis client injected — distributed limiting active")
    else:
        logger.warning(
            "RateLimiter: app.state.redis not set at startup — "
            "using in-memory fallback (NOT safe for multi-replica)"
        )


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting per IP."""
    client_ip = request.client.host if request.client else "unknown"
    if not await rate_limiter.check(client_ip):
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Rate limit exceeded. Try again later."},
        )
    return await call_next(request)


# ── Auth Endpoints ──────────────────────────────────────────────

@app.post("/api/v1/auth/register")
async def register(
    req: RegisterRequest,
    session: AsyncSession = Depends(get_db),
):
    """Register new user — hash password et persiste via SQLAlchemy ORM."""
    hashed = pwd_context.hash(req.password)

    # Vérification email dupliqué — ORM, zéro SQL brut
    existing = await _user_repo.get_by_email(session, req.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    try:
        user = await _user_repo.create(session, {
            "email": req.email,
            "hashed_password": hashed,
            "role": req.role,
        })
    except Exception as exc:
        logger.error("Register DB error: %s", exc)
        raise HTTPException(status_code=500, detail="Registration failed")

    token = create_access_token({"sub": req.email, "role": req.role, "email": req.email})
    return {"access_token": token, "token_type": "bearer", "user_id": str(user.id)}


@app.post("/api/v1/auth/login")
async def login(
    req: LoginRequest,
    session: AsyncSession = Depends(get_db),
):
    """Issue JWT après vérification du mot de passe — SQLAlchemy ORM."""
    user = await _user_repo.get_by_email(session, req.email)

    if user is None or not user.hashed_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not pwd_context.verify(req.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token({"sub": req.email, "role": user.role, "email": req.email})
    return {"access_token": token, "token_type": "bearer"}


@app.get("/api/v1/me")
async def get_me(token: TokenData = Depends(verify_token)):
    return {"user_id": token.user_id, "role": token.role, "email": token.email}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "api-gateway",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
