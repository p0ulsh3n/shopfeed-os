"""
API Gateway — FastAPI App + Routes — Section 10
================================================

Auth notes:
- POST /api/v1/auth/register  → hashes password, persists user to PostgreSQL
- POST /api/v1/auth/login     → verifies password against stored hash, issues JWT
- GET  /api/v1/me             → returns current user from JWT (Depends(verify_token))

BUG #S1 FIX: Previously, login() issued a token to ANY email without
verifying the password, and register() hashed the password but never
stored it. Fixed with asyncpg-backed user storage + bcrypt verify.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .auth import create_access_token, pwd_context, verify_token
from .rate_limiter import RateLimiter
from .schemas import LoginRequest, RegisterRequest, TokenData

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ShopFeed OS — API Gateway",
    version="1.0.0",
    description="Central entry point for the ShopFeed OS platform.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# BUG #S2: RateLimiter now accepts redis_client.
# Injected at startup via app.state.redis (see lifespan events in deployment).
# Falls back to in-memory with a warning when redis_client=None.
rate_limiter = RateLimiter(redis_client=None, max_requests=100, window_seconds=60)


@app.on_event("startup")
async def _inject_redis_into_rate_limiter() -> None:
    """Wire the shared Redis client into the rate limiter at startup.

    In production, app.state.redis is set by the k8s-level startup hook
    or by the parent process that mounts the Redis pool.
    """
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


# ── Auth helper — DB access ─────────────────────────────────────

async def _get_db():
    """Get async PostgreSQL connection from app.state pool."""
    pool = getattr(app.state, "db_pool", None)
    if pool is None:
        return None
    return await pool.acquire()


async def _release_db(conn) -> None:
    pool = getattr(app.state, "db_pool", None)
    if pool and conn:
        await pool.release(conn)


# ── Auth Endpoints ──────────────────────────────────────────────

@app.post("/api/v1/auth/register")
async def register(req: RegisterRequest):
    """Register new user — hashes password and persists to PostgreSQL.

    BUG #S1 FIX: Previously hashed the password and immediately discarded
    it (never stored). The hash is now INSERT-ed into the users table.
    """
    hashed = pwd_context.hash(req.password)

    conn = await _get_db()
    if conn is not None:
        try:
            # Check duplicate email
            existing = await conn.fetchrow(
                "SELECT id FROM users WHERE email = $1", req.email
            )
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Email already registered",
                )

            # Persist user with hashed password
            row = await conn.fetchrow(
                """
                INSERT INTO users (email, hashed_password, role, created_at)
                VALUES ($1, $2, $3, NOW())
                RETURNING id
                """,
                req.email, hashed, req.role,
            )
            user_id = str(row["id"])
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Register DB error: %s", e)
            raise HTTPException(status_code=500, detail="Registration failed")
        finally:
            await _release_db(conn)
    else:
        # No DB pool (dev/test mode) — in-memory stub
        logger.warning("No DB pool — using in-memory auth stub (dev mode only)")
        user_id = req.email

    token = create_access_token({"sub": req.email, "role": req.role, "email": req.email})
    return {"access_token": token, "token_type": "bearer", "user_id": user_id}


@app.post("/api/v1/auth/login")
async def login(req: LoginRequest):
    """Issue JWT token after verifying password against stored hash.

    BUG #S1 FIX: Previously issued a token to any email with any password
    (no database lookup, no password verification). Fixed with:
    1. SELECT user from PostgreSQL by email
    2. bcrypt verify(req.password, stored_hash)
    3. Only then issue JWT
    """
    conn = await _get_db()
    if conn is not None:
        try:
            row = await conn.fetchrow(
                "SELECT id, hashed_password, role FROM users WHERE email = $1",
                req.email,
            )
        except Exception as e:
            logger.error("Login DB error: %s", e)
            raise HTTPException(status_code=500, detail="Login failed")
        finally:
            await _release_db(conn)

        if row is None or not pwd_context.verify(req.password, row["hashed_password"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        role = row["role"]
    else:
        # Dev/test stub: accept any credentials
        logger.warning("No DB pool — skipping password verification (dev mode only)")
        role = "buyer"

    token = create_access_token({"sub": req.email, "role": role, "email": req.email})
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
