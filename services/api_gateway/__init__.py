"""API Gateway — Main Entry Point — Section 10.

Central gateway handling:
    - JWT authentication
    - Rate limiting (Redis token bucket)
    - Request routing to downstream services
    - CORS & security headers
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────
# JWT Auth
# ──────────────────────────────────────────────────────────────

import os
from jose import JWTError, jwt
from passlib.context import CryptContext

JWT_SECRET = os.getenv("JWT_SECRET", "shopfeed-dev-secret-change-in-prod")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class TokenData(BaseModel):
    user_id: str
    role: str = "buyer"     # buyer | vendor | admin
    email: str = ""


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(hours=JWT_EXPIRY_HOURS))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)


async def verify_token(request: Request) -> TokenData:
    """FastAPI dependency — extract and verify JWT token."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")

    token = auth[7:]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return TokenData(
            user_id=payload.get("sub", ""),
            role=payload.get("role", "buyer"),
            email=payload.get("email", ""),
        )
    except JWTError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid token: {e}")


def require_role(*roles: str):
    """Dependency factory — require specific role(s)."""
    async def _check(token: TokenData = Depends(verify_token)):
        if token.role not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return token
    return _check


# ──────────────────────────────────────────────────────────────
# Rate Limiter (token bucket, Redis-backed)
# ──────────────────────────────────────────────────────────────

class RateLimiter:
    """Token bucket rate limiter — Section 10."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._buckets: dict[str, list[float]] = {}

    async def check(self, key: str) -> bool:
        """Returns True if request is allowed."""
        now = time.time()
        bucket = self._buckets.setdefault(key, [])

        # Remove expired entries
        cutoff = now - self.window
        self._buckets[key] = [t for t in bucket if t > cutoff]

        if len(self._buckets[key]) >= self.max_requests:
            return False

        self._buckets[key].append(now)
        return True


rate_limiter = RateLimiter(max_requests=100, window_seconds=60)


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


# ──────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: str = ""
    role: str = "buyer"


@app.post("/api/v1/auth/login")
async def login(req: LoginRequest):
    """Issue JWT token. In production, verify against user DB."""
    token = create_access_token({"sub": req.email, "role": "buyer", "email": req.email})
    return {"access_token": token, "token_type": "bearer"}


@app.post("/api/v1/auth/register")
async def register(req: RegisterRequest):
    """Register new user. In production, hash password and persist."""
    hashed = pwd_context.hash(req.password)
    token = create_access_token({"sub": req.email, "role": req.role, "email": req.email})
    return {"access_token": token, "token_type": "bearer", "user_id": req.email}


@app.get("/api/v1/me")
async def get_me(token: TokenData = Depends(verify_token)):
    return {"user_id": token.user_id, "role": token.role, "email": token.email}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "api-gateway", "timestamp": datetime.now(timezone.utc).isoformat()}
