"""API Gateway — FastAPI App + Routes — Section 10."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import FastAPI, Depends, Request, status
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
