"""JWT Authentication — Section 10."""

from __future__ import annotations

import hmac
import logging
import os
import sys
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, Request, status
from jose import JWTError, jwt
from passlib.context import CryptContext

from .schemas import TokenData

logger = logging.getLogger(__name__)

# C-04 FIX: Fallback silencieux sur le secret par défaut en production est une
# vulnérabilité critique — tout attaquant connaissant le code peut forger des tokens.
_ENV = os.getenv("APP_ENV", "development")
_DEFAULT_SECRET = "shopfeed-dev-secret-change-in-prod"

JWT_SECRET = os.getenv("JWT_SECRET", _DEFAULT_SECRET)
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24

if JWT_SECRET == _DEFAULT_SECRET:
    if _ENV in ("production", "staging", "prod"):
        logger.critical(
            "FATAL: JWT_SECRET is set to the default development value in %s environment. "
            "The application will not start. Generate a secure secret: openssl rand -hex 32",
            _ENV,
        )
        sys.exit(1)
    else:
        logger.warning(
            "JWT_SECRET is using the default development value. "
            "This is acceptable in development but MUST be changed before deploying to staging/production."
        )

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


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
