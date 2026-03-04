"""JWT Authentication — Section 10."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, Request, status
from jose import JWTError, jwt
from passlib.context import CryptContext

from .schemas import TokenData

JWT_SECRET = os.getenv("JWT_SECRET", "shopfeed-dev-secret-change-in-prod")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24

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
