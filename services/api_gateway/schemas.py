"""Pydantic schemas for API Gateway — Section 10."""

from __future__ import annotations

from pydantic import BaseModel


class TokenData(BaseModel):
    user_id: str
    role: str = "buyer"     # buyer | vendor | admin
    email: str = ""


class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: str = ""
    role: str = "buyer"
