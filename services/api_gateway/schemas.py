"""Pydantic schemas for API Gateway — Section 10."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, EmailStr, field_validator


class TokenData(BaseModel):
    user_id: str
    role: str = "buyer"     # buyer | vendor | admin
    email: str = ""


class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    # H-02 FIX: email: str acceptait n'importe quelle chaîne, pas de validation format.
    # EmailStr valide automatiquement le format RFC 5322.
    # Requiert: pip install pydantic[email]  (email-validator)
    email: EmailStr

    # H-02 FIX: password: str sans contrainte acceptait "a" comme mot de passe.
    password: str

    full_name: str = ""

    # H-02 FIX: role: str acceptait "admin" — escalade de privilèges possible via l'API.
    # Literal permet uniquement "buyer" ou "vendor". Admin uniquement via K8s/CLI.
    role: Literal["buyer", "vendor"] = "buyer"

    @field_validator("email")
    @classmethod
    def email_lowercase(cls, v: str) -> str:
        """Normalise l'email en minuscules pour éviter les doublons case-insensitive."""
        return v.lower().strip()

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        """Valide la force du mot de passe — minimum OWASP 2026."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        return v
