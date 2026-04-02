"""Pydantic schemas for User Service — Section 05 / 40."""

from __future__ import annotations

from pydantic import BaseModel


class CreateUserRequest(BaseModel):
    email: str
    full_name: str = ""
    phone: str | None = None
    country: str = ""
    city: str | None = None


class CreateVendorRequest(BaseModel):
    user_id: str
    shop_name: str
    description: str = ""
    city: str = ""
    commune: str = ""
    country: str = ""


class UpdateProfileRequest(BaseModel):
    category_prefs: dict[str, float] | None = None
    price_ranges: dict | None = None
    active_categories: list[str] | None = None
