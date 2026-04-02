"""Pydantic schemas for Product Service — Section 39."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class CreateProductRequest(BaseModel):
    vendor_id: str
    title: str = Field(..., min_length=10, max_length=200)
    description_short: str = Field(default="", max_length=280)
    description_full: str = ""
    category_id: int = 0
    subcategory_id: int | None = None
    brand: str | None = None
    base_price: float
    compare_at_price: float | None = None
    currency: str = ""                      # ISO 4217 — set from vendor's country
    base_stock: int = 0
    tags: list[str] = []
    weight_g: int | None = None
    processing_days: int = 3


class UpdateProductRequest(BaseModel):
    title: str | None = None
    description_short: str | None = None
    description_full: str | None = None
    base_price: float | None = None
    base_stock: int | None = None
    status: str | None = None


class AddVariantRequest(BaseModel):
    variant_sku: str
    type1_value: str | None = None
    type2_value: str | None = None
    price: float
    stock: int = 0


class FlashSaleRequest(BaseModel):
    flash_sale_price: float
    duration_hours: int = 24


class ProductListResponse(BaseModel):
    products: list[dict]
    total: int
    page: int
    limit: int
