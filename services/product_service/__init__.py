"""Product Service — CRUD + Vision Pipeline — Section 39.

Handles:
    - Product CRUD (create, read, update, delete)
    - Variant management
    - Photo upload + async CV pipeline (CLIP + SightEngine)
    - Stock management
    - Flash sale management
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from pydantic import BaseModel, Field

from shared.models.product import (
    Product, ProductStatus, ProductVariant, PoolLevel, ContentType,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="ShopFeed OS — Product Service", version="1.0.0")

# In-memory storage (would be PostgreSQL in production)
_products: dict[str, dict] = {}
_variants: dict[str, list[dict]] = {}


# ──────────────────────────────────────────────────────────────
# Request / Response Models
# ──────────────────────────────────────────────────────────────

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
    currency: str = "EUR"
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


# ──────────────────────────────────────────────────────────────
# Vision Pipeline (async) — Section 15
# ──────────────────────────────────────────────────────────────

async def run_vision_pipeline(product_id: str, image_url: str) -> dict:
    """Async CV pipeline — <60s total.

    Steps:
        1. SightEngine quality check → cv_score [0, 1]
        2. CLIP embedding → 512-dim vector
        3. BLIP-2 auto-caption → text description

    In production this runs as a Celery task or Kafka consumer.
    """
    result = {
        "cv_score": 0.0,
        "clip_embedding": None,
        "caption": None,
    }

    # 1. Quality score (would call SightEngine API)
    result["cv_score"] = 0.75  # Placeholder, would be real API call

    # 2. CLIP embedding (would run on GPU worker)
    # In production: clip_model.encode_image(image)
    result["clip_embedding"] = [0.0] * 512  # Placeholder

    # 3. Auto-caption (would run BLIP-2)
    result["caption"] = f"Product image for {product_id}"

    logger.info("Vision pipeline complete: product=%s, cv_score=%.2f", product_id, result["cv_score"])
    return result


# ──────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────

@app.post("/api/v1/products")
async def create_product(req: CreateProductRequest):
    """Create a new product — auto-enters L1 pool."""
    product_id = str(uuid.uuid4())

    product_data = req.model_dump()
    product_data["id"] = product_id
    product_data["status"] = ProductStatus.DRAFT
    product_data["pool_level"] = PoolLevel.L1
    product_data["created_at"] = datetime.now(timezone.utc).isoformat()

    _products[product_id] = product_data
    _variants[product_id] = []

    return {"product_id": product_id, "status": "created", "pool_level": "L1"}


@app.get("/api/v1/products/{product_id}")
async def get_product(product_id: str):
    product = _products.get(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    product["variants"] = _variants.get(product_id, [])
    return product


@app.patch("/api/v1/products/{product_id}")
async def update_product(product_id: str, req: UpdateProductRequest):
    if product_id not in _products:
        raise HTTPException(status_code=404, detail="Product not found")

    updates = req.model_dump(exclude_unset=True)
    _products[product_id].update(updates)
    _products[product_id]["updated_at"] = datetime.now(timezone.utc).isoformat()

    return _products[product_id]


@app.delete("/api/v1/products/{product_id}")
async def delete_product(product_id: str):
    if product_id not in _products:
        raise HTTPException(status_code=404, detail="Product not found")

    _products[product_id]["status"] = ProductStatus.DELETED
    return {"status": "deleted"}


@app.post("/api/v1/products/{product_id}/variants")
async def add_variant(product_id: str, req: AddVariantRequest):
    if product_id not in _products:
        raise HTTPException(status_code=404, detail="Product not found")

    variant = req.model_dump()
    variant["id"] = str(uuid.uuid4())
    variant["product_id"] = product_id
    _variants.setdefault(product_id, []).append(variant)

    _products[product_id]["has_variants"] = True
    return variant


@app.post("/api/v1/products/{product_id}/publish")
async def publish_product(product_id: str):
    """Publish product — enters moderation queue then L1 pool."""
    if product_id not in _products:
        raise HTTPException(status_code=404, detail="Product not found")

    _products[product_id]["status"] = ProductStatus.PENDING_MODERATION
    _products[product_id]["published_at"] = datetime.now(timezone.utc).isoformat()

    return {"status": "pending_moderation", "pool_level": "L1"}


@app.post("/api/v1/products/{product_id}/flash-sale")
async def activate_flash_sale(product_id: str, req: FlashSaleRequest):
    """Activate flash sale — triggers L6 pool boost."""
    if product_id not in _products:
        raise HTTPException(status_code=404, detail="Product not found")

    from datetime import timedelta

    _products[product_id]["flash_sale_active"] = True
    _products[product_id]["flash_sale_price"] = req.flash_sale_price
    _products[product_id]["flash_sale_ends_at"] = (
        datetime.now(timezone.utc) + timedelta(hours=req.duration_hours)
    ).isoformat()

    return {"status": "flash_sale_active", "ends_at": _products[product_id]["flash_sale_ends_at"]}


@app.get("/api/v1/products", response_model=ProductListResponse)
async def list_products(
    vendor_id: Optional[str] = None,
    status: Optional[str] = None,
    category_id: Optional[int] = None,
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, le=50),
):
    """List products with filters and pagination."""
    filtered = list(_products.values())

    if vendor_id:
        filtered = [p for p in filtered if p.get("vendor_id") == vendor_id]
    if status:
        filtered = [p for p in filtered if p.get("status") == status]
    if category_id is not None:
        filtered = [p for p in filtered if p.get("category_id") == category_id]

    total = len(filtered)
    start = (page - 1) * limit
    paged = filtered[start:start + limit]

    return ProductListResponse(products=paged, total=total, page=page, limit=limit)
