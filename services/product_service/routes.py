"""Product Service — FastAPI App + Routes — Section 39."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Query

from shared.models.product import ProductStatus, PoolLevel

from .schemas import (
    AddVariantRequest,
    CreateProductRequest,
    FlashSaleRequest,
    ProductListResponse,
    UpdateProductRequest,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="ShopFeed OS — Product Service", version="1.0.0")

# In-memory storage (would be PostgreSQL in production)
_products: dict[str, dict] = {}
_variants: dict[str, list[dict]] = {}


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
