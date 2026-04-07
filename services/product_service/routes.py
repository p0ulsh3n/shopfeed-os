"""
Product Service — FastAPI App + Routes — Section 39.

Migration: dicts Python in-memory (_products, _variants)
→ SQLAlchemy 2.0 ORM via ProductRepository + ProductVariantRepository.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from shared.db.session import get_db
from shared.models.product import PoolLevel, ProductStatus
from shared.repositories.product_repository import ProductRepository, ProductVariantRepository

from .schemas import (
    AddVariantRequest,
    CreateProductRequest,
    FlashSaleRequest,
    ProductListResponse,
    UpdateProductRequest,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="ShopFeed OS — Product Service", version="1.0.0")

_product_repo = ProductRepository()
_variant_repo = ProductVariantRepository()


def _fmt_product(p) -> dict:
    """Sérialise un ProductORM en dict JSON-compatible."""
    return {
        "id": str(p.id),
        "vendor_id": str(p.vendor_id),
        "title": p.title,
        "description_short": p.description_short,
        "base_price": p.base_price,
        "currency": p.currency,
        "status": p.status,
        "pool_level": p.pool_level,
        "category_id": p.category_id,
        "has_variants": p.has_variants,
        "flash_sale_active": p.flash_sale_active,
        "flash_sale_price": p.flash_sale_price,
        "flash_sale_ends_at": str(p.flash_sale_ends_at) if p.flash_sale_ends_at else None,
        "created_at": p.created_at.isoformat() if p.created_at else None,
        "updated_at": p.updated_at.isoformat() if p.updated_at else None,
        "variants": [
            {
                "id": str(v.id),
                "variant_sku": v.variant_sku,
                "price": v.price,
                "stock": v.stock,
                "is_active": v.is_active,
            }
            for v in (p.variants or [])
        ],
    }


# ──────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────

@app.post("/api/v1/products")
async def create_product(
    req: CreateProductRequest,
    session: AsyncSession = Depends(get_db),
):
    """Create a new product — auto-enters L1 pool."""
    data = req.model_dump()
    data["status"] = ProductStatus.DRAFT
    data["pool_level"] = PoolLevel.L1
    if "vendor_id" in data and isinstance(data["vendor_id"], str):
        data["vendor_id"] = uuid.UUID(data["vendor_id"])

    product = await _product_repo.create(session, data)
    return {"product_id": str(product.id), "status": "created", "pool_level": "L1"}


@app.get("/api/v1/products/{product_id}")
async def get_product(
    product_id: str,
    session: AsyncSession = Depends(get_db),
):
    product = await _product_repo.get_by_id(session, product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return _fmt_product(product)


@app.patch("/api/v1/products/{product_id}")
async def update_product(
    product_id: str,
    req: UpdateProductRequest,
    session: AsyncSession = Depends(get_db),
):
    product = await _product_repo.get_by_id(session, product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    updates = req.model_dump(exclude_unset=True)
    product = await _product_repo.update(session, product_id, updates)
    return _fmt_product(product)


@app.delete("/api/v1/products/{product_id}")
async def delete_product(
    product_id: str,
    session: AsyncSession = Depends(get_db),
):
    product = await _product_repo.get_by_id(session, product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    await _product_repo.soft_delete(session, product_id)
    return {"status": "deleted"}


@app.post("/api/v1/products/{product_id}/variants")
async def add_variant(
    product_id: str,
    req: AddVariantRequest,
    session: AsyncSession = Depends(get_db),
):
    product = await _product_repo.get_by_id(session, product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    data = req.model_dump()
    data["product_id"] = uuid.UUID(product_id)
    variant = await _variant_repo.create(session, data)

    await _product_repo.update(session, product_id, {"has_variants": True})
    return {
        "id": str(variant.id),
        "product_id": product_id,
        "variant_sku": variant.variant_sku,
        "price": variant.price,
        "stock": variant.stock,
    }


@app.post("/api/v1/products/{product_id}/publish")
async def publish_product(
    product_id: str,
    session: AsyncSession = Depends(get_db),
):
    """Publish product — enters moderation queue then L1 pool."""
    product = await _product_repo.get_by_id(session, product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    await _product_repo.update(session, product_id, {
        "status": ProductStatus.PENDING_MODERATION,
        "published_at": datetime.now(timezone.utc).isoformat(),
    })
    return {"status": "pending_moderation", "pool_level": "L1"}


@app.post("/api/v1/products/{product_id}/flash-sale")
async def activate_flash_sale(
    product_id: str,
    req: FlashSaleRequest,
    session: AsyncSession = Depends(get_db),
):
    """Activate flash sale — triggers L6 pool boost."""
    product = await _product_repo.get_by_id(session, product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    ends_at = (datetime.now(timezone.utc) + timedelta(hours=req.duration_hours)).isoformat()
    await _product_repo.update(session, product_id, {
        "flash_sale_active": True,
        "flash_sale_price": req.flash_sale_price,
        "flash_sale_ends_at": ends_at,
    })
    return {"status": "flash_sale_active", "ends_at": ends_at}


@app.get("/api/v1/products")
async def list_products(
    vendor_id: Optional[str] = None,
    status: Optional[str] = None,
    category_id: Optional[int] = None,
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, le=50),
    session: AsyncSession = Depends(get_db),
):
    """List products with filters and pagination."""
    if not vendor_id:
        return ProductListResponse(products=[], total=0, page=page, limit=limit)

    products, total = await _product_repo.list_by_vendor(
        session,
        vendor_id=vendor_id,
        status=status,
        category_id=category_id,
        offset=(page - 1) * limit,
        limit=limit,
    )
    return ProductListResponse(
        products=[_fmt_product(p) for p in products],
        total=total,
        page=page,
        limit=limit,
    )
