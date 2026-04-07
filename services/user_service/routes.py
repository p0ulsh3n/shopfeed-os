"""
User Service — FastAPI App + Routes — Section 05 / 06 / 20 / 40.

Migration: dicts Python in-memory (_users, _vendors, _follows)
→ SQLAlchemy 2.0 ORM via UserRepository + VendorRepository + UserFollowRepository.
Persistance réelle en PostgreSQL.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from shared.db.session import get_db
from shared.models.user import IntentLevel, Persona
from shared.models.vendor import VendorTier
from shared.repositories.user_repository import (
    UserFollowRepository,
    UserProfileRepository,
    UserRepository,
)
from shared.repositories.vendor_repository import VendorRepository

from .schemas import CreateUserRequest, CreateVendorRequest, UpdateProfileRequest

logger = logging.getLogger(__name__)

app = FastAPI(title="ShopFeed OS — User Service", version="1.0.0")

_user_repo = UserRepository()
_profile_repo = UserProfileRepository()
_vendor_repo = VendorRepository()
_follow_repo = UserFollowRepository()


# ──────────────────────────────────────────────────────────────
# User Endpoints
# ──────────────────────────────────────────────────────────────

@app.post("/api/v1/users")
async def create_user(
    req: CreateUserRequest,
    session: AsyncSession = Depends(get_db),
):
    user = await _user_repo.create(session, {
        "email": req.email,
        "full_name": req.full_name,
        "phone": req.phone,
        "country": req.country or "",
        "city": req.city,
        "persona": Persona.UNKNOWN,
    })

    # Créer un profil ML vide associé
    await _profile_repo.upsert(session, user.id, {
        "category_prefs": {},
        "price_ranges": {},
        "intent_level": IntentLevel.LOW,
        "active_categories": [],
        "persona": Persona.UNKNOWN,
    })

    return {"user_id": str(user.id), "status": "created"}


@app.get("/api/v1/users/{user_id}")
async def get_user(
    user_id: str,
    session: AsyncSession = Depends(get_db),
):
    user = await _user_repo.get_by_id(session, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "id": str(user.id),
        "email": user.email,
        "full_name": user.full_name,
        "phone": user.phone,
        "country": user.country,
        "city": user.city,
        "persona": user.persona,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


@app.get("/api/v1/users/{user_id}/profile")
async def get_user_profile(
    user_id: str,
    session: AsyncSession = Depends(get_db),
):
    """Get buyer Intent Graph profile — Section 05."""
    profile = await _profile_repo.get(session, user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return {
        "user_id": str(profile.user_id),
        "category_prefs": profile.category_prefs,
        "price_ranges": profile.price_ranges,
        "intent_level": profile.intent_level,
        "active_categories": profile.active_categories,
        "persona": profile.persona,
    }


@app.patch("/api/v1/users/{user_id}/profile")
async def update_profile(
    user_id: str,
    req: UpdateProfileRequest,
    session: AsyncSession = Depends(get_db),
):
    profile = await _profile_repo.get(session, user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    updates = req.model_dump(exclude_unset=True)
    await _profile_repo.upsert(session, user_id, updates)
    profile = await _profile_repo.get(session, user_id)
    return {
        "user_id": str(profile.user_id),
        "category_prefs": profile.category_prefs,
        "intent_level": profile.intent_level,
    }


# ──────────────────────────────────────────────────────────────
# Vendor Endpoints — Section 06
# ──────────────────────────────────────────────────────────────

@app.post("/api/v1/vendors")
async def create_vendor(
    req: CreateVendorRequest,
    session: AsyncSession = Depends(get_db),
):
    vendor = await _vendor_repo.create(session, {
        "user_id": uuid.UUID(req.user_id) if req.user_id else uuid.uuid4(),
        "shop_name": req.shop_name,
        "description": req.description or "",
        "tier": VendorTier.BRONZE,
        "account_weight": 1.0,
        "total_sales": 0,
        "avg_rating": 0.0,
        "geo_city": req.city,
        "geo_commune": getattr(req, "commune", None),
        "geo_country": req.country or "",
    })
    return {"vendor_id": str(vendor.id), "tier": "bronze", "status": "created"}


@app.get("/api/v1/vendors/{vendor_id}")
async def get_vendor(
    vendor_id: str,
    session: AsyncSession = Depends(get_db),
):
    vendor = await _vendor_repo.get_by_id(session, vendor_id)
    if not vendor:
        raise HTTPException(status_code=404, detail="Vendor not found")
    return {
        "id": str(vendor.id),
        "user_id": str(vendor.user_id),
        "shop_name": vendor.shop_name,
        "description": vendor.description,
        "tier": vendor.tier,
        "account_weight": vendor.account_weight,
        "total_sales": vendor.total_sales,
        "avg_rating": vendor.avg_rating,
        "city": vendor.geo_city,
        "country": vendor.geo_country,
        "created_at": vendor.created_at.isoformat() if vendor.created_at else None,
    }


@app.get("/api/v1/vendors/{vendor_id}/tier")
async def get_vendor_tier(
    vendor_id: str,
    session: AsyncSession = Depends(get_db),
):
    """Get vendor tier & account weight — Section 06."""
    vendor = await _vendor_repo.get_by_id(session, vendor_id)
    if not vendor:
        raise HTTPException(status_code=404, detail="Vendor not found")
    return {
        "vendor_id": vendor_id,
        "tier": vendor.tier,
        "account_weight": vendor.account_weight,
        "total_sales": vendor.total_sales,
    }


# ──────────────────────────────────────────────────────────────
# Follow System — Section 20
# ──────────────────────────────────────────────────────────────

@app.post("/api/v1/users/{user_id}/follow/{vendor_id}")
async def follow_vendor(
    user_id: str,
    vendor_id: str,
    session: AsyncSession = Depends(get_db),
):
    """Follow a vendor — adds to 'following' feed."""
    user = await _user_repo.get_by_id(session, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    vendor = await _vendor_repo.get_by_id(session, vendor_id)
    if not vendor:
        raise HTTPException(status_code=404, detail="Vendor not found")

    await _follow_repo.follow(session, user_id, vendor_id)
    return {"status": "followed", "vendor_id": vendor_id}


@app.delete("/api/v1/users/{user_id}/follow/{vendor_id}")
async def unfollow_vendor(
    user_id: str,
    vendor_id: str,
    session: AsyncSession = Depends(get_db),
):
    await _follow_repo.unfollow(session, user_id, vendor_id)
    return {"status": "unfollowed"}


@app.get("/api/v1/users/{user_id}/following")
async def get_following(
    user_id: str,
    session: AsyncSession = Depends(get_db),
):
    vendor_ids = await _follow_repo.get_following(session, user_id)
    return {
        "user_id": user_id,
        "following": [str(v) for v in vendor_ids],
        "count": len(vendor_ids),
    }
