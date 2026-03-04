"""User Service — Auth + Profiles + Follows — Section 05 / 40.

Handles:
    - User registration & authentication
    - Buyer profiles (Intent Graph)
    - Vendor profiles & tier management
    - Follow/unfollow relationships
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from shared.models.user import User, UserProfile, Persona, IntentLevel
from shared.models.vendor import Vendor, VendorTier

logger = logging.getLogger(__name__)

app = FastAPI(title="ShopFeed OS — User Service", version="1.0.0")

# In-memory storage
_users: dict[str, dict] = {}
_profiles: dict[str, dict] = {}
_vendors: dict[str, dict] = {}
_follows: dict[str, set[str]] = {}  # user_id → {vendor_ids}


# ──────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────

class CreateUserRequest(BaseModel):
    email: str
    full_name: str = ""
    phone: str | None = None
    country: str = "CI"
    city: str | None = None


class CreateVendorRequest(BaseModel):
    user_id: str
    shop_name: str
    description: str = ""
    city: str = ""
    commune: str = ""
    country: str = "CI"


class UpdateProfileRequest(BaseModel):
    category_prefs: dict[str, float] | None = None
    price_ranges: dict | None = None
    active_categories: list[str] | None = None


# ──────────────────────────────────────────────────────────────
# User Endpoints
# ──────────────────────────────────────────────────────────────

@app.post("/api/v1/users")
async def create_user(req: CreateUserRequest):
    user_id = str(uuid.uuid4())
    _users[user_id] = {
        "id": user_id,
        "email": req.email,
        "full_name": req.full_name,
        "phone": req.phone,
        "country": req.country,
        "city": req.city,
        "persona": Persona.UNKNOWN,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    # Initialize empty profile
    _profiles[user_id] = {
        "user_id": user_id,
        "category_prefs": {},
        "price_ranges": {},
        "intent_level": IntentLevel.LOW,
        "active_categories": [],
        "persona": Persona.UNKNOWN,
    }
    return {"user_id": user_id, "status": "created"}


@app.get("/api/v1/users/{user_id}")
async def get_user(user_id: str):
    if user_id not in _users:
        raise HTTPException(status_code=404, detail="User not found")
    return _users[user_id]


@app.get("/api/v1/users/{user_id}/profile")
async def get_user_profile(user_id: str):
    """Get buyer Intent Graph profile — Section 05."""
    if user_id not in _profiles:
        raise HTTPException(status_code=404, detail="Profile not found")
    return _profiles[user_id]


@app.patch("/api/v1/users/{user_id}/profile")
async def update_profile(user_id: str, req: UpdateProfileRequest):
    if user_id not in _profiles:
        raise HTTPException(status_code=404, detail="Profile not found")
    updates = req.model_dump(exclude_unset=True)
    _profiles[user_id].update(updates)
    return _profiles[user_id]


# ──────────────────────────────────────────────────────────────
# Vendor Endpoints — Section 06
# ──────────────────────────────────────────────────────────────

@app.post("/api/v1/vendors")
async def create_vendor(req: CreateVendorRequest):
    vendor_id = str(uuid.uuid4())
    _vendors[vendor_id] = {
        "id": vendor_id,
        "user_id": req.user_id,
        "shop_name": req.shop_name,
        "description": req.description,
        "tier": VendorTier.BRONZE,
        "account_weight": 1.0,
        "total_sales": 0,
        "avg_rating": 0.0,
        "city": req.city,
        "commune": req.commune,
        "country": req.country,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return {"vendor_id": vendor_id, "tier": "bronze", "status": "created"}


@app.get("/api/v1/vendors/{vendor_id}")
async def get_vendor(vendor_id: str):
    if vendor_id not in _vendors:
        raise HTTPException(status_code=404, detail="Vendor not found")
    return _vendors[vendor_id]


@app.get("/api/v1/vendors/{vendor_id}/tier")
async def get_vendor_tier(vendor_id: str):
    """Get vendor tier & account weight — Section 06."""
    if vendor_id not in _vendors:
        raise HTTPException(status_code=404, detail="Vendor not found")
    v = _vendors[vendor_id]
    return {
        "vendor_id": vendor_id,
        "tier": v["tier"],
        "account_weight": v["account_weight"],
        "total_sales": v["total_sales"],
    }


# ──────────────────────────────────────────────────────────────
# Follow System — Section 20
# ──────────────────────────────────────────────────────────────

@app.post("/api/v1/users/{user_id}/follow/{vendor_id}")
async def follow_vendor(user_id: str, vendor_id: str):
    """Follow a vendor — adds to 'following' feed."""
    if user_id not in _users:
        raise HTTPException(status_code=404, detail="User not found")
    if vendor_id not in _vendors:
        raise HTTPException(status_code=404, detail="Vendor not found")

    _follows.setdefault(user_id, set()).add(vendor_id)
    return {"status": "followed", "vendor_id": vendor_id}


@app.delete("/api/v1/users/{user_id}/follow/{vendor_id}")
async def unfollow_vendor(user_id: str, vendor_id: str):
    if user_id in _follows:
        _follows[user_id].discard(vendor_id)
    return {"status": "unfollowed"}


@app.get("/api/v1/users/{user_id}/following")
async def get_following(user_id: str):
    vendor_ids = list(_follows.get(user_id, set()))
    return {"user_id": user_id, "following": vendor_ids, "count": len(vendor_ids)}
