"""Pydantic schemas for Moderation Service — Section 38."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class ModerationResult(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    HUMAN_REVIEW = "human_review"


class ModerationRequest(BaseModel):
    product_id: str
    vendor_id: str
    title: str
    description: str = ""
    image_urls: list[str] = []
    category_id: int = 0


class ModerationResponse(BaseModel):
    product_id: str
    result: str
    reasons: list[str] = []
    cv_score: float = 0.0
    category_match: bool = True
    phone_detected: bool = False
    is_nsfw: bool = False
