"""Moderation Service — Content Safety Pipeline — Section 38.

Handles:
    - SightEngine API integration (NSFW, violence, quality)
    - CLIP zero-shot category verification
    - NumberGuard: phone/WhatsApp detection in descriptions
    - Automated approval or human review queue
"""

from __future__ import annotations

import logging
import re
from enum import Enum

from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="ShopFeed OS — Moderation Service", version="1.0.0")


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


# ──────────────────────────────────────────────────────────────
# NumberGuard — phone/WhatsApp detection
# ──────────────────────────────────────────────────────────────

# Patterns for phone numbers in various formats
_PHONE_PATTERNS = [
    r"(\+?\d{1,3}[\s-]?\d{2,3}[\s-]?\d{2,3}[\s-]?\d{2,4})",   # International: +225 07 89 12 34
    r"(0[1-9]\d{8})",                                             # French: 0612345678
    r"(\d{2}\s\d{2}\s\d{2}\s\d{2}\s\d{2})",                     # Spaced: 07 89 12 34 56
    r"(whatsapp|watsap|whats\s*app|wa\.me)",                      # WhatsApp keywords
    r"(@gmail|@yahoo|@hotmail|@outlook)",                          # Email to bypass
]


def number_guard(text: str) -> tuple[bool, list[str]]:
    """Detect phone numbers and contact info in product descriptions.

    Returns (has_violation, list_of_matches).
    """
    violations: list[str] = []
    text_lower = text.lower()

    for pattern in _PHONE_PATTERNS:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        if matches:
            violations.extend(matches[:3])  # Cap at 3

    return bool(violations), violations


# ──────────────────────────────────────────────────────────────
# SightEngine stub (would call real API in production)
# ──────────────────────────────────────────────────────────────

async def check_image_safety(image_url: str) -> dict:
    """Call SightEngine API for NSFW + quality check.

    Returns: {is_safe: bool, nsfw_score: float, quality_score: float}
    """
    # In production: httpx.post("https://api.sightengine.com/1.0/check.json", ...)
    return {
        "is_safe": True,
        "nsfw_score": 0.01,
        "quality_score": 0.85,
    }


# ──────────────────────────────────────────────────────────────
# Category verification (CLIP zero-shot)
# ──────────────────────────────────────────────────────────────

async def verify_category(image_url: str, category_id: int) -> bool:
    """CLIP zero-shot: does the image match the declared category?

    In production: encode image with CLIP, compare to category text embeddings.
    Reject if cosine similarity < 0.3 (clearly wrong category).
    """
    return True  # Placeholder


# ──────────────────────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────────────────────

@app.post("/api/v1/moderation/check", response_model=ModerationResponse)
async def moderate_product(req: ModerationRequest):
    """Run full moderation pipeline — <60s.

    Pipeline:
        1. NumberGuard: check description for phone numbers → REJECT
        2. SightEngine: check images for NSFW → REJECT
        3. CLIP: verify category match → HUMAN_REVIEW
        4. All clear → APPROVED
    """
    reasons: list[str] = []
    result = ModerationResult.APPROVED

    # 1. NumberGuard
    phone_detected, phone_matches = number_guard(req.title + " " + req.description)
    if phone_detected:
        reasons.append(f"Phone/contact detected: {phone_matches[:2]}")
        result = ModerationResult.REJECTED

    # 2. Image safety
    cv_scores = []
    is_nsfw = False
    for img_url in req.image_urls[:5]:  # Process max 5 images
        safety = await check_image_safety(img_url)
        cv_scores.append(safety["quality_score"])
        if not safety["is_safe"]:
            is_nsfw = True
            reasons.append(f"NSFW content detected in image")
            result = ModerationResult.REJECTED

    avg_cv = sum(cv_scores) / len(cv_scores) if cv_scores else 0.5

    # 3. Category verification
    category_match = True
    if req.image_urls:
        category_match = await verify_category(req.image_urls[0], req.category_id)
        if not category_match:
            reasons.append("Image doesn't match declared category")
            if result != ModerationResult.REJECTED:
                result = ModerationResult.HUMAN_REVIEW

    logger.info(
        "Moderation: product=%s, result=%s, reasons=%s",
        req.product_id, result, reasons,
    )

    return ModerationResponse(
        product_id=req.product_id,
        result=result,
        reasons=reasons,
        cv_score=avg_cv,
        category_match=category_match,
        phone_detected=phone_detected,
        is_nsfw=is_nsfw,
    )
