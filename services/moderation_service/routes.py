"""Moderation Service — FastAPI App + Routes — Section 38."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from .pipeline import check_image_safety, number_guard, verify_category
from .schemas import ModerationRequest, ModerationResponse, ModerationResult

logger = logging.getLogger(__name__)

app = FastAPI(title="ShopFeed OS — Moderation Service", version="1.0.0")


@app.post("/api/v1/moderation/check", response_model=ModerationResponse)
async def moderate_product(req: ModerationRequest):
    """Run full moderation pipeline — <60s.

    Pipeline:
        1. NumberGuard: check description for phone numbers → REJECT
        2. Llama Scout vision: check images for NSFW → REJECT
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
