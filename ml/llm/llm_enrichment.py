"""
LLM Enrichment — 7 high-value tasks powered by Llama 4 Scout
================================================================
ALL tasks use Llama 4 Scout (17B active / 109B total MoE, multimodal).
Single model on 1× A100 80GB handles vision, text gen, AND reasoning.

Functions:
    1. score_photo_quality()        — Vision: multimodal quality scorer
    2. score_ad_creative()          — Vision: ad creative quality for EPSILON
    3. enrich_product()             — Vision: auto SEO + tags + attributes
    4. generate_ad_copy()           — Text: ad copy variants
    5. conversational_search()      — Text: NL query → structured filters
    6. explain_moderation()         — Vision: why product was flagged
    7. generate_vendor_insights()   — Reasoning: campaign strategy
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _get_router():
    from ml.llm import get_router
    return get_router()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1. PHOTO QUALITY SCORING (Scout vision)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def score_photo_quality(
    image_url: str,
    product_title: str = "",
    category: str = "",
) -> dict[str, Any]:
    """Score product photo quality using Llama 4 Scout multimodal vision.

    Returns enriched quality data:
        - score, sharpness, lighting, framing (standard metrics)
        - composition_score, brand_coherence (advanced analysis)
        - styling_advice, category_fit (actionable recommendations)
    """
    router = _get_router()

    prompt = f"""Analyze this product photo for e-commerce listing quality.
Product: {product_title or "unknown"}
Category: {category or "general"}

Score each criterion from 0.0 (worst) to 1.0 (best). Be strict — most amateur photos score 0.3-0.6.

Return ONLY valid JSON:
{{
    "score": overall_quality_0_to_1,
    "sharpness": focus_clarity_0_to_1,
    "lighting": even_lighting_quality_0_to_1,
    "framing": subject_centered_margins_0_to_1,
    "background_clean": true_if_clean_background,
    "text_overlay_detected": true_if_text_watermarks_visible,
    "face_detected": true_if_human_face_present,
    "composition_score": visual_balance_rule_of_thirds_0_to_1,
    "brand_coherence": matches_professional_ecommerce_standards_0_to_1,
    "recommendations": ["specific actionable tip 1", "tip 2"],
    "styling_advice": "one sentence professional improvement advice",
    "category_fit": "how well photo matches category standard (one sentence)"
}}"""

    result = await router.vision_json(
        prompt=prompt,
        image_url=image_url,
        system_prompt=(
            "You are a strict product photography quality assessor for a premium "
            "e-commerce platform. Analyze images with professional precision. "
            "Score conservatively. Always respond with valid JSON only."
        ),
        max_tokens=400,
    )

    if not result or not isinstance(result, dict):
        return _default_quality_score()

    # Ensure all required fields exist with safe defaults
    result.setdefault("score", 0.5)
    result.setdefault("sharpness", 0.5)
    result.setdefault("lighting", 0.5)
    result.setdefault("framing", 0.5)
    result.setdefault("background_clean", True)
    result.setdefault("text_overlay_detected", False)
    result.setdefault("face_detected", False)
    result.setdefault("composition_score", 0.5)
    result.setdefault("brand_coherence", 0.5)
    result.setdefault("recommendations", [])
    result.setdefault("styling_advice", "")
    result.setdefault("category_fit", "")

    # Clamp numeric values
    for key in ["score", "sharpness", "lighting", "framing", "composition_score", "brand_coherence"]:
        val = result.get(key, 0.5)
        result[key] = round(max(0.0, min(1.0, float(val))), 4)

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2. AD CREATIVE QUALITY (Scout vision) — for EPSILON ad_ranker
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def score_ad_creative(
    image_url: str,
    ad_title: str = "",
    target_audience: str = "",
) -> dict[str, float]:
    """Score ad creative quality for EPSILON eCPM calculation.

    Feeds directly into AdScore.creative_quality and AdScore.landing_page_score.
    Higher creative quality = lower CPC in GSP auction.
    """
    router = _get_router()

    prompt = f"""Rate this ad creative for social commerce performance.
Ad title: {ad_title or "untitled"}
Target audience: {target_audience or "general"}

Score 0.0 (terrible) to 1.0 (perfect):
Return ONLY valid JSON:
{{
    "creative_quality": overall_ad_quality_0_to_1,
    "visual_appeal": eye_catching_scroll_stopping_0_to_1,
    "clarity": product_clearly_visible_0_to_1,
    "cta_strength": call_to_action_compelling_0_to_1,
    "brand_safety": no_offensive_misleading_content_0_to_1
}}"""

    result = await router.vision_json(
        prompt=prompt,
        image_url=image_url,
        system_prompt=(
            "You are an ad creative quality assessor for a social commerce platform. "
            "Score strictly — most ad creatives score 0.4-0.7. "
            "Always respond with valid JSON only."
        ),
        max_tokens=200,
    )

    if not result or not isinstance(result, dict):
        return {"creative_quality": 0.5, "visual_appeal": 0.5, "clarity": 0.5, "cta_strength": 0.5, "brand_safety": 0.9}

    for key in ["creative_quality", "visual_appeal", "clarity", "cta_strength", "brand_safety"]:
        val = result.get(key, 0.5)
        result[key] = round(max(0.0, min(1.0, float(val))), 4)

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3. PRODUCT ENRICHMENT (Scout vision) — auto SEO + tags
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def enrich_product(
    product_title: str,
    product_image_url: str,
    category_name: str = "",
    vendor_description: str = "",
) -> dict[str, Any]:
    """Enrich product listing from image using Scout vision.

    Outputs: SEO description, auto-tags, attribute extraction,
    search keywords, suggested price tier.
    """
    router = _get_router()

    prompt = f"""Analyze this product image and generate enriched listing data.
Title: {product_title}
Category: {category_name}
Vendor notes: {vendor_description}

Return ONLY valid JSON:
{{
    "seo_description": "200-char SEO-optimized description based on what you SEE",
    "auto_tags": ["tag1", "tag2", ...],
    "attributes": {{"color": "...", "material": "...", "style": "...", "occasion": "...", "season": "..."}},
    "search_keywords": ["keyword1", "keyword2", ...],
    "suggested_price_tier": "budget|mid|premium|luxury"
}}"""

    result = await router.vision_json(
        prompt=prompt,
        image_url=product_image_url,
        system_prompt=(
            "You are a product listing expert. Analyze images precisely. "
            "Never hallucinate details not visible in the image. "
            "Always respond with valid JSON only."
        ),
        max_tokens=500,
    )

    return result if isinstance(result, dict) else {}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  4. AD COPY GENERATION (Scout text) — fast ad copy
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def generate_ad_copy(
    product_title: str,
    product_description: str,
    target_audience: str = "general",
    vendor_name: str = "",
    num_variants: int = 5,
) -> list[dict[str, str]]:
    """Generate ad copy variants for EPSILON campaigns.

    Returns [{headline, body, cta, tone}, ...]
    """
    router = _get_router()

    prompt = f"""Generate {num_variants} ad copy variants for this product.
Each variant uses a different tone: luxury, urgent, friendly, informative, playful.

Product: {product_title}
Description: {product_description}
Vendor: {vendor_name}
Target: {target_audience}

Return ONLY a valid JSON array:
[{{"headline": "max 40 chars", "body": "max 120 chars", "cta": "action text", "tone": "luxury|urgent|friendly|informative|playful"}}]"""

    result = await router.text_json(
        prompt=prompt,
        system_prompt=(
            "You are an expert social commerce ad copywriter. "
            "Write compelling copy that drives clicks AND vendor store visits. "
            "Always respond with a valid JSON array only."
        ),
        max_tokens=800,
    )

    return result if isinstance(result, list) else []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  5. CONVERSATIONAL SEARCH (Scout text) — NL → filters
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def conversational_search(
    user_query: str,
    available_categories: list[str] | None = None,
) -> dict[str, Any]:
    """Convert natural language search to structured product filters.

    "robe d'ete pour un mariage, soie, moins de 200EUR"
    → {category: "robes", attributes: {season: "ete", material: "soie"}, price_range: {max: 200}}
    """
    router = _get_router()
    cats = ", ".join(available_categories or [
        "fashion", "beauty", "electronics", "home", "food",
        "sports", "baby", "automotive", "health",
    ])

    prompt = f"""Convert this search query into structured product filters.
Query: "{user_query}"
Categories: {cats}

Return ONLY valid JSON:
{{
    "category": "best match",
    "attributes": {{"key": "value"}},
    "price_range": {{"min": null, "max": null}},
    "sort_by": "relevance|price_asc|price_desc|newest",
    "keywords": ["term1", "term2"],
    "intent": "browse|compare|buy"
}}"""

    result = await router.text_json(
        prompt=prompt,
        system_prompt=(
            "You are a search query parser for e-commerce. "
            "Extract structured filters from natural language. "
            "Always respond with valid JSON only."
        ),
        max_tokens=300,
    )

    if not isinstance(result, dict):
        return {"keywords": user_query.split(), "intent": "browse"}
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  6. MODERATION EXPLANATION (Scout vision) — transparency
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def explain_moderation(
    product_title: str,
    image_url: str,
    moderation_scores: dict[str, float],
    flagged_reasons: list[str],
) -> dict[str, Any]:
    """Explain WHY a product was flagged using Scout vision.

    Provides transparent, vendor-friendly explanations.
    """
    router = _get_router()
    scores_str = ", ".join(f"{k}: {v:.2f}" for k, v in moderation_scores.items())

    prompt = f"""A product was flagged by content moderation. Analyze the image and explain why.
Product: {product_title}
Scores: {scores_str}
Flagged for: {', '.join(flagged_reasons)}

Return ONLY valid JSON:
{{
    "explanation": "clear 2-sentence explanation of why flagged",
    "severity": "low|medium|high|critical",
    "appeal_recommendation": "approve|needs_review|reject",
    "suggested_fixes": ["fix 1", "fix 2"],
    "false_positive_likelihood": 0.0_to_1.0
}}"""

    result = await router.vision_json(
        prompt=prompt,
        image_url=image_url,
        system_prompt=(
            "You are a fair content moderation reviewer. "
            "Analyze objectively. Give vendors actionable fixes. "
            "Always respond with valid JSON only."
        ),
        max_tokens=300,
    )

    if not isinstance(result, dict):
        return {"explanation": "Manual review required", "severity": "medium", "appeal_recommendation": "needs_review"}
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  7. VENDOR INSIGHTS (Scout reasoning) — campaign strategy
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def generate_vendor_insights(
    vendor_name: str,
    campaign_metrics: dict[str, Any],
    product_count: int = 0,
    avg_cv_score: float = 0.5,
    monthly_views: int = 0,
    conversion_rate: float = 0.0,
) -> str:
    """Generate strategy insights for vendor dashboard.

    Scout's 17B MoE handles complex reasoning for campaign optimization.
    """
    router = _get_router()

    prompt = f"""Analyze this vendor's performance and provide specific recommendations
to DOUBLE their store traffic.

Store: {vendor_name}
Products: {product_count}
Photo quality: {avg_cv_score:.2f}/1.0
Monthly views: {monthly_views:,}
Conversion rate: {conversion_rate:.1%}
Campaign data: {json.dumps(campaign_metrics, indent=2)}

Provide:
1. Store health (2 sentences)
2. Top 3 quick wins for immediate traffic boost
3. Photo quality plan (if score < 0.7)
4. Recommended EPSILON ad plan (STARTER/GROWTH/PREMIUM)

Keep under 300 words, vendor-friendly language."""

    return await router.reason(
        prompt=prompt,
        system_prompt=(
            "You are an e-commerce growth strategist. "
            "Give specific, actionable advice tied to concrete metrics."
        ),
        max_tokens=600,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _default_quality_score() -> dict[str, Any]:
    """Default quality score when Scout is unavailable."""
    return {
        "score": 0.5,
        "sharpness": 0.5,
        "lighting": 0.5,
        "framing": 0.5,
        "background_clean": True,
        "text_overlay_detected": False,
        "face_detected": False,
        "composition_score": 0.5,
        "brand_coherence": 0.5,
        "recommendations": [],
        "styling_advice": "",
        "category_fit": "",
    }
