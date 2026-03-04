"""Content Safety Pipeline — NumberGuard + SightEngine + CLIP — Section 38."""

from __future__ import annotations

import re


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


async def verify_category(image_url: str, category_id: int) -> bool:
    """CLIP zero-shot: does the image match the declared category?

    In production: encode image with CLIP, compare to category text embeddings.
    Reject if cosine similarity < 0.3 (clearly wrong category).
    """
    return True  # Placeholder
