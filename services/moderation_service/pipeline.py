"""Content Safety Pipeline — NumberGuard + SightEngine + CLIP — Section 38."""

from __future__ import annotations

import re


# Patterns for phone numbers in various global formats
_PHONE_PATTERNS = [
    # International with country code: +225 07 89 12 34, +1-555-123-4567
    r"(\+?\d{1,3}[\s\-.]?\d{2,4}[\s\-.]?\d{2,4}[\s\-.]?\d{2,4})",
    # French: 06/07 + 8 digits
    r"(0[1-9]\d{8})",
    # Spaced FR: 07 89 12 34 56
    r"(\d{2}\s\d{2}\s\d{2}\s\d{2}\s\d{2})",
    # US/Canada: (555) 123-4567 or 555.123.4567
    r"(\(\d{3}\)\s?\d{3}[\s\-.]?\d{4})",
    # UK: 07xxx xxxxxx
    r"(07\d{3}\s?\d{6})",
    # Nigeria/Ghana: 080x xxx xxxx, 024x xxx xxxx
    r"(0[2-9]\d{1,2}\s?\d{3}\s?\d{4})",
    # Arabic numerals phone (٠١٢٣٤٥٦٧٨٩)
    r"([٠-٩]{10,11})",
    # Social media bypass keywords (all languages)
    r"(whatsapp|watsap|whats\s*app|wa\.me|watsapp)",
    r"(telegram|t\.me|telegrm|telegrame)",
    r"(signal|viber|imo\b)",
    r"(wechat|weixin|微信)",
    r"(instagram\s*dm|insta\s*dm|snap\s*chat|snapchat)",
    r"(appelle[z]?\s*moi|call\s*me|اتصل\s*بي|contactez)",
    # Email bypass
    r"(@gmail|@yahoo|@hotmail|@outlook|@proton|@icloud|@aol)",
    # Obfuscated numbers: zero-sept-huit or z3r0
    r"(zero|un|deux|trois|quatre|cinq|six|sept|huit|neuf)[\s\-]+"
    r"(zero|un|deux|trois|quatre|cinq|six|sept|huit|neuf)",
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
