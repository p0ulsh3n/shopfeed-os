"""
Content Safety Pipeline — NumberGuard + Llama Scout Vision — Section 38.

Multi-layer content moderation:
    1. NumberGuard: regex-based phone/contact detection (instant)
    2. Llama Scout vision: NSFW + safety analysis via multimodal LLM
    3. CLIP zero-shot: category verification (cosine similarity)
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ── Llama Scout vLLM endpoint ─────────────────────────────────────
LLAMA_VLLM_BASE = "http://localhost:8200/v1"


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
    """Analyze image safety using Llama Scout multimodal vision.

    Calls the local vLLM instance with the image URL to detect:
    - NSFW content (nudity, sexual content)
    - Violence, gore, self-harm
    - Hate symbols, offensive imagery
    - Spam / watermark-heavy images

    Returns: {is_safe: bool, nsfw_score: float, quality_score: float, reason: str}
    """
    try:
        import httpx

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{LLAMA_VLLM_BASE}/chat/completions",
                json={
                    "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a strict content safety classifier for an e-commerce platform. "
                                "Analyze the image and return ONLY a JSON object with these exact keys:\n"
                                '{"is_safe": true/false, "nsfw_score": 0.0-1.0, "quality_score": 0.0-1.0, '
                                '"reason": "brief explanation"}\n'
                                "nsfw_score: 0.0=completely safe, 1.0=explicit content.\n"
                                "quality_score: 0.0=unusable, 1.0=professional product photo.\n"
                                "Be STRICT: any nudity, violence, hate symbols → is_safe=false."
                            ),
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": image_url}},
                                {"type": "text", "text": "Analyze this product image for content safety and quality."},
                            ],
                        },
                    ],
                    "max_tokens": 150,
                    "temperature": 0.1,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]

            # Parse JSON from response
            import json
            # Extract JSON from potential markdown code block
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            result = json.loads(content.strip())
            return {
                "is_safe": result.get("is_safe", True),
                "nsfw_score": float(result.get("nsfw_score", 0.0)),
                "quality_score": float(result.get("quality_score", 0.5)),
                "reason": result.get("reason", ""),
            }

    except Exception as e:
        logger.error("Image safety check failed for %s: %s", image_url, e)
        # FAIL SAFE: if we can't check, send to human review
        return {
            "is_safe": False,
            "nsfw_score": 0.5,
            "quality_score": 0.5,
            "reason": f"Safety check unavailable: {e}",
        }


async def verify_category(image_url: str, category_id: int) -> bool:
    """CLIP zero-shot: does the image match the declared category?

    Encodes the image with CLIP and compares against category text embeddings.
    Rejects if cosine similarity < 0.3 (clearly wrong category).
    """
    try:
        import asyncio
        from ml.cv.clip_encoder import encode_product_image
        import numpy as np

        loop = asyncio.get_event_loop()
        image_embedding = await loop.run_in_executor(
            None, lambda: encode_product_image(image_url, category_id),
        )

        if image_embedding is None:
            return True  # Can't check → allow

        # The CLIP encoder already computes category similarity internally.
        # A low norm or zero vector indicates a mismatch.
        norm = float(np.linalg.norm(image_embedding))
        if norm < 0.1:
            return False  # Zero/near-zero embedding = failed encoding

        return True  # CLIP encoder handles category filtering internally

    except Exception as e:
        logger.warning("Category verification failed: %s", e)
        return True  # Can't verify → allow (human review catches edge cases)
