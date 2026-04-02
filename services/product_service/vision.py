"""
Vision Pipeline — CLIP + Llama Scout — Section 15.

Production pipeline triggered after every product photo upload.
All processing handled by on-premise CLIP + Llama Scout models.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


async def run_vision_pipeline(product_id: str, image_url: str) -> dict:
    """Async CV pipeline — <60s total.

    Steps:
        1. Llama Scout quality scoring → cv_score [0, 1]
        2. CLIP embedding → 512-dim vector
        3. Llama Scout product enrichment → auto description + tags

    In production this runs as a Celery task or Kafka consumer.
    """
    result = {
        "cv_score": 0.0,
        "clip_embedding": None,
        "caption": None,
        "auto_tags": [],
    }

    # 1. Quality score (Llama Scout multimodal vision)
    try:
        from ml.cv.quality_scorer import score_product_photo
        score_result = await score_product_photo(image_url, product_id, "")
        result["cv_score"] = score_result.get("score", 0.5)
    except Exception as e:
        logger.warning("Quality score failed for product=%s: %s", product_id, e)
        result["cv_score"] = 0.5  # Neutral score on failure

    # 2. CLIP embedding (GPU worker)
    try:
        import asyncio
        from ml.cv.clip_encoder import encode_product_image
        loop = asyncio.get_event_loop()
        clip_np = await loop.run_in_executor(
            None, lambda: encode_product_image(image_url, 0),
        )
        result["clip_embedding"] = clip_np.tolist()
    except Exception as e:
        logger.warning("CLIP encode failed for product=%s: %s", product_id, e)

    # 3. Product enrichment (Llama Scout auto-description + tags)
    try:
        from ml.llm.llm_enrichment import enrich_product
        enrichment = await enrich_product(
            product_title=product_id,
            product_image_url=image_url,
        )
        result["caption"] = enrichment.get("seo_description", "")
        result["auto_tags"] = enrichment.get("auto_tags", [])
    except Exception as e:
        logger.warning("LLM enrichment failed for product=%s: %s", product_id, e)

    logger.info(
        "Vision pipeline complete: product=%s, cv_score=%.2f",
        product_id, result["cv_score"],
    )
    return result
