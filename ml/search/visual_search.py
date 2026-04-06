"""
Visual Search Pipeline — Gap 1: Image upload → similar products + videos
=========================================================================
Full pipeline for visual product search:

    User uploads image
      → CLIP encode (512d, FashionSigLIP or ViT-B/32)
      → Category prediction (CategoryRouter, Gap 5)
      → Milvus ANN search on visual_embeddings (pre-filtered by category)
      → Enrich candidates with business metadata
      → LambdaMART re-rank (Gap 4) with personalization
      → Cross-modal enrichment (Gap 3) — attach associated videos
      → Response with products + associated videos

Best practices 2026 (verified):
    - Signed URL pattern for image uploads (no base64 in JSON)
    - Embedding cache in Redis (TTL 5min) to avoid recomputation
    - Pre-filter by predicted category before ANN (Pailitao pattern)
    - L2-normalized embeddings + cosine metric for visual similarity
    - Parallel metadata fetch while scoring
    - SLA target: <200ms total pipeline
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class VisualSearchPipeline:
    """End-to-end visual search: image → similar products + videos.

    Connects the existing CLIP encoder, FAISS/Milvus index, and
    the new LambdaMART reranker + cross-modal bridge.

    Usage:
        pipeline = VisualSearchPipeline(milvus_client, reranker, cross_modal, category_router)
        results = await pipeline.search(
            image_url="https://cdn.shopfeed.com/uploads/query.jpg",
            user_profile=user_profile,
            category_filter=None,  # auto-predict
            limit=30,
        )
    """

    def __init__(
        self,
        milvus_client=None,
        faiss_index=None,
        reranker=None,
        cross_modal=None,
        category_router=None,
        redis_client=None,
    ):
        self.milvus = milvus_client
        self.faiss = faiss_index
        self.reranker = reranker
        self.cross_modal = cross_modal
        self.category_router = category_router
        self.redis = redis_client
        self._embedding_cache: dict[str, list[float]] = {}

    async def search(
        self,
        image_url: str,
        user_profile: dict[str, Any] | None = None,
        category_filter: int | None = None,
        limit: int = 30,
        include_videos: bool = True,
    ) -> dict[str, Any]:
        """Execute full visual search pipeline.

        Args:
            image_url: URL of the query image
            user_profile: user profile for personalization
            category_filter: explicit category filter (None = auto-predict)
            limit: max products to return
            include_videos: whether to attach associated videos

        Returns:
            {
                "products": [...],
                "associated_videos": [...],
                "predicted_category": int | None,
                "pipeline_ms": float,
            }
        """
        t_start = time.perf_counter()

        # ── Step 1: Encode query image → CLIP 512d ──────────────
        query_embedding = await self._encode_image(image_url)
        if query_embedding is None:
            return self._empty_response(time.perf_counter() - t_start)

        # ── Step 2: Category prediction (Gap 5) ─────────────────
        filter_expr = None
        predicted_category = None
        if self.category_router and category_filter is None:
            filter_expr = self.category_router.predict_from_image(image_url)
            if filter_expr:
                # Extract category IDs from filter for response
                try:
                    import re
                    nums = re.findall(r'\d+', filter_expr)
                    predicted_category = int(nums[0]) if nums else None
                except Exception:
                    pass
        elif category_filter is not None:
            filter_expr = f"category_id == {category_filter}"
            predicted_category = category_filter

        # ── Step 3: ANN search (Milvus or FAISS fallback) ───────
        ann_k = min(limit * 5, 500)  # Retrieve 5x for re-ranking headroom
        raw_candidates = await self._ann_search(
            query_embedding, top_k=ann_k, filter_expr=filter_expr
        )

        if not raw_candidates:
            # Retry without category filter if no results
            if filter_expr:
                logger.info("Visual search: no results with filter, retrying full index")
                raw_candidates = await self._ann_search(
                    query_embedding, top_k=ann_k, filter_expr=None
                )

        if not raw_candidates:
            return self._empty_response(time.perf_counter() - t_start)

        # ── Step 4: Enrich candidates with metadata ─────────────
        candidates = self._enrich_candidates(raw_candidates, query_embedding)

        # ── Step 5: LambdaMART re-ranking (Gap 4) ───────────────
        if self.reranker:
            candidates = self.reranker.rerank(
                candidates,
                query_type="visual",
                user_profile=user_profile,
                limit=limit,
            )
        else:
            # Fallback: sort by visual similarity
            candidates.sort(
                key=lambda x: x.get("visual_similarity", 0), reverse=True
            )
            candidates = candidates[:limit]

        # ── Step 6: Cross-modal enrichment (Gap 3) ──────────────
        associated_videos = []
        if include_videos and self.cross_modal:
            product_ids = [c["item_id"] for c in candidates[:10]]
            associated_videos = await self.cross_modal.find_videos_for_products(
                product_ids
            )

        pipeline_ms = (time.perf_counter() - t_start) * 1000

        logger.info(
            "VisualSearch: %d results in %.1fms (filter=%s, reranked=%s)",
            len(candidates),
            pipeline_ms,
            filter_expr or "none",
            self.reranker is not None,
        )

        return {
            "products": candidates,
            "associated_videos": associated_videos,
            "predicted_category": predicted_category,
            "total_candidates": len(raw_candidates),
            "pipeline_ms": pipeline_ms,
        }

    async def _encode_image(self, image_url: str) -> Optional[np.ndarray]:
        """Encode image via CLIP with caching."""
        cache_key = hashlib.md5(image_url.encode()).hexdigest()[:16]

        # Check in-memory cache
        if cache_key in self._embedding_cache:
            return np.array(self._embedding_cache[cache_key], dtype=np.float32)

        # Check Redis cache
        if self.redis:
            try:
                import json
                cached = await self.redis.get(f"vsearch:emb:{cache_key}")
                if cached:
                    emb = json.loads(cached)
                    self._embedding_cache[cache_key] = emb
                    return np.array(emb, dtype=np.float32)
            except Exception:
                pass

        # Compute embedding
        try:
            import asyncio
            from ml.cv.clip_encoder import encode_product_image

            loop = asyncio.get_event_loop()
            emb = await loop.run_in_executor(
                None, lambda: encode_product_image(image_url, category_id=0)
            )

            if emb is not None and np.any(emb != 0):
                emb_list = emb.tolist()
                self._embedding_cache[cache_key] = emb_list

                # Cache in Redis (TTL 5 minutes)
                if self.redis:
                    try:
                        import json
                        await self.redis.set(
                            f"vsearch:emb:{cache_key}",
                            json.dumps(emb_list),
                            ex=300,
                        )
                    except Exception:
                        pass

                return emb
        except Exception as e:
            logger.error("Visual search encode failed: %s", e)

        return None

    async def _ann_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 200,
        filter_expr: str | None = None,
    ) -> list[dict[str, Any]]:
        """ANN search via Milvus (preferred) or FAISS fallback."""
        import asyncio

        loop = asyncio.get_event_loop()

        # Try Milvus first
        if self.milvus:
            try:
                results = await loop.run_in_executor(
                    None,
                    lambda: self.milvus.search(
                        collection_name="visual_embeddings",
                        query_vectors=[query_embedding.tolist()],
                        top_k=top_k,
                        filter_expr=filter_expr,
                    ),
                )
                if results and results[0]:
                    return [
                        {
                            "item_id": r["id"],
                            "visual_similarity": r["score"],
                            "metadata": r.get("metadata", {}),
                        }
                        for r in results[0]
                    ]
            except Exception as e:
                logger.warning("Milvus visual search failed: %s, trying FAISS", e)

        # FAISS fallback (no filter support)
        if self.faiss:
            try:
                item_ids, scores = await loop.run_in_executor(
                    None,
                    lambda: self.faiss.search(query_embedding, k=top_k),
                )
                return [
                    {"item_id": str(iid), "visual_similarity": float(s), "metadata": {}}
                    for iid, s in zip(item_ids, scores)
                    if iid != -1
                ]
            except Exception as e:
                logger.error("FAISS visual search failed: %s", e)

        return []

    def _enrich_candidates(
        self, raw: list[dict], query_embedding: np.ndarray
    ) -> list[dict[str, Any]]:
        """Enrich raw ANN results with metadata for re-ranking.

        In production, this fetches from catalog DB / Redis.
        Here we extract what's available from Milvus metadata.
        """
        enriched = []
        for r in raw:
            meta = r.get("metadata", {})
            enriched.append({
                "item_id": r["item_id"],
                "visual_similarity": r.get("visual_similarity", 0.0),
                "text_similarity": 0.0,  # Not applicable for visual search
                "category_match": True,  # Assume pre-filtered
                "category_id": meta.get("category_id", 0),
                "vendor_id": meta.get("vendor_id", ""),
                "price": meta.get("price", 0.0),
                "cv_score": meta.get("cv_score", 0.5),
                "total_sold": meta.get("total_sold", 0),
                "review_rating": meta.get("review_rating", 0.0),
                "review_count": meta.get("review_count", 0),
                "vendor_rating": meta.get("vendor_rating", 0.0),
                "avg_category_price": meta.get("avg_category_price", 0.0),
                "freshness": meta.get("freshness", 0.5),
                "pool_level": meta.get("pool_level", "L1"),
                "conversion_rate": meta.get("conversion_rate", 0.01),
                "title": meta.get("title", ""),
                "image_url": meta.get("image_url", ""),
            })
        return enriched

    def _empty_response(self, elapsed: float) -> dict[str, Any]:
        return {
            "products": [],
            "associated_videos": [],
            "predicted_category": None,
            "total_candidates": 0,
            "pipeline_ms": elapsed * 1000,
        }
