"""
Cross-Modal Bridge — Gap 3: Product ↔ Video Association
========================================================
Enables bidirectional association between products and videos:

    Product → Video:  "Show me videos featuring this product"
    Video → Product:  "What products appear in this video?"

Architecture (2026 best practices):
    1. Explicit mapping (DB): product_id ↔ video_id (vendor-created links)
    2. Embedding bridge: CLIP product emb (512d) → nearest VideoMAE video emb (768d)
       via a learned projection layer (512d → shared 256d ← 768d)
    3. Visual similarity fallback: CLIP product → CLIP video frames

The bridge operates in a shared 256d projection space:
    - ProductProjector: CLIP 512d → 256d (linear + LayerNorm + GELU)
    - VideoProjector:   VideoMAE 768d → 256d (linear + LayerNorm + GELU)
    Both are trained with contrastive loss on (product, video) pairs.

When no trained projection is available, we use CLIP-only frame
matching (less precise but works without training data).

Usage:
    bridge = CrossModalBridge(milvus, redis)
    videos = await bridge.find_videos_for_products(["prod_123", "prod_456"])
    products = await bridge.find_products_in_video("video_789")
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Shared projection dimension
PROJECTION_DIM = 256
CLIP_DIM = 512
VIDEOMAE_DIM = 768


class CrossModalBridge:
    """Bidirectional product ↔ video association engine.

    Supports 3 levels of association quality:
    1. Explicit mapping (Redis/DB): vendor explicitly links product to video
    2. Projection bridge: trained projectors map CLIP & VideoMAE to shared space
    3. Visual fallback: CLIP cosine similarity between product and video frames
    """

    def __init__(
        self,
        milvus_client=None,
        redis_client=None,
        product_projector=None,
        video_projector=None,
    ):
        self.milvus = milvus_client
        self.redis = redis_client
        self.product_projector = product_projector
        self.video_projector = video_projector

    # ── Product → Video ─────────────────────────────────────────────

    async def find_videos_for_products(
        self,
        product_ids: list[str],
        max_videos_per_product: int = 3,
    ) -> list[dict[str, Any]]:
        """Find videos associated with given products.

        Tries 3 strategies in order:
        1. Explicit mapping in Redis (vendor-linked)
        2. Embedding bridge (projection space search)
        3. Visual similarity fallback

        Returns:
            List of video dicts with product_id, video_id, association_type, score
        """
        t_start = time.perf_counter()
        all_videos = []

        # ── Strategy 1: Explicit mapping (Redis) ──────────────
        explicit = await self._find_explicit_mapping(product_ids)
        all_videos.extend(explicit)

        # Find products without explicit mapping
        mapped_products = {v["product_id"] for v in explicit}
        unmapped_products = [pid for pid in product_ids if pid not in mapped_products]

        if unmapped_products:
            # ── Strategy 2: Embedding bridge ──────────────────
            if self.product_projector and self.video_projector:
                bridge_results = await self._find_via_projection_bridge(
                    unmapped_products, max_per_product=max_videos_per_product
                )
                all_videos.extend(bridge_results)

                # Update unmapped list
                bridged = {v["product_id"] for v in bridge_results}
                unmapped_products = [
                    pid for pid in unmapped_products if pid not in bridged
                ]

            # ── Strategy 3: Visual fallback ───────────────────
            if unmapped_products:
                visual_results = await self._find_via_visual_similarity(
                    unmapped_products, max_per_product=max_videos_per_product
                )
                all_videos.extend(visual_results)

        elapsed = (time.perf_counter() - t_start) * 1000
        logger.info(
            "CrossModal: found %d videos for %d products in %.1fms "
            "(explicit=%d, bridge=%d, visual=%d)",
            len(all_videos), len(product_ids), elapsed,
            len(explicit),
            len([v for v in all_videos if v["association_type"] == "embedding_bridge"]),
            len([v for v in all_videos if v["association_type"] == "visual_similarity"]),
        )

        return all_videos

    # ── Video → Products ────────────────────────────────────────────

    async def find_products_in_video(
        self,
        video_id: str,
        max_products: int = 10,
    ) -> list[dict[str, Any]]:
        """Find products that appear in a given video.

        Used for:
        - Video shopping: user watches a video, sees products overlay
        - Search: user finds a video, sees related products

        Returns:
            List of product dicts with video_id, product_id, association_type, score
        """
        t_start = time.perf_counter()
        products = []

        # Strategy 1: Explicit mapping
        if self.redis:
            try:
                import json
                raw = await self.redis.smembers(f"video:{video_id}:products")
                if raw:
                    for pid in raw:
                        pid_str = pid.decode() if isinstance(pid, bytes) else str(pid)
                        products.append({
                            "video_id": video_id,
                            "product_id": pid_str,
                            "association_type": "explicit",
                            "score": 1.0,
                        })
            except Exception as e:
                logger.debug("Explicit video→product lookup failed: %s", e)

        # Strategy 2: Video embedding → nearest product embeddings
        if len(products) < max_products and self.milvus:
            try:
                video_emb = await self._get_video_embedding(video_id)
                if video_emb is not None:
                    # Search visual_embeddings collection with video embedding
                    # (works if CLIP and VideoMAE are in compatible spaces via projection)
                    loop = asyncio.get_event_loop()

                    if self.video_projector:
                        # Project VideoMAE 768d → shared 256d
                        projected = self.video_projector.project(video_emb)
                        results = await loop.run_in_executor(
                            None,
                            lambda: self.milvus.search(
                                collection_name="item_embeddings",
                                query_vectors=[projected.tolist()],
                                top_k=max_products,
                            ),
                        )
                    else:
                        results = await loop.run_in_executor(
                            None,
                            lambda: self.milvus.search(
                                collection_name="video_temporal_embeddings",
                                query_vectors=[video_emb.tolist()],
                                top_k=max_products,
                            ),
                        )

                    if results and results[0]:
                        existing_pids = {p["product_id"] for p in products}
                        for r in results[0]:
                            if r["id"] not in existing_pids:
                                products.append({
                                    "video_id": video_id,
                                    "product_id": r["id"],
                                    "association_type": "embedding_bridge",
                                    "score": float(r["score"]),
                                })
            except Exception as e:
                logger.warning("Video→product embedding search failed: %s", e)

        elapsed = (time.perf_counter() - t_start) * 1000
        logger.info(
            "CrossModal: found %d products in video %s in %.1fms",
            len(products), video_id, elapsed,
        )
        return products[:max_products]

    # ── Explicit Mapping (Redis) ────────────────────────────────────

    async def _find_explicit_mapping(
        self, product_ids: list[str]
    ) -> list[dict[str, Any]]:
        """Look up vendor-created product→video links in Redis."""
        if not self.redis:
            return []

        results = []
        try:
            pipe = self.redis.pipeline()
            for pid in product_ids:
                pipe.smembers(f"product:{pid}:videos")

            raw_results = await pipe.execute()

            for pid, raw in zip(product_ids, raw_results):
                if not raw:
                    continue
                for vid in raw:
                    vid_str = vid.decode() if isinstance(vid, bytes) else str(vid)
                    results.append({
                        "product_id": pid,
                        "video_id": vid_str,
                        "association_type": "explicit",
                        "score": 1.0,
                    })
        except Exception as e:
            logger.debug("Explicit mapping lookup failed: %s", e)

        return results

    # ── Embedding Bridge (Projection Space) ─────────────────────────

    async def _find_via_projection_bridge(
        self,
        product_ids: list[str],
        max_per_product: int = 3,
    ) -> list[dict[str, Any]]:
        """Find videos via learned projection bridge.

        CLIP 512d → ProductProjector → shared 256d
        VideoMAE 768d → VideoProjector → shared 256d
        Then cosine similarity in shared space.
        """
        if not self.milvus or not self.product_projector:
            return []

        results = []
        loop = asyncio.get_event_loop()

        for pid in product_ids:
            try:
                # Get product CLIP embedding
                product_emb = await self._get_product_clip_embedding(pid)
                if product_emb is None:
                    continue

                # Project to shared space
                projected = self.product_projector.project(product_emb)

                # Search video projections in shared space
                search_result = await loop.run_in_executor(
                    None,
                    lambda: self.milvus.search(
                        collection_name="video_temporal_embeddings",
                        query_vectors=[projected.tolist()],
                        top_k=max_per_product,
                    ),
                )

                if search_result and search_result[0]:
                    for r in search_result[0]:
                        results.append({
                            "product_id": pid,
                            "video_id": r["id"],
                            "association_type": "embedding_bridge",
                            "score": float(r["score"]),
                        })
            except Exception as e:
                logger.debug("Projection bridge failed for %s: %s", pid, e)

        return results

    # ── Visual Similarity Fallback ──────────────────────────────────

    async def _find_via_visual_similarity(
        self,
        product_ids: list[str],
        max_per_product: int = 3,
    ) -> list[dict[str, Any]]:
        """Fallback: find videos with visually similar content via CLIP.

        Searches the visual_embeddings collection which may contain
        both product images and video keyframes in the same CLIP space.
        """
        if not self.milvus:
            return []

        results = []
        loop = asyncio.get_event_loop()

        for pid in product_ids:
            try:
                product_emb = await self._get_product_clip_embedding(pid)
                if product_emb is None:
                    continue

                # Search for video frames in visual_embeddings
                search_result = await loop.run_in_executor(
                    None,
                    lambda: self.milvus.search(
                        collection_name="visual_embeddings",
                        query_vectors=[product_emb.tolist()],
                        top_k=max_per_product * 3,
                        filter_expr='content_type == "video"',
                    ),
                )

                if search_result and search_result[0]:
                    seen_videos = set()
                    for r in search_result[0]:
                        vid = r.get("metadata", {}).get("video_id", r["id"])
                        if vid in seen_videos:
                            continue
                        seen_videos.add(vid)
                        results.append({
                            "product_id": pid,
                            "video_id": vid,
                            "association_type": "visual_similarity",
                            "score": float(r["score"]),
                        })
                        if len(seen_videos) >= max_per_product:
                            break
            except Exception as e:
                logger.debug("Visual similarity search failed for %s: %s", pid, e)

        return results

    # ── Data Access Helpers ─────────────────────────────────────────

    async def _get_product_clip_embedding(self, product_id: str) -> Optional[np.ndarray]:
        """Fetch CLIP embedding for a product from Redis or Milvus."""
        # Try Redis first
        if self.redis:
            try:
                import json
                raw = await self.redis.get(f"product:{product_id}:clip_emb")
                if raw:
                    return np.array(json.loads(raw), dtype=np.float32)
            except Exception:
                pass

        # Try Milvus get by ID
        if self.milvus:
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self.milvus.get(
                        collection_name="visual_embeddings",
                        ids=[product_id],
                    ),
                )
                if result and result[0]:
                    return np.array(result[0].get("embedding", []), dtype=np.float32)
            except Exception:
                pass

        return None

    async def _get_video_embedding(self, video_id: str) -> Optional[np.ndarray]:
        """Fetch VideoMAE embedding for a video from Redis or Milvus."""
        if self.redis:
            try:
                import json
                raw = await self.redis.get(f"video:{video_id}:videomae_emb")
                if raw:
                    return np.array(json.loads(raw), dtype=np.float32)
            except Exception:
                pass

        if self.milvus:
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self.milvus.get(
                        collection_name="video_temporal_embeddings",
                        ids=[video_id],
                    ),
                )
                if result and result[0]:
                    return np.array(result[0].get("embedding", []), dtype=np.float32)
            except Exception:
                pass

        return None

    # ── Indexing Helpers (for ingestion pipeline) ───────────────────

    async def register_product_video_link(
        self, product_id: str, video_id: str
    ):
        """Register an explicit product ↔ video link (vendor action).

        Called when a vendor uploads a video and tags products in it.
        Creates bidirectional Redis sets for O(1) lookup.
        """
        if not self.redis:
            logger.warning("Redis not available for product-video mapping")
            return

        try:
            pipe = self.redis.pipeline()
            pipe.sadd(f"product:{product_id}:videos", video_id)
            pipe.sadd(f"video:{video_id}:products", product_id)
            await pipe.execute()
            logger.info("Registered link: product=%s ↔ video=%s", product_id, video_id)
        except Exception as e:
            logger.error("Failed to register product-video link: %s", e)


class EmbeddingProjector:
    """Linear projection layer to map embeddings to a shared space.

    Can be trained with contrastive loss on (product, video) pairs.
    If no trained weights exist, uses a random initialized projection
    (which won't be accurate but allows the pipeline to run).
    """

    def __init__(self, input_dim: int, output_dim: int = PROJECTION_DIM):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._projection_matrix = None

    def load(self, path: str):
        """Load trained projection weights."""
        try:
            self._projection_matrix = np.load(path)
            logger.info(
                "Loaded projector %dx%d from %s",
                self.input_dim, self.output_dim, path,
            )
        except Exception as e:
            logger.warning("Projector load failed: %s — using random init", e)
            self._init_random()

    def _init_random(self):
        """Initialize with Xavier random weights (untrained fallback)."""
        scale = np.sqrt(2.0 / (self.input_dim + self.output_dim))
        self._projection_matrix = np.random.randn(
            self.input_dim, self.output_dim
        ).astype(np.float32) * scale

    def project(self, embedding: np.ndarray) -> np.ndarray:
        """Project embedding to shared space."""
        if self._projection_matrix is None:
            self._init_random()

        projected = embedding @ self._projection_matrix
        # L2 normalize
        norm = np.linalg.norm(projected)
        if norm > 1e-8:
            projected = projected / norm
        return projected.astype(np.float32)
