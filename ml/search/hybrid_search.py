"""
Hybrid Search Pipeline — Gap 2: BM25 + Vector + RRF Fusion
============================================================
Combines keyword (BM25) and semantic (vector) search using
Reciprocal Rank Fusion (RRF k=60) — the 2026 industry standard.

Architecture:
    User text query
      ├──→ BM25 keyword search (exact terms, SKUs, brands) → ranked list A
      ├──→ Text encoder → 768d → Milvus ANN search → ranked list B
      └──→ CLIP text → 512d → Milvus visual ANN → ranked list C (cross-modal)
      │
      ├──→ RRF fusion (k=60) of lists A + B + C
      ├──→ CategoryRouter pre-filter (Gap 5)
      ├──→ LambdaMART re-rank (Gap 4)
      └──→ Cross-modal enrichment (Gap 3)

RRF (Reciprocal Rank Fusion) — Why k=60:
    Score(doc) = Σ 1 / (k + rank_in_list_i)
    k=60 is the standard constant (Cormack et al., 2009).
    It rewards documents appearing high in multiple lists
    without letting a single rank-1 outlier dominate.
    It operates on ranks, not scores — no normalization needed.

Best practices 2026:
    - Parallel retrieval: BM25 ∥ vector ∥ cross-modal (max latency, not sum)
    - RRF k=60 for fusion (no score normalization headaches)
    - Cross-encoder reranking on top-N if latency allows
    - Pre-filter by predicted category
    - Metadata filtering (price, brand) applied pre-retrieval
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re
import time
from collections import defaultdict
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# RRF constant — industry standard 2026
RRF_K = 60


def reciprocal_rank_fusion(
    *ranked_lists: list[tuple[str, float]],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion of multiple ranked lists.

    Args:
        ranked_lists: each list is [(item_id, score), ...] sorted by score desc
        k: RRF constant (default 60)

    Returns:
        Fused list [(item_id, rrf_score), ...] sorted by rrf_score desc
    """
    scores: dict[str, float] = defaultdict(float)

    for ranked_list in ranked_lists:
        for rank, (item_id, _) in enumerate(ranked_list, start=1):
            scores[item_id] += 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused


class HybridSearchPipeline:
    """Hybrid BM25 + vector search with RRF fusion.

    Searches products using 3 parallel signals:
    1. BM25 keyword match (title, description, tags, brand)
    2. Semantic vector search (text embedding → Milvus)
    3. Cross-modal visual search (CLIP text → visual embedding Milvus)

    Results are fused via RRF(k=60) then re-ranked by LambdaMART.

    Usage:
        pipeline = HybridSearchPipeline(milvus, reranker, cross_modal, category_router)
        results = await pipeline.search(
            query="robe d'été en soie",
            user_profile=user_profile,
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
        product_index: list[dict] | None = None,
        es_backend=None,           # Elasticsearch backend (production BM25)
        clip_onnx=None,            # ONNX-optimized CLIP (2-5x faster)
    ):
        self.milvus = milvus_client
        self.faiss = faiss_index
        self.reranker = reranker
        self.cross_modal = cross_modal
        self.category_router = category_router
        self.redis = redis_client
        self.es_backend = es_backend   # ElasticsearchBackend instance
        self.clip_onnx = clip_onnx     # CLIPOnnxInference instance
        # In-memory product index for BM25 (fallback when ES unavailable)
        self._product_index = product_index or []
        self._idf_cache: dict[str, float] = {}

    def set_product_index(self, products: list[dict]):
        """Set the in-memory product index for BM25 search."""
        self._product_index = products
        self._idf_cache.clear()
        logger.info("HybridSearch: BM25 index set with %d products", len(products))

    async def search(
        self,
        query: str,
        user_profile: dict[str, Any] | None = None,
        category_filter: int | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
        limit: int = 30,
        include_videos: bool = True,
    ) -> dict[str, Any]:
        """Execute hybrid search pipeline.

        Args:
            query: natural language search query
            user_profile: user profile for personalization
            category_filter: explicit category filter
            min_price/max_price: price range filter
            limit: max products to return
            include_videos: attach associated videos

        Returns:
            {
                "products": [...],
                "associated_videos": [...],
                "query_intent": str,
                "pipeline_ms": float,
            }
        """
        t_start = time.perf_counter()
        query = query.strip()
        if not query:
            return self._empty_response(0)

        # ── Step 1: Category prediction for pre-filtering (Gap 5) ──
        filter_expr = None
        if self.category_router and category_filter is None:
            filter_expr = self.category_router.predict_from_text(query)
        elif category_filter is not None:
            filter_expr = f"category_id == {category_filter}"

        # Add price filters
        price_filters = []
        if min_price is not None:
            price_filters.append(f"price >= {min_price}")
        if max_price is not None:
            price_filters.append(f"price <= {max_price}")

        if price_filters:
            price_expr = " and ".join(price_filters)
            if filter_expr:
                filter_expr = f"({filter_expr}) and {price_expr}"
            else:
                filter_expr = price_expr

        # ── Step 2: Parallel retrieval (BM25 ∥ Vector ∥ Cross-modal) ──
        ann_k = min(limit * 5, 500)

        # Run all 3 retrievals in parallel
        bm25_task = asyncio.create_task(
            self._bm25_search(query, top_k=ann_k)
        )
        vector_task = asyncio.create_task(
            self._vector_search(query, top_k=ann_k, filter_expr=filter_expr)
        )
        cross_modal_task = asyncio.create_task(
            self._cross_modal_visual_search(query, top_k=ann_k, filter_expr=filter_expr)
        )

        bm25_results, vector_results, cross_modal_results = await asyncio.gather(
            bm25_task, vector_task, cross_modal_task,
            return_exceptions=True,
        )

        # Handle exceptions from parallel tasks
        if isinstance(bm25_results, Exception):
            logger.warning("BM25 search failed: %s", bm25_results)
            bm25_results = []
        if isinstance(vector_results, Exception):
            logger.warning("Vector search failed: %s", vector_results)
            vector_results = []
        if isinstance(cross_modal_results, Exception):
            logger.warning("Cross-modal search failed: %s", cross_modal_results)
            cross_modal_results = []

        # ── Step 3: RRF Fusion (k=60) ──────────────────────────
        fused = reciprocal_rank_fusion(
            bm25_results, vector_results, cross_modal_results, k=RRF_K
        )

        if not fused:
            return self._empty_response(time.perf_counter() - t_start)

        # ── Step 4: Build candidate dicts for re-ranking ────────
        # Merge scores from each retrieval into the candidate
        bm25_scores = {item_id: score for item_id, score in bm25_results}
        vector_scores = {item_id: score for item_id, score in vector_results}
        cross_scores = {item_id: score for item_id, score in cross_modal_results}

        candidates = []
        for item_id, rrf_score in fused[: ann_k]:
            candidates.append({
                "item_id": item_id,
                "rrf_score": rrf_score,
                "text_similarity": vector_scores.get(item_id, 0.0),
                "visual_similarity": cross_scores.get(item_id, 0.0),
                "bm25_score": bm25_scores.get(item_id, 0.0),
                # Business signals (fetched from catalog in production)
                "category_match": True,
                "cv_score": 0.5,
                "total_sold": 0,
                "review_rating": 0.0,
                "review_count": 0,
                "vendor_rating": 0.0,
                "price": 0.0,
                "avg_category_price": 0.0,
                "freshness": 0.5,
                "pool_level": "L1",
                "conversion_rate": 0.01,
                "title": "",
                "image_url": "",
            })

        # Enrich with metadata from BM25 product index
        self._enrich_from_index(candidates)

        # ── Step 5: LambdaMART re-ranking (Gap 4) ──────────────
        if self.reranker:
            candidates = self.reranker.rerank(
                candidates,
                query_type="text",
                user_profile=user_profile,
                limit=limit,
            )
        else:
            candidates.sort(key=lambda x: x["rrf_score"], reverse=True)
            candidates = candidates[:limit]

        # ── Step 6: Cross-modal enrichment (Gap 3) ──────────────
        associated_videos = []
        if include_videos and self.cross_modal:
            product_ids = [c["item_id"] for c in candidates[:10]]
            associated_videos = await self.cross_modal.find_videos_for_products(
                product_ids
            )

        pipeline_ms = (time.perf_counter() - t_start) * 1000

        # Detect query intent
        intent = self._detect_intent(query)

        logger.info(
            "HybridSearch: '%s' → %d results in %.1fms "
            "(bm25=%d, vector=%d, cross=%d, intent=%s)",
            query[:50], len(candidates), pipeline_ms,
            len(bm25_results), len(vector_results), len(cross_modal_results),
            intent,
        )

        return {
            "products": candidates,
            "associated_videos": associated_videos,
            "query_intent": intent,
            "total_candidates": len(fused),
            "retrieval_counts": {
                "bm25": len(bm25_results),
                "vector": len(vector_results),
                "cross_modal": len(cross_modal_results),
            },
            "pipeline_ms": pipeline_ms,
        }

    # ── BM25 Keyword Search ────────────────────────────────────────

    async def _bm25_search(
        self, query: str, top_k: int = 200
    ) -> list[tuple[str, float]]:
        """BM25 keyword search: Elasticsearch (production) or in-memory (dev).

        Uses AsyncElasticsearch when available for sub-10ms search
        across millions of products. Falls back to in-memory BM25
        for development and testing without external dependencies.
        """
        # Production: use Elasticsearch
        if self.es_backend and self.es_backend._connected:
            try:
                return await self.es_backend.bm25_search(
                    query, top_k=top_k
                )
            except Exception as e:
                logger.warning("ES BM25 failed, falling back to in-memory: %s", e)

        # Dev/fallback: in-memory BM25
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self._bm25_search_sync(query, top_k)
        )

    def _bm25_search_sync(
        self, query: str, top_k: int = 200
    ) -> list[tuple[str, float]]:
        """Synchronous BM25 scoring."""
        if not self._product_index:
            return []

        terms = self._tokenize(query)
        if not terms:
            return []

        # BM25 parameters
        k1 = 1.5
        b = 0.75
        N = len(self._product_index)
        avgdl = sum(
            len(self._get_doc_text(p).split())
            for p in self._product_index
        ) / max(N, 1)

        scores = []
        for product in self._product_index:
            doc_text = self._get_doc_text(product)
            doc_terms = doc_text.lower().split()
            dl = len(doc_terms)

            score = 0.0
            for term in terms:
                tf = doc_terms.count(term)
                if tf == 0:
                    continue

                # IDF with cache
                if term not in self._idf_cache:
                    df = sum(
                        1 for p in self._product_index
                        if term in self._get_doc_text(p).lower()
                    )
                    self._idf_cache[term] = math.log(
                        (N - df + 0.5) / (df + 0.5) + 1
                    )
                idf = self._idf_cache[term]

                # BM25 TF normalization
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
                score += idf * tf_norm

            if score > 0:
                product_id = str(product.get("id", product.get("product_id", "")))
                scores.append((product_id, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    # ── Vector Semantic Search ─────────────────────────────────────

    async def _vector_search(
        self, query: str, top_k: int = 200, filter_expr: str | None = None
    ) -> list[tuple[str, float]]:
        """Semantic vector search via text encoder → Milvus ANN."""
        loop = asyncio.get_event_loop()

        try:
            # Encode query text → 768d embedding
            from ml.feature_store.encoders import get_text_encoder
            encoder = get_text_encoder()
            query_emb = await loop.run_in_executor(
                None, lambda: encoder.encode([query])[0]
            )

            if query_emb is None or not np.any(query_emb):
                return []

            query_emb = query_emb.astype(np.float32)

            # Search Milvus item_embeddings (256d Two-Tower space)
            # Try the item_embeddings collection which uses combined embeddings
            if self.milvus:
                results = await loop.run_in_executor(
                    None,
                    lambda: self.milvus.search(
                        collection_name="item_embeddings",
                        query_vectors=[query_emb.tolist()],
                        top_k=top_k,
                        filter_expr=filter_expr,
                    ),
                )
                if results and results[0]:
                    return [
                        (r["id"], float(r["score"])) for r in results[0]
                    ]

        except Exception as e:
            logger.warning("Vector search failed: %s", e)

        return []

    # ── Cross-Modal Visual Search (text → images) ──────────────────

    async def _cross_modal_visual_search(
        self, query: str, top_k: int = 100, filter_expr: str | None = None
    ) -> list[tuple[str, float]]:
        """CLIP text -> visual embedding search.

        Uses CLIP's shared text-image space to find products
        visually matching the text query (e.g., "red sneakers"
        finds red sneaker images even without the word "red" in metadata).

        Uses ONNX Runtime when available (2-5x faster than PyTorch).
        """
        loop = asyncio.get_event_loop()

        try:
            query_emb = None

            # Try ONNX CLIP first (2-5x faster)
            if self.clip_onnx and self.clip_onnx.is_loaded:
                query_emb = await loop.run_in_executor(
                    None, lambda: self.clip_onnx.encode_text(query)
                )

            # Fallback to PyTorch CLIP
            if query_emb is None:
                import open_clip
                import torch

                model, _, _ = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="openai"
                )
                model.eval()
                tokenizer = open_clip.get_tokenizer("ViT-B-32")
                tokens = tokenizer([query])

                with torch.no_grad():
                    text_features = model.encode_text(tokens)
                    text_features /= text_features.norm(dim=-1, keepdim=True)

                query_emb = text_features.squeeze(0).numpy().astype(np.float32)

            if query_emb is None:
                return []

            # Search Milvus visual_embeddings (CLIP 512d space)
            if self.milvus:
                results = await loop.run_in_executor(
                    None,
                    lambda: self.milvus.search(
                        collection_name="visual_embeddings",
                        query_vectors=[query_emb.tolist()],
                        top_k=top_k,
                        filter_expr=filter_expr,
                    ),
                )
                if results and results[0]:
                    return [
                        (r["id"], float(r["score"])) for r in results[0]
                    ]

        except Exception as e:
            logger.debug("Cross-modal visual search failed: %s", e)

        return []

    # ── Helpers ─────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace tokenizer with basic normalization."""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = [t for t in text.split() if len(t) > 1]
        return tokens

    def _get_doc_text(self, product: dict) -> str:
        """Combine searchable fields into one string."""
        parts = [
            product.get("title", ""),
            product.get("description_short", ""),
            " ".join(product.get("tags", [])),
            product.get("brand", ""),
            product.get("auto_description", ""),
        ]
        return " ".join(parts)

    def _enrich_from_index(self, candidates: list[dict]):
        """Enrich candidate dicts with metadata from BM25 product index."""
        if not self._product_index:
            return

        index_map = {}
        for p in self._product_index:
            pid = str(p.get("id", p.get("product_id", "")))
            index_map[pid] = p

        for c in candidates:
            p = index_map.get(c["item_id"])
            if not p:
                continue
            c["title"] = p.get("title", "")
            c["image_url"] = p.get("image_url", "")
            c["price"] = float(p.get("base_price", 0.0))
            c["category_id"] = p.get("category_id", 0)
            c["vendor_id"] = str(p.get("vendor_id", ""))
            c["cv_score"] = float(p.get("cv_score", 0.5))
            c["total_sold"] = p.get("total_sold", 0)
            c["review_rating"] = float(p.get("review_rating", 0.0))
            c["review_count"] = p.get("review_count", 0)

    def _detect_intent(self, query: str) -> str:
        """Detect query intent for analytics."""
        q = query.lower()
        if any(kw in q for kw in ["acheter", "buy", "commander", "prix"]):
            return "purchase"
        if any(kw in q for kw in ["pas cher", "promo", "solde", "discount", "cheap"]):
            return "deal_seeking"
        if any(kw in q for kw in ["comparer", "versus", "vs", "meilleur"]):
            return "compare"
        if any(kw in q for kw in ["comment", "how to", "tutoriel", "tutorial"]):
            return "informational"
        return "browse"

    def _empty_response(self, elapsed: float) -> dict[str, Any]:
        return {
            "products": [],
            "associated_videos": [],
            "query_intent": "browse",
            "total_candidates": 0,
            "retrieval_counts": {"bm25": 0, "vector": 0, "cross_modal": 0},
            "pipeline_ms": elapsed * 1000,
        }
