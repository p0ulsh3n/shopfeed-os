"""
Hybrid Search Pipeline — BM25 + Vector + RRF Fusion
=====================================================
Combines keyword (BM25) and semantic (vector) search using
Reciprocal Rank Fusion (RRF k=60) — the 2026 industry standard.

MIGRATION SÉCURITÉ:
- AVANT: filter_expr = f"category_id == {category_filter}"
         filter_expr = f"price >= {min_price}"
  → injection directe dans les filtres Milvus (valeurs non typées)
- APRÈS: MilvusFilterBuilder avec types Python stricts et validation
  → opérateurs fixes, valeurs int/float validées, zéro interpolation arbitraire

Architecture:
    User text query
      ├──→ BM25 keyword search (exact terms, SKUs, brands)  → ranked list A
      ├──→ Text encoder → 768d → Milvus ANN search          → ranked list B
      └──→ CLIP text → 512d → Milvus visual ANN             → ranked list C
      │
      ├──→ RRF fusion (k=60) of lists A + B + C
      ├──→ CategoryRouter pre-filter (Gap 5)
      ├──→ LambdaMART re-rank (Gap 4)
      └──→ Cross-modal enrichment (Gap 3)

RRF k=60 — Cormack et al., 2009.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from collections import defaultdict
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

RRF_K = 60


# ─────────────────────── MILVUS FILTER BUILDER ───────────────────

class MilvusFilterBuilder:
    """
    Construit des expressions de filtre Milvus de façon sécurisée.

    FIX INJECTION:
    - AVANT: f"category_id == {category_filter}" puis f"price >= {min_price}"
      → n'importe quelle valeur peut être injectée (ex: "1 OR 1==1")
    - APRÈS: opérateurs et noms de champs fixes (hardcodés),
      valeurs typées Python (int/float) serializées explicitement

    Milvus n'a pas de parameterized expressions type SQL ($1).
    La protection correcte est:
      1. Noms de champs et opérateurs HARDCODÉS dans le code
      2. Valeurs validées et castées en types Python stricts avant expr
      3. Jamais de construction via f-string sur des inputs utilisateur

    Référence: https://milvus.io/docs/boolean.md
    """

    @staticmethod
    def category_eq(category_id: int) -> str:
        """category_id == N — int strict, jamais string user."""
        safe = int(category_id)      # cast + validation implicite
        return f"category_id == {safe}"

    @staticmethod
    def price_gte(price: float) -> str:
        """price >= N.N — float strict."""
        safe = float(price)
        if safe < 0:
            raise ValueError(f"Price must be non-negative, got {safe}")
        return f"price >= {safe:.6f}"

    @staticmethod
    def price_lte(price: float) -> str:
        """price <= N.N — float strict."""
        safe = float(price)
        if safe < 0:
            raise ValueError(f"Price must be non-negative, got {safe}")
        return f"price <= {safe:.6f}"

    @staticmethod
    def and_(*expressions: str) -> str:
        """Combine des expressions avec AND — jamais de string user dans les args."""
        valid = [e for e in expressions if isinstance(e, str) and e.strip()]
        return " and ".join(f"({e})" for e in valid) if valid else ""

    @classmethod
    def build(
        cls,
        category_id: Optional[int] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
    ) -> Optional[str]:
        """Construit l'expression de filtre Milvus complète."""
        parts: list[str] = []
        if category_id is not None:
            parts.append(cls.category_eq(category_id))
        if min_price is not None:
            parts.append(cls.price_gte(min_price))
        if max_price is not None:
            parts.append(cls.price_lte(max_price))
        return cls.and_(*parts) if parts else None


# ─────────────────────── RRF ─────────────────────────────────────

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

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ─────────────────────── PIPELINE ────────────────────────────────

class HybridSearchPipeline:
    """Hybrid BM25 + vector search with RRF fusion.

    Searches products using 3 parallel signals:
    1. BM25 keyword match (title, description, tags, brand)
    2. Semantic vector search (text embedding → Milvus)
    3. Cross-modal visual search (CLIP text → visual embedding Milvus)

    Results fused via RRF(k=60) then re-ranked by LambdaMART.
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
        es_backend=None,
        clip_onnx=None,
    ):
        self.milvus = milvus_client
        self.faiss = faiss_index
        self.reranker = reranker
        self.cross_modal = cross_modal
        self.category_router = category_router
        self.redis = redis_client
        self.es_backend = es_backend
        self.clip_onnx = clip_onnx
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

        FIX INJECTION:
        - AVANT: filter_expr = f"category_id == {category_filter}"
                 filter_expr += f" and price >= {min_price}"
          → valeurs utilisateur interpolées directement dans l'expr Milvus
        - APRÈS: MilvusFilterBuilder.build() — types stricts, opérateurs hardcodés
        """
        t_start = time.perf_counter()
        query = query.strip()
        if not query:
            return self._empty_response(0)

        # ── Step 1: Milvus filter — construit de façon sécurisée ──
        try:
            filter_expr = MilvusFilterBuilder.build(
                category_id=category_filter,
                min_price=min_price,
                max_price=max_price,
            )
        except (ValueError, TypeError) as exc:
            logger.warning("Invalid filter params, ignoring: %s", exc)
            filter_expr = None

        # Category router (retourne déjà un int category_id, pas un expr string)
        if self.category_router and category_filter is None:
            predicted_cat = self.category_router.predict_from_text(query)
            if predicted_cat and isinstance(predicted_cat, int):
                existing_filters = {}
                if min_price is not None:
                    existing_filters["min_price"] = min_price
                if max_price is not None:
                    existing_filters["max_price"] = max_price
                filter_expr = MilvusFilterBuilder.build(
                    category_id=predicted_cat,
                    **existing_filters,
                )

        # ── Step 2: Parallel retrieval (BM25 ∥ Vector ∥ Cross-modal) ──
        ann_k = min(limit * 5, 500)

        bm25_task = asyncio.create_task(self._bm25_search(query, top_k=ann_k))
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

        if isinstance(bm25_results, Exception):
            logger.warning("BM25 search failed: %s", bm25_results)
            bm25_results = []
        if isinstance(vector_results, Exception):
            logger.warning("Vector search failed: %s", vector_results)
            vector_results = []
        if isinstance(cross_modal_results, Exception):
            logger.warning("Cross-modal search failed: %s", cross_modal_results)
            cross_modal_results = []

        # ── Step 3: RRF Fusion (k=60) ──
        fused = reciprocal_rank_fusion(
            bm25_results, vector_results, cross_modal_results, k=RRF_K
        )

        if not fused:
            return self._empty_response(time.perf_counter() - t_start)

        # ── Step 4: Build candidates ──
        bm25_scores = {item_id: score for item_id, score in bm25_results}
        vector_scores = {item_id: score for item_id, score in vector_results}
        cross_scores = {item_id: score for item_id, score in cross_modal_results}

        candidates = [
            {
                "item_id": item_id,
                "rrf_score": rrf_score,
                "text_similarity": vector_scores.get(item_id, 0.0),
                "visual_similarity": cross_scores.get(item_id, 0.0),
                "bm25_score": bm25_scores.get(item_id, 0.0),
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
            }
            for item_id, rrf_score in fused[:ann_k]
        ]

        self._enrich_from_index(candidates)

        # ── Step 5: LambdaMART re-ranking ──
        if self.reranker:
            candidates = self.reranker.rerank(
                candidates, query_type="text", user_profile=user_profile, limit=limit
            )
        else:
            candidates.sort(key=lambda x: x["rrf_score"], reverse=True)
            candidates = candidates[:limit]

        # ── Step 6: Cross-modal enrichment ──
        associated_videos = []
        if include_videos and self.cross_modal:
            product_ids = [c["item_id"] for c in candidates[:10]]
            associated_videos = await self.cross_modal.find_videos_for_products(product_ids)

        pipeline_ms = (time.perf_counter() - t_start) * 1000
        intent = self._detect_intent(query)

        logger.info(
            "HybridSearch: '%s' → %d results in %.1fms "
            "(bm25=%d, vector=%d, cross=%d, intent=%s)",
            query[:50], len(candidates), pipeline_ms,
            len(bm25_results), len(vector_results), len(cross_modal_results), intent,
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

    # ── BM25 ─────────────────────────────────────────────────────

    async def _bm25_search(self, query: str, top_k: int = 200) -> list[tuple[str, float]]:
        if self.es_backend and getattr(self.es_backend, "_connected", False):
            try:
                return await self.es_backend.bm25_search(query, top_k=top_k)
            except Exception as e:
                logger.warning("ES BM25 failed, falling back to in-memory: %s", e)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._bm25_search_sync(query, top_k))

    def _bm25_search_sync(self, query: str, top_k: int = 200) -> list[tuple[str, float]]:
        if not self._product_index:
            return []
        terms = self._tokenize(query)
        if not terms:
            return []

        k1 = 1.5
        b = 0.75
        N = len(self._product_index)
        avgdl = sum(len(self._get_doc_text(p).split()) for p in self._product_index) / max(N, 1)

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
                if term not in self._idf_cache:
                    df = sum(1 for p in self._product_index if term in self._get_doc_text(p).lower())
                    self._idf_cache[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)
                idf = self._idf_cache[term]
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
                score += idf * tf_norm
            if score > 0:
                pid = str(product.get("id", product.get("product_id", "")))
                scores.append((pid, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    # ── Vector Search ─────────────────────────────────────────────

    async def _vector_search(
        self, query: str, top_k: int = 200, filter_expr: str | None = None
    ) -> list[tuple[str, float]]:
        loop = asyncio.get_event_loop()
        try:
            from ml.feature_store.encoders import get_text_encoder
            encoder = get_text_encoder()
            query_emb = await loop.run_in_executor(None, lambda: encoder.encode([query])[0])

            if query_emb is None or not np.any(query_emb):
                return []
            query_emb = query_emb.astype(np.float32)

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
                    return [(r["id"], float(r["score"])) for r in results[0]]
        except Exception as e:
            logger.warning("Vector search failed: %s", e)
        return []

    # ── Cross-Modal ───────────────────────────────────────────────

    async def _cross_modal_visual_search(
        self, query: str, top_k: int = 100, filter_expr: str | None = None
    ) -> list[tuple[str, float]]:
        loop = asyncio.get_event_loop()
        try:
            query_emb = None
            if self.clip_onnx and self.clip_onnx.is_loaded:
                query_emb = await loop.run_in_executor(
                    None, lambda: self.clip_onnx.encode_text(query)
                )
            if query_emb is None:
                import open_clip
                import torch
                model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
                model.eval()
                tokenizer = open_clip.get_tokenizer("ViT-B-32")
                tokens = tokenizer([query])
                with torch.no_grad():
                    text_features = model.encode_text(tokens)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                query_emb = text_features.squeeze(0).numpy().astype(np.float32)

            if query_emb is None:
                return []
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
                    return [(r["id"], float(r["score"])) for r in results[0]]
        except Exception as e:
            logger.debug("Cross-modal visual search failed: %s", e)
        return []

    # ── Helpers ───────────────────────────────────────────────────

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", " ", text)
        return [t for t in text.split() if len(t) > 1]

    def _get_doc_text(self, product: dict) -> str:
        parts = [
            product.get("title", ""),
            product.get("description_short", ""),
            " ".join(product.get("tags", [])),
            product.get("brand", ""),
            product.get("auto_description", ""),
        ]
        return " ".join(parts)

    def _enrich_from_index(self, candidates: list[dict]):
        if not self._product_index:
            return
        index_map = {
            str(p.get("id", p.get("product_id", ""))): p for p in self._product_index
        }
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
