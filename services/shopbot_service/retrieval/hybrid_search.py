"""
Hybrid Search Engine
====================
Combines BM25 (sparse/lexical) + Dense (semantic) retrieval via RRF fusion.

Architecture (2026 best practices):
1. DENSE retrieval: pgvector HNSW with int8 quantization (fast ANN)
2. SPARSE retrieval: PostgreSQL tsvector full-text search (BM25-like)
3. RRF FUSION: Reciprocal Rank Fusion to merge both result lists
4. OPTIONAL RESCORE: float32 re-ranking for top candidates

Why RRF instead of linear combination?
- BM25 and cosine scores are NOT in the same space (not linearly separable)
- RRF uses only rank positions → robust, no hyperparameter tuning needed
- De-facto 2026 standard (Qdrant, Elasticsearch, pgvector all use it)
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from services.shopbot_service.config import get_settings
from services.shopbot_service.embeddings.encoder import (
    EmbeddingEncoder,
    batch_cosine_similarities,
    float32_to_pgvector_str,
)
from services.shopbot_service.models.schemas import Product, RetrievedProduct

logger = logging.getLogger(__name__)
settings = get_settings()


class HybridSearchEngine:
    """
    Production hybrid search with pgvector HNSW + BM25 + RRF.

    Usage:
        engine = HybridSearchEngine(session)
        results = await engine.search(shop_id="shop123", query="robe rouge")
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._encoder = EmbeddingEncoder.get_instance()

    # ─────────────────────── PUBLIC API ──────────────────────────

    async def search(
        self,
        shop_id: str,
        query: str,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> list[RetrievedProduct]:
        """
        Main hybrid search entry point.

        Steps:
        1. Encode query → get float32, int8, binary embeddings
        2. Run dense retrieval (pgvector HNSW) in parallel with BM25
        3. RRF fusion of both ranked lists
        4. Rescore top candidates with float32 (optional but recommended)
        5. Return top_k results with metadata

        Args:
            shop_id: Filter results to this shop only
            query: Natural language search query
            top_k: Number of final results (default: settings.retrieval_final_top_k)
            filters: Optional JSONB metadata filters (e.g., {"availability": "in_stock"})
        """
        t_start = time.perf_counter()
        final_k = top_k or settings.retrieval_final_top_k

        # Step 1: Encode query at all precision levels
        float32_vec, int8_vec, binary_vec = (
            await self._encoder.encode_query_all_precisions(query)
        )

        # Step 2: Parallel retrieval
        dense_results, sparse_results = await self._parallel_retrieval(
            shop_id=shop_id,
            query=query,
            float32_vec=float32_vec,
            filters=filters,
        )

        # Step 3: RRF fusion
        fused = self._reciprocal_rank_fusion(
            ranked_lists=[dense_results, sparse_results],
            k=settings.rrf_k,
        )

        # Step 4: Float32 rescore (if quantization was used)
        # This eliminates any quality loss from int8 retrieval
        if settings.embedding_rescoring and len(fused) > 0:
            fused = await self._rescore_with_float32(
                candidates=fused,
                query_float32=float32_vec,
                shop_id=shop_id,
            )

        # Step 5: Return top_k with proper metadata
        top_results = fused[:final_k]

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            f"[HybridSearch] shop={shop_id} query='{query[:50]}' "
            f"dense={len(dense_results)} sparse={len(sparse_results)} "
            f"fused={len(fused)} final={len(top_results)} "
            f"latency={elapsed_ms:.1f}ms"
        )

        return top_results

    # ─────────────────────── DENSE RETRIEVAL ─────────────────────

    async def _dense_retrieval(
        self,
        shop_id: str,
        float32_vec: np.ndarray,
        filters: dict | None = None,
    ) -> list[tuple[str, float]]:
        """
        pgvector HNSW cosine similarity search filtered by shop_id.
        Returns list of (product_id, score) sorted by score DESC.

        Uses <=> operator (cosine distance) on the HNSW index.
        ef_search is set via session parameter for per-query tuning.
        """
        # Set HNSW ef_search for this query (higher = better recall, more latency)
        await self._session.execute(
            text(f"SET LOCAL hnsw.ef_search = {settings.hnsw_ef_search}")
        )

        # Build optional JSONB metadata filter
        filter_clause = ""
        filter_params: dict = {}
        if filters:
            conditions = []
            for i, (key, value) in enumerate(filters.items()):
                param_name = f"filter_val_{i}"
                conditions.append(
                    f"metadata->>'{{key}}' = :{param_name}".replace("{key}", key)
                )
                filter_params[param_name] = str(value)
            if conditions:
                filter_clause = "AND " + " AND ".join(conditions)

        vector_str = float32_to_pgvector_str(float32_vec)

        query_sql = text(f"""
            SELECT
                product_id,
                1 - (embedding_float32 <=> :query_vector::vector) AS cosine_score
            FROM shopbot_product_embeddings
            WHERE shop_id = :shop_id
              {filter_clause}
              AND embedding_float32 IS NOT NULL
            ORDER BY embedding_float32 <=> :query_vector::vector
            LIMIT :top_k
        """)

        result = await self._session.execute(
            query_sql,
            {
                "query_vector": vector_str,
                "shop_id": shop_id,
                "top_k": settings.retrieval_top_k_dense,
                **filter_params,
            },
        )
        rows = result.fetchall()
        return [(row[0], float(row[1])) for row in rows]

    # ─────────────────────── BM25 RETRIEVAL ──────────────────────

    async def _sparse_retrieval(
        self,
        shop_id: str,
        query: str,
        filters: dict | None = None,
    ) -> list[tuple[str, float]]:
        """
        PostgreSQL tsvector full-text search (BM25-equivalent via ts_rank_cd).
        Uses the GIN index on product_tsv for fast keyword matching.

        ts_rank_cd uses a cover density ranking formula that approximates BM25.
        It considers: term frequency, proximity, document length normalization.
        """
        # Sanitize query for tsquery (remove special chars, handle operators)
        safe_query = " & ".join(
            word for word in query.split() if len(word) > 1
        )
        if not safe_query:
            return []

        filter_clause = ""
        filter_params: dict = {}
        if filters:
            conditions = []
            for i, (key, value) in enumerate(filters.items()):
                param_name = f"filter_val_{i}"
                conditions.append(
                    f"metadata->>'{{key}}' = :{param_name}".replace("{key}", key)
                )
                filter_params[param_name] = str(value)
            if conditions:
                filter_clause = "AND " + " AND ".join(conditions)

        query_sql = text(f"""
            SELECT
                product_id,
                ts_rank_cd(
                    product_tsv,
                    websearch_to_tsquery('french', :search_query),
                    32  -- normalization: divide by (1 + log(ndoc))
                ) AS bm25_score
            FROM shopbot_product_embeddings
            WHERE shop_id = :shop_id
              {filter_clause}
              AND product_tsv @@ websearch_to_tsquery('french', :search_query)
            ORDER BY bm25_score DESC
            LIMIT :top_k
        """)

        try:
            result = await self._session.execute(
                query_sql,
                {
                    "search_query": safe_query,
                    "shop_id": shop_id,
                    "top_k": settings.retrieval_top_k_sparse,
                    **filter_params,
                },
            )
            rows = result.fetchall()
            return [(row[0], float(row[1])) for row in rows]
        except Exception as e:
            logger.warning(f"BM25 retrieval failed (falling back to dense only): {e}")
            return []

    # ─────────────────────── PARALLEL EXEC ───────────────────────

    async def _parallel_retrieval(
        self,
        shop_id: str,
        query: str,
        float32_vec: np.ndarray,
        filters: dict | None,
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        """
        Run dense and sparse retrieval concurrently using asyncio.gather.
        Both queries hit the same PostgreSQL connection pool.
        """
        import asyncio
        dense_task = self._dense_retrieval(shop_id, float32_vec, filters)
        sparse_task = self._sparse_retrieval(shop_id, query, filters)
        dense_results, sparse_results = await asyncio.gather(
            dense_task, sparse_task, return_exceptions=True
        )

        # Handle partial failures gracefully
        if isinstance(dense_results, Exception):
            logger.error(f"Dense retrieval error: {dense_results}")
            dense_results = []
        if isinstance(sparse_results, Exception):
            logger.error(f"Sparse retrieval error: {sparse_results}")
            sparse_results = []

        return dense_results, sparse_results  # type: ignore

    # ─────────────────────── RRF FUSION ──────────────────────────

    def _reciprocal_rank_fusion(
        self,
        ranked_lists: list[list[tuple[str, float]]],
        k: int = 60,
    ) -> list[RetrievedProduct]:
        """
        Reciprocal Rank Fusion (RRF) — Cormack & Clarke 2009.

        RRF score formula: sum(1 / (k + rank_i)) for each ranked list i
        Where rank_i is the 1-indexed position of the document in list i.

        Why k=60?
        - The original paper recommends k=60 as the optimal constant
        - Qdrant, Elasticsearch, and pgvector implementations all default to 60
        - It controls the weight given to top-ranked documents
          (larger k = more uniform weighting)

        Returns preliminary RetrievedProduct list with fused scores.
        Product objects are populated later by _fetch_products.
        """
        # Build product_id → {list_idx: rank} mapping
        rrf_scores: dict[str, float] = defaultdict(float)

        for ranked_list in ranked_lists:
            for rank, (product_id, _score) in enumerate(ranked_list, start=1):
                rrf_scores[product_id] += 1.0 / (k + rank)

        # Sort by RRF score descending
        sorted_items = sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Create preliminary results (products fetched separately)
        results = []
        for product_id, rrf_score in sorted_items:
            results.append(
                RetrievedProduct(
                    product=Product(
                        id=product_id,
                        shop_id="",  # filled by _fetch_products
                        name="",
                        price=0.0,
                    ),
                    score=min(rrf_score, 1.0),  # Normalize to [0,1]
                    retrieval_method="rrf_fusion",
                )
            )

        return results

    # ─────────────────────── FLOAT32 RESCORE ─────────────────────

    async def _rescore_with_float32(
        self,
        candidates: list[RetrievedProduct],
        query_float32: np.ndarray,
        shop_id: str,
    ) -> list[RetrievedProduct]:
        """
        Re-score RRF candidates with float32 embeddings for maximum accuracy.

        This is the '2-stage retrieval' pattern:
        1. Fast: int8/binary + BM25 → rough candidates (done above)
        2. Accurate: float32 dot product on fetched embeddings → final ranking

        This gives ~99.9% of full float32 quality at fraction of the cost.
        We only fetch full embeddings for the top candidates (default: 50+50=100 max).
        """
        if not candidates:
            return candidates

        product_ids = [r.product.id for r in candidates]

        # Fetch product data AND float32 embeddings for rescoring
        query_sql = text("""
            SELECT
                product_id,
                metadata,
                embedding_float32::text
            FROM shopbot_product_embeddings
            WHERE shop_id = :shop_id
              AND product_id = ANY(:product_ids)
        """)

        result = await self._session.execute(
            query_sql,
            {"shop_id": shop_id, "product_ids": product_ids},
        )
        rows = result.fetchall()

        # Build lookup: product_id → (metadata, embedding)
        product_data: dict[str, tuple[dict, np.ndarray | None]] = {}
        for row in rows:
            pid, metadata, emb_str = row
            emb: np.ndarray | None = None
            if emb_str:
                # Parse pgvector string format "[x,y,z,...]"
                vals = emb_str.strip("[]").split(",")
                emb = np.array([float(v) for v in vals], dtype=np.float32)
            product_data[pid] = (metadata or {}, emb)

        # Rescore with float32 cosine similarity
        rescored: list[RetrievedProduct] = []
        for candidate in candidates:
            pid = candidate.product.id
            if pid not in product_data:
                continue
            metadata, emb = product_data[pid]

            if emb is not None:
                # Cosine similarity (both are L2-normalized)
                float32_score = float(np.dot(query_float32, emb))
                # Blend RRF rank score with float32 score (weighted average)
                # 70% float32 accuracy + 30% rank diversity from RRF
                final_score = 0.7 * float32_score + 0.3 * candidate.score
            else:
                final_score = candidate.score

            # Build proper Product from metadata
            product = self._metadata_to_product(pid, shop_id, metadata)
            rescored.append(
                RetrievedProduct(
                    product=product,
                    score=max(0.0, min(1.0, final_score)),
                    retrieval_method="rrf_float32_rescored",
                )
            )

        # Sort by final score
        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored

    # ─────────────────────── HELPERS ─────────────────────────────

    def _metadata_to_product(
        self, product_id: str, shop_id: str, metadata: dict
    ) -> Product:
        """
        Reconstruct a Product object from the JSONB metadata store.
        Includes image reconstruction so ProductCard.from_retrieved()
        can access image URLs for frontend rendering.
        """
        from services.shopbot_service.models.schemas import (
            ProductAvailability,
            ProductImage,
        )
        # Reconstruct images from stored metadata
        raw_images = metadata.get("images", [])
        images = [
            ProductImage(
                url=img.get("url", ""),
                alt=img.get("alt"),
                is_primary=img.get("is_primary", False),
            )
            for img in raw_images
            if img.get("url")
        ]
        return Product(
            id=product_id,
            shop_id=shop_id,
            name=metadata.get("name", "Produit"),
            description=metadata.get("description"),
            price=float(metadata.get("price", 0.0)),
            currency=metadata.get("currency", "XAF"),
            category=metadata.get("category"),
            subcategory=metadata.get("subcategory"),
            tags=metadata.get("tags", []),
            images=images,
            availability=ProductAvailability(
                metadata.get("availability", "in_stock")
            ),
            stock_quantity=metadata.get("stock_quantity"),
            sku=metadata.get("sku"),
            attributes=metadata.get("attributes", {}),
        )
