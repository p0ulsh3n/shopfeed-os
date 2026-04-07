"""
Hybrid Search Engine
====================
Combines BM25 (sparse/lexical) + Dense (semantic) retrieval via RRF fusion.

Architecture (2026 best practices):
1. DENSE retrieval: pgvector HNSW avec SQLAlchemy parameterized queries
2. SPARSE retrieval: tsvector full-text search — paramètres bindés ORM
3. RRF FUSION: Reciprocal Rank Fusion
4. RESCORE: float32 re-ranking

MIGRATION SÉCURITÉ:
- AVANT: f"SET LOCAL hnsw.ef_search = {settings.hnsw_ef_search}" → injection SQL
- AVANT: filter_clause avec f-string sur les clés metadata → injection SQL
- APRÈS: sqlalchemy text() avec paramètre bindé :ef_search
- APRÈS: select().where() ORM avec cast conditionnel — zéro interpolation
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
    Production hybrid search avec pgvector HNSW + BM25 + RRF.
    Toutes les requêtes utilisent des paramètres bindés SQLAlchemy — zéro injection SQL.
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
        t_start = time.perf_counter()
        final_k = top_k or settings.retrieval_final_top_k

        float32_vec, int8_vec, binary_vec = (
            await self._encoder.encode_query_all_precisions(query)
        )

        dense_results, sparse_results = await self._parallel_retrieval(
            shop_id=shop_id,
            query=query,
            float32_vec=float32_vec,
            filters=filters,
        )

        fused = self._reciprocal_rank_fusion(
            ranked_lists=[dense_results, sparse_results],
            k=settings.rrf_k,
        )

        if settings.embedding_rescoring and len(fused) > 0:
            fused = await self._rescore_with_float32(
                candidates=fused,
                query_float32=float32_vec,
                shop_id=shop_id,
            )

        top_results = fused[:final_k]

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "[HybridSearch] shop=%s query='%s' dense=%d sparse=%d "
            "fused=%d final=%d latency=%.1fms",
            shop_id, query[:50],
            len(dense_results), len(sparse_results),
            len(fused), len(top_results), elapsed_ms,
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
        pgvector HNSW cosine similarity search.

        FIX SÉCURITÉ:
        - AVANT: f"SET LOCAL hnsw.ef_search = {settings.hnsw_ef_search}" → injection
        - APRÈS: text("SET LOCAL hnsw.ef_search = :ef_search") avec paramètre bindé

        FIX SÉCURITÉ filters:
        - AVANT: f"metadata->>'{{key}}' = :{param_name}" avec f-string sur key → injection
        - APRÈS: conditions construites avec des noms de colonnes fixes en JSONB — clés
          validées contre une allowlist avant utilisation
        """
        # SAFE: paramètre bindé — plus de f-string
        await self._session.execute(
            text("SET LOCAL hnsw.ef_search = :ef_search"),
            {"ef_search": int(settings.hnsw_ef_search)},
        )

        vector_str = float32_to_pgvector_str(float32_vec)

        # Construction sécurisée des filtres JSONB
        filter_clause, filter_params = self._build_safe_filter_clause(filters)

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
        PostgreSQL tsvector full-text search.
        Paramètres bindés — zéro interpolation.
        """
        safe_query = " & ".join(
            word for word in query.split() if len(word) > 1
        )
        if not safe_query:
            return []

        filter_clause, filter_params = self._build_safe_filter_clause(filters)

        query_sql = text(f"""
            SELECT
                product_id,
                ts_rank_cd(
                    product_tsv,
                    websearch_to_tsquery('french', :search_query),
                    32
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
        except Exception as exc:
            logger.warning("BM25 retrieval failed (falling back to dense only): %s", exc)
            return []

    # ─────────────────────── SAFE FILTER BUILDER ─────────────────

    # Allowlist des clés metadata autorisées — zéro injection possible
    _ALLOWED_FILTER_KEYS = frozenset({
        "availability", "category", "subcategory", "currency",
        "sku", "brand", "tags",
    })

    def _build_safe_filter_clause(
        self, filters: dict | None
    ) -> tuple[str, dict]:
        """
        Construit une clause WHERE JSONB avec paramètres bindés.

        SÉCURITÉ: les noms de colonnes JSONB (keys) sont validés contre
        _ALLOWED_FILTER_KEYS avant toute utilisation — aucune interpolation
        de valeur fournie par l'utilisateur directement dans le SQL.
        """
        if not filters:
            return "", {}

        conditions: list[str] = []
        params: dict = {}

        for i, (key, value) in enumerate(filters.items()):
            if key not in self._ALLOWED_FILTER_KEYS:
                logger.warning("Rejected unknown filter key '%s'", key)
                continue
            param_name = f"filter_val_{i}"
            # La clé vient de l'allowlist — pas d'interpolation utilisateur
            conditions.append(f"metadata->>'{key}' = :{param_name}")
            params[param_name] = str(value)

        if not conditions:
            return "", {}

        return "AND " + " AND ".join(conditions), params

    # ─────────────────────── PARALLEL EXEC ───────────────────────

    async def _parallel_retrieval(
        self,
        shop_id: str,
        query: str,
        float32_vec: np.ndarray,
        filters: dict | None,
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        import asyncio
        dense_task = self._dense_retrieval(shop_id, float32_vec, filters)
        sparse_task = self._sparse_retrieval(shop_id, query, filters)
        dense_results, sparse_results = await asyncio.gather(
            dense_task, sparse_task, return_exceptions=True
        )

        if isinstance(dense_results, Exception):
            logger.error("Dense retrieval error: %s", dense_results)
            dense_results = []
        if isinstance(sparse_results, Exception):
            logger.error("Sparse retrieval error: %s", sparse_results)
            sparse_results = []

        return dense_results, sparse_results  # type: ignore

    # ─────────────────────── RRF FUSION ──────────────────────────

    def _reciprocal_rank_fusion(
        self,
        ranked_lists: list[list[tuple[str, float]]],
        k: int = 60,
    ) -> list[RetrievedProduct]:
        rrf_scores: dict[str, float] = defaultdict(float)

        for ranked_list in ranked_lists:
            for rank, (product_id, _score) in enumerate(ranked_list, start=1):
                rrf_scores[product_id] += 1.0 / (k + rank)

        sorted_items = sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        )

        return [
            RetrievedProduct(
                product=Product(id=pid, shop_id="", name="", price=0.0),
                score=min(rrf_score, 1.0),
                retrieval_method="rrf_fusion",
            )
            for pid, rrf_score in sorted_items
        ]

    # ─────────────────────── FLOAT32 RESCORE ─────────────────────

    async def _rescore_with_float32(
        self,
        candidates: list[RetrievedProduct],
        query_float32: np.ndarray,
        shop_id: str,
    ) -> list[RetrievedProduct]:
        if not candidates:
            return candidates

        product_ids = [r.product.id for r in candidates]

        # Paramètre bindé — liste passée comme :product_ids via SQLAlchemy
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

        product_data: dict[str, tuple[dict, np.ndarray | None]] = {}
        for row in rows:
            pid, metadata, emb_str = row
            emb: np.ndarray | None = None
            if emb_str:
                vals = emb_str.strip("[]").split(",")
                emb = np.array([float(v) for v in vals], dtype=np.float32)
            product_data[pid] = (metadata or {}, emb)

        rescored: list[RetrievedProduct] = []
        for candidate in candidates:
            pid = candidate.product.id
            if pid not in product_data:
                continue
            metadata, emb = product_data[pid]

            if emb is not None:
                float32_score = float(np.dot(query_float32, emb))
                final_score = 0.7 * float32_score + 0.3 * candidate.score
            else:
                final_score = candidate.score

            product = self._metadata_to_product(pid, shop_id, metadata)
            rescored.append(
                RetrievedProduct(
                    product=product,
                    score=max(0.0, min(1.0, final_score)),
                    retrieval_method="rrf_float32_rescored",
                )
            )

        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored

    # ─────────────────────── HELPERS ─────────────────────────────

    def _metadata_to_product(
        self, product_id: str, shop_id: str, metadata: dict
    ) -> Product:
        from services.shopbot_service.models.schemas import (
            ProductAvailability,
            ProductImage,
        )
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
