"""
Milvus Vector Search Client (archi-2026 §9.2)
================================================
Distributed vector search — replaces in-memory FAISS for production scale.

FAISS vs Milvus:
    - FAISS: excellent for <10M vectors, single-machine, no persistence
    - Milvus: distributed, persistent, scales to billions. Similar query speed.

In shopfeed-os, Milvus and FAISS coexist:
    - Dev/testing → FAISS (no external dependencies)
    - Production → Milvus (distributed, persistent)

The retrieval pipeline auto-selects based on configuration.

Requires:
    pip install pymilvus>=2.4
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from pymilvus import (
        MilvusClient,
        Collection,
        CollectionSchema,
        FieldSchema,
        DataType,
        connections,
        utility,
    )
    HAS_PYMILVUS = True
except ImportError:
    HAS_PYMILVUS = False
    logger.warning("pymilvus not installed — using FAISS fallback. pip install pymilvus>=2.4")


# ── Configuration ──────────────────────────────────────────────

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_ALIAS = "default"


# ── Collection Definitions ────────────────────────────────────

COLLECTIONS = {
    "item_embeddings": {
        "description": "Product/video embeddings from Two-Tower item encoder (256D)",
        "dim": 256,
        "metric": "COSINE",
        "index_type": "IVF_SQ8",
        "nlist": 4096,
        "partition_key": True,  # Gap 5: auto-partition by category_id
    },
    "visual_embeddings": {
        "description": "CLIP visual embeddings for content similarity (512D)",
        "dim": 512,
        "metric": "COSINE",
        "index_type": "IVF_SQ8",
        "nlist": 4096,
        "partition_key": True,  # Gap 5: auto-partition by category_id
    },
    "video_temporal_embeddings": {
        "description": "VideoMAE temporal embeddings (768D)",
        "dim": 768,
        "metric": "COSINE",
        "index_type": "IVF_SQ8",
        "nlist": 2048,
        "partition_key": False,  # Videos are not category-specific
    },
}


class MilvusVectorSearch:
    """Production vector search using Milvus.

    Handles:
        - Auto-connection with retry
        - Collection creation with proper indexing
        - Batch upsert for embedding updates
        - ANN search for retrieval (<5ms on 100M vectors)
        - Graceful fallback to FAISS when Milvus is unavailable
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        use_faiss_fallback: bool = True,
    ):
        self.host = host or MILVUS_HOST
        self.port = port or MILVUS_PORT
        self.use_faiss_fallback = use_faiss_fallback
        self._connected = False
        self._collections: dict[str, Any] = {}

    def connect(self, max_retries: int = 3) -> bool:
        """Connect to Milvus server with retry logic."""
        if not HAS_PYMILVUS:
            logger.warning("pymilvus not installed — vector search will use FAISS")
            return False

        for attempt in range(max_retries):
            try:
                connections.connect(
                    alias=MILVUS_ALIAS,
                    host=self.host,
                    port=self.port,
                    timeout=10,
                )
                self._connected = True
                logger.info("Connected to Milvus: %s:%s", self.host, self.port)
                return True
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(
                    "Milvus connection attempt %d failed: %s (retrying in %ds)",
                    attempt + 1, e, wait,
                )
                time.sleep(wait)

        logger.error("Failed to connect to Milvus after %d attempts", max_retries)
        return False

    def ensure_collection(self, collection_name: str) -> bool:
        """Create a collection if it doesn't exist."""
        if not self._connected:
            return False

        if collection_name not in COLLECTIONS:
            logger.error("Unknown collection: %s", collection_name)
            return False

        try:
            if utility.has_collection(collection_name):
                self._collections[collection_name] = Collection(collection_name)
                self._collections[collection_name].load()
                return True

            config = COLLECTIONS[collection_name]

            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=config["dim"]),
                FieldSchema(name="metadata", dtype=DataType.JSON),
                # Gap 5: category_id for partition routing (Pailitao pattern)
                FieldSchema(
                    name="category_id",
                    dtype=DataType.INT64,
                    is_partition_key=config.get("partition_key", False),
                ),
                # Gap 3: content_type for cross-modal filtering (product|video|video_frame)
                FieldSchema(
                    name="content_type",
                    dtype=DataType.VARCHAR,
                    max_length=32,
                    default_value="product",
                ),
            ]
            schema = CollectionSchema(fields=fields, description=config["description"])

            collection = Collection(name=collection_name, schema=schema)

            # Build vector index for fast ANN search
            index_params = {
                "index_type": config["index_type"],
                "metric_type": config["metric"],
                "params": {"nlist": config["nlist"]},
            }
            collection.create_index("embedding", index_params)

            # Gap 5: Scalar index on category_id for fast pre-filtering
            if config.get("partition_key", False):
                collection.create_index(
                    "category_id",
                    {"index_type": "STL_SORT"},
                )
                logger.info(
                    "Milvus partition_key enabled on category_id for %s",
                    collection_name,
                )

            # Scalar index on content_type for cross-modal filtering
            collection.create_index(
                "content_type",
                {"index_type": "Trie"},
            )

            collection.load()

            self._collections[collection_name] = collection
            logger.info("Milvus collection created: %s (%dD)", collection_name, config["dim"])
            return True

        except Exception as e:
            logger.error("Failed to create collection %s: %s", collection_name, e)
            return False

    def upsert(
        self,
        collection_name: str,
        ids: list[str],
        embeddings: list[list[float]],
        metadata: Optional[list[dict]] = None,
    ) -> int:
        """Insert or update embeddings.

        Args:
            collection_name: target collection
            ids: list of unique IDs (item_id, video_id, etc.)
            embeddings: list of embedding vectors
            metadata: optional metadata dicts per item

        Returns:
            Number of successfully upserted items.
        """
        if not self._connected or collection_name not in self._collections:
            return 0

        try:
            collection = self._collections[collection_name]
            meta = metadata or [{}] * len(ids)

            data = [ids, embeddings, meta]
            result = collection.upsert(data)

            logger.debug("Milvus upsert: %s → %d items", collection_name, len(ids))
            return len(ids)

        except Exception as e:
            logger.error("Milvus upsert failed: %s", e)
            return 0

    def search(
        self,
        collection_name: str,
        query_vectors: list[list[float]],
        top_k: int = 500,
        nprobe: int = 16,
        filter_expr: Optional[str] = None,
    ) -> list[list[dict]]:
        """Search for nearest neighbors.

        Args:
            collection_name: collection to search
            query_vectors: list of query embeddings
            top_k: number of results per query
            nprobe: search precision (higher = more accurate, slower)
            filter_expr: optional Milvus filter expression

        Returns:
            List of lists of {id, score, metadata} dicts.
        """
        if not self._connected or collection_name not in self._collections:
            if self.use_faiss_fallback:
                return self._faiss_fallback_search(collection_name, query_vectors, top_k)
            return [[] for _ in query_vectors]

        try:
            collection = self._collections[collection_name]

            search_params = {
                "metric_type": COLLECTIONS[collection_name]["metric"],
                "params": {"nprobe": nprobe},
            }

            results = collection.search(
                data=query_vectors,
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=["metadata"],
            )

            formatted = []
            for hits in results:
                formatted.append([
                    {
                        "id": hit.id,
                        "score": hit.score,
                        "metadata": hit.entity.get("metadata", {}),
                    }
                    for hit in hits
                ])

            return formatted

        except Exception as e:
            logger.error("Milvus search failed: %s", e)
            if self.use_faiss_fallback:
                return self._faiss_fallback_search(collection_name, query_vectors, top_k)
            return [[] for _ in query_vectors]

    def _faiss_fallback_search(
        self,
        collection_name: str,
        query_vectors: list[list[float]],
        top_k: int,
    ) -> list[list[dict]]:
        """Fall back to FAISS when Milvus is unavailable."""
        try:
            from ml.serving.faiss_index import FaissIndex
            import numpy as np

            index = FaissIndex.get_instance()
            if index is None or index.index is None:
                return [[] for _ in query_vectors]

            logger.info("Using FAISS fallback for %s (top_k=%d)", collection_name, top_k)
            results = []
            for qv in query_vectors:
                query = np.array([qv], dtype=np.float32)
                item_ids, scores = index.search(query, k=top_k)
                batch_results = []
                for item_id, score in zip(item_ids[0], scores[0]):
                    if item_id != -1:
                        batch_results.append({
                            "id": str(item_id),
                            "score": float(score),
                        })
                results.append(batch_results)
            return results
        except (ImportError, Exception) as e:
            logger.warning("FAISS fallback search failed: %s", e)
            return [[] for _ in query_vectors]

    def delete(self, collection_name: str, ids: list[str]) -> int:
        """Delete embeddings by ID."""
        if not self._connected or collection_name not in self._collections:
            return 0

        try:
            collection = self._collections[collection_name]
            expr = f'id in {ids}'
            collection.delete(expr)
            return len(ids)
        except Exception as e:
            logger.error("Milvus delete failed: %s", e)
            return 0

    def get_count(self, collection_name: str) -> int:
        """Get total number of vectors in a collection."""
        if not self._connected or collection_name not in self._collections:
            return 0
        try:
            return self._collections[collection_name].num_entities
        except Exception:
            return 0
