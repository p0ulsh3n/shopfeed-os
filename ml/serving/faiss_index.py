"""
FAISS ANN Index — build + search pour le retrieval Two-Tower.
Gère la construction, le chargement depuis S3, et la recherche ANN.

Usage:
  index = FaissIndex()
  index.build(embeddings, item_ids)  # build depuis embeddings numpy
  index.save(s3_path)               # persiste sur S3
  index.load(s3_path)               # charge depuis S3 ou disque
  results = index.search(query, k)  # ANN search
"""

from __future__ import annotations
import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss not installed. FAISS index will be unavailable.")


class FaissIndex:
    """
    Wrapper FAISS HNSW pour ANN search Two-Tower retrieval.

    Architecture HNSW (Hierarchical Navigable Small World):
      - M=16 connexions par noeud
      - ef_construction=64 construction time vs quality
      - ef_search=64 query time vs recall trade-off
      - Distance: cosine (via normalisation L2 + dotproduct)
      - Capacité: jusqu'à 2M+ items

    SLA: <10ms pour k=2000 sur 500K produits indexés.
    """

    def __init__(
        self,
        dim: int = 512,
        m: int = 16,
        ef_construction: int = 64,
        ef_search: int = 64,
        index_path: str | None = None,
    ):
        self.dim = dim
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.index_path = index_path or os.environ.get(
            "FAISS_INDEX_PATH", "data/faiss_index.bin"
        )
        self._index: Optional["faiss.Index"] = None
        self._item_ids: list[str] = []  # mapping int64 faiss idx → item_id str

    def _build_index(self) -> "faiss.Index":
        """Construit un index HNSW + PQ pour cosine similarity."""
        if not FAISS_AVAILABLE:
            raise RuntimeError("faiss not installed")
        # Flat inner product (après normalisation L2 = cosine)
        base_index = faiss.IndexFlatIP(self.dim)
        # Wrapper HNSW pour ANN
        index = faiss.IndexHNSWFlat(self.dim, self.m)
        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.efSearch = self.ef_search
        return index

    def build(self, embeddings: np.ndarray, item_ids: list[str]) -> None:
        """
        Construit l'index depuis un array d'embeddings.

        Args:
            embeddings: [N, dim] float32
            item_ids:   liste de N item_id strings (ordre correspondant)
        """
        if not FAISS_AVAILABLE:
            raise RuntimeError("faiss not installed")

        assert embeddings.shape[1] == self.dim, (
            f"Embedding dim mismatch: expected {self.dim}, got {embeddings.shape[1]}"
        )
        assert len(item_ids) == embeddings.shape[0]

        logger.info(f"Building FAISS HNSW index for {len(item_ids)} items (dim={self.dim})")

        # Normalisation L2 pour cosine similarity via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-8, norms)
        emb_norm = embeddings / norms
        emb_norm = emb_norm.astype(np.float32)

        self._index = self._build_index()
        self._index.add(emb_norm)
        self._item_ids = list(item_ids)

        logger.info(f"FAISS index built: {self._index.ntotal} vectors indexed.")

    def search(
        self,
        query_vec: np.ndarray,
        k: int = 2000,
    ) -> tuple[list[str], list[float]]:
        """
        ANN search: retourne les k plus proches voisins.

        Args:
            query_vec: [1, dim] ou [dim] float32
            k:         nombre de candidats à retourner

        Returns:
            (item_ids, scores) — triés par score décroissant
        """
        if self._index is None:
            raise RuntimeError("FAISS index not loaded. Call load() or build() first.")

        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        # Normalisation L2
        norm = np.linalg.norm(query_vec, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1e-8, norm)
        query_norm = (query_vec / norm).astype(np.float32)

        k = min(k, self._index.ntotal)
        distances, indices = self._index.search(query_norm, k)

        result_ids = []
        result_scores = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            result_ids.append(self._item_ids[idx])
            result_scores.append(float(dist))

        return result_ids, result_scores

    def size(self) -> int:
        """Retourne le nombre de vecteurs indexés."""
        if self._index is None:
            return 0
        return self._index.ntotal

    def save(self, path: str | None = None) -> None:
        """Sauvegarde l'index + le mapping item_ids sur disque."""
        save_path = path or self.index_path
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        if not FAISS_AVAILABLE or self._index is None:
            raise RuntimeError("No index to save")

        faiss.write_index(self._index, save_path)
        ids_path = save_path + ".ids"
        with open(ids_path, "wb") as f:
            pickle.dump(self._item_ids, f)

        logger.info(f"FAISS index saved to {save_path} ({self.size()} vectors)")

    def load(self, path: str | None = None) -> None:
        """Charge l'index depuis le disque."""
        load_path = path or self.index_path

        if not FAISS_AVAILABLE:
            raise RuntimeError("faiss not installed")

        if not os.path.exists(load_path):
            logger.warning(f"FAISS index not found at {load_path}. Skipping load.")
            return

        self._index = faiss.read_index(load_path)
        ids_path = load_path + ".ids"
        if os.path.exists(ids_path):
            with open(ids_path, "rb") as f:
                self._item_ids = pickle.load(f)

        logger.info(f"FAISS index loaded from {load_path}: {self.size()} vectors")

    def load_from_s3(self, s3_path: str, bucket: str = "shopfeed-ml-models") -> None:
        """Télécharge l'index depuis S3 puis charge en mémoire."""
        try:
            import boto3
            s3 = boto3.client("s3")
            with tempfile.TemporaryDirectory() as tmpdir:
                local_path = os.path.join(tmpdir, "index.faiss")
                s3.download_file(bucket, s3_path, local_path)
                ids_s3_path = s3_path + ".ids"
                ids_local_path = local_path + ".ids"
                try:
                    s3.download_file(bucket, ids_s3_path, ids_local_path)
                except Exception:
                    pass
                self.load(local_path)
        except Exception as e:
            logger.error(f"Failed to load FAISS index from S3 {s3_path}: {e}")
            raise

    def add_items(self, new_embeddings: np.ndarray, new_ids: list[str]) -> None:
        """
        Ajoute de nouveaux items à l'index (pour les nouveaux produits publiés).
        Note: HNSW ne supporte pas la suppression — rebuild nécessaire pour les retraits.
        """
        if self._index is None:
            raise RuntimeError("Index not initialized. Call build() first.")

        norms = np.linalg.norm(new_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-8, norms)
        emb_norm = (new_embeddings / norms).astype(np.float32)

        self._index.add(emb_norm)
        self._item_ids.extend(new_ids)
        logger.info(f"Added {len(new_ids)} items to FAISS index. Total: {self.size()}")


def get_index_size() -> int:
    """Helper pour le health endpoint — retourne la taille de l'index global."""
    try:
        idx = FaissIndex()
        idx.load()
        return idx.size()
    except Exception:
        return 0
