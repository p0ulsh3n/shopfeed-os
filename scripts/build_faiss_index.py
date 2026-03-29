"""
Build FAISS Index — Rebuild de l'index ANN depuis la base de données.

Usage:
  python -m scripts.build_faiss_index --source catalog_db --output faiss_indexes/products_clip.faiss
  python -m scripts.build_faiss_index --source feed_db --dim 512 --batch 10000

Pipeline:
  1. Query PostgreSQL → clip_embedding pour tous les items actifs
  2. Build FAISS HNSW index
  3. Save localement + upload S3
  4. Update ml_embedding_index table
"""

from __future__ import annotations
import argparse
import asyncio
import logging
import os
import time
from typing import Generator

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

S3_BUCKET = os.environ.get("S3_BUCKET_ML", "shopfeed-ml-models")
INDEX_OUTPUT_DIR = os.environ.get("FAISS_OUTPUT_DIR", "/tmp/faiss_indexes")


async def fetch_product_embeddings(
    db_dsn: str,
    batch_size: int = 10000,
) -> Generator[tuple[list[str], np.ndarray], None, None]:
    """
    Fetch clip_embedding depuis catalog_db par batch.
    Yields (item_ids, embeddings_np) par batch.
    """
    import asyncpg

    pool = await asyncpg.create_pool(db_dsn, min_size=2, max_size=5)
    try:
        offset = 0
        while True:
            rows = await pool.fetch(
                """
                SELECT id::text, clip_embedding
                FROM products
                WHERE status = 'active'
                  AND clip_embedding IS NOT NULL
                ORDER BY published_at DESC
                LIMIT $1 OFFSET $2
                """,
                batch_size, offset
            )
            if not rows:
                break

            ids = [r["id"] for r in rows]
            embs = np.array([list(r["clip_embedding"]) for r in rows], dtype=np.float32)
            yield ids, embs
            offset += batch_size
            logger.info(f"Fetched {offset} products so far...")

    finally:
        await pool.close()


def build_index_from_db(
    db_dsn: str,
    dim: int,
    batch_size: int,
    output_path: str,
) -> int:
    """
    Build complet depuis la DB.
    Returns: nombre de vecteurs indexés.
    """
    from ml.serving.faiss_index import FaissIndex
    import faiss

    all_ids = []
    all_embs = []

    logger.info("Fetching embeddings from database...")
    t_fetch = time.time()

    async def collect():
        async for ids, embs in fetch_product_embeddings(db_dsn, batch_size):
            all_ids.extend(ids)
            all_embs.append(embs)

    asyncio.run(collect())

    if not all_ids:
        logger.warning("No embeddings found. Empty index will be created.")
        return 0

    combined_embs = np.vstack(all_embs)
    logger.info(f"Fetched {len(all_ids)} embeddings in {time.time()-t_fetch:.1f}s")

    # Build FAISS index
    t_build = time.time()
    idx = FaissIndex(dim=combined_embs.shape[1])
    idx.build(combined_embs, all_ids)
    logger.info(f"FAISS index built in {time.time()-t_build:.1f}s")

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    idx.save(output_path)
    logger.info(f"Index saved to {output_path}")

    return len(all_ids)


def upload_to_s3(local_path: str, s3_key: str, bucket: str = S3_BUCKET) -> None:
    """Upload l'index FAISS vers S3."""
    import boto3
    s3 = boto3.client("s3")
    s3.upload_file(local_path, bucket, s3_key)
    # Aussi les .ids
    ids_path = local_path + ".ids"
    if os.path.exists(ids_path):
        s3.upload_file(ids_path, bucket, s3_key + ".ids")
    logger.info(f"Uploaded to s3://{bucket}/{s3_key}")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS ANN Index")
    parser.add_argument(
        "--db_dsn",
        default=os.environ.get("CATALOG_DB_DSN", "postgresql://localhost/catalog_db"),
        help="DSN PostgreSQL de la base de données"
    )
    parser.add_argument("--dim", type=int, default=512, help="Dimension des embeddings")
    parser.add_argument("--batch", type=int, default=10000, help="Batch size pour fetch DB")
    parser.add_argument(
        "--output",
        default=os.path.join(INDEX_OUTPUT_DIR, "products_clip_index.faiss"),
        help="Chemin de sortie local"
    )
    parser.add_argument("--s3_key", default="faiss_indexes/products_clip_index.faiss")
    parser.add_argument("--upload_s3", action="store_true", help="Upload vers S3 après build")
    args = parser.parse_args()

    logger.info(f"Building FAISS index: dim={args.dim}, batch={args.batch}")
    n_vectors = build_index_from_db(
        db_dsn=args.db_dsn,
        dim=args.dim,
        batch_size=args.batch,
        output_path=args.output,
    )

    if args.upload_s3 and n_vectors > 0:
        upload_to_s3(args.output, args.s3_key)

    logger.info(f"Done. {n_vectors} vectors indexed.")


if __name__ == "__main__":
    main()
