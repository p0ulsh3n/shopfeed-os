"""
Build FAISS Index — Rebuild de l'index ANN depuis la base de données.

Usage:
  python -m scripts.build_faiss_index --output faiss_indexes/products_clip.faiss

Pipeline:
  1. Query PostgreSQL via SQLAlchemy ORM → clip_embedding pour tous les items actifs
  2. Build FAISS HNSW index
  3. Save localement + upload S3
  4. Update ml_embedding_index table

MIGRATION: asyncpg brut → SQLAlchemy 2.0 select(ProductORM)
"""

from __future__ import annotations
import argparse
import asyncio
import logging
import os
import time
from typing import AsyncGenerator

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

S3_BUCKET = os.environ.get("S3_BUCKET_ML", "shopfeed-ml-models")
INDEX_OUTPUT_DIR = os.environ.get("FAISS_OUTPUT_DIR", "/tmp/faiss_indexes")


async def fetch_product_embeddings(
    batch_size: int = 10000,
) -> AsyncGenerator[tuple[list[str], np.ndarray], None]:
    """
    Fetch clip_embedding depuis catalog_db via SQLAlchemy ORM.
    Yields (item_ids, embeddings_np) par batch.

    MIGRATION:
    - AVANT: asyncpg.create_pool() + pool.fetch(raw SQL SELECT ...)
    - APRÈS: SQLAlchemy select(ProductORM).where(...)
    """
    from sqlalchemy import select, func
    from shared.db.session import AsyncSessionLocal
    from shared.db.models.product import ProductORM

    offset = 0
    while True:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(ProductORM.id, ProductORM.clip_embedding)
                .where(
                    ProductORM.status == "active",
                    ProductORM.clip_embedding.isnot(None),
                    ProductORM.deleted_at.is_(None),
                )
                .order_by(ProductORM.published_at.desc())
                .limit(batch_size)
                .offset(offset)
            )
            rows = result.fetchall()

        if not rows:
            break

        ids = [str(row[0]) for row in rows]
        embs = np.array(
            [row[1] for row in rows if row[1]],
            dtype=np.float32,
        )
        if len(ids) != len(embs):
            logger.warning("Mismatch ids/embeddings at offset=%d — skipping batch", offset)
            break

        yield ids, embs
        offset += batch_size
        logger.info("Fetched %d products so far...", offset)


def build_index_from_db(
    dim: int,
    batch_size: int,
    output_path: str,
) -> int:
    """
    Build complet depuis la DB via ORM.
    Returns: nombre de vecteurs indexés.
    """
    from ml.serving.faiss_index import FaissIndex

    all_ids: list[str] = []
    all_embs: list[np.ndarray] = []

    logger.info("Fetching embeddings from database (SQLAlchemy ORM)...")
    t_fetch = time.time()

    async def collect() -> None:
        async for ids, embs in fetch_product_embeddings(batch_size):
            all_ids.extend(ids)
            all_embs.append(embs)

    asyncio.run(collect())

    if not all_ids:
        logger.warning("No embeddings found. Empty index will be created.")
        return 0

    combined_embs = np.vstack(all_embs)
    logger.info("Fetched %d embeddings in %.1fs", len(all_ids), time.time() - t_fetch)

    # Build FAISS index
    t_build = time.time()
    idx = FaissIndex(dim=combined_embs.shape[1])
    idx.build(combined_embs, all_ids)
    logger.info("FAISS index built in %.1fs", time.time() - t_build)

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    idx.save(output_path)
    logger.info("Index saved to %s", output_path)

    return len(all_ids)


def upload_to_s3(local_path: str, s3_key: str, bucket: str = S3_BUCKET) -> None:
    """Upload l'index FAISS vers S3."""
    import boto3
    s3 = boto3.client("s3")
    s3.upload_file(local_path, bucket, s3_key)
    ids_path = local_path + ".ids"
    if os.path.exists(ids_path):
        s3.upload_file(ids_path, bucket, s3_key + ".ids")
    logger.info("Uploaded to s3://%s/%s", bucket, s3_key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS ANN Index")
    parser.add_argument("--dim", type=int, default=512, help="Dimension des embeddings")
    parser.add_argument("--batch", type=int, default=10000, help="Batch size pour fetch DB")
    parser.add_argument(
        "--output",
        default=os.path.join(INDEX_OUTPUT_DIR, "products_clip_index.faiss"),
        help="Chemin de sortie local",
    )
    parser.add_argument("--s3_key", default="faiss_indexes/products_clip_index.faiss")
    parser.add_argument("--upload_s3", action="store_true", help="Upload vers S3 après build")
    args = parser.parse_args()

    logger.info("Building FAISS index: dim=%d, batch=%d", args.dim, args.batch)
    n_vectors = build_index_from_db(
        dim=args.dim,
        batch_size=args.batch,
        output_path=args.output,
    )

    if args.upload_s3 and n_vectors > 0:
        upload_to_s3(args.output, args.s3_key)

    logger.info("Done. %d vectors indexed.", n_vectors)


if __name__ == "__main__":
    main()
