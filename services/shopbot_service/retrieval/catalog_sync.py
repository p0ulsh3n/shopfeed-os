"""
Real-Time Catalog Synchronization
===================================
Listens to PostgreSQL NOTIFY events for instant catalog updates.
No polling — zero-latency sync via asyncpg LISTEN.

Strategy:
- FULL SYNC: On startup or manual trigger — indexes all products for a shop
- INCREMENTAL SYNC: Via LISTEN/NOTIFY — handles INSERT/UPDATE/DELETE in real-time
- BATCH UPSERT: Efficient bulk insert using asyncpg's executemany

This ensures the ShopBot NEVER serves stale product data.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

import asyncpg
import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from services.shopbot_service.config import get_settings
from services.shopbot_service.database.connection import (
    AsyncSessionLocal,
    get_raw_pool,
)
from services.shopbot_service.embeddings.encoder import (
    EmbeddingEncoder,
    float32_to_pgvector_str,
)
from services.shopbot_service.models.schemas import (
    CatalogEvent,
    CatalogEventType,
    CatalogSyncResponse,
    Product,
)

logger = logging.getLogger(__name__)
settings = get_settings()


class CatalogSyncService:
    """
    Manages real-time and bulk synchronization of shop catalogs
    into the ShopBot vector index.

    Lifecycle:
        sync = CatalogSyncService()
        await sync.start_listener()   # Background LISTEN task
        # ... app runs ...
        await sync.stop()             # Graceful shutdown
    """

    def __init__(self) -> None:
        self._encoder = EmbeddingEncoder.get_instance()
        self._listener_task: asyncio.Task | None = None
        self._running = False
        self._processing_semaphore = asyncio.Semaphore(4)  # Max 4 concurrent syncs

    # ─────────────────────── LIFECYCLE ───────────────────────────

    async def start_listener(self) -> None:
        """
        Start the background LISTEN task.
        Connects to PostgreSQL and listens on catalog_notify_channel.
        Auto-reconnects on connection loss.
        """
        self._running = True
        self._listener_task = asyncio.create_task(
            self._listen_loop(), name="shopbot-catalog-listener"
        )
        logger.info(
            f"Catalog sync listener started "
            f"[channel={settings.catalog_notify_channel}]"
        )

    async def stop(self) -> None:
        """Gracefully stop the listener."""
        self._running = False
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        logger.info("Catalog sync listener stopped")

    # ─────────────────────── LISTEN LOOP ─────────────────────────

    async def _listen_loop(self) -> None:
        """
        Persistent LISTEN loop with auto-reconnect.
        Uses asyncpg directly (not SQLAlchemy) for LISTEN/NOTIFY support.
        """
        backoff = 1
        while self._running:
            try:
                pool = await get_raw_pool()
                async with pool.acquire() as conn:
                    await conn.add_listener(
                        settings.catalog_notify_channel,
                        self._on_notification,
                    )
                    logger.info(
                        f"Listening on channel: {settings.catalog_notify_channel}"
                    )
                    backoff = 1  # Reset backoff on successful connect

                    # Keep the connection alive
                    while self._running:
                        await asyncio.sleep(5)
                        # Heartbeat to detect dead connections
                        await conn.execute("SELECT 1")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    f"Catalog listener error (retrying in {backoff}s): {e}"
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)  # Exponential backoff, max 60s

    async def _on_notification(
        self,
        connection: asyncpg.Connection,
        pid: int,
        channel: str,
        payload: str,
    ) -> None:
        """
        Callback fired when PostgreSQL sends a NOTIFY.
        Parses the JSON payload and dispatches to the appropriate handler.
        """
        try:
            data = json.loads(payload)
            event = CatalogEvent(**data)
            logger.debug(
                f"Catalog event: {event.event_type} "
                f"shop={event.shop_id} product={event.product_id}"
            )

            # Schedule processing without blocking the listener
            asyncio.create_task(
                self._process_event(event),
                name=f"shopbot-sync-{event.product_id[:8]}",
            )

        except Exception as e:
            logger.error(f"Failed to process catalog notification: {e}")

    async def _process_event(self, event: CatalogEvent) -> None:
        """Process a single catalog event with semaphore-limited concurrency."""
        async with self._processing_semaphore:
            try:
                if event.event_type == CatalogEventType.DELETE:
                    await self._delete_product_from_index(
                        event.shop_id, event.product_id
                    )
                elif event.event_type in (
                    CatalogEventType.INSERT, CatalogEventType.UPDATE
                ):
                    if event.product_data:
                        await self._upsert_product_to_index(
                            event.shop_id, event.product_id, event.product_data
                        )
                    else:
                        # Fetch from DB if payload was truncated
                        logger.warning(
                            f"No product data in event for {event.product_id}, "
                            f"fetching from DB..."
                        )
            except Exception as e:
                logger.error(
                    f"Failed to process event "
                    f"{event.event_type} {event.product_id}: {e}"
                )

    # ─────────────────────── FULL SYNC ───────────────────────────

    async def full_sync_shop(
        self,
        shop_id: str,
        products: list[Product],
        force_rebuild: bool = False,
    ) -> CatalogSyncResponse:
        """
        Full catalog re-indexing for a shop.
        Called on:
        - First-time shop onboarding
        - Manual sync trigger (POST /catalog/sync)
        - Recovery after missed events

        Steps:
        1. Optionally clear existing embeddings for this shop
        2. Batch encode all products (parallel)
        3. Bulk upsert into shopbot_product_embeddings
        """
        t_start = datetime.now(timezone.utc)
        indexed = 0
        failed = 0

        logger.info(
            f"Starting full sync for shop={shop_id} "
            f"products={len(products)} force={force_rebuild}"
        )

        if force_rebuild:
            await self._clear_shop_index(shop_id)

        # Process in batches to avoid memory overflow
        batch_size = settings.sync_batch_size
        for batch_start in range(0, len(products), batch_size):
            batch = products[batch_start: batch_start + batch_size]
            try:
                batch_indexed = await self._upsert_product_batch(shop_id, batch)
                indexed += batch_indexed
            except Exception as e:
                logger.error(
                    f"Batch sync failed for shop={shop_id} "
                    f"batch={batch_start}: {e}"
                )
                failed += len(batch)

        duration_ms = (
            datetime.now(timezone.utc) - t_start
        ).total_seconds() * 1000

        # Log to sync table
        await self._log_sync(shop_id, "full", indexed, failed, duration_ms)

        logger.info(
            f"Full sync complete: shop={shop_id} "
            f"indexed={indexed} failed={failed} "
            f"duration={duration_ms:.0f}ms"
        )

        return CatalogSyncResponse(
            shop_id=shop_id,
            products_indexed=indexed,
            products_failed=failed,
            duration_ms=duration_ms,
            status="success" if failed == 0 else "partial",
        )

    # ─────────────────────── UPSERT BATCH ────────────────────────

    async def _upsert_product_batch(
        self, shop_id: str, products: list[Product]
    ) -> int:
        """
        Encode and upsert a batch of products.
        Returns number of successfully indexed products.
        """
        # Prepare texts for batch encoding
        product_texts = [p.to_text_for_embedding() for p in products]
        bm25_texts = [p.to_bm25_text() for p in products]

        # Batch encode (runs in thread pool)
        float32_embeddings, _, _ = await self._encoder.encode_passages_batch(
            product_texts
        )

        # Prepare metadata JSONB for each product
        records = []
        for i, product in enumerate(products):
            emb_vec = float32_embeddings[i]
            metadata = self._product_to_metadata(product)

            records.append({
                "shop_id": shop_id,
                "product_id": product.id,
                "product_text": bm25_texts[i],
                "embedding_vector": float32_to_pgvector_str(emb_vec),
                "metadata": json.dumps(metadata),
            })

        # Bulk upsert via SQLAlchemy
        async with AsyncSessionLocal() as session:
            for record in records:
                await session.execute(
                    text("""
                        INSERT INTO shopbot_product_embeddings
                            (shop_id, product_id, product_text,
                             embedding_float32, metadata)
                        VALUES
                            (:shop_id, :product_id, :product_text,
                             :embedding_vector::vector, :metadata::jsonb)
                        ON CONFLICT (shop_id, product_id) DO UPDATE SET
                            product_text      = EXCLUDED.product_text,
                            embedding_float32 = EXCLUDED.embedding_float32,
                            metadata          = EXCLUDED.metadata,
                            updated_at        = NOW()
                    """),
                    record,
                )
            await session.commit()

        return len(records)

    async def _upsert_product_to_index(
        self, shop_id: str, product_id: str, product_data: dict
    ) -> None:
        """
        Single-product upsert from a NOTIFY event.
        Triggered by INSERT or UPDATE on the products table.
        """
        product = self._event_data_to_product(product_id, shop_id, product_data)
        await self._upsert_product_batch(shop_id, [product])
        logger.debug(f"Upserted product {product_id} for shop {shop_id}")

    # ─────────────────────── DELETE ──────────────────────────────

    async def _delete_product_from_index(
        self, shop_id: str, product_id: str
    ) -> None:
        """
        Remove a product from the vector index.
        Called when a product is deleted from the main catalog.
        """
        async with AsyncSessionLocal() as session:
            await session.execute(
                text("""
                    DELETE FROM shopbot_product_embeddings
                    WHERE shop_id = :shop_id AND product_id = :product_id
                """),
                {"shop_id": shop_id, "product_id": product_id},
            )
            await session.commit()
        logger.debug(f"Deleted product {product_id} from shop {shop_id} index")

    async def _clear_shop_index(self, shop_id: str) -> None:
        """Remove ALL product embeddings for a shop (used on full rebuild)."""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                text("""
                    DELETE FROM shopbot_product_embeddings
                    WHERE shop_id = :shop_id
                    RETURNING product_id
                """),
                {"shop_id": shop_id},
            )
            count = len(result.fetchall())
            await session.commit()
        logger.info(f"Cleared {count} embeddings for shop={shop_id}")

    # ─────────────────────── SYNC LOG ────────────────────────────

    async def _log_sync(
        self,
        shop_id: str,
        sync_type: str,
        indexed: int,
        failed: int,
        duration_ms: float,
    ) -> None:
        """Write sync result to the shopbot_sync_log table."""
        status = "success" if failed == 0 else ("failed" if indexed == 0 else "partial")
        async with AsyncSessionLocal() as session:
            await session.execute(
                text("""
                    INSERT INTO shopbot_sync_log
                        (shop_id, sync_type, products_count, status, completed_at)
                    VALUES
                        (:shop_id, :sync_type, :count, :status, NOW())
                """),
                {
                    "shop_id": shop_id,
                    "sync_type": sync_type,
                    "count": indexed + failed,
                    "status": status,
                },
            )
            await session.commit()

    # ─────────────────────── HELPERS ─────────────────────────────

    def _product_to_metadata(self, product: Product) -> dict:
        """Serialize product to JSONB metadata for storage."""
        return {
            "name": product.name,
            "description": product.description,
            "price": product.price,
            "currency": product.currency,
            "category": product.category,
            "subcategory": product.subcategory,
            "tags": product.tags,
            "availability": product.availability.value,
            "stock_quantity": product.stock_quantity,
            "sku": product.sku,
            "attributes": product.attributes,
            "images": [
                {"url": img.url, "alt": img.alt, "is_primary": img.is_primary}
                for img in product.images
            ],
        }

    def _event_data_to_product(
        self, product_id: str, shop_id: str, data: dict
    ) -> Product:
        """Convert raw NOTIFY event data to a Product object."""
        from services.shopbot_service.models.schemas import ProductAvailability
        return Product(
            id=product_id,
            shop_id=shop_id,
            name=data.get("name", ""),
            description=data.get("description"),
            price=float(data.get("price", 0)),
            currency=data.get("currency", "XAF"),
            category=data.get("category"),
            subcategory=data.get("subcategory"),
            tags=data.get("tags", []),
            availability=ProductAvailability(
                data.get("availability", "in_stock")
            ),
            stock_quantity=data.get("stock_quantity"),
            sku=data.get("sku"),
            attributes=data.get("attributes", {}),
        )
