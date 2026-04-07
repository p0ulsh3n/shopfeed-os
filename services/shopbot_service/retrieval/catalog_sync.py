"""
Real-Time Catalog Synchronization
===================================
Listens to PostgreSQL NOTIFY events for instant catalog updates.
No polling — zero-latency sync via asyncpg LISTEN.

Strategy:
- FULL SYNC: On startup or manual trigger — indexes all products for a shop
- INCREMENTAL SYNC: Via LISTEN/NOTIFY — handles INSERT/UPDATE/DELETE in real-time
- BATCH UPSERT: pg_insert ON CONFLICT DO UPDATE via SQLAlchemy ORM

MIGRATION:
- AVANT: session.execute(text("INSERT INTO shopbot_product_embeddings ..."))
  → SQL brut inline dans le code applicatif
- APRÈS: ShopbotProductEmbeddingRepository avec pg_insert ON CONFLICT DO UPDATE
  → ORM SQLAlchemy 2.0, paramètres bindés, zéro SQL brut
- GARDÉ: asyncpg LISTEN/NOTIFY (justifié — SQLAlchemy ne supporte pas LISTEN nativement)
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

import asyncpg
import numpy as np
from sqlalchemy.dialects.postgresql import insert as pg_insert
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


# ─────────────────────── ORM Table Reference ─────────────────────
# On importe la table via SQLAlchemy core pour les pg_insert.
# Les modèles ORM shopbot sont locaux au service (pas dans shared/db/models).
# Ils seront créés par Alembic depuis ces classes.

from sqlalchemy import Column, String, DateTime, Text, BigInteger, Integer
from sqlalchemy.dialects.postgresql import BYTEA, JSONB, UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase

from services.shopbot_service.database.connection import Base


class ShopbotProductEmbeddingORM(Base):
    """
    ORM model pour shopbot_product_embeddings.
    Alembic génère la migration depuis ce modèle.
    """
    __tablename__ = "shopbot_product_embeddings"
    __table_args__ = {"extend_existing": True}

    from sqlalchemy import UniqueConstraint
    import uuid as _uuid

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=_uuid.uuid4)
    shop_id = Column(String(100), nullable=False, index=True)
    product_id = Column(String(100), nullable=False)
    product_text = Column(Text, nullable=False)
    embedding_float32 = Column(Text, nullable=True)   # pgvector stocké comme text
    embedding_int8 = Column(BYTEA, nullable=True)
    embedding_binary = Column(BYTEA, nullable=True)
    metadata_ = Column("metadata", JSONB, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)


class ShopbotSyncLogORM(Base):
    """ORM model pour shopbot_sync_log."""
    __tablename__ = "shopbot_sync_log"
    __table_args__ = {"extend_existing": True}

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    shop_id = Column(String(100), nullable=False)
    sync_type = Column(String(30), nullable=False)
    products_count = Column(Integer, nullable=False, default=0)
    status = Column(String(30), nullable=False)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)


class ShopbotSessionORM(Base):
    """ORM model pour shopbot_sessions."""
    __tablename__ = "shopbot_sessions"
    __table_args__ = {"extend_existing": True}

    import uuid as _uuid2
    session_id = Column(String(200), primary_key=True)
    shop_id = Column(String(100), nullable=False, index=True)
    customer_id = Column(String(200), nullable=True)
    history = Column(JSONB, nullable=False, default=list)
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)


# ─────────────────────── Catalog Sync Service ────────────────────

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
        self._processing_semaphore = asyncio.Semaphore(4)

    # ─────────────────────── LIFECYCLE ───────────────────────────

    async def start_listener(self) -> None:
        """
        Start the background LISTEN task.
        asyncpg LISTEN utilisé ici car SQLAlchemy ne supporte pas LISTEN nativement.
        """
        self._running = True
        self._listener_task = asyncio.create_task(
            self._listen_loop(), name="shopbot-catalog-listener"
        )
        logger.info(
            "Catalog sync listener started [channel=%s]",
            settings.catalog_notify_channel,
        )

    async def stop(self) -> None:
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
        asyncpg JUSTIFIÉ ici — seule utilisation légitime de la pool brute.
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
                    logger.info("Listening on channel: %s", settings.catalog_notify_channel)
                    backoff = 1

                    while self._running:
                        await asyncio.sleep(5)
                        # Heartbeat — connexion asyncpg maintenue pour LISTEN
                        await conn.execute("SELECT 1")

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Catalog listener error (retrying in %ds): %s", backoff, exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    async def _on_notification(
        self,
        connection: asyncpg.Connection,
        pid: int,
        channel: str,
        payload: str,
    ) -> None:
        try:
            data = json.loads(payload)
            event = CatalogEvent(**data)
            logger.debug(
                "Catalog event: %s shop=%s product=%s",
                event.event_type, event.shop_id, event.product_id,
            )
            asyncio.create_task(
                self._process_event(event),
                name=f"shopbot-sync-{event.product_id[:8]}",
            )
        except Exception as exc:
            logger.error("Failed to process catalog notification: %s", exc)

    async def _process_event(self, event: CatalogEvent) -> None:
        async with self._processing_semaphore:
            try:
                if event.event_type == CatalogEventType.DELETE:
                    await self._delete_product_from_index(event.shop_id, event.product_id)
                elif event.event_type in (CatalogEventType.INSERT, CatalogEventType.UPDATE):
                    if event.product_data:
                        await self._upsert_product_to_index(
                            event.shop_id, event.product_id, event.product_data
                        )
            except Exception as exc:
                logger.error(
                    "Failed to process event %s %s: %s",
                    event.event_type, event.product_id, exc,
                )

    # ─────────────────────── FULL SYNC ───────────────────────────

    async def full_sync_shop(
        self,
        shop_id: str,
        products: list[Product],
        force_rebuild: bool = False,
    ) -> CatalogSyncResponse:
        t_start = datetime.now(timezone.utc)
        indexed = 0
        failed = 0

        logger.info(
            "Starting full sync for shop=%s products=%d force=%s",
            shop_id, len(products), force_rebuild,
        )

        if force_rebuild:
            await self._clear_shop_index(shop_id)

        batch_size = settings.sync_batch_size
        for batch_start in range(0, len(products), batch_size):
            batch = products[batch_start: batch_start + batch_size]
            try:
                batch_indexed = await self._upsert_product_batch(shop_id, batch)
                indexed += batch_indexed
            except Exception as exc:
                logger.error(
                    "Batch sync failed for shop=%s batch=%d: %s",
                    shop_id, batch_start, exc,
                )
                failed += len(batch)

        duration_ms = (datetime.now(timezone.utc) - t_start).total_seconds() * 1000
        await self._log_sync(shop_id, "full", indexed, failed, duration_ms)

        logger.info(
            "Full sync complete: shop=%s indexed=%d failed=%d duration=%.0fms",
            shop_id, indexed, failed, duration_ms,
        )

        return CatalogSyncResponse(
            shop_id=shop_id,
            products_indexed=indexed,
            products_failed=failed,
            duration_ms=duration_ms,
            status="success" if failed == 0 else "partial",
        )

    # ─────────────────────── UPSERT BATCH (ORM) ──────────────────

    async def _upsert_product_batch(
        self, shop_id: str, products: list[Product]
    ) -> int:
        """
        Encode and upsert a batch of products.

        MIGRATION:
        - AVANT: session.execute(text("INSERT INTO shopbot_product_embeddings ... ON CONFLICT ..."))
          → SQL brut inline
        - APRÈS: pg_insert(ShopbotProductEmbeddingORM).on_conflict_do_update(...)
          → SQLAlchemy dialects.postgresql.insert — paramètres bindés, zéro SQL brut
        """
        product_texts = [p.to_text_for_embedding() for p in products]
        bm25_texts = [p.to_bm25_text() for p in products]

        float32_embeddings, _, _ = await self._encoder.encode_passages_batch(product_texts)

        now = datetime.now(timezone.utc)
        records = []
        for i, product in enumerate(products):
            emb_vec = float32_embeddings[i]
            metadata = self._product_to_metadata(product)

            records.append({
                "shop_id": shop_id,
                "product_id": product.id,
                "product_text": bm25_texts[i],
                "embedding_float32": float32_to_pgvector_str(emb_vec),
                "metadata": json.dumps(metadata),
                "created_at": now,
                "updated_at": now,
            })

        async with AsyncSessionLocal() as session:
            stmt = (
                pg_insert(ShopbotProductEmbeddingORM)
                .values(records)
                .on_conflict_do_update(
                    index_elements=["shop_id", "product_id"],
                    set_={
                        "product_text": pg_insert(ShopbotProductEmbeddingORM).excluded.product_text,
                        "embedding_float32": pg_insert(ShopbotProductEmbeddingORM).excluded.embedding_float32,
                        "metadata": pg_insert(ShopbotProductEmbeddingORM).excluded.metadata,
                        "updated_at": now,
                    },
                )
            )
            await session.execute(stmt)
            await session.commit()

        return len(records)

    async def _upsert_product_to_index(
        self, shop_id: str, product_id: str, product_data: dict
    ) -> None:
        """Single-product upsert from a NOTIFY event."""
        product = self._event_data_to_product(product_id, shop_id, product_data)
        await self._upsert_product_batch(shop_id, [product])
        logger.debug("Upserted product %s for shop %s", product_id, shop_id)

    # ─────────────────────── DELETE (ORM) ────────────────────────

    async def _delete_product_from_index(self, shop_id: str, product_id: str) -> None:
        """
        Remove a product from the vector index.

        MIGRATION:
        - AVANT: session.execute(text("DELETE FROM shopbot_product_embeddings WHERE ..."))
        - APRÈS: session.execute(delete(ShopbotProductEmbeddingORM).where(...))
        """
        from sqlalchemy import delete

        async with AsyncSessionLocal() as session:
            await session.execute(
                delete(ShopbotProductEmbeddingORM).where(
                    ShopbotProductEmbeddingORM.shop_id == shop_id,
                    ShopbotProductEmbeddingORM.product_id == product_id,
                )
            )
            await session.commit()
        logger.debug("Deleted product %s from shop %s index", product_id, shop_id)

    async def _clear_shop_index(self, shop_id: str) -> None:
        """Remove ALL product embeddings for a shop (used on full rebuild)."""
        from sqlalchemy import delete, func, select

        async with AsyncSessionLocal() as session:
            count_result = await session.execute(
                select(func.count()).where(
                    ShopbotProductEmbeddingORM.shop_id == shop_id
                )
            )
            count = count_result.scalar_one()
            await session.execute(
                delete(ShopbotProductEmbeddingORM).where(
                    ShopbotProductEmbeddingORM.shop_id == shop_id
                )
            )
            await session.commit()
        logger.info("Cleared %d embeddings for shop=%s", count, shop_id)

    # ─────────────────────── SYNC LOG (ORM) ──────────────────────

    async def _log_sync(
        self,
        shop_id: str,
        sync_type: str,
        indexed: int,
        failed: int,
        duration_ms: float,
    ) -> None:
        """
        Write sync result to shopbot_sync_log.

        MIGRATION:
        - AVANT: session.execute(text("INSERT INTO shopbot_sync_log ..."))
        - APRÈS: session.add(ShopbotSyncLogORM(...)) — ORM pur
        """
        status = "success" if failed == 0 else ("failed" if indexed == 0 else "partial")
        now = datetime.now(timezone.utc)

        async with AsyncSessionLocal() as session:
            session.add(ShopbotSyncLogORM(
                shop_id=shop_id,
                sync_type=sync_type,
                products_count=indexed + failed,
                status=status,
                started_at=now,
                completed_at=now,
            ))
            await session.commit()

    # ─────────────────────── HELPERS ─────────────────────────────

    def _product_to_metadata(self, product: Product) -> dict:
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
            availability=ProductAvailability(data.get("availability", "in_stock")),
            stock_quantity=data.get("stock_quantity"),
            sku=data.get("sku"),
            attributes=data.get("attributes", {}),
        )
