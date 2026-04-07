"""
FastAPI Routes — ShopBot HTTP API
===================================
All public endpoints for the ShopBot service.

Endpoints:
  POST   /chat              — Send a message (streaming SSE or JSON)
  POST   /catalog/sync      — Full catalog re-index for a shop
  PUT    /catalog/product   — Upsert a single product
  DELETE /catalog/product   — Remove a product from index
  GET    /health            — Service health check
  GET    /metrics           — Prometheus metrics (optional)
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from services.shopbot_service.bot.shopbot import ShopBot
from services.shopbot_service.config import get_settings
from services.shopbot_service.database.connection import check_db_health
from services.shopbot_service.embeddings.encoder import EmbeddingEncoder
from services.shopbot_service.llm.client import VLLMClient
from services.shopbot_service.models.schemas import (
    SyncCatalogRequest,
    CatalogSyncResponse,
    ChatRequest,
    ChatResponse,
    DeleteProductRequest,
    ErrorResponse,
    HealthResponse,
    Product,
)
from services.shopbot_service.retrieval.catalog_sync import CatalogSyncService

logger = logging.getLogger(__name__)
settings = get_settings()

# ─────────────────────── DEPENDENCY INJECTION ────────────────────

# Module-level singletons (created once at startup)
_vllm_client: VLLMClient | None = None
_catalog_sync: CatalogSyncService | None = None
_startup_time: float = 0.0


def get_vllm_client() -> VLLMClient:
    if _vllm_client is None:
        raise HTTPException(status_code=503, detail="vLLM client not initialized")
    return _vllm_client


def get_catalog_sync() -> CatalogSyncService:
    if _catalog_sync is None:
        raise HTTPException(status_code=503, detail="Catalog sync not initialized")
    return _catalog_sync


async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> None:
    """Simple API key authentication for internal service calls."""
    if x_api_key != settings.internal_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )


# ─────────────────────── ROUTERS ─────────────────────────────────

# Public (with API key) — used by app backend
router = APIRouter(prefix="/api/v1/shopbot", tags=["ShopBot"])

# Internal admin — used by ops/devops
admin_router = APIRouter(prefix="/admin/shopbot", tags=["ShopBot Admin"])


# ─────────────────────── HEALTH ──────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
)
async def health_check() -> HealthResponse:
    """
    Returns service health status.
    Called by load balancer and Kubernetes liveness probes.
    """
    encoder = EmbeddingEncoder.get_instance()
    vllm_ok = False
    if _vllm_client:
        vllm_ok = await _vllm_client.health_check()

    db_ok = await check_db_health()

    return HealthResponse(
        status="healthy" if (db_ok and encoder.is_loaded()) else "degraded",
        version="1.0.0",
        vllm_connected=vllm_ok,
        db_connected=db_ok,
        embedding_model_loaded=encoder.is_loaded(),
        uptime_seconds=time.time() - _startup_time,
    )


# ─────────────────────── CHAT ────────────────────────────────────

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a message to the ShopBot",
    dependencies=[Depends(verify_api_key)],
)
async def chat(
    request: ChatRequest,
    vllm: VLLMClient = Depends(get_vllm_client),
) -> StreamingResponse | ChatResponse:
    """
    Main chat endpoint. Supports both streaming (SSE) and non-streaming modes.

    **Streaming mode** (recommended):
    Set `stream: true` in the request body.
    Returns Server-Sent Events with incremental tokens.

    **Non-streaming mode**:
    Set `stream: false`.
    Returns complete response as JSON.

    The ShopBot will:
    1. Search the shop's product catalog using hybrid retrieval
    2. Generate a contextually accurate response using Qwen2.5-VL-7B-AWQ
    3. Never hallucinate products not in the catalog
    """
    bot = ShopBot(vllm_client=vllm)

    if request.stream:
        # Return SSE stream
        return StreamingResponse(
            bot.stream_message(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable Nginx buffering for SSE
            },
        )
    else:
        response = await bot.handle_message(request)
        return response


# ─────────────────────── CATALOG MANAGEMENT ──────────────────────

@admin_router.post(
    "/catalog/sync",
    response_model=CatalogSyncResponse,
    summary="Full catalog re-sync for a shop",
    dependencies=[Depends(verify_api_key)],
)
async def sync_catalog(
    request: SyncCatalogRequest,
    sync_service: CatalogSyncService = Depends(get_catalog_sync),
) -> CatalogSyncResponse:
    """
    Trigger a full catalog re-indexing for a shop.

    **When to use:**
    - First-time shop onboarding
    - After a bulk product import
    - After recovering from sync failures
    - Scheduled nightly re-sync for data integrity

    **Note:** Incremental updates happen automatically via PostgreSQL LISTEN/NOTIFY.
    You rarely need to call this endpoint for normal operations.
    """
    from services.shopbot_service.database.connection import AsyncSessionLocal
    from sqlalchemy import text

    # Fetch all products for this shop from the main DB
    products = []
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("""
                SELECT
                    p.id, p.shop_id, p.name, p.description,
                    p.price, p.currency, p.category, p.availability,
                    p.stock_quantity, p.sku, p.attributes, p.tags,
                    p.created_at, p.updated_at
                FROM products p
                WHERE p.shop_id = :shop_id
                  AND p.deleted_at IS NULL
                ORDER BY p.created_at
            """),
            {"shop_id": request.shop_id},
        )
        rows = result.fetchall()

    for row in rows:
        try:
            product = Product(
                id=str(row[0]),
                shop_id=str(row[1]),
                name=row[2] or "",
                description=row[3],
                price=float(row[4] or 0),
                currency=row[5] or "XAF",
                category=row[6],
                availability=row[7] or "in_stock",
                stock_quantity=row[8],
                sku=row[9],
                attributes=row[10] or {},
                tags=row[11] or [],
            )
            products.append(product)
        except Exception as e:
            logger.warning(f"Skipping malformed product {row[0]}: {e}")

    if not products:
        raise HTTPException(
            status_code=404,
            detail=f"No products found for shop {request.shop_id}",
        )

    return await sync_service.full_sync_shop(
        shop_id=request.shop_id,
        products=products,
        force_rebuild=request.force_full_rebuild,
    )


@admin_router.put(
    "/catalog/product",
    summary="Upsert a single product into the index",
    dependencies=[Depends(verify_api_key)],
)
async def upsert_product(
    product: Product,
    sync_service: CatalogSyncService = Depends(get_catalog_sync),
) -> dict:
    """
    Manually upsert a single product into the ShopBot index.
    Usually handled automatically via LISTEN/NOTIFY — use this for testing.
    """
    await sync_service._upsert_product_batch(product.shop_id, [product])
    return {"status": "ok", "product_id": product.id}


@admin_router.delete(
    "/catalog/product",
    summary="Remove a product from the index",
    dependencies=[Depends(verify_api_key)],
)
async def delete_product(
    request: DeleteProductRequest,
    sync_service: CatalogSyncService = Depends(get_catalog_sync),
) -> dict:
    """
    Remove a product from the ShopBot vector index.
    Usually handled automatically via LISTEN/NOTIFY DELETE event.
    """
    await sync_service._delete_product_from_index(request.shop_id, request.product_id)
    return {"status": "ok", "product_id": request.product_id}


# ─────────────────────── STARTUP/SHUTDOWN ────────────────────────

async def startup(app) -> None:
    """
    Application startup:
    1. Initialize database schema (idempotent)
    2. Load embedding model (lazy, in background)
    3. Start catalog LISTEN/NOTIFY listener
    4. Warm up vLLM client
    """
    global _vllm_client, _catalog_sync, _startup_time
    _startup_time = time.time()

    logger.info("ShopBot service starting up...")

    # 1. Database schema
    from services.shopbot_service.database.connection import initialize_schema
    await initialize_schema()

    # 2. Load embedding model in background (don't block startup)
    encoder = EmbeddingEncoder.get_instance()
    import asyncio
    asyncio.create_task(encoder.load())

    # 3. vLLM client
    _vllm_client = VLLMClient()

    # 4. Catalog sync service + listener
    _catalog_sync = CatalogSyncService()
    await _catalog_sync.start_listener()

    logger.info("ShopBot service ready ✓")


async def shutdown(app) -> None:
    """Graceful shutdown."""
    if _catalog_sync:
        await _catalog_sync.stop()

    encoder = EmbeddingEncoder.get_instance()
    await encoder.shutdown()

    from services.shopbot_service.database.connection import (
        close_raw_pool, engine
    )
    await close_raw_pool()
    await engine.dispose()

    logger.info("ShopBot service shut down cleanly")
