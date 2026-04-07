"""
ShopBot Service — FastAPI Entry Point
=====================================
Run with:
    uvicorn services.shopbot_service.main:app --host 0.0.0.0 --port 8070 --workers 4

Or via Docker:
    docker compose up shopbot
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from services.shopbot_service.api.routes import (
    admin_router,
    router,
    shutdown,
    startup,
)
from services.shopbot_service.config import get_settings

# ─────────────────────── LOGGING SETUP ───────────────────────────

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("shopbot")


# ─────────────────────── LIFESPAN ────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler (replaces deprecated on_event).
    Handles startup and graceful shutdown.
    """
    await startup(app)
    yield
    await shutdown(app)


# ─────────────────────── APP FACTORY ─────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="ShopFeed ShopBot API",
        description=(
            "Self-hosted AI shop assistant with RAG retrieval. "
            "Each shop gets its own catalog-aware bot powered by "
            "Qwen2.5-VL-7B-Instruct-AWQ + pgvector hybrid search."
        ),
        version="1.0.0",
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url="/redoc" if settings.environment != "production" else None,
        lifespan=lifespan,
    )

    # ── Middleware ──────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],        # Restrict in production via env
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-Id"],
    )
    # Compress large responses (product lists, etc.)
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # ── Request ID middleware ───────────────────────────────────
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        import uuid
        request_id = str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response

    # ── Global exception handler ────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(
            f"Unhandled exception on {request.method} {request.url}: {exc}",
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "details": []},
        )

    # ── Routers ─────────────────────────────────────────────────
    app.include_router(router)
    app.include_router(admin_router)

    # ── Root ────────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": "ShopFeed ShopBot",
            "version": "1.0.0",
            "status": "running",
            "docs": "/docs",
        }

    return app


app = create_app()
