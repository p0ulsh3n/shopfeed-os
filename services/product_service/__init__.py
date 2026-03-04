"""Product Service — CRUD + Vision Pipeline — Section 39."""

from .routes import app
from .schemas import (
    AddVariantRequest,
    CreateProductRequest,
    FlashSaleRequest,
    ProductListResponse,
    UpdateProductRequest,
)
from .vision import run_vision_pipeline

__all__ = [
    "app",
    "CreateProductRequest",
    "UpdateProductRequest",
    "AddVariantRequest",
    "FlashSaleRequest",
    "ProductListResponse",
    "run_vision_pipeline",
]
