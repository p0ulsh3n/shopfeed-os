"""ShopBot models package."""
from services.shopbot_service.models.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    CatalogEvent,
    CatalogSyncResponse,
    DeleteProductRequest,
    ErrorResponse,
    HealthResponse,
    MessageRole,
    Product,
    ProductAvailability,
    ProductCard,
    ProductImage,
    RetrievedProduct,
    StreamChunk,
    SyncCatalogRequest,
)

__all__ = [
    "ChatMessage", "ChatRequest", "ChatResponse",
    "CatalogEvent", "CatalogSyncResponse",
    "DeleteProductRequest", "ErrorResponse", "HealthResponse",
    "MessageRole", "Product", "ProductAvailability",
    "ProductCard", "ProductImage",
    "RetrievedProduct", "StreamChunk", "SyncCatalogRequest",
]
