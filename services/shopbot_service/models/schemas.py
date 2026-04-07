"""
ShopBot Pydantic Schemas
========================
All API request/response models + internal data structures.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, HttpUrl


# ─────────────────────── ENUMS ───────────────────────────────────

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ProductAvailability(str, Enum):
    IN_STOCK = "in_stock"
    OUT_OF_STOCK = "out_of_stock"
    PRE_ORDER = "pre_order"
    DISCONTINUED = "discontinued"


class CatalogEventType(str, Enum):
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"


# ─────────────────────── PRODUCT CATALOG ─────────────────────────

class ProductImage(BaseModel):
    url: str
    alt: str | None = None
    is_primary: bool = False


class Product(BaseModel):
    """
    Core product entity — mirrors the products table in PostgreSQL.
    """
    id: str
    shop_id: str
    name: str
    description: str | None = None
    price: float
    currency: str = "XAF"   # Central African Franc (ShopFeed's primary market)
    category: str | None = None
    subcategory: str | None = None
    tags: list[str] = Field(default_factory=list)
    images: list[ProductImage] = Field(default_factory=list)
    availability: ProductAvailability = ProductAvailability.IN_STOCK
    stock_quantity: int | None = None
    sku: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)  # color, size, weight, etc.
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def primary_image_url(self) -> str | None:
        for img in self.images:
            if img.is_primary:
                return img.url
        return self.images[0].url if self.images else None

    def to_text_for_embedding(self) -> str:
        """
        Create a rich text representation for embedding.
        Used by the encoder to build the vector index.
        Instruction-tuned format for multilingual-e5-large-instruct.
        """
        parts = [f"Produit: {self.name}"]
        if self.category:
            parts.append(f"Catégorie: {self.category}")
        if self.subcategory:
            parts.append(f"Sous-catégorie: {self.subcategory}")
        if self.description:
            parts.append(f"Description: {self.description}")
        parts.append(f"Prix: {self.price} {self.currency}")
        parts.append(f"Disponibilité: {self.availability.value}")
        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")
        if self.attributes:
            attr_str = ", ".join(f"{k}: {v}" for k, v in self.attributes.items())
            parts.append(f"Attributs: {attr_str}")
        return " | ".join(parts)

    def to_bm25_text(self) -> str:
        """
        Flat text blob for BM25 keyword indexing.
        Optimized for sparse retrieval (exact keyword matches).
        """
        tokens = [self.name]
        if self.description:
            tokens.append(self.description)
        if self.category:
            tokens.append(self.category)
        if self.subcategory:
            tokens.append(self.subcategory)
        tokens.extend(self.tags)
        tokens.extend(str(v) for v in self.attributes.values())
        if self.sku:
            tokens.append(self.sku)
        return " ".join(tokens)


# ─────────────────────── CHAT MESSAGES ───────────────────────────

class ChatMessage(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatContext(BaseModel):
    """
    Full conversation context passed to the LLM.
    The system prompt is built from shop context + RAG results.
    """
    shop_id: str
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    history: list[ChatMessage] = Field(default_factory=list)
    retrieved_products: list[Product] = Field(default_factory=list)
    customer_language: str = "fr"


# ─────────────────────── API REQUESTS ────────────────────────────

class ChatRequest(BaseModel):
    """
    POST /chat — send a message to the ShopBot.
    """
    shop_id: str = Field(..., description="Unique shop identifier")
    session_id: str | None = Field(
        default=None,
        description="Session ID for conversation continuity. Create new if null.",
    )
    message: str = Field(..., min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=20,
        description="Previous conversation turns (last N messages)",
    )
    # Optional: client can provide images (product photos) for visual search
    image_urls: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Image URLs for visual product search",
    )
    language: str | None = Field(
        default=None,
        description="Force response language (iso-639-1). Auto-detected if null.",
    )
    stream: bool = Field(default=True, description="Stream the response via SSE")


class SyncCatalogRequest(BaseModel):
    """
    POST /catalog/sync — trigger full catalog re-indexing for a shop.
    """
    shop_id: str
    force_full_rebuild: bool = False


class DeleteProductRequest(BaseModel):
    """
    DELETE /catalog/product — remove a product from the shopbot index.
    """
    shop_id: str
    product_id: str


# ─────────────────────── API RESPONSES ───────────────────────────

class RetrievedProduct(BaseModel):
    """
    A product retrieved from the RAG index, with relevance score.
    """
    product: Product
    score: float = Field(ge=0.0, le=1.0)
    retrieval_method: str  # "dense", "sparse", "rrf_fusion"


class ChatResponse(BaseModel):
    """
    POST /chat response (non-streaming).

    Frontend usage:
        - Display `message` in the chat bubble
        - Render `product_cards` below the text as a horizontal scroll
          or grid of product cards with images
        - If product_cards is empty, show text only
    """
    session_id: str
    message: str
    # Frontend-ready product cards (images + price + name)
    product_cards: list[ProductCard] = Field(
        default_factory=list,
        description="Product cards with images for frontend rendering",
    )
    # Full source products (for clients that need complete data)
    sources: list[RetrievedProduct] = Field(default_factory=list)
    latency_ms: float
    model: str
    retrieved_count: int = 0


class ProductCard(BaseModel):
    """
    Frontend-ready product card with image URLs.
    Sent in the final SSE chunk / JSON response for UI rendering.
    The frontend displays: bot text response → product card(s) with image(s).
    """
    product_id: str
    name: str
    price: float
    currency: str = "XAF"
    availability: str
    category: str | None = None
    # All product images, primary image first
    images: list[str] = Field(default_factory=list, description="Ordered image URLs")
    primary_image: str | None = Field(default=None, description="Primary image URL")
    description: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    score: float = Field(ge=0.0, le=1.0, description="Relevance score")

    @classmethod
    def from_retrieved(cls, retrieved: "RetrievedProduct") -> "ProductCard":
        """Build a ProductCard from a RetrievedProduct for frontend consumption."""
        p = retrieved.product
        all_images = [img.url for img in p.images if img.url]
        primary = next(
            (img.url for img in p.images if img.is_primary),
            all_images[0] if all_images else None,
        )
        return cls(
            product_id=p.id,
            name=p.name,
            price=p.price,
            currency=p.currency,
            availability=p.availability.value,
            category=p.category,
            images=all_images,
            primary_image=primary,
            description=p.description,
            attributes=p.attributes,
            score=retrieved.score,
        )


class StreamChunk(BaseModel):
    """
    SSE streaming chunk format.

    During streaming:  delta contains text tokens, is_final=False
    On final chunk:    delta="", is_final=True, product_cards populated

    Frontend usage:
        - Append each delta to the chat bubble
        - On is_final=True, render product_cards below the text
    """
    session_id: str
    delta: str = ""        # Incremental text token
    is_final: bool = False
    # Only populated on the final chunk
    product_cards: list[ProductCard] | None = None
    # Raw sources (for advanced clients that want full product data)
    sources: list[RetrievedProduct] | None = None


class CatalogSyncResponse(BaseModel):
    shop_id: str
    products_indexed: int
    products_failed: int
    duration_ms: float
    status: str


class HealthResponse(BaseModel):
    status: str
    version: str
    vllm_connected: bool
    db_connected: bool
    embedding_model_loaded: bool
    uptime_seconds: float


# ─────────────────────── CATALOG EVENTS ──────────────────────────

class CatalogEvent(BaseModel):
    """
    Real-time catalog update event from PostgreSQL LISTEN/NOTIFY.
    Payload is JSON from the trigger function.
    """
    event_type: CatalogEventType
    shop_id: str
    product_id: str
    product_data: dict[str, Any] | None = None  # None on DELETE
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────── EMBEDDING RESULTS ───────────────────────

class EmbeddingResult(BaseModel):
    """
    Result from the encoder — includes both float32 and quantized forms.
    """
    text: str
    embedding_float32: list[float]
    embedding_int8: list[int] | None = None
    embedding_binary: bytes | None = None


# ─────────────────────── ERROR MODELS ────────────────────────────

class ErrorDetail(BaseModel):
    code: str
    message: str
    field: str | None = None


class ErrorResponse(BaseModel):
    error: str
    details: list[ErrorDetail] = Field(default_factory=list)
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
