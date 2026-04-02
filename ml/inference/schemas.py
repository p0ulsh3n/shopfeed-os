"""
Pydantic v2 schemas for ML Inference API.
Couvre les 9 endpoints : /v1/feed/rank, /v1/embed/user, /v1/embed/product,
/v1/session/intent-vector, /v1/moderation/clip-check, /v1/health,
/v1/llm/search, /v1/llm/vendor-insights, /v1/llm/generate-ad-copy
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 1 — POST /v1/feed/rank
# ─────────────────────────────────────────────────────────────────────────────

class ActionEvent(BaseModel):
    type: str = Field(..., description="view|zoom|add_to_cart|buy_now|save|skip|pause|watch_pct")
    product_id: str
    category: int
    price: float
    dwell_ms: int = 0
    watch_pct: float = 0.0
    timestamp: str


class RankRequest(BaseModel):
    user_id: str
    session_vector: list[float] = Field(..., description="128d intent vector from Redis session")
    candidates: Optional[list[str]] = Field(
        None,
        description="Pre-fetched candidate IDs. If None, Two-Tower retrieves candidates."
    )
    session_actions: list[ActionEvent] = Field(default_factory=list, description="Recent session actions for DIN context")
    intent_level: str = Field("low", description="low|medium|high|buying_now")
    limit: int = Field(15, ge=1, le=100)
    context: str = Field("feed", description="feed|marketplace|live")


class MTLScores(BaseModel):
    p_buy_now: float
    p_purchase: float
    p_add_to_cart: float
    p_save: float
    p_share: float
    e_watch_time: float
    p_negative: float


class DiversityFlags(BaseModel):
    is_new_vendor: bool = False
    is_regional: bool = False
    is_cold_start: bool = False


class RankedCandidate(BaseModel):
    item_id: str
    score: float
    pool_level: str
    mtl_scores: MTLScores
    diversity_flags: DiversityFlags


class RankResponse(BaseModel):
    candidates: list[RankedCandidate]
    pipeline_ms: float


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 2 — POST /v1/embed/user
# ─────────────────────────────────────────────────────────────────────────────

class InteractionHistoryItem(BaseModel):
    item_id: str
    action: str
    timestamp: str


class ProfileFeatures(BaseModel):
    category_prefs: dict = Field(default_factory=dict)
    price_ranges: dict = Field(default_factory=dict)
    purchase_history: list = Field(default_factory=list)


class EmbedUserRequest(BaseModel):
    user_id: str
    interaction_history: list[InteractionHistoryItem] = Field(
        default_factory=list,
        description="Max 200 items"
    )
    profile_features: ProfileFeatures = Field(default_factory=ProfileFeatures)


class EmbedUserResponse(BaseModel):
    embedding: list[float] = Field(..., description="256d user tower output")
    updated_at: str


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 3 — POST /v1/embed/product
# ─────────────────────────────────────────────────────────────────────────────

class EmbedProductRequest(BaseModel):
    product_id: str
    image_url: str
    title: str
    description: str
    price: float
    category_id: int
    attributes: dict = Field(default_factory=dict)


class EmbedProductResponse(BaseModel):
    clip_embedding: list[float] = Field(..., description="512d — stored in catalog_db.products")
    cv_score: float
    auto_description: str
    auto_tags: list[str]
    category_verified: bool
    pipeline_ms: float


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 4 — POST /v1/session/intent-vector
# ─────────────────────────────────────────────────────────────────────────────

class SessionAction(BaseModel):
    type: str = Field(..., description="view|zoom|add_to_cart|buy_now|save|skip|pause|watch_pct")
    product_id: str
    category: int
    price: float
    dwell_ms: int = 0
    watch_pct: float = 0.0
    timestamp: str


class IntentVectorRequest(BaseModel):
    session_id: str
    session_actions: list[SessionAction]


class IntentVectorResponse(BaseModel):
    session_vector: list[float] = Field(..., description="128d — injected into /v1/feed/rank")
    intent_level: str
    active_categories: list[int]
    price_range_signal: dict
    negative_categories: list[int]


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 5 — POST /v1/moderation/clip-check
# ─────────────────────────────────────────────────────────────────────────────

class ClipCheckRequest(BaseModel):
    image_url: str
    declared_category_id: int


class ClipCheckResponse(BaseModel):
    match: bool
    similarity_score: float
    closest_category_id: int
    confidence: float
    explanation: str = Field("", description="Llama 4 Scout explanation of WHY flagged (empty if match=True)")
    suggested_fixes: list[str] = Field(default_factory=list, description="Actionable fixes for the vendor")


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 6 — POST /v1/llm/search  (Conversational Search)
# ─────────────────────────────────────────────────────────────────────────────

class ConversationalSearchRequest(BaseModel):
    query: str = Field(..., description="Natural language search query")
    categories: list[str] = Field(default_factory=list, description="Available categories")


class ConversationalSearchResponse(BaseModel):
    category: str = ""
    attributes: dict = Field(default_factory=dict)
    price_range: dict = Field(default_factory=dict)
    sort_by: str = "relevance"
    keywords: list[str] = Field(default_factory=list)
    intent: str = "browse"


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 7 — POST /v1/llm/vendor-insights
# ─────────────────────────────────────────────────────────────────────────────

class VendorInsightsRequest(BaseModel):
    vendor_name: str
    product_count: int = 0
    avg_cv_score: float = 0.5
    monthly_views: int = 0
    conversion_rate: float = 0.0
    campaign_metrics: dict = Field(default_factory=dict)


class VendorInsightsResponse(BaseModel):
    insights: str = Field(..., description="Natural language strategy recommendations")


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 8 — POST /v1/llm/generate-ad-copy
# ─────────────────────────────────────────────────────────────────────────────

class AdCopyRequest(BaseModel):
    product_title: str
    product_description: str = ""
    target_audience: str = "general"
    vendor_name: str = ""
    num_variants: int = Field(5, ge=1, le=10)


class AdCopyVariant(BaseModel):
    headline: str = ""
    body: str = ""
    cta: str = ""
    tone: str = ""


class AdCopyResponse(BaseModel):
    variants: list[AdCopyVariant] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 9 — GET /v1/health
# ─────────────────────────────────────────────────────────────────────────────

class ModelVersions(BaseModel):
    two_tower: str = "0.0.0"
    mtl_ple: str = "0.0.0"
    din: str = "0.0.0"
    dien: str = "0.0.0"
    bst: str = "0.0.0"
    geo_classifier: str = "0.0.0"


class HealthResponse(BaseModel):
    status: str = Field(..., description="ok|degraded|down")
    model_versions: ModelVersions
    faiss_index_size: int
    last_trained_at: Optional[str] = None
    monolith_lag_s: int
    uptime_s: int
