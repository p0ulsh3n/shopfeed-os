"""
Pydantic v2 schemas for ML Inference API.

Endpoints:
  POST /v1/feed/scroll              ← Feed scroll infini (TikTok-style)
  POST /v1/feed/marketplace         ← Feed marketplace (product grid)
  POST /v1/feed/live                ← Feed live shopping (FOMO)
  POST /v1/feed/rank                ← Generic (backward-compat, context field)
  POST /v1/embed/user
  POST /v1/embed/product
  POST /v1/session/intent-vector
  POST /v1/moderation/clip-check
  GET  /v1/health
  POST /v1/llm/search
  POST /v1/llm/vendor-insights
  POST /v1/llm/generate-ad-copy

Architecture:
  Feed Scroll + Marketplace + Live partagent le MÊME pipeline ML
  mais retournent des réponses structurées différemment.
  Les données temps réel (session_vector, interactions) sont partagées.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


# ═════════════════════════════════════════════════════════════════════════════
# Shared Models (used across all feed endpoints)
# ═════════════════════════════════════════════════════════════════════════════

class ActionEvent(BaseModel):
    """Real-time user action within current session."""
    type: str = Field(..., description="view|zoom|add_to_cart|buy_now|save|skip|pause|watch_pct")
    product_id: str
    category: int
    price: float
    dwell_ms: int = 0
    watch_pct: float = 0.0
    timestamp: str


class MTLScores(BaseModel):
    """Multi-task prediction scores from PLE model."""
    p_buy_now: float = Field(0.0, description="Probability of immediate purchase")
    p_purchase: float = Field(0.0, description="Probability of eventual purchase")
    p_add_to_cart: float = Field(0.0, description="Probability of adding to cart")
    p_save: float = Field(0.0, description="Probability of saving/wishlisting")
    p_share: float = Field(0.0, description="Probability of sharing")
    e_watch_time: float = Field(0.0, description="Expected watch time (normalized)")
    p_negative: float = Field(0.0, description="Probability of skip/hide/report")


class DiversityFlags(BaseModel):
    """Diversity metadata for ranking transparency."""
    is_new_vendor: bool = False
    is_regional: bool = False
    is_cold_start: bool = False


# ═════════════════════════════════════════════════════════════════════════════
# Shared Request (used by all 3 feed endpoints + generic)
# ═════════════════════════════════════════════════════════════════════════════

class RankRequest(BaseModel):
    """Shared ranking request — used by all feed endpoints.

    The `context` field is set automatically by the endpoint handler:
      /v1/feed/scroll      → context="feed"
      /v1/feed/marketplace → context="marketplace"
      /v1/feed/live        → context="live"
    """
    user_id: str
    session_vector: list[float] = Field(..., description="128d intent vector from Redis session")
    candidates: Optional[list[str]] = Field(
        None,
        description="Pre-fetched candidate IDs. If None, Two-Tower retrieves candidates."
    )
    session_actions: list[ActionEvent] = Field(
        default_factory=list,
        description="Recent session actions for DIN attention context"
    )
    intent_level: str = Field("low", description="low|medium|high|buying_now")
    limit: int = Field(15, ge=1, le=100)
    context: str = Field("feed", description="feed|marketplace|live — auto-set by endpoint")


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINT: POST /v1/feed/scroll — Feed Scroll Infini
# ═════════════════════════════════════════════════════════════════════════════
# TikTok-style: engagement-first, rich media, vertical swipe
# Le frontend utilise cette réponse pour les cartes vidéo/image plein écran

class ScrollFeedItem(BaseModel):
    """Single item in the scroll feed — optimized for full-screen card display."""
    item_id: str
    rank: int = Field(..., description="Position in feed (1-indexed)")
    score: float = Field(..., description="Engagement-weighted composite score")

    # ── Content (what the frontend displays) ──
    title: str = ""
    image_url: str = ""
    video_url: Optional[str] = Field(None, description="Video URL if content type is video")
    content_type: str = Field("image", description="image|video|live_replay")
    vendor_name: str = ""
    vendor_id: str = ""
    vendor_avatar_url: str = ""

    # ── Pricing ──
    price: float = 0.0
    original_price: Optional[float] = Field(None, description="Crossed-out price if on sale")
    currency: str = "EUR"
    discount_pct: Optional[int] = Field(None, description="Discount percentage if applicable")

    # ── Engagement signals (for frontend UI decisions) ──
    engagement: ScrollEngagement = Field(default_factory=lambda: ScrollEngagement())

    # ── ML transparency ──
    pool_level: str = Field("L1", description="Traffic pool L1-L6")
    mtl_scores: MTLScores = Field(default_factory=MTLScores)
    diversity_flags: DiversityFlags = Field(default_factory=DiversityFlags)
    reason: str = Field("", description="Why this item was shown: personalized|trending|discovery|cross_sell")


class ScrollEngagement(BaseModel):
    """Engagement metrics displayed on the feed card."""
    likes_count: int = 0
    comments_count: int = 0
    shares_count: int = 0
    saves_count: int = 0
    is_liked_by_user: bool = False
    is_saved_by_user: bool = False


class ScrollFeedResponse(BaseModel):
    """Response for /v1/feed/scroll — engagement-optimized feed."""
    items: list[ScrollFeedItem]
    total_candidates: int = Field(0, description="Number of candidates scored by the ML pipeline")
    has_more: bool = Field(True, description="Whether more items can be loaded (infinite scroll)")
    next_cursor: Optional[str] = Field(None, description="Cursor for pagination (opaque token)")
    pipeline_ms: float = Field(0.0, description="ML pipeline latency")
    context: str = Field("feed", description="Always 'feed' for this endpoint")


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINT: POST /v1/feed/marketplace — Marketplace Product Grid
# ═════════════════════════════════════════════════════════════════════════════
# E-commerce grid: conversion-first, product cards, price prominent
# Le frontend utilise cette réponse pour la grille produit (2 colonnes)

class MarketplaceProductItem(BaseModel):
    """Single product in marketplace feed — optimized for purchase conversion."""
    item_id: str
    rank: int = Field(..., description="Position in grid (1-indexed)")
    score: float = Field(..., description="Conversion-weighted composite score")

    # ── Product Info ──
    title: str = ""
    image_url: str = ""
    images: list[str] = Field(default_factory=list, description="All product images (gallery)")
    description_short: str = Field("", description="Truncated description for card preview")
    category_id: int = 0
    category_name: str = ""

    # ── Vendor ──
    vendor_id: str = ""
    vendor_name: str = ""
    vendor_avatar_url: str = ""
    vendor_rating: float = Field(0.0, description="Vendor rating 0-5")
    vendor_is_verified: bool = False

    # ── Pricing (prominent in marketplace) ──
    price: float
    original_price: Optional[float] = Field(None, description="Crossed-out original price")
    currency: str = "EUR"
    discount_pct: Optional[int] = None
    free_shipping: bool = False
    shipping_estimate: str = Field("", description="e.g. '2-3 days' or 'Express 24h'")

    # ── Social proof (drives conversion) ──
    social_proof: MarketplaceSocialProof = Field(default_factory=lambda: MarketplaceSocialProof())

    # ── Badges (visual urgency cues) ──
    badges: list[str] = Field(
        default_factory=list,
        description="Visual badges: bestseller|new|sale|low_stock|trending|top_rated|eco_friendly"
    )

    # ── Cross-sell ──
    is_cross_sell: bool = Field(False, description="True if injected as cross-sell complement")
    cross_sell_reason: str = Field("", description="e.g. 'Bought together with Air Max 90'")

    # ── ML transparency ──
    pool_level: str = "L1"
    mtl_scores: MTLScores = Field(default_factory=MTLScores)
    diversity_flags: DiversityFlags = Field(default_factory=DiversityFlags)
    reason: str = Field("", description="personalized|trending|discovery|cross_sell|price_variety")


class MarketplaceSocialProof(BaseModel):
    """Social proof signals that drive conversion in marketplace."""
    total_sold: int = 0
    views_today: int = 0
    in_carts_now: int = Field(0, description="Number of users with this item in cart right now")
    rating: float = Field(0.0, description="Product rating 0-5")
    review_count: int = 0
    recent_buyer_avatars: list[str] = Field(
        default_factory=list,
        description="URLs of 3 most recent buyer avatars (social proof)"
    )


class MarketplaceFeedResponse(BaseModel):
    """Response for /v1/feed/marketplace — conversion-optimized product grid."""
    items: list[MarketplaceProductItem]
    total_candidates: int = 0
    has_more: bool = True
    next_cursor: Optional[str] = None
    pipeline_ms: float = 0.0
    context: str = Field("marketplace", description="Always 'marketplace' for this endpoint")

    # ── Marketplace-specific metadata ──
    active_filters: dict = Field(default_factory=dict, description="Applied category/price filters")
    sort_mode: str = Field("relevance", description="relevance|price_asc|price_desc|newest|bestseller")
    price_range: dict = Field(
        default_factory=dict,
        description="Min/max price in result set for filter UI"
    )


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINT: POST /v1/feed/live — Live Shopping Feed
# ═════════════════════════════════════════════════════════════════════════════
# Live shopping: urgency-first, countdown, limited stock
# Le frontend utilise cette réponse pour le carrousel produit pendant le live

class LiveProductItem(BaseModel):
    """Single product shown during a live stream — urgency-optimized."""
    item_id: str
    rank: int
    score: float

    # ── Product ──
    title: str = ""
    image_url: str = ""
    category_id: int = 0

    # ── Pricing (with live-exclusive deals) ──
    price: float
    original_price: Optional[float] = None
    live_price: Optional[float] = Field(None, description="Live-exclusive price (lower than regular)")
    currency: str = "EUR"
    discount_pct: Optional[int] = None

    # ── Urgency signals (FOMO) ──
    urgency: LiveUrgency = Field(default_factory=lambda: LiveUrgency())

    # ── ML transparency ──
    pool_level: str = "L1"
    mtl_scores: MTLScores = Field(default_factory=MTLScores)


class LiveUrgency(BaseModel):
    """Urgency signals for live shopping — drives impulse buying."""
    stock_remaining: Optional[int] = Field(None, description="Items left (null if unlimited)")
    buyers_count: int = Field(0, description="How many bought during this live session")
    viewers_watching: int = Field(0, description="Current live viewer count")
    deal_expires_at: Optional[str] = Field(None, description="ISO timestamp when deal expires")
    is_flash_deal: bool = False
    flash_deal_label: str = Field("", description="e.g. 'Flash: -40% pendant 5 min'")


class LiveFeedResponse(BaseModel):
    """Response for /v1/feed/live — urgency-optimized product carousel."""
    items: list[LiveProductItem]
    stream_id: str = Field("", description="Current live stream ID")
    vendor_id: str = Field("", description="Streaming vendor ID")
    vendor_name: str = ""
    viewers_count: int = 0
    pipeline_ms: float = 0.0
    context: str = Field("live", description="Always 'live' for this endpoint")


# ═════════════════════════════════════════════════════════════════════════════
# Generic /v1/feed/rank (backward-compatible)
# ═════════════════════════════════════════════════════════════════════════════

class RankedCandidate(BaseModel):
    """Generic ranked item — used by /v1/feed/rank (backward-compat)."""
    item_id: str
    score: float
    pool_level: str
    mtl_scores: MTLScores
    diversity_flags: DiversityFlags


class RankResponse(BaseModel):
    """Generic ranking response — backward-compatible."""
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


# ═════════════════════════════════════════════════════════════════════════════
# Endpoint 10 — POST /v1/ads/serve  (EPSILON Ad Pipeline)
# ═════════════════════════════════════════════════════════════════════════════

class AdServeRequest(BaseModel):
    """Request to serve ads within the feed."""
    user_id: str
    session_vector: list[float] = Field(..., description="128d user intent vector")
    feed_items: list[str] = Field(default_factory=list, description="Organic item IDs in current page")
    placement_slots: int = Field(2, ge=1, le=5, description="Max ad slots to fill")
    context: str = Field("feed", description="feed|marketplace|live")


class ServedAd(BaseModel):
    """Single ad placement returned by EPSILON."""
    ad_id: str
    campaign_id: str
    product_id: str
    vendor_id: str
    position: int = Field(..., description="Insert position in organic feed (0-indexed)")
    bid_price: float = Field(0.0, description="Winning bid price (GSP auction)")
    predicted_ctr: float = Field(0.0, description="Calibrated pCTR")
    predicted_cvr: float = Field(0.0, description="Predicted conversion rate")
    predicted_roas: float = Field(0.0, description="Predicted return on ad spend")
    creative_type: str = Field("product_card", description="product_card|video_ad|banner|story")
    is_native: bool = Field(True, description="True if ad looks like organic content")
    fatigue_discount: float = Field(1.0, description="Discount factor for user fatigue (0-1)")


class AdServeResponse(BaseModel):
    """Response from EPSILON ad serving pipeline."""
    ads: list[ServedAd] = Field(default_factory=list)
    total_eligible: int = Field(0, description="Total eligible campaigns for this user")
    pipeline_ms: float = 0.0
    auction_type: str = Field("gsp", description="gsp (Generalized Second Price)")


# ═════════════════════════════════════════════════════════════════════════════
# Endpoint 11 — POST /v1/fraud/check
# ═════════════════════════════════════════════════════════════════════════════

class FraudCheckRequest(BaseModel):
    """Check a user action for fraud signals."""
    user_id: str
    action: str = Field(..., description="login|purchase|like|comment|follow|signup")
    ip_address: str = ""
    device_fingerprint: str = ""
    session_duration_s: int = 0
    actions_last_hour: int = 0
    metadata: dict = Field(default_factory=dict, description="Additional signals (user_agent, geo, etc.)")


class FraudCheckResponse(BaseModel):
    """Fraud detection result."""
    fraud_score: float = Field(..., ge=0, le=1, description="0=legit, 1=fraud")
    decision: str = Field(..., description="allow|captcha|shadowban|block")
    risk_factors: list[str] = Field(default_factory=list, description="Detected risk signals")
    is_bot: bool = False
    is_emulator: bool = False
    velocity_alert: bool = Field(False, description="True if unusual action velocity detected")


# ═════════════════════════════════════════════════════════════════════════════
# Endpoint 12 — POST /v1/audio/transcribe
# ═════════════════════════════════════════════════════════════════════════════

class TranscribeRequest(BaseModel):
    """Transcribe vendor video/live audio via Whisper."""
    audio_url: str = Field(..., description="URL to audio/video file")
    language: Optional[str] = Field(None, description="ISO language code (auto-detect if null)")
    extract_entities: bool = Field(True, description="Extract products, brands, prices from transcript")


class TranscribedEntity(BaseModel):
    """Entity extracted from audio transcript."""
    type: str = Field(..., description="product|brand|price|urgency")
    value: str
    confidence: float = 0.0


class TranscribeResponse(BaseModel):
    """Audio transcription result."""
    transcript: str = ""
    language: str = ""
    duration_s: float = 0.0
    entities: list[TranscribedEntity] = Field(default_factory=list)
    word_count: int = 0
    pipeline_ms: float = 0.0


# ═════════════════════════════════════════════════════════════════════════════
# Endpoint 13 — POST /v1/moderation/duplicate-check
# ═════════════════════════════════════════════════════════════════════════════

class DuplicateCheckRequest(BaseModel):
    """Check if a product image is a duplicate of existing products."""
    image_url: str
    vendor_id: str = Field("", description="Vendor uploading the image")
    check_scope: str = Field("vendor", description="vendor|global — check within vendor or entire catalog")


class DuplicateMatch(BaseModel):
    """A matching duplicate found."""
    matched_product_id: str
    similarity: float = Field(..., description="0-1 similarity score (>0.95 = duplicate)")
    hash_distance: int = Field(..., description="Hamming distance between perceptual hashes")


class DuplicateCheckResponse(BaseModel):
    """Duplicate detection result."""
    is_duplicate: bool = False
    matches: list[DuplicateMatch] = Field(default_factory=list)
    phash: str = Field("", description="Perceptual hash of the uploaded image")
    pipeline_ms: float = 0.0


# ═════════════════════════════════════════════════════════════════════════════
# Endpoint 14 — POST /v1/embed/video
# ═════════════════════════════════════════════════════════════════════════════

class EmbedVideoRequest(BaseModel):
    """Extract temporal video embedding via VideoMAE."""
    video_url: str = Field(..., description="URL to video file")
    product_id: str = ""
    n_frames: int = Field(16, description="Number of frames to sample")


class EmbedVideoResponse(BaseModel):
    """Video embedding result."""
    embedding: list[float] = Field(..., description="768d VideoMAE spatio-temporal embedding")
    duration_s: float = 0.0
    content_type: str = Field("", description="Detected: unboxing|demo|lifestyle|review|tutorial")
    pipeline_ms: float = 0.0

