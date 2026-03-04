"""Kafka event definitions — Section 14 / 30.

All inter-service communication flows through Kafka topics.
Each event is a Pydantic model serialized to JSON.
"""

from __future__ import annotations

import enum
from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────
# Topics
# ──────────────────────────────────────────────────────────────

class Topic(str, enum.Enum):
    USER_EVENTS = "shopfeed.user.events"
    COMMERCE_EVENTS = "shopfeed.commerce.events"
    VENDOR_EVENTS = "shopfeed.vendor.events"
    LIVE_EVENTS = "shopfeed.live.events"
    ORDERS_CREATED = "shopfeed.orders.created"
    ML_FEATURES = "shopfeed.ml.features"
    NOTIFICATIONS_SEND = "shopfeed.notifications.send"
    SEARCH_INDEX = "shopfeed.search.index"


# ──────────────────────────────────────────────────────────────
# Base event
# ──────────────────────────────────────────────────────────────

class BaseEvent(BaseModel):
    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    topic: str = ""


# ──────────────────────────────────────────────────────────────
# Feed events (Section 30 — Zone Feed)
# ──────────────────────────────────────────────────────────────

class FeedEventType(str, enum.Enum):
    CONTENT_SHOWN = "feed.content_shown"
    SCROLL_PAST = "feed.scroll_past"
    PAUSE_3S = "feed.pause_3s"
    ZOOM = "feed.zoom"
    VIDEO_WATCH_PERCENT = "feed.video_watch_percent"
    VIDEO_REPLAY = "feed.video_replay"
    NOT_INTERESTED = "feed.not_interested"
    SHARE = "feed.share"
    SAVE_WISHLIST = "feed.save_wishlist"


class FeedEvent(BaseEvent):
    topic: str = Topic.USER_EVENTS
    event_type: FeedEventType
    user_id: UUID
    content_id: UUID
    product_id: UUID | None = None
    session_id: str = ""

    # Video-specific
    watch_percent: float | None = None      # 0.0 to 1.0

    # Carousel-specific
    slides_viewed: int | None = None


# ──────────────────────────────────────────────────────────────
# Commerce events (Section 30 — Zone Commerce)
# ──────────────────────────────────────────────────────────────

class CommerceEventType(str, enum.Enum):
    PRODUCT_VIEWED = "product.viewed"
    PHOTO_ZOOM = "product.photo_zoom"
    ADD_TO_CART = "product.add_to_cart"
    BUY_NOW = "product.buy_now"
    CHECKOUT_STARTED = "cart.checkout_started"
    CART_ABANDONED = "cart.abandoned"
    ORDER_COMPLETED = "order.completed"
    ORDER_DELIVERED = "order.delivered"
    REVIEW_POSTED = "review.posted"


class CommerceEvent(BaseEvent):
    topic: str = Topic.COMMERCE_EVENTS
    event_type: CommerceEventType
    user_id: UUID
    product_id: UUID
    vendor_id: UUID | None = None
    session_id: str = ""
    order_id: UUID | None = None
    amount: float | None = None


# ──────────────────────────────────────────────────────────────
# Live events (Section 30 — Zone Live)
# ──────────────────────────────────────────────────────────────

class LiveEventType(str, enum.Enum):
    VIEWER_JOINED = "live.joined"
    VIEWER_LEFT = "live.left"
    COMMENT = "live.comment"
    GIFT_SENT = "live.gift_sent"
    PRODUCT_CLICK = "live.product_click"
    BUY_NOW_LIVE = "live.buy_now"


class LiveEvent(BaseEvent):
    topic: str = Topic.LIVE_EVENTS
    event_type: LiveEventType
    live_id: UUID
    user_id: UUID
    product_id: UUID | None = None
    gift_value: float | None = None
    comment_text: str | None = None


# ──────────────────────────────────────────────────────────────
# Vendor events
# ──────────────────────────────────────────────────────────────

class VendorEventType(str, enum.Enum):
    PRODUCT_PUBLISHED = "product.published"
    LIVE_STARTED = "live.started"
    LIVE_ENDED = "live.ended"
    FLASH_SALE_ACTIVATED = "flash_sale.activated"


class VendorEvent(BaseEvent):
    topic: str = Topic.VENDOR_EVENTS
    event_type: VendorEventType
    vendor_id: UUID
    product_id: UUID | None = None
    live_id: UUID | None = None
