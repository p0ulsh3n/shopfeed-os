"""Live commerce models — Section 07."""

from __future__ import annotations

import enum
from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class LiveStatus(str, enum.Enum):
    SCHEDULED = "scheduled"
    LIVE = "live"
    ENDED = "ended"


class LiveType(str, enum.Enum):
    INSTANT = "instant"         # ×3.0 boost — launches immediately
    SCHEDULED = "scheduled"     # Push at J-24h, J-1h, J-15min
    FLASH_SALE = "flash_sale"   # ×2.0 additional during flash window


class LiveSession(BaseModel):
    """Live stream session — Section 07 / 40."""
    id: UUID = Field(default_factory=uuid4)
    vendor_id: UUID
    title: str = Field(default="", max_length=200)
    status: LiveStatus = LiveStatus.SCHEDULED
    live_type: LiveType = LiveType.INSTANT

    # Streaming
    stream_key: str | None = None       # RTMP key
    playback_url: str | None = None     # HLS URL

    # Schedule
    scheduled_at: datetime | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None

    # Metrics
    peak_concurrent: int = 0
    total_viewers: int = 0
    total_gmv: float = 0.0

    # Section 07 — LiveScore updated every 60s
    live_score: float = 0.0
    pool_level: str = "L1"

    # Flash sale
    flash_sale_active: bool = False
    flash_sale_ends_at: datetime | None = None

    # Products pinned during live
    pinned_product_ids: list[UUID] = Field(default_factory=list)

    def compute_live_score(
        self,
        concurrent_viewers: int,
        purchase_rate: float,
        gift_value: float,
        comment_rate: float,
    ) -> float:
        """Compute LiveScore — Section 07 formula.

        LiveScore = Concurrent × 1.0 + Peak × 0.5
                  + PurchaseRate × 100 + GiftValue × 2.0
                  + CommentRate × 10
        """
        self.live_score = (
            concurrent_viewers * 1.0
            + self.peak_concurrent * 0.5
            + purchase_rate * 100.0
            + gift_value * 2.0
            + comment_rate * 10.0
        )
        return self.live_score
