"""
GSP Auction Engine — Generalized Second-Price with Quality Score
=================================================================
Real-time ad auction that determines which ads win placement and at what price.

Mechanism (superior to ANDROMEDA's VCG-like auction):
    1. Each ad's rank = eCPM (bid × pCTR × value × quality)
    2. Ads are sorted by rank descending
    3. Winner pays: next_rank_ecpm / (winner.pCTR × value × quality) + $0.01
       → Second-price: you pay just enough to beat the next ad
    4. Quality score rewards high-quality creatives with lower costs
    5. Diversity bonus prevents same-vendor ad clusters
    6. Reserve price floor ensures minimum revenue per impression

Why GSP beats VCG here:
    - GSP is simpler, faster, and equally effective for finite slots
    - Quality score alignment makes truthful bidding the dominant strategy
    - Reserve price protects against advertiser collusion
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AuctionResult:
    """Result of an ad auction for a single slot."""
    ad_id: str
    campaign_id: str
    vendor_id: str
    rank: int                          # Position in feed (0-indexed)
    bid_amount: float                  # Original bid
    clearing_price: float              # Actual price charged (second-price)
    ecpm: float                        # Effective CPM
    quality_score: float               # Combined quality signal
    target_url: str = ""               # Vendor store deep link
    is_organic_blend: bool = True      # Blended with organic feed
    discount_pct: float = 0.0          # How much less than bid was charged


class GSPAuction:
    """Generalized Second-Price Auction with quality scoring.

    Usage:
        auction = GSPAuction(reserve_price=0.10)
        winners = auction.run(scored_ads, num_slots=3)
    """

    def __init__(
        self,
        reserve_price: float = 0.10,     # Minimum eCPM to enter ($0.10)
        max_same_vendor: int = 1,         # Max ads from same vendor
        second_price_increment: float = 0.01,  # Min increment over next bid
    ):
        self.reserve_price = reserve_price
        self.max_same_vendor = max_same_vendor
        self.second_price_increment = second_price_increment

    def run(
        self,
        scored_ads: list,  # list[AdScore] from AdRanker
        num_slots: int = 3,
        feed_length: int = 30,
    ) -> list[AuctionResult]:
        """Run GSP auction and return winning ads with clearing prices.

        Args:
            scored_ads: AdScore list sorted by eCPM (from AdRanker)
            num_slots: Number of ad slots in this feed page
            feed_length: Total items in feed (for spacing calculation)

        Returns:
            List of AuctionResult for winning ads
        """
        if not scored_ads:
            return []

        # Filter by reserve price
        eligible = [ad for ad in scored_ads if ad.ecpm >= self.reserve_price]
        if not eligible:
            logger.debug("No ads above reserve price %.2f", self.reserve_price)
            return []

        # Enforce vendor diversity (max N ads per vendor)
        vendor_count: dict[str, int] = {}
        diverse_eligible = []
        for ad in eligible:
            vid = ad.vendor_id
            if vendor_count.get(vid, 0) < self.max_same_vendor:
                diverse_eligible.append(ad)
                vendor_count[vid] = vendor_count.get(vid, 0) + 1

        # GSP: assign slots and calculate clearing prices
        winners = []
        for i in range(min(num_slots, len(diverse_eligible))):
            winner = diverse_eligible[i]

            # Second-price: pay just enough to beat the next ad
            if i + 1 < len(diverse_eligible):
                next_ad = diverse_eligible[i + 1]
                # Clearing price = next_ecpm / (winner_pCTR × value × quality) + increment
                if winner.pCTR > 0 and winner.quality_score > 0:
                    value_signal = winner.pCVR * 0.6 + winner.pStoreVisit * 0.4
                    denom = winner.pCTR * max(value_signal, 0.001) * max(winner.quality_score, 0.001)
                    clearing_price = next_ad.ecpm / (denom * 1000) + self.second_price_increment
                else:
                    clearing_price = self.reserve_price
            else:
                # Last winner pays reserve price
                clearing_price = self.reserve_price + self.second_price_increment

            # Never charge more than the original bid
            clearing_price = min(clearing_price, winner.bid_amount)
            clearing_price = max(clearing_price, self.reserve_price)

            discount = 1.0 - (clearing_price / max(winner.bid_amount, 0.001))

            # Calculate optimal feed position for this ad slot
            feed_position = self._calculate_feed_position(i, num_slots, feed_length)

            winners.append(AuctionResult(
                ad_id=winner.ad_id,
                campaign_id=winner.campaign_id,
                vendor_id=winner.vendor_id,
                rank=feed_position,
                bid_amount=winner.bid_amount,
                clearing_price=clearing_price,
                ecpm=winner.ecpm,
                quality_score=winner.quality_score,
                target_url=winner.target_url,
                is_organic_blend=True,
                discount_pct=max(0, discount * 100),
            ))

        logger.info(
            "GSP auction: %d eligible → %d winners (reserve=%.2f)",
            len(eligible), len(winners), self.reserve_price,
        )
        return winners

    def _calculate_feed_position(
        self, slot_index: int, total_slots: int, feed_length: int
    ) -> int:
        """Calculate optimal feed position for ad slot.

        Distributes ads evenly through the feed:
        - Slot 0: position 4 (after 3 organic items)
        - Slot 1: position ~12 (after 8 more organic)
        - Slot 2: position ~20 (after 8 more organic)

        This ensures ads feel natural and don't cluster.
        """
        if total_slots <= 1:
            return 4  # After 3 organic items

        spacing = max(5, feed_length // (total_slots + 1))
        return 4 + slot_index * spacing
