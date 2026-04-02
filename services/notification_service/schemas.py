"""Pydantic schemas for Notification Service — Section 24."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class NotificationType(str, Enum):
    LIVE_REMINDER = "live_reminder"
    ORDER_UPDATE = "order_update"
    FLASH_SALE = "flash_sale"
    VENDOR_NEW_PRODUCT = "vendor_new_product"
    PROMO = "promo"
    # t.md §6: Pavlovian conditioned notification types
    NARRATIVE_TEASER = "narrative_teaser"         # "Ton histoire est en pause..."
    COLLECTION_REMINDER = "collection_reminder"   # Zeigarnik "3/7 pièces complétées"
    SOCIAL_PROOF_LOCAL = "social_proof_local"      # "Tes voisins achètent..."
    DESIRE_TRIGGER = "desire_trigger"             # Circadian-timed desire activation


class NotificationRequest(BaseModel):
    user_id: str
    notification_type: str
    title: str
    body: str
    data: dict = {}
    channels: list[str] = ["push", "in_app"]  # push | in_app | email


class NotificationResponse(BaseModel):
    notification_id: str
    delivered_to: list[str] = []
    status: str = "sent"
