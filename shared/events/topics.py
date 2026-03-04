"""Kafka topic definitions — Section 14 / 30."""

from __future__ import annotations

import enum


class Topic(str, enum.Enum):
    USER_EVENTS = "shopfeed.user.events"
    COMMERCE_EVENTS = "shopfeed.commerce.events"
    VENDOR_EVENTS = "shopfeed.vendor.events"
    LIVE_EVENTS = "shopfeed.live.events"
    ORDERS_CREATED = "shopfeed.orders.created"
    ML_FEATURES = "shopfeed.ml.features"
    NOTIFICATIONS_SEND = "shopfeed.notifications.send"
    SEARCH_INDEX = "shopfeed.search.index"
