"""
shared/repositories/__init__.py
Expose tous les repositories pour import simple.
"""
from shared.repositories.analytics_repository import AnalyticsRepository
from shared.repositories.live_repository import LiveSessionRepository
from shared.repositories.order_repository import CartRepository, OrderRepository
from shared.repositories.product_repository import ProductRepository, ProductVariantRepository
from shared.repositories.user_repository import (
    UserFollowRepository,
    UserProfileRepository,
    UserRepository,
)
from shared.repositories.vendor_repository import VendorRepository

__all__ = [
    "UserRepository",
    "UserProfileRepository",
    "UserFollowRepository",
    "VendorRepository",
    "ProductRepository",
    "ProductVariantRepository",
    "CartRepository",
    "OrderRepository",
    "LiveSessionRepository",
    "AnalyticsRepository",
]
