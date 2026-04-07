"""
shared/db/models/__init__.py
Importe tous les modèles ORM dans le bon ordre pour qu'Alembic
les découvre automatiquement lors de `alembic revision --autogenerate`.
"""
from shared.db.models.analytics import (
    FeedVideoORM,
    ProductEventCounterORM,
    VendorMetricORM,
)
from shared.db.models.live_session import LiveSessionORM
from shared.db.models.order import (
    CartItemORM,
    OrderItemORM,
    OrderORM,
    ShipmentORM,
)
from shared.db.models.product import ProductORM, ProductVariantORM
from shared.db.models.user import UserFollowORM, UserORM, UserProfileORM
from shared.db.models.vendor import VendorORM

__all__ = [
    "UserORM",
    "UserProfileORM",
    "UserFollowORM",
    "VendorORM",
    "ProductORM",
    "ProductVariantORM",
    "CartItemORM",
    "OrderORM",
    "OrderItemORM",
    "ShipmentORM",
    "LiveSessionORM",
    "VendorMetricORM",
    "ProductEventCounterORM",
    "FeedVideoORM",
]
