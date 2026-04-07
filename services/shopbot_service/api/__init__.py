"""ShopBot API package."""
from services.shopbot_service.api.routes import router, admin_router, startup, shutdown

__all__ = ["router", "admin_router", "startup", "shutdown"]
