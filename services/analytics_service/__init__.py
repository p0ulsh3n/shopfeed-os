"""Analytics Service — Vendor Dashboard + Audience — Section 21 / 36."""

from .routes import app
from .schemas import AudienceSegment, DashboardMetrics, ProductAnalytics

__all__ = [
    "app",
    "DashboardMetrics",
    "ProductAnalytics",
    "AudienceSegment",
]
