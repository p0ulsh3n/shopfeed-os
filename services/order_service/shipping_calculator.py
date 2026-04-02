"""
Shipping Rate Calculator — Multi-Vendor Zone + Weight Pricing.

Calculates shipping costs for a multi-vendor cart by:
1. Grouping cart items by vendor_id
2. Calling geosort to determine geographic zone (A/B/C) per vendor
3. Looking up the vendor's rate table for that zone
4. Calculating cost based on total weight + weight tiers
5. Applying free shipping if order subtotal exceeds threshold

Architecture:
    App React Native (buyer lat/lon)
        → geosort (zone A/B/C per vendor)
        → shipping_calculator (vendor rate table + weight)
        → checkout (total + breakdown)

IMPORTANT: No default rates. Vendors MUST configure their shipping
rates for each zone they serve. If a vendor hasn't configured a zone,
that zone is unavailable (vendor doesn't ship there).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Errors ───────────────────────────────────────────────────────

class ShippingUnavailableError(Exception):
    """Raised when a vendor has no shipping config for a zone."""
    def __init__(self, vendor_id: str, zone: str):
        self.vendor_id = vendor_id
        self.zone = zone
        super().__init__(
            f"Vendor {vendor_id} does not ship to Zone {zone}. "
            f"Please configure shipping rates for this zone."
        )


# ── Result dataclasses ───────────────────────────────────────────

@dataclass
class VendorShippingResult:
    """Shipping cost calculation result for one vendor."""
    vendor_id: str
    zone: str                   # "A", "B", "C"
    zone_label: str             # "local_delivery" / "national_shipping" / "international_shipping"
    total_weight_g: int         # Total weight of items + packaging
    item_count: int
    subtotal: float             # Price subtotal for this vendor's items
    shipping_cost: float        # Calculated shipping cost
    is_free: bool = False       # Was free shipping applied?
    free_reason: str = ""       # Why it's free (e.g. "order_above_50000")
    distance_km: float = 0.0
    vendor_city: str = ""
    buyer_city: str = ""
    shipping_suggestion: str = ""
    available: bool = True      # False if vendor doesn't ship to this zone
    error: str = ""             # Error message if unavailable


@dataclass
class CartShippingResult:
    """Full cart shipping calculation — all vendors combined."""
    total_shipping_cost: float
    total_items_cost: float
    grand_total: float
    vendor_count: int
    vendors: list[VendorShippingResult] = field(default_factory=list)
    free_shipping_hints: list[str] = field(default_factory=list)
    has_unavailable: bool = False  # True if any vendor can't ship


# ── Core calculation functions ───────────────────────────────────

def calculate_vendor_shipping(
    items: list[dict],
    zone: str,
    vendor_shipping_config: dict | None = None,
    package_weight_g: int = 100,
) -> VendorShippingResult:
    """Calculate shipping cost for a single vendor's items in a given zone.

    Args:
        items: List of cart items from this vendor. Each must have:
            - weight_g (int): product weight in grams
            - quantity (int): number of units
            - unit_price (float): price per unit
        zone: Geographic zone ("A", "B", or "C") from geosort
        vendor_shipping_config: Vendor's zone rate config (dict with
            base_price, base_weight_g, price_per_extra_kg, free_above).
            REQUIRED — no defaults. If None, shipping is unavailable.
        package_weight_g: Additional packaging weight (grams)

    Returns:
        VendorShippingResult with calculated cost
    """
    # 1. Calculate total weight and subtotal
    total_weight_g = package_weight_g
    subtotal = 0.0
    item_count = 0

    for item in items:
        qty = item.get("quantity", 1)
        weight = item.get("weight_g", 500)  # Default 500g if not specified
        price = item.get("unit_price", 0.0)
        total_weight_g += weight * qty
        subtotal += price * qty
        item_count += qty

    zone_label = "local_delivery" if zone == "A" else "national_shipping" if zone == "B" else "international_shipping"

    # 2. Check vendor has configured shipping for this zone
    if not vendor_shipping_config:
        return VendorShippingResult(
            vendor_id="",
            zone=zone,
            zone_label=zone_label,
            total_weight_g=total_weight_g,
            item_count=item_count,
            subtotal=round(subtotal, 2),
            shipping_cost=0.0,
            available=False,
            error=f"Shipping not available for Zone {zone}",
        )

    rate = vendor_shipping_config
    base_price = rate.get("base_price", 0.0)
    base_weight_g = rate.get("base_weight_g", 2000)
    price_per_extra_kg = rate.get("price_per_extra_kg", 0.0)
    free_above = rate.get("free_above")

    # 3. Check free shipping threshold
    is_free = False
    free_reason = ""
    if free_above is not None and subtotal >= free_above:
        is_free = True
        free_reason = f"order_above_{int(free_above)}"

    # 4. Calculate shipping cost
    if is_free:
        shipping_cost = 0.0
    else:
        # Base price covers up to base_weight_g
        if total_weight_g <= base_weight_g:
            shipping_cost = base_price
        else:
            # Extra weight charged per kg
            extra_weight_kg = (total_weight_g - base_weight_g) / 1000.0
            shipping_cost = base_price + (extra_weight_kg * price_per_extra_kg)


    return VendorShippingResult(
        vendor_id="",  # Set by caller
        zone=zone,
        zone_label=zone_label,
        total_weight_g=total_weight_g,
        item_count=item_count,
        subtotal=round(subtotal, 2),
        shipping_cost=round(shipping_cost, 2),
        is_free=is_free,
        free_reason=free_reason,
    )


def calculate_cart_shipping(
    cart_items: list[dict],
    vendors_configs: dict[str, dict] | None = None,
    buyer_lat: float | None = None,
    buyer_lon: float | None = None,
) -> CartShippingResult:
    """Calculate shipping for an entire multi-vendor cart.

    This is the main entry point called from checkout.

    Process:
    1. Group items by vendor_id
    2. For each vendor: call geosort → get zone → calculate shipping
    3. Return combined result with per-vendor breakdown

    Args:
        cart_items: Full cart. Each item must have:
            - vendor_id (str)
            - product_id (str)
            - weight_g (int)
            - quantity (int)
            - unit_price (float)
            - vendor_lat (float): vendor's GPS latitude
            - vendor_lon (float): vendor's GPS longitude
        vendors_configs: Dict mapping vendor_id → their shipping configs.
            Each config has zone_rates: list of {zone, base_price, ...}
        buyer_lat, buyer_lon: Buyer's GPS coordinates

    Returns:
        CartShippingResult with total + per-vendor breakdown
    """
    from services.geosort_service.classifier import classify_order

    vendors_configs = vendors_configs or {}

    # 1. Group items by vendor
    vendor_groups: dict[str, list[dict]] = {}
    for item in cart_items:
        vid = str(item.get("vendor_id", "unknown"))
        vendor_groups.setdefault(vid, []).append(item)

    # 2. Calculate per vendor
    vendor_results: list[VendorShippingResult] = []
    free_shipping_hints: list[str] = []

    for vendor_id, items in vendor_groups.items():
        # Get vendor GPS from first item
        vendor_lat = items[0].get("vendor_lat", 0.0)
        vendor_lon = items[0].get("vendor_lon", 0.0)

        # Classify zone via geosort
        if buyer_lat and buyer_lon and vendor_lat and vendor_lon:
            geo_result = classify_order(
                vendor_lat=vendor_lat,
                vendor_lon=vendor_lon,
                buyer_lat=buyer_lat,
                buyer_lon=buyer_lon,
            )
            zone = geo_result.zone
            distance_km = geo_result.distance_km
            vendor_city = geo_result.vendor_city
            buyer_city = geo_result.buyer_city
            shipping_suggestion = geo_result.shipping_suggestion
        else:
            # Fallback: no GPS → assume Zone C (worst case)
            zone = "C"
            distance_km = 0.0
            vendor_city = ""
            buyer_city = ""
            shipping_suggestion = ""

        # Get vendor's rate config for this zone
        vendor_config = vendors_configs.get(vendor_id, {})
        zone_rates = vendor_config.get("zone_rates", [])
        zone_rate = None
        for zr in zone_rates:
            if zr.get("zone") == zone:
                zone_rate = zr
                break

        # Calculate package weight
        package_weight_g = vendor_config.get("package_weight_g", 100)

        # Calculate shipping
        result = calculate_vendor_shipping(
            items=items,
            zone=zone,
            vendor_shipping_config=zone_rate,
            package_weight_g=package_weight_g,
        )
        result.vendor_id = vendor_id
        result.distance_km = distance_km
        result.vendor_city = vendor_city
        result.buyer_city = buyer_city
        result.shipping_suggestion = shipping_suggestion

        vendor_results.append(result)

        # Generate free shipping hint if applicable
        if not result.is_free and zone_rate:
            free_above = zone_rate.get("free_above")
            if free_above and result.subtotal < free_above:
                remaining = free_above - result.subtotal
                free_shipping_hints.append(
                    f"Add {remaining:.0f} more from this vendor for free shipping"
                )

    # 3. Combine
    total_shipping = sum(r.shipping_cost for r in vendor_results)
    total_items = sum(r.subtotal for r in vendor_results)
    has_unavailable = any(not r.available for r in vendor_results)

    return CartShippingResult(
        total_shipping_cost=round(total_shipping, 2),
        total_items_cost=round(total_items, 2),
        grand_total=round(total_items + total_shipping, 2),
        vendor_count=len(vendor_results),
        vendors=vendor_results,
        free_shipping_hints=free_shipping_hints,
        has_unavailable=has_unavailable,
    )
