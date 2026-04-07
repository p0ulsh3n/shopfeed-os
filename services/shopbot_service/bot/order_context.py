"""
Order Context Module — Tool-Use Pattern for Order/Shipping Queries
===================================================================
2026 Best Practice: Pour les données structurées (commandes, livraisons),
on utilise PAS le RAG (vectoriel). On utilise des "Tools" prédéfinis
que le LLM peut appeler via function calling.

Pourquoi pas RAG pour les commandes ?
- Les commandes sont des données relationnelles en temps réel
- Le RAG utilise la similarité (probabiliste) → mauvais pour les IDs exacts
- Une requête SQL est déterministe : ORDER #12345 retourne toujours #12345
- Pas besoin de re-embedder à chaque changement de statut

Architecture:
1. Intent Classifier → détecte si la question concerne une commande
2. Info Extractor    → extrait order_id / customer_id du message
3. Safe DB Queries  → fonctions paramétrées READ-ONLY (jamais d'écriture)
4. Context Builder  → formate les données pour injection dans le prompt LLM

Sécurité:
- Toutes les requêtes sont READ-ONLY
- Paramètres typés et validés (pas d'interpolation de SQL)
- Le customer_id doit matcher le shop_id (no cross-shop leaks)
- Limite stricte sur le nombre de commandes retournées
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# ─────────────────────── QUERY INTENT ────────────────────────────

class QueryIntent(str, Enum):
    PRODUCT_SEARCH = "product_search"   # Recherche produit → RAG
    ORDER_STATUS = "order_status"       # Statut commande → DB direct
    SHIPPING_INFO = "shipping_info"     # Info livraison → DB direct
    RETURN_REQUEST = "return_request"   # Retour/remboursement → DB direct
    GENERAL_SHOP = "general_shop"       # Infos boutique générales → LLM
    POLICY = "policy"                   # Politique boutique → LLM


# ─────────────────────── ORDER DATA MODELS ───────────────────────

@dataclass
class OrderItem:
    product_name: str
    quantity: int
    unit_price: float
    currency: str = "XAF"


@dataclass
class OrderInfo:
    order_id: str
    status: str
    created_at: str
    total_amount: float
    currency: str
    items: list[OrderItem] = field(default_factory=list)
    shipping_address: str | None = None
    tracking_number: str | None = None
    estimated_delivery: str | None = None
    carrier: str | None = None
    payment_status: str | None = None


@dataclass
class IntentResult:
    intent: QueryIntent
    order_id: str | None = None         # Extracted from user message
    customer_identifier: str | None = None  # Phone, email, or customer_id
    confidence: float = 0.0


# ─────────────────────── INTENT CLASSIFIER ───────────────────────

# Patterns pour détecter les intentions liées aux commandes
_ORDER_PATTERNS = [
    re.compile(
        r"\b(commande?|order|achat|purchase|transaction)\b.*"
        r"(numéro|number|ref|référence|#|no\.?|id)?\s*([A-Z0-9\-]{4,20})",
        re.IGNORECASE,
    ),
    re.compile(r"\b(statut|status|où en est|ou en est|suivre|track|suivi)\b", re.IGNORECASE),
    re.compile(r"\b(ma commande|my order|mon achat|j'ai commandé|i ordered)\b", re.IGNORECASE),
]

_SHIPPING_PATTERNS = [
    re.compile(
        r"\b(livraison|delivery|expédition|shipping|expedition|colis|"
        r"package|colissimo|DHL|chronopost|tracking)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(quand|when|délai|delay|arrive|arriver|reçu|received|"
        r"en route|en chemin|en transit)\b",
        re.IGNORECASE,
    ),
]

_RETURN_PATTERNS = [
    re.compile(
        r"\b(retour|return|remboursement|refund|rembourser|renvoyer|"
        r"retourner|échange|exchange|annuler|cancel|annulation)\b",
        re.IGNORECASE,
    ),
]

# Pattern pour extraire un order ID du message
_ORDER_ID_PATTERN = re.compile(
    r"\b(?:commande?|order|ref|#|numéro)?\s*([A-Z]{0,3}[\-]?[0-9]{4,12}[A-Z0-9\-]*)\b",
    re.IGNORECASE,
)


def classify_intent(message: str, history_text: str = "") -> IntentResult:
    """
    Classify the user's query intent.

    Product search → RAG pipeline
    Order/shipping/return → Direct DB query (Tool-use pattern)

    This classifier is fast (~0.5ms) and runs entirely locally.
    No LLM call needed for intent detection.
    """
    combined = f"{message} {history_text}".lower()

    # Check return/refund intent first (most specific)
    for pattern in _RETURN_PATTERNS:
        if pattern.search(message):
            order_id = _extract_order_id(message)
            return IntentResult(
                intent=QueryIntent.RETURN_REQUEST,
                order_id=order_id,
                confidence=0.85,
            )

    # Check order status
    for pattern in _ORDER_PATTERNS:
        if pattern.search(message):
            order_id = _extract_order_id(message)
            return IntentResult(
                intent=QueryIntent.ORDER_STATUS,
                order_id=order_id,
                confidence=0.90,
            )

    # Check shipping info
    for pattern in _SHIPPING_PATTERNS:
        if pattern.search(message):
            order_id = _extract_order_id(message)
            return IntentResult(
                intent=QueryIntent.SHIPPING_INFO,
                order_id=order_id,
                confidence=0.80,
            )

    # Default: product search (goes through RAG)
    return IntentResult(
        intent=QueryIntent.PRODUCT_SEARCH,
        confidence=0.70,
    )


def _extract_order_id(message: str) -> str | None:
    """Extract order ID from user message. Returns None if not found."""
    match = _ORDER_ID_PATTERN.search(message)
    if match:
        candidate = match.group(1).upper().strip()
        # Minimum 4 chars to avoid false positives
        if len(candidate) >= 4:
            return candidate
    return None


# ─────────────────────── SAFE DB QUERIES ─────────────────────────

class OrderContextService:
    """
    Safe, read-only DB query service for order/shipping data.

    All queries:
    - Use SQLAlchemy parameterized queries (no SQL injection possible)
    - Filter by shop_id to prevent cross-shop data leaks
    - Return minimal fields (no payment card data, no private info)
    - Hard-limit result counts
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_order_by_id(
        self, order_id: str, shop_id: str, customer_id: str | None = None
    ) -> OrderInfo | None:
        """
        Fetch order details by order ID.
        Enforces shop_id filter to prevent cross-shop data access.
        """
        params: dict[str, Any] = {
            "order_id": order_id.upper(),
            "shop_id": shop_id,
        }
        customer_filter = ""
        if customer_id:
            customer_filter = "AND o.customer_id = :customer_id"
            params["customer_id"] = customer_id

        result = await self._session.execute(
            text(f"""
                SELECT
                    o.id,
                    o.status,
                    o.created_at,
                    o.total_amount,
                    o.currency,
                    o.shipping_address,
                    o.payment_status,
                    s.tracking_number,
                    s.carrier,
                    s.estimated_delivery_date
                FROM orders o
                LEFT JOIN shipments s ON s.order_id = o.id
                WHERE (o.id = :order_id OR o.order_number = :order_id)
                  AND o.shop_id = :shop_id
                  AND o.deleted_at IS NULL
                  {customer_filter}
                LIMIT 1
            """),
            params,
        )
        row = result.fetchone()
        if not row:
            return None

        # Fetch order items
        items = await self._get_order_items(str(row[0]))

        return OrderInfo(
            order_id=str(row[0]),
            status=str(row[1] or "en_traitement"),
            created_at=self._format_date(row[2]),
            total_amount=float(row[3] or 0),
            currency=str(row[4] or "XAF"),
            shipping_address=self._mask_address(row[5]),  # Mask for privacy
            payment_status=str(row[6] or "inconnu"),
            tracking_number=str(row[7]) if row[7] else None,
            carrier=str(row[8]) if row[8] else None,
            estimated_delivery=self._format_date(row[9]) if row[9] else None,
            items=items,
        )

    async def get_recent_orders(
        self, shop_id: str, customer_id: str, limit: int = 3
    ) -> list[OrderInfo]:
        """
        Fetch customer's most recent orders for a shop.
        Maximum 3 orders to keep context manageable.
        """
        result = await self._session.execute(
            text("""
                SELECT
                    o.id, o.status, o.created_at, o.total_amount,
                    o.currency, o.payment_status,
                    s.tracking_number, s.carrier, s.estimated_delivery_date
                FROM orders o
                LEFT JOIN shipments s ON s.order_id = o.id
                WHERE o.shop_id = :shop_id
                  AND o.customer_id = :customer_id
                  AND o.deleted_at IS NULL
                ORDER BY o.created_at DESC
                LIMIT :limit
            """),
            {"shop_id": shop_id, "customer_id": customer_id, "limit": min(limit, 3)},
        )
        rows = result.fetchall()
        orders = []
        for row in rows:
            items = await self._get_order_items(str(row[0]))
            orders.append(OrderInfo(
                order_id=str(row[0]),
                status=str(row[1] or "en_traitement"),
                created_at=self._format_date(row[2]),
                total_amount=float(row[3] or 0),
                currency=str(row[4] or "XAF"),
                payment_status=str(row[5] or "inconnu"),
                tracking_number=str(row[6]) if row[6] else None,
                carrier=str(row[7]) if row[7] else None,
                estimated_delivery=self._format_date(row[8]) if row[8] else None,
                items=items,
            ))
        return orders

    async def _get_order_items(self, order_id: str) -> list[OrderItem]:
        """Fetch line items for an order."""
        result = await self._session.execute(
            text("""
                SELECT oi.quantity, oi.unit_price, oi.currency, p.name
                FROM order_items oi
                LEFT JOIN products p ON p.id = oi.product_id
                WHERE oi.order_id = :order_id
                LIMIT 10
            """),
            {"order_id": order_id},
        )
        return [
            OrderItem(
                product_name=str(row[3] or "Produit"),
                quantity=int(row[0] or 1),
                unit_price=float(row[1] or 0),
                currency=str(row[2] or "XAF"),
            )
            for row in result.fetchall()
        ]

    # ─────────────────────── CONTEXT BUILDER ─────────────────────

    def build_order_context(self, orders: list[OrderInfo]) -> str:
        """
        Format order data as plain text for LLM context injection.
        This replaces the product catalog section when the query is order-related.

        The format is designed to be unambiguous and factual.
        No placeholders or approximations.
        """
        if not orders:
            return (
                "\n[COMMANDES]\n"
                "Aucune commande trouvée pour cette boutique avec les informations fournies.\n"
            )

        lines = ["\n[COMMANDES DU CLIENT]\n"]
        for order in orders:
            lines.append(f"Commande : {order.order_id}")
            lines.append(f"  Statut           : {self._translate_status(order.status)}")
            lines.append(f"  Date             : {order.created_at}")
            lines.append(f"  Montant total    : {order.total_amount:,.0f} {order.currency}")
            lines.append(f"  Paiement         : {self._translate_payment(order.payment_status)}")

            if order.items:
                lines.append("  Articles         :")
                for item in order.items[:5]:  # Max 5 items
                    lines.append(
                        f"    - {item.product_name} x{item.quantity} "
                        f"({item.unit_price:,.0f} {item.currency})"
                    )

            if order.tracking_number:
                lines.append(f"  Numéro de suivi  : {order.tracking_number}")
            if order.carrier:
                lines.append(f"  Transporteur     : {order.carrier}")
            if order.estimated_delivery:
                lines.append(f"  Livraison prévue : {order.estimated_delivery}")
            if order.shipping_address:
                lines.append(f"  Adresse          : {order.shipping_address}")
            lines.append("")

        lines.append(
            "[IMPORTANT] Répondez uniquement sur la base de ces données. "
            "N'inventez aucune information supplémentaire.\n"
        )
        return "\n".join(lines)

    # ─────────────────────── HELPERS ─────────────────────────────

    def _translate_status(self, status: str) -> str:
        """Translate DB status codes to human-readable text."""
        status_map = {
            "pending": "En attente de confirmation",
            "confirmed": "Confirmée",
            "processing": "En cours de préparation",
            "shipped": "Expédiée",
            "delivered": "Livrée",
            "cancelled": "Annulée",
            "refunded": "Remboursée",
            "failed": "Echec de paiement",
            "en_traitement": "En cours de traitement",
        }
        return status_map.get(status.lower(), status)

    def _translate_payment(self, status: str | None) -> str:
        if not status:
            return "Inconnu"
        pay_map = {
            "paid": "Payée",
            "pending": "En attente",
            "failed": "Echec",
            "refunded": "Remboursée",
            "partial": "Partiellement payée",
        }
        return pay_map.get(status.lower(), status)

    def _format_date(self, dt: Any) -> str:
        if dt is None:
            return "Non disponible"
        if isinstance(dt, datetime):
            return dt.strftime("%d/%m/%Y à %H:%M")
        return str(dt)

    def _mask_address(self, address: str | None) -> str | None:
        """Partially mask address for privacy (show city only)."""
        if not address:
            return None
        # Return only the city portion to avoid full address disclosure
        parts = [p.strip() for p in address.split(",")]
        if len(parts) >= 2:
            return parts[-2] + ", " + parts[-1]  # City + Country
        return parts[0] if parts else None
