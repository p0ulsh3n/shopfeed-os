"""
Order Context Module — Tool-Use Pattern for Order/Shipping Queries
===================================================================
2026 Best Practice: Pour les données structurées (commandes, livraisons),
on utilise PAS le RAG (vectoriel). On utilise des "Tools" prédéfinis
que le LLM peut appeler via function calling.

Migration sécurité:
- AVANT: text(f"... {customer_filter}") — injection SQL possible
- APRÈS: SQLAlchemy ORM select() avec .where() conditionnel — sûr à 100%
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from shared.repositories.order_repository import OrderRepository

logger = logging.getLogger(__name__)

_order_repo = OrderRepository()


# ─────────────────────── QUERY INTENT ────────────────────────────

class QueryIntent(str, Enum):
    PRODUCT_SEARCH = "product_search"
    ORDER_STATUS = "order_status"
    SHIPPING_INFO = "shipping_info"
    RETURN_REQUEST = "return_request"
    GENERAL_SHOP = "general_shop"
    POLICY = "policy"


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
    order_id: str | None = None
    customer_identifier: str | None = None
    confidence: float = 0.0


# ─────────────────────── INTENT CLASSIFIER ───────────────────────

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

_ORDER_ID_PATTERN = re.compile(
    r"\b(?:commande?|order|ref|#|numéro)?\s*([A-Z]{0,3}[\-]?[0-9]{4,12}[A-Z0-9\-]*)\b",
    re.IGNORECASE,
)


def classify_intent(message: str, history_text: str = "") -> IntentResult:
    """Classify the user's query intent. Runs locally ~0.5ms, no LLM."""
    for pattern in _RETURN_PATTERNS:
        if pattern.search(message):
            return IntentResult(
                intent=QueryIntent.RETURN_REQUEST,
                order_id=_extract_order_id(message),
                confidence=0.85,
            )

    for pattern in _ORDER_PATTERNS:
        if pattern.search(message):
            return IntentResult(
                intent=QueryIntent.ORDER_STATUS,
                order_id=_extract_order_id(message),
                confidence=0.90,
            )

    for pattern in _SHIPPING_PATTERNS:
        if pattern.search(message):
            return IntentResult(
                intent=QueryIntent.SHIPPING_INFO,
                order_id=_extract_order_id(message),
                confidence=0.80,
            )

    return IntentResult(intent=QueryIntent.PRODUCT_SEARCH, confidence=0.70)


def _extract_order_id(message: str) -> str | None:
    match = _ORDER_ID_PATTERN.search(message)
    if match:
        candidate = match.group(1).upper().strip()
        if len(candidate) >= 4:
            return candidate
    return None


# ─────────────────────── SAFE DB QUERIES ─────────────────────────

class OrderContextService:
    """
    Safe, read-only DB query service for order/shipping data.

    MIGRATION SÉCURITÉ:
    - AVANT : text(f"SELECT ... {customer_filter}") — injection SQL via f-string
    - APRÈS : OrderRepository.get_by_order_number() avec .where() conditionnel — sûr

    Toutes les requêtes :
    - Utilisent SQLAlchemy parameterized queries (zéro injection SQL possible)
    - Filtrent par customer_id de façon conditionnelle sans f-string
    - Retournent des champs minimaux (pas de données de paiement privées)
    - Limitent strictement le nombre de résultats
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_order_by_id(
        self, order_id: str, shop_id: str, customer_id: str | None = None
    ) -> OrderInfo | None:
        """
        Fetch order details by order ID (ORM, paramétré, sans injection).
        customer_id filtré conditionnellement via .where() ORM — pas de f-string.
        """
        order = await _order_repo.get_by_order_number(
            self._session,
            order_number=order_id,
            customer_id=customer_id,
        )
        if not order:
            return None

        items = [
            OrderItem(
                product_name=str(
                    (item.product_snapshot or {}).get("name", "Produit")
                ),
                quantity=item.quantity,
                unit_price=item.unit_price,
                currency=item.currency,
            )
            for item in (order.items or [])[:10]
        ]

        tracking = order.shipment.tracking_number if order.shipment else None
        carrier = order.shipment.carrier if order.shipment else None
        estimated = (
            self._format_date(order.shipment.estimated_delivery_date)
            if order.shipment
            else None
        )

        return OrderInfo(
            order_id=str(order.order_number or order.id),
            status=str(order.status or "en_traitement"),
            created_at=self._format_date(order.created_at),
            total_amount=float(order.total_amount or 0),
            currency=str(order.currency or "XAF"),
            shipping_address=self._mask_address(
                str(order.shipping_address) if order.shipping_address else None
            ),
            payment_status=str(order.payment_status or "inconnu"),
            tracking_number=tracking,
            carrier=carrier,
            estimated_delivery=estimated,
            items=items,
        )

    async def get_recent_orders(
        self, shop_id: str, customer_id: str, limit: int = 3
    ) -> list[OrderInfo]:
        """
        Fetch recent orders for a customer in a shop.
        ORM select() avec filtres .where() — aucun f-string, aucune interpolation.
        """
        orders = await _order_repo.list_recent_by_customer(
            self._session,
            shop_id=shop_id,
            customer_id=customer_id,
            limit=min(limit, 3),
        )

        result = []
        for order in orders:
            items = [
                OrderItem(
                    product_name=str(
                        (item.product_snapshot or {}).get("name", "Produit")
                    ),
                    quantity=item.quantity,
                    unit_price=item.unit_price,
                    currency=item.currency,
                )
                for item in (order.items or [])[:10]
            ]
            tracking = order.shipment.tracking_number if order.shipment else None
            carrier = order.shipment.carrier if order.shipment else None
            estimated = (
                self._format_date(order.shipment.estimated_delivery_date)
                if order.shipment
                else None
            )
            result.append(OrderInfo(
                order_id=str(order.order_number or order.id),
                status=str(order.status or "en_traitement"),
                created_at=self._format_date(order.created_at),
                total_amount=float(order.total_amount or 0),
                currency=str(order.currency or "XAF"),
                payment_status=str(order.payment_status or "inconnu"),
                tracking_number=tracking,
                carrier=carrier,
                estimated_delivery=estimated,
                items=items,
            ))
        return result

    # ─────────────────────── CONTEXT BUILDER ─────────────────────

    def build_order_context(self, orders: list[OrderInfo]) -> str:
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
                for item in order.items[:5]:
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
        if not address:
            return None
        parts = [p.strip() for p in address.split(",")]
        if len(parts) >= 2:
            return parts[-2] + ", " + parts[-1]
        return parts[0] if parts else None
