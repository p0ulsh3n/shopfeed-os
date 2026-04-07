"""
ShopBot Core Orchestrator — Full Pipeline
==========================================
Pipeline complet par requête:

1. GUARDRAIL INPUT    → détecte jailbreak / injection avant tout traitement
2. LANGUAGE DETECT   → detect langue du message + langue de la boutique
3. INTENT CLASSIFY   → produit (RAG) ou commande (DB direct) ?
4. RETRIEVAL
   - Si produit → Hybrid Search (BM25 + Dense + RRF)
   - Si commande → Order DB query (Tool-use pattern)
5. LLM GENERATION    → vLLM avec contexte RAG ou commande
6. GUARDRAIL OUTPUT  → strip emojis, vérif auto-divulgation
7. SESSION SAVE      → historique persisté en DB
8. RESPOND           → ProductCards + images si produits, sinon texte pur
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from services.shopbot_service.bot.guardrails import (
    InputGuardrail,
    OutputGuardrail,
    ThreatLevel,
    detect_language,
)
from services.shopbot_service.bot.order_context import (
    OrderContextService,
    QueryIntent,
    classify_intent,
)
from services.shopbot_service.config import get_settings
from services.shopbot_service.database.connection import AsyncSessionLocal
from services.shopbot_service.llm.client import VLLMClient
from services.shopbot_service.models.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    MessageRole,
    ProductCard,
    RetrievedProduct,
    StreamChunk,
)
from services.shopbot_service.retrieval.hybrid_search import HybridSearchEngine

logger = logging.getLogger(__name__)
settings = get_settings()

_input_guardrail = InputGuardrail()
_output_guardrail = OutputGuardrail()


class ShopBot:
    """
    Main ShopBot orchestrator.
    One instance per request — lightweight creation.
    Shared state (encoder singleton, vLLM client) held externally.
    """

    def __init__(self, vllm_client: VLLMClient) -> None:
        self._llm = vllm_client

    # ─────────────────────── NON-STREAMING ───────────────────────

    async def handle_message(self, request: ChatRequest) -> ChatResponse:
        """
        Full pipeline — non-streaming mode.
        Returns complete ChatResponse with product cards and images.
        """
        t_start = time.perf_counter()
        session_id = request.session_id or str(uuid.uuid4())

        # Step 1: Detect language + run input guardrail
        user_lang = detect_language(request.message)
        guard = _input_guardrail.check(request.message, detected_lang=user_lang)

        if not guard.is_safe:
            # Early return — do NOT call the LLM for jailbreak attempts
            logger.info(
                f"Guardrail blocked: shop={request.shop_id} "
                f"threat={guard.threat_type} level={guard.threat_level}"
            )
            return ChatResponse(
                session_id=session_id,
                message=_output_guardrail.check_and_clean(guard.safe_response or ""),
                product_cards=[],
                sources=[],
                latency_ms=(time.perf_counter() - t_start) * 1000,
                model=settings.vllm_model,
                retrieved_count=0,
            )

        # Use sanitized input from now on
        clean_message = guard.sanitized_input

        # Step 2: Load history + shop info concurrently
        history_task = asyncio.create_task(
            self._load_history(session_id, request.shop_id)
        )
        shop_task = asyncio.create_task(
            self._get_shop_info(request.shop_id)
        )
        history, shop_name = await asyncio.gather(history_task, shop_task)
        history.extend(request.history)

        # Step 3: Intent classification
        history_text = " ".join(m.content for m in history[-4:])
        intent_result = classify_intent(clean_message, history_text)

        # Step 4: Route to appropriate retrieval strategy
        retrieved: list[RetrievedProduct] = []
        order_context: str | None = None

        if intent_result.intent in (
            QueryIntent.ORDER_STATUS,
            QueryIntent.SHIPPING_INFO,
            QueryIntent.RETURN_REQUEST,
        ):
            order_context = await self._fetch_order_context(
                shop_id=request.shop_id,
                order_id=intent_result.order_id,
                customer_id=request.session_id,  # session carries customer identity
                intent=intent_result.intent,
            )
        else:
            # Product search via RAG
            retrieved = await self._retrieve_products(
                shop_id=request.shop_id,
                query=clean_message,
                image_urls=request.image_urls,
            )

        # Step 5: LLM generation
        raw_response = await self._llm.chat(
            shop_name=shop_name,
            retrieved_products=retrieved,
            history=history,
            user_message=clean_message,
            image_urls=request.image_urls or None,
            order_context=order_context,
        )

        # Step 6: Output guardrail (strip emojis, check self-disclosure)
        response_text = _output_guardrail.check_and_clean(raw_response)

        # Step 7: Build product cards (only for product queries)
        product_cards = self._build_product_cards(retrieved) if retrieved else []

        # Step 8: Persist session
        new_user_msg = ChatMessage(role=MessageRole.USER, content=clean_message)
        new_bot_msg = ChatMessage(role=MessageRole.ASSISTANT, content=response_text)
        await self._save_history(
            session_id, request.shop_id, history + [new_user_msg, new_bot_msg]
        )

        latency_ms = (time.perf_counter() - t_start) * 1000
        return ChatResponse(
            session_id=session_id,
            message=response_text,
            product_cards=product_cards,
            sources=retrieved[:3],
            latency_ms=latency_ms,
            model=settings.vllm_model,
            retrieved_count=len(retrieved),
        )

    # ─────────────────────── STREAMING ───────────────────────────

    async def stream_message(
        self, request: ChatRequest
    ) -> AsyncGenerator[str, None]:
        """
        Full pipeline — streaming SSE mode.
        Applies guardrails, intent detection, and retrieval BEFORE streaming.
        Sends product cards in the final chunk.
        """
        session_id = request.session_id or str(uuid.uuid4())

        # Step 1: Guardrail check (runs synchronously — fast regex, < 1ms)
        user_lang = detect_language(request.message)
        guard = _input_guardrail.check(request.message, detected_lang=user_lang)

        if not guard.is_safe:
            logger.info(
                f"Guardrail blocked (stream): shop={request.shop_id} "
                f"threat={guard.threat_type}"
            )
            clean_refusal = _output_guardrail.check_and_clean(
                guard.safe_response or ""
            )
            # Stream the refusal as a single complete chunk
            final = StreamChunk(
                session_id=session_id,
                delta=clean_refusal,
                is_final=True,
                product_cards=[],
            )
            yield f"data: {final.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
            return

        clean_message = guard.sanitized_input

        # Load all context concurrently
        history_task = asyncio.create_task(
            self._load_history(session_id, request.shop_id)
        )
        shop_task = asyncio.create_task(self._get_shop_info(request.shop_id))

        history, shop_name = await asyncio.gather(history_task, shop_task)
        history.extend(request.history)

        # Intent + retrieval
        history_text = " ".join(m.content for m in history[-4:])
        intent_result = classify_intent(clean_message, history_text)

        retrieved: list[RetrievedProduct] = []
        order_context: str | None = None

        if intent_result.intent in (
            QueryIntent.ORDER_STATUS,
            QueryIntent.SHIPPING_INFO,
            QueryIntent.RETURN_REQUEST,
        ):
            order_context = await self._fetch_order_context(
                shop_id=request.shop_id,
                order_id=intent_result.order_id,
                customer_id=request.session_id,
                intent=intent_result.intent,
            )
        else:
            retrieved = await self._retrieve_products(
                shop_id=request.shop_id,
                query=clean_message,
                image_urls=request.image_urls,
            )

        # Stream tokens with output guardrail applied per-token
        full_response_parts: list[str] = []

        async for token in self._llm.stream_chat(
            shop_name=shop_name,
            retrieved_products=retrieved,
            history=history,
            user_message=clean_message,
            image_urls=request.image_urls or None,
            order_context=order_context,
        ):
            # Strip emojis from each token (fast regex)
            from services.shopbot_service.bot.guardrails import strip_emojis
            clean_token = strip_emojis(token)
            if clean_token:
                full_response_parts.append(clean_token)
                chunk = StreamChunk(
                    session_id=session_id,
                    delta=clean_token,
                    is_final=False,
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

        # Final chunk with product cards
        full_response = "".join(full_response_parts)
        product_cards = self._build_product_cards(retrieved) if retrieved else []

        final_chunk = StreamChunk(
            session_id=session_id,
            delta="",
            is_final=True,
            product_cards=product_cards,
            sources=retrieved[:3] if retrieved else None,
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

        # Persist session asynchronously
        new_user_msg = ChatMessage(role=MessageRole.USER, content=clean_message)
        new_bot_msg = ChatMessage(role=MessageRole.ASSISTANT, content=full_response)
        asyncio.create_task(
            self._save_history(
                session_id,
                request.shop_id,
                history + [new_user_msg, new_bot_msg],
            )
        )

    # ─────────────────────── ORDER CONTEXT ───────────────────────

    async def _fetch_order_context(
        self,
        shop_id: str,
        order_id: str | None,
        customer_id: str | None,
        intent: QueryIntent,
    ) -> str:
        """
        Fetch order data from DB using the Tool-use pattern.
        Returns formatted string for injection into LLM context.
        """
        async with AsyncSessionLocal() as session:
            order_service = OrderContextService(session)

            if order_id:
                # Fetch specific order
                order = await order_service.get_order_by_id(
                    order_id=order_id,
                    shop_id=shop_id,
                    customer_id=customer_id,
                )
                orders = [order] if order else []
            elif customer_id:
                # Fetch recent orders for this customer
                orders = await order_service.get_recent_orders(
                    shop_id=shop_id,
                    customer_id=customer_id,
                    limit=3,
                )
            else:
                orders = []

            return order_service.build_order_context(orders)

    # ─────────────────────── PRODUCT RETRIEVAL ───────────────────

    async def _retrieve_products(
        self,
        shop_id: str,
        query: str,
        image_urls: list[str] | None = None,
    ) -> list[RetrievedProduct]:
        """Hybrid search retrieval (RAG pipeline)."""
        async with AsyncSessionLocal() as session:
            search_engine = HybridSearchEngine(session)
            results = await search_engine.search(
                shop_id=shop_id,
                query=query,
                top_k=settings.max_context_products,
            )

        # In-stock first
        in_stock = [r for r in results if r.product.availability.value == "in_stock"]
        out_of_stock = [r for r in results if r.product.availability.value != "in_stock"]
        return (in_stock + out_of_stock)[: settings.max_context_products]

    # ─────────────────────── PRODUCT CARDS ───────────────────────

    def _build_product_cards(
        self, retrieved: list[RetrievedProduct]
    ) -> list[ProductCard]:
        """Convert retrieved products into frontend-ready ProductCard objects."""
        cards = []
        for result in retrieved[: settings.max_context_products]:
            try:
                cards.append(ProductCard.from_retrieved(result))
            except Exception as e:
                logger.warning(f"Failed to build ProductCard for {result.product.id}: {e}")
        return cards

    # ─────────────────────── SESSION MANAGEMENT ──────────────────

    async def _load_history(
        self, session_id: str, shop_id: str
    ) -> list[ChatMessage]:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                text("""
                    SELECT history FROM shopbot_sessions
                    WHERE session_id = :session_id
                      AND shop_id = :shop_id
                      AND expires_at > NOW()
                """),
                {"session_id": session_id, "shop_id": shop_id},
            )
            row = result.fetchone()

        if not row:
            return []
        try:
            raw = row[0] if isinstance(row[0], list) else json.loads(row[0])
            return [
                ChatMessage(**m)
                for m in raw[-(settings.max_history_turns * 2):]
            ]
        except Exception as e:
            logger.warning(f"Failed to parse session history: {e}")
            return []

    async def _save_history(
        self, session_id: str, shop_id: str, history: list[ChatMessage]
    ) -> None:
        max_turns = settings.max_history_turns * 2
        truncated = history[-max_turns:]
        history_json = json.dumps([m.model_dump(mode="json") for m in truncated])

        async with AsyncSessionLocal() as session:
            await session.execute(
                text("""
                    INSERT INTO shopbot_sessions
                        (session_id, shop_id, history, expires_at)
                    VALUES
                        (:session_id, :shop_id, :history::jsonb,
                         NOW() + INTERVAL '24 hours')
                    ON CONFLICT (session_id) DO UPDATE SET
                        history    = EXCLUDED.history,
                        updated_at = NOW(),
                        expires_at = NOW() + INTERVAL '24 hours'
                """),
                {
                    "session_id": session_id,
                    "shop_id": shop_id,
                    "history": history_json,
                },
            )
            await session.commit()

    # ─────────────────────── SHOP INFO ───────────────────────────

    async def _get_shop_info(self, shop_id: str) -> str:
        """
        Fetch shop name from the main DB.
        Returns shop display name only.
        The bot auto-detects and responds in the CLIENT's language,
        not a shop-level language (the shop serves an international audience).
        """
        async with AsyncSessionLocal() as session:
            try:
                result = await session.execute(
                    text("""
                        SELECT name FROM shops
                        WHERE id = :shop_id
                        LIMIT 1
                    """),
                    {"shop_id": shop_id},
                )
                row = result.fetchone()
                if row and row[0]:
                    return str(row[0])
            except Exception as e:
                logger.warning(f"Could not fetch shop info for {shop_id}: {e}")

        return "the shop"
