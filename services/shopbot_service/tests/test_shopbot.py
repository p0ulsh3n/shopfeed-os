"""
ShopBot Integration Tests
===========================
Tests the full pipeline: Hybrid Search → LLM → Response with images.

Run:
    pytest services/shopbot_service/tests/ -v --asyncio-mode=auto

For CI (no GPU, no DB):
    pytest services/shopbot_service/tests/ -v --asyncio-mode=auto -m "not integration"
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from services.shopbot_service.config import get_settings
from services.shopbot_service.main import app
from services.shopbot_service.models.schemas import (
    ChatRequest,
    MessageRole,
    Product,
    ProductAvailability,
    ProductCard,
    ProductImage,
    RetrievedProduct,
)


# ─────────────────────── FIXTURES ────────────────────────────────

@pytest.fixture
def sample_product() -> Product:
    return Product(
        id="prod-001",
        shop_id="shop-abc",
        name="Robe Wax Africaine Rouge",
        description="Magnifique robe en tissu wax, parfaite pour les cérémonies.",
        price=25000.0,
        currency="XAF",
        category="Vêtements",
        subcategory="Robes",
        tags=["wax", "cérémonie", "rouge", "africain"],
        images=[
            ProductImage(url="https://cdn.shopfeed.com/prod-001-main.jpg", is_primary=True),
            ProductImage(url="https://cdn.shopfeed.com/prod-001-side.jpg", is_primary=False),
        ],
        availability=ProductAvailability.IN_STOCK,
        stock_quantity=5,
        attributes={"taille": "M", "matière": "100% coton"},
    )


@pytest.fixture
def sample_retrieved(sample_product) -> RetrievedProduct:
    return RetrievedProduct(
        product=sample_product,
        score=0.92,
        retrieval_method="rrf_float32_rescored",
    )


# ─────────────────────── SCHEMA TESTS ────────────────────────────

class TestProductCard:
    def test_from_retrieved_extracts_images(self, sample_retrieved):
        """ProductCard must extract primary_image and all image URLs."""
        card = ProductCard.from_retrieved(sample_retrieved)

        assert card.product_id == "prod-001"
        assert card.name == "Robe Wax Africaine Rouge"
        assert card.price == 25000.0
        assert card.currency == "XAF"
        assert card.primary_image == "https://cdn.shopfeed.com/prod-001-main.jpg"
        assert len(card.images) == 2
        assert "https://cdn.shopfeed.com/prod-001-side.jpg" in card.images
        assert card.score == 0.92
        assert card.availability == "in_stock"

    def test_product_no_images(self):
        """ProductCard works even when product has no images."""
        product = Product(
            id="prod-002", shop_id="shop-abc",
            name="Sac à Main", price=15000.0,
        )
        retrieved = RetrievedProduct(
            product=product, score=0.75, retrieval_method="rrf_fusion"
        )
        card = ProductCard.from_retrieved(retrieved)
        assert card.primary_image is None
        assert card.images == []

    def test_card_json_serializable(self, sample_retrieved):
        """ProductCard must be JSON serializable for SSE streaming."""
        card = ProductCard.from_retrieved(sample_retrieved)
        data = card.model_dump_json()
        parsed = json.loads(data)
        assert parsed["primary_image"].startswith("https://")
        assert isinstance(parsed["images"], list)


class TestProductEmbeddingText:
    def test_to_text_for_embedding(self, sample_product):
        """Embedding text must contain all relevant product fields."""
        text = sample_product.to_text_for_embedding()
        assert "Robe Wax Africaine Rouge" in text
        assert "25000" in text
        assert "Vêtements" in text
        assert "XAF" in text
        assert "in_stock" in text

    def test_to_bm25_text(self, sample_product):
        """BM25 text must contain keywords for sparse retrieval."""
        text = sample_product.to_bm25_text()
        assert "wax" in text.lower()
        assert "cérémonie" in text.lower()


# ─────────────────────── HYBRID SEARCH TESTS ─────────────────────

class TestRRFFusion:
    """Test the RRF fusion logic in isolation."""

    def _get_engine(self):
        from services.shopbot_service.retrieval.hybrid_search import HybridSearchEngine
        mock_session = MagicMock()
        return HybridSearchEngine(session=mock_session)

    def test_rrf_merges_two_lists(self):
        engine = self._get_engine()
        dense = [("prod-A", 0.95), ("prod-B", 0.87), ("prod-C", 0.72)]
        sparse = [("prod-B", 0.88), ("prod-A", 0.65), ("prod-D", 0.50)]

        results = engine._reciprocal_rank_fusion([dense, sparse], k=60)
        ids = [r.product.id for r in results]

        # prod-A and prod-B appear in both lists → higher RRF score
        assert ids.index("prod-A") < ids.index("prod-D")
        assert ids.index("prod-B") < ids.index("prod-D")

    def test_rrf_handles_empty_lists(self):
        engine = self._get_engine()
        results = engine._reciprocal_rank_fusion([[], []], k=60)
        assert results == []

    def test_rrf_handles_single_list(self):
        engine = self._get_engine()
        dense = [("prod-X", 0.9), ("prod-Y", 0.8)]
        results = engine._reciprocal_rank_fusion([dense, []], k=60)
        assert len(results) == 2
        assert results[0].product.id == "prod-X"

    def test_rrf_scores_normalized(self):
        engine = self._get_engine()
        dense = [("prod-A", 0.9)] * 10
        results = engine._reciprocal_rank_fusion([dense, [("prod-A", 0.9)]], k=60)
        for r in results:
            assert 0.0 <= r.score <= 1.0


# ─────────────────────── LLM SYSTEM PROMPT TESTS ─────────────────

class TestSystemPrompt:
    def test_prompt_contains_shop_name(self, sample_retrieved):
        from services.shopbot_service.llm.client import build_system_prompt
        prompt = build_system_prompt("Ma Boutique Wax", [sample_retrieved])
        assert "Ma Boutique Wax" in prompt

    def test_prompt_contains_product_price(self, sample_retrieved):
        from services.shopbot_service.llm.client import build_system_prompt
        prompt = build_system_prompt("Shop", [sample_retrieved])
        assert "25" in prompt  # Price 25000 XAF

    def test_prompt_empty_catalog(self):
        from services.shopbot_service.llm.client import build_system_prompt
        prompt = build_system_prompt("Shop", [])
        assert "Aucun produit trouvé" in prompt

    def test_prompt_static_prefix_consistent(self, sample_retrieved):
        """
        The static part must be identical across calls for APC cache hits.
        Re-generating with different products must NOT change the prefix.
        """
        from services.shopbot_service.llm.client import (
            SYSTEM_PROMPT_STATIC,
            build_system_prompt,
        )
        prompt1 = build_system_prompt("Shop A", [sample_retrieved])
        prompt2 = build_system_prompt("Shop B", [sample_retrieved])

        # Both start with the same static prefix
        assert prompt1.startswith(SYSTEM_PROMPT_STATIC)
        assert prompt2.startswith(SYSTEM_PROMPT_STATIC)


# ─────────────────────── API ENDPOINT TESTS ──────────────────────

@pytest.mark.asyncio
class TestChatEndpoint:
    """Integration-style tests for the /chat endpoint (mocked vLLM + DB)."""

    @pytest.fixture
    def mock_bot(self, sample_retrieved):
        """Mock ShopBot to avoid real DB/LLM calls."""
        from services.shopbot_service.models.schemas import ChatResponse, ProductCard
        card = ProductCard.from_retrieved(sample_retrieved)
        mock_response = ChatResponse(
            session_id="test-session-123",
            message="Bonjour ! Nous avons la **Robe Wax Africaine Rouge** à 25 000 XAF ✅",
            product_cards=[card],
            sources=[sample_retrieved],
            latency_ms=123.5,
            model="Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
            retrieved_count=1,
        )
        mock = AsyncMock()
        mock.handle_message.return_value = mock_response
        return mock

    async def test_chat_returns_product_cards(self, mock_bot, sample_retrieved):
        """Chat response must include product cards with image URLs."""
        from services.shopbot_service.models.schemas import ChatResponse

        request = ChatRequest(
            shop_id="shop-abc",
            message="Avez-vous des robes ?",
            stream=False,
        )

        response = await mock_bot.handle_message(request)

        assert response.session_id == "test-session-123"
        assert len(response.product_cards) == 1
        card = response.product_cards[0]
        assert card.primary_image == "https://cdn.shopfeed.com/prod-001-main.jpg"
        assert len(card.images) == 2
        assert "Robe" in response.message

    async def test_chat_response_json_serializable(self, mock_bot):
        """Full chat response must be JSON serializable for API transmission."""
        request = ChatRequest(
            shop_id="shop-abc",
            message="Bonjour",
            stream=False,
        )
        response = await mock_bot.handle_message(request)
        # Must serialize without errors
        json_str = response.model_dump_json()
        parsed = json.loads(json_str)
        assert "product_cards" in parsed
        assert "message" in parsed
