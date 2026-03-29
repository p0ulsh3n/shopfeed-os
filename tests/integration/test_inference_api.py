"""
Integration test — ML Inference API end-to-end.
Tests les 6 endpoints via httpx async client.
"""

from __future__ import annotations
import asyncio
import pytest
import pytest_asyncio

import httpx

BASE_URL = "http://localhost:8100"


@pytest_asyncio.fixture
async def client():
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=5.0) as c:
        yield c


# ── Endpoint 1: /v1/feed/rank ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_feed_rank_basic(client: httpx.AsyncClient):
    payload = {
        "user_id": "test-user-001",
        "session_vector": [0.1] * 128,
        "session_actions": [
            {"type": "view", "product_id": "prod-001", "category": 1,
             "price": 29.99, "dwell_ms": 3000, "watch_pct": 0.5, "timestamp": "2026-03-05T10:00:00Z"}
        ],
        "intent_level": "medium",
        "limit": 10,
        "context": "feed",
    }
    resp = await client.post("/v1/feed/rank", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "candidates" in data
    assert "pipeline_ms" in data
    assert data["pipeline_ms"] < 5000  # SLA 5s (test env sans GPU)


@pytest.mark.asyncio
async def test_feed_rank_with_candidates(client: httpx.AsyncClient):
    payload = {
        "user_id": "test-user-002",
        "session_vector": [0.0] * 128,
        "candidates": ["prod-001", "prod-002", "prod-003"],
        "session_actions": [],
        "intent_level": "low",
        "limit": 3,
    }
    resp = await client.post("/v1/feed/rank", json=payload)
    assert resp.status_code == 200


# ── Endpoint 2: /v1/embed/user ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_embed_user(client: httpx.AsyncClient):
    payload = {
        "user_id": "test-user-003",
        "interaction_history": [
            {"item_id": "prod-001", "action": "view", "timestamp": "2026-03-05T09:00:00Z"}
        ],
        "profile_features": {
            "category_prefs": {"1": 0.8, "3": 0.5},
            "price_ranges": {},
            "purchase_history": [],
        },
    }
    resp = await client.post("/v1/embed/user", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["embedding"]) == 256
    assert "updated_at" in data


# ── Endpoint 3: /v1/embed/product ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_embed_product(client: httpx.AsyncClient):
    payload = {
        "product_id": "test-prod-001",
        "image_url": "https://via.placeholder.com/400x400",
        "title": "Robe fleurie bleue",
        "description": "Belle robe en coton avec motifs floraux",
        "price": 29.99,
        "category_id": 1,
        "attributes": {"color": "bleu", "material": "coton"},
    }
    resp = await client.post("/v1/embed/product", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["clip_embedding"]) == 512
    assert 0.0 <= data["cv_score"] <= 1.0
    assert "pipeline_ms" in data


# ── Endpoint 4: /v1/session/intent-vector ────────────────────────────────────

@pytest.mark.asyncio
async def test_session_intent_vector(client: httpx.AsyncClient):
    payload = {
        "session_id": "test-session-001",
        "session_actions": [
            {"type": "view", "product_id": "p1", "category": 1, "price": 50.0,
             "dwell_ms": 5000, "watch_pct": 0.7, "timestamp": "2026-03-05T10:00:00Z"},
            {"type": "add_to_cart", "product_id": "p1", "category": 1, "price": 50.0,
             "dwell_ms": 0, "watch_pct": 0.0, "timestamp": "2026-03-05T10:01:00Z"},
        ],
    }
    resp = await client.post("/v1/session/intent-vector", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["session_vector"]) == 128
    assert data["intent_level"] in ("low", "medium", "high", "buying_now")


# ── Endpoint 5: /v1/moderation/clip-check ────────────────────────────────────

@pytest.mark.asyncio
async def test_clip_check(client: httpx.AsyncClient):
    payload = {
        "image_url": "https://via.placeholder.com/400x400",
        "declared_category_id": 1,
    }
    resp = await client.post("/v1/moderation/clip-check", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "match" in data
    assert "similarity_score" in data
    assert "confidence" in data


# ── Endpoint 6: /v1/health ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health(client: httpx.AsyncClient):
    resp = await client.get("/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("ok", "degraded", "down")
    assert "model_versions" in data
    assert "faiss_index_size" in data
    assert "uptime_s" in data
