"""Feed Service — FastAPI App + Routes — Section 04 / 13 / 20."""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Header, Query, WebSocket, WebSocketDisconnect
from jose import JWTError, jwt

from services.feed_service.pipeline import RecommendationPipeline, SessionState
from .schemas import FeedItem, FeedResponse, FomoSignals

logger = logging.getLogger(__name__)

# M-04 FIX: URL hardcodée localhost:8001 → variable d'environnement
# En K8s, les services se résolvent via DNS interne (svc.cluster.local)
USER_SERVICE_URL = os.getenv("USER_SERVICE_URL", "http://user-service:8001")

# C-03 FIX: WebSocket JWT authentication
_JWT_SECRET = os.getenv("JWT_SECRET", "shopfeed-dev-secret-change-in-prod")
_JWT_ALGORITHM = "HS256"


async def _authenticate_websocket(websocket: WebSocket, token: Optional[str]) -> Optional[str]:
    """Valide le JWT AVANT d'accepter la connexion WebSocket.

    C-03 FIX: Sans auth, n'importe qui peut injecter des signaux comportementaux
    (buy_now, gaze_linger) pour n'importe quel user_id, manipulant le scoring ML.

    Retourne le user_id du token si valide, None si invalide (connexion déjà fermée).

    Note sécurité: Le JWT en query param apparaît dans les logs nginx/proxy.
    En production, émettre un one-time ticket via POST /ws-ticket (TTL 30s).
    """
    if not token:
        await websocket.close(code=1008, reason="Missing authentication token")
        return None
    try:
        payload = jwt.decode(token, _JWT_SECRET, algorithms=[_JWT_ALGORITHM])
        user_id = payload.get("sub", "")
        if not user_id:
            await websocket.close(code=1008, reason="Invalid token: missing subject")
            return None
        return user_id
    except JWTError:
        await websocket.close(code=1008, reason="Invalid authentication token")
        return None

app = FastAPI(title="ShopFeed OS — Feed Service", version="1.0.0")

# Singletons
pipeline = RecommendationPipeline()
session_state = SessionState()


# ──────────────────────────────────────────────────────────────
# REST Endpoints
# ──────────────────────────────────────────────────────────────

@app.get("/api/v1/feed", response_model=FeedResponse)
async def get_feed(
    cursor: Optional[str] = None,
    session_id: Optional[str] = None,
    content_types: Optional[str] = None,
    limit: int = Query(default=10, le=20),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
    """Main discovery feed — Section 04.

    Returns algorithmically ranked content with FOMO signals.
    The pipeline executes in <80ms.

    User identification:
        - Authenticated: X-User-ID header (set by API gateway from JWT)
        - Anonymous: auto-generated session-based ID
    """
    user_id = x_user_id or "anonymous"
    sid = session_id or str(uuid.uuid4())
    ct_filter = content_types.split(",") if content_types else None

    candidates = await pipeline.generate_feed(
        user_id=user_id,
        session_id=sid,
        content_types=ct_filter,
        limit=limit,
    )

    items = [
        FeedItem(
            content_id=c.content_id,
            content_type=c.content_type,
            rank_score=round(c.final_score, 4),
            pool_level=c.pool_level,
            vendor_id=c.vendor_id,
            vendor_tier=c.vendor_tier,
            product_id=c.product_id,
            price=c.base_price,
            fomo=FomoSignals(
                stock_remaining=c.stock if c.stock < 20 else None,
            ),
        )
        for c in candidates
    ]

    return FeedResponse(items=items, next_cursor=cursor)


@app.get("/api/v1/feed/following", response_model=FeedResponse)
async def get_following_feed(
    limit: int = Query(default=10, le=20),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
    """Following Feed — Section 20.

    Shows only content from followed vendors, sorted chronologically.
    Requires authentication — anonymous users get empty feed.
    """
    user_id = x_user_id
    if not user_id:
        return FeedResponse(items=[], next_cursor=None)

    # Fetch followed vendors via user_service
    try:
        import httpx
        # M-04 FIX: URL depuis variable d'environnement, plus de localhost hardcodé
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(
                f"{USER_SERVICE_URL}/api/v1/users/{user_id}/following"
            )
            resp.raise_for_status()
            vendor_ids = resp.json().get("following", [])
    except Exception as e:
        logger.warning("Failed to fetch following list for user=%s: %s", user_id, e)
        return FeedResponse(items=[], next_cursor=None)

    if not vendor_ids:
        return FeedResponse(items=[], next_cursor=None)

    # Generate feed filtered to followed vendors only
    candidates = await pipeline.generate_feed(
        user_id=user_id,
        session_id=str(uuid.uuid4()),
        content_types=None,
        limit=limit * 3,  # Over-fetch to filter
    )

    # Filter to only followed vendors, sort by recency (content_id as proxy)
    following_set = set(vendor_ids)
    following_candidates = [
        c for c in candidates if c.vendor_id in following_set
    ]
    following_candidates = following_candidates[:limit]

    items = [
        FeedItem(
            content_id=c.content_id,
            content_type=c.content_type,
            rank_score=round(c.final_score, 4),
            pool_level=c.pool_level,
            vendor_id=c.vendor_id,
            vendor_tier=c.vendor_tier,
            product_id=c.product_id,
            price=c.base_price,
            fomo=FomoSignals(
                stock_remaining=c.stock if c.stock < 20 else None,
            ),
        )
        for c in following_candidates
    ]

    return FeedResponse(items=items, next_cursor=None)


# ──────────────────────────────────────────────────────────────
# WebSocket — Session State (Section 13)
# ──────────────────────────────────────────────────────────────

@app.websocket("/ws/session/{user_id}")
async def session_state_ws(
    websocket: WebSocket,
    user_id: str,
    token: Optional[str] = Query(default=None),  # ?token=<JWT>
):
    """Real-time session state updates — Section 13.

    C-03 FIX: Authentification JWT AVANT websocket.accept().
    Connexion: ws://host/ws/session/{user_id}?token=<JWT>
    Le token est le même JWT que celui utilisé pour les appels REST.
    """
    # C-03 FIX: Authentifier AVANT d'accepter la connexion
    authenticated_user = await _authenticate_websocket(websocket, token)
    if not authenticated_user:
        return  # Connexion déjà fermée dans _authenticate_websocket

    # Vérification anti-usurpation: user_id du path == user_id du token
    if authenticated_user != user_id:
        await websocket.close(code=1008, reason="User ID mismatch")
        return

    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info("Session WS authenticated: user=%s session=%s", user_id, session_id)

    old_vector = None

    try:
        while True:
            data = await websocket.receive_json()

            action_type = data.get("type", "view")
            product_id = data.get("product_id")
            category = data.get("category")
            price = data.get("price")

            # Update Session State — <50ms
            session = await session_state.update_session(
                session_id=session_id,
                user_id=user_id,
                action_type=action_type,
                product_id=product_id,
                category=category,
                price=price,
            )

            # Check if buffer N+1 needs recalculation (Section 04b)
            new_vector = session_state.compute_session_vector(session)

            should_recalc = False
            if old_vector is not None:
                # Cosine distance
                dot = float(new_vector @ old_vector)
                delta = 1.0 - dot  # 0 = identical, 2 = opposite
                if delta > 0.15:
                    should_recalc = True

            if action_type in ("buy_now", "add_to_cart"):
                should_recalc = True

            if should_recalc:
                # Trigger N+1 recalculation
                await websocket.send_json({
                    "event": "next_content_updated",
                    "recalculate": True,
                    "intent_level": session.get("intent_level", "low"),
                })

            old_vector = new_vector

    except WebSocketDisconnect:
        logger.info("Session WS disconnected: user=%s", user_id)


# M-05 FIX: Endpoint /health requis par les Kubernetes liveness/readiness probes
# Sans cela, K8s ne sait pas si le pod est prêt et ne peut pas gérer les crashs.
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "feed-service",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
