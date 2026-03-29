"""
ShopFeed-OS ML Inference API — FastAPI Application
Port 8100 — appelé par shop-backend/shared/infrastructure/ml_client.py

6 endpoints HTTP synchrones:
  POST /v1/feed/rank              ← SLA <80ms
  POST /v1/embed/user
  POST /v1/embed/product
  POST /v1/session/intent-vector  ← SLA <100ms
  POST /v1/moderation/clip-check
  GET  /v1/health
"""

from __future__ import annotations
import logging
import time
import asyncio
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse

from ml.inference.schemas import (
    RankRequest, RankResponse,
    EmbedUserRequest, EmbedUserResponse,
    EmbedProductRequest, EmbedProductResponse,
    IntentVectorRequest, IntentVectorResponse,
    ClipCheckRequest, ClipCheckResponse,
    HealthResponse,
)
from ml.inference.health import get_health
from ml.inference.pipeline import RankingPipeline

logger = logging.getLogger(__name__)

# ── Globals chargés au startup ──────────────────────────────────────────────

_registry = None
_faiss_index = None
_pipeline: RankingPipeline | None = None
_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Chargement des modèles au démarrage, cleanup à l'arrêt."""
    global _registry, _faiss_index, _pipeline

    logger.info("Loading ML models...")

    try:
        from ml.serving.registry import ModelRegistry
        _registry = ModelRegistry()
        _registry.load_all()
        logger.info("ModelRegistry loaded.")
    except Exception as e:
        logger.error(f"Failed to load ModelRegistry: {e}")
        _registry = None

    try:
        from ml.serving.faiss_index import FaissIndex
        _faiss_index = FaissIndex()
        _faiss_index.load()
        logger.info(f"FAISS index loaded: {_faiss_index.size()} vectors.")
    except Exception as e:
        logger.warning(f"FAISS index not loaded: {e}")
        _faiss_index = None

    _pipeline = RankingPipeline(
        registry=_registry,
        faiss_index=_faiss_index,
        redis_client=None,  # injecté si Redis disponible
    )

    logger.info("ML Inference API ready.")
    yield

    logger.info("Shutting down ML Inference API.")


# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ShopFeed OS — ML Inference API",
    version="1.0.0",
    description="Pont ML ↔ shop-backend. SLA global <200ms.",
    lifespan=lifespan,
)


def get_pipeline() -> RankingPipeline:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="ML pipeline not ready")
    return _pipeline


def get_registry():
    if _registry is None:
        raise HTTPException(status_code=503, detail="Model registry not loaded")
    return _registry


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 1 — POST /v1/feed/rank  ⚡ SLA <80ms
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/v1/feed/rank",
    response_model=RankResponse,
    summary="Feed ranking pipeline — Two-Tower → DeepFM → MTL/PLE → Diversity",
    tags=["Feed"],
)
async def feed_rank(
    request: RankRequest,
    pipeline: RankingPipeline = Depends(get_pipeline),
) -> RankResponse:
    """
    Pipeline de ranking complet en <80ms:
    1. Two-Tower ANN FAISS → 2000 candidats
    2. DeepFM pre-ranking → 400 items
    3. MTL/PLE scoring 7 objectifs
    4. DPP diversity re-ranking (anti-monopole)
    5. Pool-aware filtering
    6. Cross-sell injection si buy_now trigger
    """
    return await pipeline.rank(request)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 2 — POST /v1/embed/user
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/v1/embed/user",
    response_model=EmbedUserResponse,
    summary="Compute/refresh user tower embedding (256d)",
    tags=["Embeddings"],
)
async def embed_user(
    request: EmbedUserRequest,
    registry=Depends(get_registry),
) -> EmbedUserResponse:
    """
    Calcule l'embedding user 256d via le user tower Two-Tower.
    Stocké dans Redis ml_user_features:{user_id} (TTL 3600).
    """
    loop = asyncio.get_event_loop()

    try:
        from ml.feature_store.pipeline import user_to_features
        features = user_to_features(
            user_profile=request.profile_features.model_dump(),
            interaction_history=[i.model_dump() for i in request.interaction_history],
        )
        embedding = await loop.run_in_executor(
            None,
            lambda: registry.encode_user(features),
        )
        embedding_list = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
    except Exception as e:
        logger.warning(f"User embed failed: {e}, returning random embedding")
        embedding_list = np.random.randn(256).tolist()

    # Mise en cache Redis
    try:
        from ml.monolith.redis_store import cache_user_features
        import datetime
        await loop.run_in_executor(
            None,
            lambda: cache_user_features(request.user_id, embedding_list),
        )
    except Exception:
        pass

    import datetime
    return EmbedUserResponse(
        embedding=embedding_list,
        updated_at=datetime.datetime.utcnow().isoformat(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 3 — POST /v1/embed/product  ⚡ Déclenché à chaque publication
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/v1/embed/product",
    response_model=EmbedProductResponse,
    summary="Product multi-modal embedding pipeline (CLIP + SightEngine + BLIP-2)",
    tags=["Embeddings"],
)
async def embed_product(
    request: EmbedProductRequest,
) -> EmbedProductResponse:
    """
    Pipeline complet pour un produit:
    1. FashionSigLIP/CLIP → 512d visual embedding
    2. SentenceTransformer → text embedding
    3. SightEngine → cv_score
    4. BLIP-2 → auto_description
    5. CLIP zero-shot → category verification
    """
    t_start = time.perf_counter()
    loop = asyncio.get_event_loop()

    clip_embedding = []
    cv_score = 0.5
    auto_description = ""
    auto_tags = []
    category_verified = False

    # 1. Visual embedding (CLIP / FashionSigLIP)
    try:
        from ml.cv.clip_encoder import encode_product_image
        clip_np = await loop.run_in_executor(
            None,
            lambda: encode_product_image(request.image_url, request.category_id),
        )
        clip_embedding = clip_np.tolist()
    except Exception as e:
        logger.warning(f"CLIP encode failed for {request.product_id}: {e}")
        clip_embedding = np.zeros(512).tolist()

    # 2. CV Quality Score (SightEngine)
    try:
        from ml.cv.quality_scorer import score_product_photo
        result = await score_product_photo(request.image_url)
        cv_score = result.get("score", 0.5)
    except Exception as e:
        logger.warning(f"Quality score failed: {e}")

    # 3. BLIP-2 auto description
    try:
        from ml.cv.blip2_describer import describe_product
        auto_description = await describe_product(request.image_url, request.title)
    except Exception as e:
        logger.warning(f"BLIP-2 failed: {e}")
        auto_description = request.description[:200] if request.description else ""

    # 4. Auto-tags depuis titre + description
    auto_tags = [w.lower() for w in request.title.split() if len(w) > 3][:10]

    # 5. Category verification
    try:
        from ml.cv.category_verifier import verify_category
        ver = await loop.run_in_executor(
            None,
            lambda: verify_category(request.image_url, request.category_id),
        )
        category_verified = ver.get("match", False)
    except Exception as e:
        logger.warning(f"Category verify failed: {e}")

    pipeline_ms = (time.perf_counter() - t_start) * 1000

    return EmbedProductResponse(
        clip_embedding=clip_embedding,
        cv_score=cv_score,
        auto_description=auto_description,
        auto_tags=auto_tags,
        category_verified=category_verified,
        pipeline_ms=pipeline_ms,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 4 — POST /v1/session/intent-vector  ⚡ SLA <100ms
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/v1/session/intent-vector",
    response_model=IntentVectorResponse,
    summary="Compute session intent vector 128d via BST attention",
    tags=["Session"],
)
async def session_intent_vector(
    request: IntentVectorRequest,
    registry=Depends(get_registry),
) -> IntentVectorResponse:
    """
    Encode la séquence d'actions session en vecteur d'intent 128d.
    Stocké dans Redis ml_session_intent:{session_id} (TTL 1800).
    """
    loop = asyncio.get_event_loop()
    actions = [a.model_dump() for a in request.session_actions]

    # Calcul du vecteur d'intent via BST
    session_vector = np.zeros(128).tolist()
    try:
        from ml.feature_store.pipeline import session_to_features
        features = session_to_features(actions)
        vec = await loop.run_in_executor(
            None,
            lambda: registry.encode_session(features),
        )
        session_vector = vec.tolist() if hasattr(vec, "tolist") else list(vec)
    except Exception as e:
        logger.warning(f"Session encode failed: {e}")

    # Déduction de l'intent_level
    has_buy_now = any(a["type"] in ("buy_now",) for a in actions)
    has_cart = any(a["type"] in ("add_to_cart", "buy_now") for a in actions)
    has_zoom = any(a["type"] in ("zoom", "save") for a in actions)

    if has_buy_now:
        intent_level = "buying_now"
    elif has_cart:
        intent_level = "high"
    elif has_zoom:
        intent_level = "medium"
    else:
        intent_level = "low"

    # Catégories actives dans la session
    cat_weights: dict[int, int] = {}
    negative_cats: set[int] = set()
    skip_counts: dict[int, int] = {}

    for a in actions:
        cat = a.get("category", 0)
        if a["type"] == "skip":
            skip_counts[cat] = skip_counts.get(cat, 0) + 1
            if skip_counts[cat] >= 2:
                negative_cats.add(cat)
        else:
            cat_weights[cat] = cat_weights.get(cat, 0) + 1

    active_categories = sorted(cat_weights, key=lambda c: cat_weights[c], reverse=True)[:10]

    # Price range signal
    prices = [a["price"] for a in actions if a.get("price", 0) > 0]
    price_range = {}
    if prices:
        price_range = {"min": min(prices), "max": max(prices)}

    # Cache Redis
    try:
        from ml.monolith.redis_store import cache_session_intent
        await loop.run_in_executor(
            None,
            lambda: cache_session_intent(request.session_id, session_vector),
        )
    except Exception:
        pass

    return IntentVectorResponse(
        session_vector=session_vector,
        intent_level=intent_level,
        active_categories=active_categories,
        price_range_signal=price_range,
        negative_categories=list(negative_cats),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 5 — POST /v1/moderation/clip-check
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/v1/moderation/clip-check",
    response_model=ClipCheckResponse,
    summary="Verify product category via CLIP zero-shot matching",
    tags=["Moderation"],
)
async def moderation_clip_check(
    request: ClipCheckRequest,
) -> ClipCheckResponse:
    """
    CLIP zero-shot: vérifie que l'image correspond à la catégorie déclarée.
    Utilisé par moderation_service step 3 (category verification).
    """
    loop = asyncio.get_event_loop()
    try:
        from ml.cv.category_verifier import verify_category
        result = await loop.run_in_executor(
            None,
            lambda: verify_category(request.image_url, request.declared_category_id),
        )
        return ClipCheckResponse(
            match=result.get("match", False),
            similarity_score=result.get("score", 0.0),
            closest_category_id=result.get("closest_category", request.declared_category_id),
            confidence=result.get("confidence", 0.0),
        )
    except Exception as e:
        logger.error(f"CLIP check failed: {e}")
        return ClipCheckResponse(
            match=False,
            similarity_score=0.0,
            closest_category_id=request.declared_category_id,
            confidence=0.0,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 6 — GET /v1/health
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/v1/health",
    response_model=HealthResponse,
    summary="Health check — model versions, FAISS size, monolith lag",
    tags=["System"],
)
async def health(registry=Depends(get_registry)) -> HealthResponse:
    return await get_health(registry)


# ── Exception handlers ───────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal ML server error", "error": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "ml.inference.app:app",
        host="0.0.0.0",
        port=8100,
        workers=4,
        log_level="info",
    )
