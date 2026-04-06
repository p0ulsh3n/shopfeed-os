"""
RankingPipeline — orchestrateur du pipeline de ranking ML.
Gère les 6 étapes : Two-Tower retrieve → DeepFM pre-rank → MTL/PLE score
→ Diversity re-rank → Pool filter → Session cross-sell injection.
SLA: <80ms total
"""

from __future__ import annotations
import asyncio
import time
import logging
from typing import Optional

import numpy as np

from ml.inference.schemas import (
    RankRequest, RankResponse, RankedCandidate, MTLScores, DiversityFlags
)

logger = logging.getLogger(__name__)

# ── Context-Aware MTL Weights ────────────────────────────────────────────────
#
# Feed (scroll infini TikTok-style): Engagement-first → keep them scrolling
#   High: watch_time, share, save → "c'est intéressant, je reste"
#   Medium: purchase signals → naturel, pas forcé
#   Low: negative penalty → tolérant (exploration mode)
#
# Marketplace (product pages): Conversion-first + addictive discovery
#   High: purchase, add_to_cart, buy_now → "j'achète"
#   Medium: save, share → intention future + viralité
#   Very high negative penalty → ne PAS montrer ce qui rebute
#
# Live (live shopping): Urgency + conversion → FOMO
#   Highest: buy_now → limité dans le temps
#   High: purchase, add_to_cart → conversion immédiate

CONTEXT_WEIGHTS = {
    "feed": {
        "e_watch_time": 10.0,   # Priorité #1: garder l'attention
        "p_share": 8.0,         # Viralité → croissance organique
        "p_save": 7.0,          # Intention future → rétention
        "p_add_to_cart": 5.0,   # Conversion soft
        "p_purchase": 4.0,      # Conversion naturelle (pas forcée)
        "p_buy_now": 3.0,       # Rarement dans le feed scroll
        "p_negative": -5.0,     # Tolérant: exploration > précision
    },
    "marketplace": {
        "p_purchase": 14.0,     # Priorité #1: CONVERSION
        "p_buy_now": 13.0,      # Achat immédiat → revenue
        "p_add_to_cart": 11.0,  # Panier → quasi-conversion
        "p_save": 8.0,          # Wishlist → retour utilisateur → addiction
        "p_share": 6.0,         # Social proof → viralité marketplace
        "e_watch_time": 5.0,    # Engagement sur la fiche produit
        "p_negative": -12.0,    # TRÈS pénalisé: chaque skip = vente perdue
    },
    "live": {
        "p_buy_now": 15.0,      # URGENCE: acheter pendant le live
        "p_purchase": 12.0,     # Conversion post-live
        "p_add_to_cart": 10.0,  # Panier pendant le live
        "e_watch_time": 9.0,    # Rester sur le live
        "p_save": 4.0,
        "p_share": 7.0,         # Inviter des amis au live
        "p_negative": -10.0,
    },
}

# Fallback = marketplace (comportement actuel)
DEFAULT_CONTEXT = "marketplace"

# ── Context-Aware Diversity Rules ────────────────────────────────────────────
#
# Feed: haute diversité → on veut surprendre → addiction par la découverte
# Marketplace: diversité agressive → on veut montrer beaucoup de catégories
#              et vendeurs pour que l'utilisateur EXPLORE et reste → addiction
#              par la variété (effet "je scrolle encore car y'a toujours du nouveau")

CONTEXT_DIVERSITY = {
    "feed": {
        "max_same_vendor": 1,     # Max 1 même vendor par page
        "max_same_category": 2,   # Max 2 même catégorie → variété
        "inject_new_vendor": True,
        "inject_regional": True,
        "discovery_boost": 1.15,  # +15% score pour items jamais vus
    },
    "marketplace": {
        "max_same_vendor": 2,     # 2 max → assez pour comparer
        "max_same_category": 3,   # 3 max → explorer dans la catégorie MAIS switcher
        "inject_new_vendor": True, # Toujours injecter des nouveaux → addiction
        "inject_regional": True,
        "discovery_boost": 1.25,  # +25% boost → forte poussée de découverte
        "cross_sell_slots": 3,    # 3 items cross-sell au lieu de 2
        "price_variety": True,    # Mélanger les prix → gamification inconsciente
    },
    "live": {
        "max_same_vendor": 5,     # Le live = 1 vendeur
        "max_same_category": 10,  # Pas de filtre catégorie en live
        "inject_new_vendor": False,
        "inject_regional": False,
        "discovery_boost": 1.0,
    },
}

# Seuils des pools de trafic
POOL_IMPRESSION_RANGES = {
    "L1": (200, 800),
    "L2": (1_000, 5_000),
    "L3": (5_000, 30_000),
    "L4": (30_000, 200_000),
    "L5": (200_000, 2_000_000),
    "L6": (2_000_000, float("inf")),
}


class RankingPipeline:
    """
    Pipeline de ranking en 6 étapes.
    SLA global: <80ms par request depuis shop-backend.
    """

    def __init__(self, registry, faiss_index, redis_client=None):
        self.registry = registry
        self.faiss_index = faiss_index
        self.redis = redis_client

    # ─── Étape 1 : Two-Tower ANN Retrieval (<10ms) ─────────────────────────

    async def _retrieve_candidates(
        self,
        user_id: str,
        session_vector: list[float],
        k: int = 2000
    ) -> list[str]:
        """
        FAISS ANN search depuis le user embedding.
        Returns liste de item_ids.
        """
        loop = asyncio.get_event_loop()
        query = np.array(session_vector, dtype=np.float32).reshape(1, -1)
        item_ids, _ = await loop.run_in_executor(
            None, lambda: self.faiss_index.search(query, k)
        )
        candidates = [str(iid) for iid in item_ids[0] if iid != -1]
        logger.debug("Retrieved %d candidates for user=%s", len(candidates), user_id)
        return candidates

    # ─── Étape 2 : DeepFM pre-ranking (<20ms) ──────────────────────────────

    async def _prerank_deepfm(
        self,
        candidates: list[str],
        user_id: str,
        top_k: int = 400
    ) -> list[str]:
        """
        DeepFM léger pour filtrer 2000 → 400 items.
        """
        loop = asyncio.get_event_loop()
        try:
            scores = await loop.run_in_executor(
                None,
                lambda: self.registry.predict_deepfm(user_id, candidates)
            )
            sorted_ids = sorted(
                zip(candidates, scores),
                key=lambda x: x[1],
                reverse=True
            )
            return [iid for iid, _ in sorted_ids[:top_k]]
        except Exception as e:
            logger.warning(f"DeepFM pre-rank failed: {e}, using raw candidates")
            return candidates[:top_k]

    # ─── Étape 3 : MTL/PLE scoring (<30ms) ─────────────────────────────────

    async def _score_mtl(
        self,
        candidates: list[str],
        user_id: str,
        session_actions: list,
        intent_level: str,
    ) -> dict[str, dict]:
        """
        MTL/PLE: score les 7 objectifs simultanément.
        Retourne {item_id: {task: score}}
        """
        loop = asyncio.get_event_loop()
        try:
            raw_scores = await loop.run_in_executor(
                None,
                lambda: self.registry.predict_mtl(
                    user_id,
                    candidates,
                    session_actions=[a.model_dump() for a in session_actions],
                    intent_level=intent_level,
                )
            )
            return raw_scores
        except Exception as e:
            logger.warning(f"MTL scoring failed: {e}, using popularity-based fallback")
            return self._popularity_fallback(candidates)

    def _compute_commerce_score(self, mtl_scores: dict, context: str = "marketplace") -> float:
        """Σ (task_weight × prediction) — context-aware scoring.

        Feed: engagement-weighted (watch_time, share dominate)
        Marketplace: conversion-weighted (purchase, cart dominate) + addictive
        Live: urgency-weighted (buy_now dominates)
        """
        weights = CONTEXT_WEIGHTS.get(context, CONTEXT_WEIGHTS[DEFAULT_CONTEXT])
        return sum(
            weights.get(task, 0.0) * mtl_scores.get(task, 0.0)
            for task in weights
        )

    def _popularity_fallback(self, candidates: list[str]) -> dict[str, dict]:
        """Popularity-based MTL fallback — uses Redis impression counts.

        Instead of giving identical scores to every item (which makes ranking
        random), we use each item's impression count as a proxy for quality.
        More-seen items get slightly higher base scores, creating a meaningful
        ranking even when the MTL model is unavailable.
        """
        fallback = {}
        impressions = {}

        # Try to fetch real popularity data from Redis
        if self.redis:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                pipe = self.redis.pipeline()
                for iid in candidates:
                    pipe.get(f"item:{iid}:impressions")
                values = loop.run_until_complete(pipe.execute()) if pipe else []
                for iid, val in zip(candidates, values):
                    impressions[iid] = int(val) if val else 0
            except Exception:
                pass

        max_imp = max(impressions.values()) if impressions else 1

        for i, iid in enumerate(candidates):
            # Normalize impression count to 0-1 range, with position decay
            pop_score = impressions.get(iid, 0) / max(max_imp, 1)
            position_decay = 1.0 - (i / max(len(candidates), 1)) * 0.1
            base = 0.01 + pop_score * 0.04  # Range: 0.01 → 0.05

            fallback[iid] = {
                "p_buy_now": base * 0.5 * position_decay,
                "p_purchase": base * 0.8 * position_decay,
                "p_add_to_cart": base * 1.5 * position_decay,
                "p_save": base * 2.0 * position_decay,
                "p_share": base * 0.5 * position_decay,
                "e_watch_time": 0.3 + pop_score * 0.4,
                "p_negative": 0.01,
            }

        return fallback

    # ─── Étape 4 : Context-Aware Diversity Re-ranking (<10ms) ─────────────────

    def _apply_diversity(
        self,
        scored_items: list[tuple[str, float, dict]],
        limit: int,
        context: str = "marketplace",
    ) -> list[tuple[str, float, dict, DiversityFlags]]:
        """Context-aware diversity re-ranking.

        Feed:        max 1 vendor, 2 categories → surprise & discovery
        Marketplace: max 2 vendor, 3 categories → variety addiction
                     + discovery boost (+25%) for unseen items
                     + price mixing for subconscious gamification
        Live:        relaxed diversity (single vendor show)
        """
        rules = CONTEXT_DIVERSITY.get(context, CONTEXT_DIVERSITY["marketplace"])
        max_vendor = rules["max_same_vendor"]
        max_cat = rules["max_same_category"]
        discovery_boost = rules.get("discovery_boost", 1.0)

        vendor_counts: dict[str, int] = {}
        cat_counts: dict[str, int] = {}
        result = []
        seen_prices = []  # For price variety tracking

        for item_id, score, meta in scored_items:
            vendor_id = meta.get("vendor_id", "")
            cat_id = str(meta.get("category_id", ""))

            # Enforce vendor/category caps
            if vendor_counts.get(vendor_id, 0) >= max_vendor:
                continue
            if cat_counts.get(cat_id, 0) >= max_cat:
                continue

            # Discovery boost: items from new vendors get a score bump
            # This creates the "there's always something new" addiction loop
            adjusted_score = score
            is_new = meta.get("is_new_vendor", False)
            is_regional = meta.get("is_regional", False)

            if is_new and rules.get("inject_new_vendor", False):
                adjusted_score *= discovery_boost
            if is_regional and rules.get("inject_regional", False):
                adjusted_score *= 1.10  # +10% regional boost

            # Price variety (marketplace): alternate between price ranges
            # to create subconscious "treasure hunt" effect
            if rules.get("price_variety") and len(result) >= 3:
                item_price = meta.get("price", 0)
                if item_price > 0 and seen_prices:
                    avg_recent = sum(seen_prices[-3:]) / len(seen_prices[-3:])
                    # Boost items with different price range than recent items
                    if abs(item_price - avg_recent) / max(avg_recent, 1) > 0.3:
                        adjusted_score *= 1.08  # +8% variety bonus
                if item_price > 0:
                    seen_prices.append(item_price)

            flags = DiversityFlags(
                is_new_vendor=is_new,
                is_regional=is_regional,
                is_cold_start=meta.get("is_cold_start", False),
            )

            vendor_counts[vendor_id] = vendor_counts.get(vendor_id, 0) + 1
            cat_counts[cat_id] = cat_counts.get(cat_id, 0) + 1
            result.append((item_id, adjusted_score, meta, flags))

            if len(result) >= limit:
                break

        return result

    # ─── Étape 5 : Pool-aware filtering (<5ms) ──────────────────────────────

    def _get_pool_level(self, meta: dict) -> str:
        return meta.get("pool_level", "L1")

    # ─── Étape 6 : Cross-sell injection (<5ms) ──────────────────────────────

    async def _inject_cross_sell(
        self,
        result: list,
        session_id: str,
        max_items: int = 2,
    ) -> list:
        """Inject complementary items when buy_now trigger is active.

        Marketplace: 3 cross-sell items at position 2 (aggressive)
        Feed:        2 cross-sell items at position 3 (natural)
        Live:        0 (disabled)

        When a user clicks 'buy now', Redis stores the purchased item's
        category + vendor. We fetch complementary items from the same
        category (different vendor to avoid monopole) and inject them
        at the optimal cross-sell slot.
        """
        if not self.redis or len(result) < 4:
            return result
        try:
            trigger_data = await self.redis.get(
                f"session:{session_id}:cross_sell_trigger"
            )
            if not trigger_data:
                return result

            import json
            trigger = json.loads(trigger_data)
            category_id = trigger.get("category_id")
            exclude_vendor = trigger.get("vendor_id", "")

            if not category_id:
                return result

            # Fetch top items in same category from Redis sorted set
            cross_sell_ids = await self.redis.zrevrange(
                f"category:{category_id}:top_items", 0, 9
            )
            if not cross_sell_ids:
                return result

            # Filter out items already in result + same vendor
            existing_ids = {r[0] for r in result}
            injected = 0
            inject_pos = 3  # After position 3 in feed

            for cs_id in cross_sell_ids:
                if isinstance(cs_id, bytes):
                    cs_id = cs_id.decode("utf-8")
                if cs_id in existing_ids:
                    continue

                # Get item metadata from Redis
                meta_raw = await self.redis.hgetall(f"item:{cs_id}:meta")
                meta = {k.decode() if isinstance(k, bytes) else k:
                        v.decode() if isinstance(v, bytes) else v
                        for k, v in meta_raw.items()} if meta_raw else {}

                # Skip same vendor (diversity)
                if meta.get("vendor_id") == exclude_vendor:
                    continue

                # Build cross-sell entry with boosted score
                cs_flags = DiversityFlags(
                    is_new_vendor=False,
                    is_regional=False,
                    is_cold_start=False,
                )
                # Score slightly below the item at insert position
                insert_score = result[min(inject_pos, len(result) - 1)][1] * 0.95

                cs_entry = (cs_id, insert_score, {
                    "p_buy_now": 0.05,
                    "p_purchase": 0.08,
                    "p_add_to_cart": 0.12,
                    "p_save": 0.06,
                    "p_share": 0.02,
                    "e_watch_time": 0.6,
                    "p_negative": 0.005,
                    "vendor_id": meta.get("vendor_id", ""),
                    "category_id": category_id,
                    "is_cross_sell": True,
                }, cs_flags)

                result.insert(inject_pos + injected, cs_entry)
                injected += 1
                if injected >= max_items:
                    break

            if injected > 0:
                logger.info(
                    "Cross-sell: injected %d items at pos %d for session %s (cat=%s)",
                    injected, inject_pos, session_id, category_id,
                )

        except Exception as e:
            logger.warning(f"Cross-sell injection failed: {e}")
        return result

    # ─── Pipeline principal ──────────────────────────────────────────────────

    async def rank(self, request: RankRequest) -> RankResponse:
        """Context-aware ranking pipeline.

        Adapts scoring, diversity, and cross-sell strategy based on context:
            feed:        engagement-first  → keep scrolling
            marketplace: conversion-first  → buy + discover (addictive)
            live:        urgency-first     → buy NOW
        """
        t_start = time.perf_counter()
        ctx = request.context if request.context in CONTEXT_WEIGHTS else DEFAULT_CONTEXT

        # Step 1 — Retrieve (si pas de candidats fournis)
        if request.candidates:
            candidates = request.candidates
        else:
            # Marketplace retrieves more candidates for better diversity
            retrieval_k = 3000 if ctx == "marketplace" else 2000
            candidates = await self._retrieve_candidates(
                request.user_id,
                request.session_vector,
                k=retrieval_k,
            )

        # Step 2 — DeepFM pre-rank → keep more candidates for marketplace
        prerank_k = 500 if ctx == "marketplace" else 400
        if len(candidates) > prerank_k:
            candidates = await self._prerank_deepfm(candidates, request.user_id, top_k=prerank_k)

        # Step 3 — MTL/PLE scoring
        mtl_scores_map = await self._score_mtl(
            candidates,
            request.user_id,
            request.session_actions,
            request.intent_level,
        )

        # Calcul du score CONTEXT-AWARE + tri
        scored = []
        for item_id in candidates:
            meta = mtl_scores_map.get(item_id, {})
            score = self._compute_commerce_score(meta, context=ctx)
            scored.append((item_id, score, meta))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Step 4 — Context-aware diversity
        # Marketplace: 3× limit for aggressive re-ranking (more variety = more addiction)
        diversity_pool = request.limit * 3 if ctx == "marketplace" else request.limit * 2
        diverse = self._apply_diversity(scored, diversity_pool, context=ctx)

        # Step 5 — Cross-sell (marketplace: 3 slots, feed: 2, live: 0)
        cross_sell_slots = CONTEXT_DIVERSITY.get(ctx, {}).get("cross_sell_slots", 2)
        if ctx != "live" and cross_sell_slots > 0:
            diverse = await self._inject_cross_sell(diverse, "", max_items=cross_sell_slots)

        # Build response
        ranked = []
        for item_id, score, meta, flags in diverse[: request.limit]:
            m = MTLScores(
                p_buy_now=meta.get("p_buy_now", 0.0),
                p_purchase=meta.get("p_purchase", 0.0),
                p_add_to_cart=meta.get("p_add_to_cart", 0.0),
                p_save=meta.get("p_save", 0.0),
                p_share=meta.get("p_share", 0.0),
                e_watch_time=meta.get("e_watch_time", 0.0),
                p_negative=meta.get("p_negative", 0.0),
            )
            ranked.append(RankedCandidate(
                item_id=item_id,
                score=score,
                pool_level=self._get_pool_level(meta),
                mtl_scores=m,
                diversity_flags=flags,
            ))

        pipeline_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "RankingPipeline [%s] completed in %.1fms for user=%s (%d items)",
            ctx, pipeline_ms, request.user_id, len(ranked),
        )

        return RankResponse(candidates=ranked, pipeline_ms=pipeline_ms)
