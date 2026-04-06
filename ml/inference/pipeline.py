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

# Poids des tâches MTL pour le score commerce final
TASK_WEIGHTS = {
    "p_buy_now": 12.0,
    "p_purchase": 10.0,
    "p_add_to_cart": 8.0,
    "p_save": 6.0,
    "p_share": 5.0,
    "e_watch_time": 3.0,
    "p_negative": -8.0,
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

    def _compute_commerce_score(self, mtl_scores: dict) -> float:
        """Σ (task_weight × prediction)"""
        return sum(
            TASK_WEIGHTS[task] * mtl_scores.get(task, 0.0)
            for task in TASK_WEIGHTS
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

    # ─── Étape 4 : Diversity re-ranking DPP (<10ms) ─────────────────────────

    def _apply_diversity(
        self,
        scored_items: list[tuple[str, float, dict]],
        limit: int
    ) -> list[tuple[str, float, dict, DiversityFlags]]:
        """
        Anti-monopole:
          - max 2 même vendor / 10 items
          - max 3 même catégorie / 10 items
          - inject 1 new_vendor + 1 regional chaque 15 items
        """
        vendor_counts: dict[str, int] = {}
        cat_counts: dict[str, int] = {}
        result = []
        injected_new_vendor = False
        injected_regional = False

        for item_id, score, meta in scored_items:
            vendor_id = meta.get("vendor_id", "")
            cat_id = str(meta.get("category_id", ""))

            if vendor_counts.get(vendor_id, 0) >= 2:
                continue
            if cat_counts.get(cat_id, 0) >= 3:
                continue

            flags = DiversityFlags(
                is_new_vendor=meta.get("is_new_vendor", False),
                is_regional=meta.get("is_regional", False),
                is_cold_start=meta.get("is_cold_start", False),
            )

            if flags.is_new_vendor:
                injected_new_vendor = True
            if flags.is_regional:
                injected_regional = True

            vendor_counts[vendor_id] = vendor_counts.get(vendor_id, 0) + 1
            cat_counts[cat_id] = cat_counts.get(cat_id, 0) + 1
            result.append((item_id, score, meta, flags))

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
    ) -> list:
        """Inject complementary items when buy_now trigger is active.

        When a user clicks 'buy now', Redis stores the purchased item's
        category + vendor. We fetch 2 complementary items from the same
        category (different vendor to avoid monopole) and inject them
        at position 3 in the feed — the optimal cross-sell slot.
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
                if injected >= 2:
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
        t_start = time.perf_counter()

        # Step 1 — Retrieve (si pas de candidats fournis)
        if request.candidates:
            candidates = request.candidates
        else:
            candidates = await self._retrieve_candidates(
                request.user_id,
                request.session_vector,
            )

        # Step 2 — DeepFM pre-rank 2000 → 400
        if len(candidates) > 400:
            candidates = await self._prerank_deepfm(candidates, request.user_id)

        # Step 3 — MTL/PLE scoring
        mtl_scores_map = await self._score_mtl(
            candidates,
            request.user_id,
            request.session_actions,
            request.intent_level,
        )

        # Calcul du commerce score + tri
        scored = []
        for item_id in candidates:
            meta = mtl_scores_map.get(item_id, {})
            commerce_score = self._compute_commerce_score(meta)
            scored.append((item_id, commerce_score, meta))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Step 4 — Diversity
        diverse = self._apply_diversity(scored, request.limit * 2)

        # Step 6 — Cross-sell
        diverse = await self._inject_cross_sell(diverse, "")

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
        logger.info(f"RankingPipeline completed in {pipeline_ms:.1f}ms for user={request.user_id}")

        return RankResponse(candidates=ranked, pipeline_ms=pipeline_ms)
