"""
Feature Engineering Pipeline — pont critique entre PostgreSQL et les modèles ML.

3 fonctions principales:
  product_to_features(product_row, vendor_emb_table) → torch.Tensor
  user_to_features(user_profile, interaction_history) → torch.Tensor
  session_to_features(actions) → torch.Tensor (128d intent vector input pour BST)
"""

from __future__ import annotations
import os
import math
import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ── Constantes — lues depuis configs/training.yaml ───────────────────────────
#
# Ne jamais hardcoder ces valeurs ici — elles doivent rester en accord avec
# configs/training.yaml (features.n_categories, embedding_dim, etc.).
# Si le YAML n'est pas disponible (tests unitaires, env minimal), les valeurs
# ci-dessous servent de défaut de sécurité aligné avec le YAML courant.

def _load_feature_constants() -> dict:
    """Charge les dimensions depuis configs/training.yaml."""
    try:
        from ml.config_loader import get_training_config
        return get_training_config().get("features", {})
    except Exception:
        return {}

_FEAT_CFG = _load_feature_constants()

# INCOHÉRENCE FIX: était hardcodé 200 alors que training.yaml dit n_categories: 500
N_CATEGORIES    = int(_FEAT_CFG.get("n_categories",   500))  # 500 catégories (YAML)
VENDOR_EMB_DIM  = int(_FEAT_CFG.get("vendor_emb_dim",  64))
VISUAL_EMB_DIM  = int(_FEAT_CFG.get("visual_emb_dim", 512))  # CLIP / SigLIP
TEXT_EMB_DIM    = int(_FEAT_CFG.get("text_emb_dim",   768))  # sentence-transformers
COMBINED_EMB_DIM = int(_FEAT_CFG.get("embedding_dim", 256))  # Two-Tower output
SESSION_VEC_DIM  = int(_FEAT_CFG.get("session_vec_dim", 128))  # BST output

# Fraîcheur: décroissance exponentielle τ (override via FRESHNESS_TAU_HOURS env var)
# Ex: 72h pour catalogues flash-sales, 336h (14j) pour catalogues stables
FRESHNESS_TAU_HOURS = float(os.environ.get("FRESHNESS_TAU_HOURS", "168.0"))

# Personas utilisateur
PERSONA_LIST = [
    "unknown", "active_buyer", "browser", "cold",
    "impulse_buyer", "researcher", "high_value",
]
PERSONA_IDX = {p: i for i, p in enumerate(PERSONA_LIST)}


# Action types → feature vector index
# t.md §1: Added micro_pause, scroll_slow, gaze_linger, scroll_reverse
# for subconscious desire detection (50-500ms hesitation signals)
ACTION_TYPES = [
    "view", "zoom", "save", "like", "share", "comment",
    "add_to_cart", "buy_now", "skip", "dwell", "pause",
    "watch_pct", "live_join", "live_buy",
    "micro_pause", "scroll_slow", "gaze_linger", "scroll_reverse",
]
ACTION_IDX = {a: i for i, a in enumerate(ACTION_TYPES)}

# Poids des actions (pour attention dans session_to_features)
# t.md §1: micro-pause signals have moderate positive weights
# (they indicate subconscious interest, lighter than explicit actions)
ACTION_WEIGHTS_MAP = {
    "buy_now": 12.0,
    "purchase": 10.0,
    "add_to_cart": 8.0,
    "save": 6.0,
    "share": 5.0,
    "like": 4.0,
    "comment": 3.0,
    "zoom": 2.0,
    "gaze_linger": 1.8,
    "scroll_reverse": 1.5,
    "micro_pause": 1.3,
    "scroll_slow": 1.1,
    "view": 1.0,
    "skip": -2.0,
    "scroll_past": -1.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. product_to_features
# ─────────────────────────────────────────────────────────────────────────────

def product_to_features(
    product_row: dict[str, Any],
    vendor_emb_table: Any | None = None,
    avg_category_price: float | None = None,
) -> torch.Tensor:
    """
    Transforme une row PostgreSQL products en feature vector pour ML.

    Dimension totale: VISUAL_EMB_DIM + 1 + N_CATEGORIES + 1 + 1 + VENDOR_EMB_DIM + 1
                    = 512 + 1 + 200 + 1 + 1 + 64 + 1 = 780

    1. visual_feat    [512] = clip_embedding (pré-calculé en DB, CLIP gelé)
    2. price_feat     [1]   = log(base_price / avg_category_price)
    3. category_feat  [N_cat] = one-hot(category_id)
    4. freshness_feat [1]   = exp(-age_hours / τ)
    5. cv_feat        [1]   = cv_score or 0.5
    6. vendor_feat    [64]  = vendor_emb_table[vendor_id]
    7. stock_feat     [1]   = min(stock, 100) / 100
    """
    parts = []

    # 1. Visual embedding (CLIP 512d)
    clip_emb = product_row.get("clip_embedding")
    if clip_emb is not None and len(clip_emb) == VISUAL_EMB_DIM:
        visual = torch.tensor(clip_emb, dtype=torch.float32)
    else:
        visual = torch.zeros(VISUAL_EMB_DIM)
    parts.append(visual)

    # 2. Price feature (log-normalized)
    price = float(product_row.get("base_price", 1.0))
    if avg_category_price is None or avg_category_price <= 0:
        avg_category_price = price  # pas de normalisation
    price_feat = math.log(max(price, 0.01) / max(avg_category_price, 0.01))
    parts.append(torch.tensor([price_feat], dtype=torch.float32))

    # 3. Category one-hot
    cat_id = int(product_row.get("category_id", 0))
    cat_onehot = torch.zeros(N_CATEGORIES)
    if 0 <= cat_id < N_CATEGORIES:
        cat_onehot[cat_id] = 1.0
    parts.append(cat_onehot)

    # 4. Freshness (décroissance exponentielle depuis published_at)
    published_at = product_row.get("published_at")
    if published_at:
        try:
            from datetime import datetime, timezone
            if isinstance(published_at, str):
                pub_dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            else:
                pub_dt = published_at
            now = datetime.now(timezone.utc)
            age_hours = (now - pub_dt).total_seconds() / 3600.0
            freshness = math.exp(-age_hours / FRESHNESS_TAU_HOURS)
        except Exception:
            freshness = 0.5
    else:
        freshness = 0.5
    parts.append(torch.tensor([freshness], dtype=torch.float32))

    # 5. CV Quality Score
    cv_score = float(product_row.get("cv_score") or 0.5)
    parts.append(torch.tensor([cv_score], dtype=torch.float32))

    # 6. Vendor embedding
    vendor_id = str(product_row.get("vendor_id", ""))
    if vendor_emb_table is not None and vendor_id:
        try:
            vendor_emb = vendor_emb_table.lookup(vendor_id)
            if isinstance(vendor_emb, torch.Tensor):
                vendor_feat = vendor_emb.float()
            else:
                vendor_feat = torch.tensor(vendor_emb, dtype=torch.float32)
        except Exception:
            vendor_feat = torch.zeros(VENDOR_EMB_DIM)
    else:
        vendor_feat = torch.zeros(VENDOR_EMB_DIM)
    parts.append(vendor_feat)

    # 7. Stock signal [0, 1]
    stock = float(product_row.get("base_stock", 0))
    stock_norm = min(stock, 100.0) / 100.0
    parts.append(torch.tensor([stock_norm], dtype=torch.float32))

    return torch.cat(parts, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# 2. user_to_features
# ─────────────────────────────────────────────────────────────────────────────

def user_to_features(
    user_profile: dict[str, Any],
    interaction_history: list[dict[str, Any]],
    global_avg_price: float = 50.0,
) -> torch.Tensor:
    """
    Transforme profil user + historique en feature vector.
    Utilisé comme input du user tower (Two-Tower).

    Dimension: N_CATEGORIES + 1 + 1 + 7 + 1 + N_CATEGORIES
             = 200 + 1 + 1 + 7 + 1 + 200 = 410

    1. category_prefs     [N_cat] — depuis user_profiles.category_prefs JSON
    2. price_sensitivity  [1]    — log(avg_price / global_avg)
    3. purchase_frequency [1]    — log(1 + purchases_30d)
    4. persona_onehot     [7]    — 7 personas
    5. geo_cluster        [1]    — index du cluster géo
    6. active_categories  [N_cat]— binary mask dernières 48h
    """
    parts = []

    # 1. Category preferences (score par catégorie)
    cat_prefs_raw = user_profile.get("category_prefs", {}) or {}
    cat_prefs = torch.zeros(N_CATEGORIES)
    for cat_id_str, score in cat_prefs_raw.items():
        try:
            cid = int(cat_id_str)
            if 0 <= cid < N_CATEGORIES:
                cat_prefs[cid] = float(score)
        except (ValueError, TypeError):
            pass
    # Normaliser [0, 1]
    max_val = cat_prefs.max()
    if max_val > 0:
        cat_prefs = cat_prefs / max_val
    parts.append(cat_prefs)

    # 2. Price sensitivity
    price_ranges = user_profile.get("price_ranges", {}) or {}
    avg_prices = [
        v.get("avg", 0) for v in price_ranges.values()
        if isinstance(v, dict) and v.get("avg", 0) > 0
    ]
    user_avg = sum(avg_prices) / len(avg_prices) if avg_prices else global_avg_price
    price_sensitivity = math.log(max(user_avg, 0.01) / max(global_avg_price, 0.01))
    parts.append(torch.tensor([price_sensitivity], dtype=torch.float32))

    # 3. Purchase frequency (30j)
    purchases_30d = sum(
        1 for item in interaction_history
        if item.get("action") in ("purchase", "buy_now")
    )
    parts.append(torch.tensor([math.log(1.0 + purchases_30d)], dtype=torch.float32))

    # 4. Persona one-hot
    persona = str(user_profile.get("persona", "unknown"))
    persona_vec = torch.zeros(len(PERSONA_LIST))
    idx = PERSONA_IDX.get(persona, 0)
    persona_vec[idx] = 1.0
    parts.append(persona_vec)

    # 5. Geo cluster (index basé sur le pays)
    country = str(user_profile.get("country", ""))
    # Simple mapping pays → cluster index [0, 50]
    geo_cluster = float(abs(hash(country)) % 51) / 50.0
    parts.append(torch.tensor([geo_cluster], dtype=torch.float32))

    # 6. Active categories (48h)
    active_cats_raw = user_profile.get("active_categories", {}) or {}
    active_cats = torch.zeros(N_CATEGORIES)
    for cat_id_str in active_cats_raw:
        try:
            cid = int(cat_id_str)
            if 0 <= cid < N_CATEGORIES:
                active_cats[cid] = 1.0
        except (ValueError, TypeError):
            pass
    parts.append(active_cats)

    return torch.cat(parts, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. session_to_features
# ─────────────────────────────────────────────────────────────────────────────

def session_to_features(
    actions: list[dict[str, Any]],
    max_seq_len: int = 50,
) -> torch.Tensor:
    """
    Transforme une liste d'actions session en feature tensor pour BST.
    Retourne: [max_seq_len, feature_dim] où feature_dim = action_dim + N_cats + 7

    Features par action:
      - action_type onehot    [18]  (14 original + 4 micro-pause types)
      - category onehot       [N_cat]
      - price_norm            [1]
      - dwell_norm            [1]
      - watch_pct             [1]
      - action_weight         [1]
      - scroll_velocity_norm  [1]  (t.md §1: slow scroll = high interest)
      - pause_pattern         [1]  (t.md §1: 50-500ms hesitation flag)
      - touch_pressure_norm   [1]  (t.md §1: pressure signal proxy)
    """
    action_dim = len(ACTION_TYPES)
    # t.md §1: expanded from +4 to +7 per-action features
    feature_dim = action_dim + N_CATEGORIES + 7

    # Padding à gauche si moins d'actions
    seq_tensor = torch.zeros(max_seq_len, feature_dim)

    valid_actions = actions[-max_seq_len:]  # garder les plus récentes

    for i, a in enumerate(valid_actions):
        pos = i  # remplir depuis le début (left-padded by zeros)

        # Action type one-hot
        atype = str(a.get("type", "view"))
        aidx = ACTION_IDX.get(atype, ACTION_IDX.get("view", 0))
        seq_tensor[pos, aidx] = 1.0

        # Category one-hot
        cat = int(a.get("category", 0))
        if 0 <= cat < N_CATEGORIES:
            seq_tensor[pos, action_dim + cat] = 1.0

        # Price normalized (log-scale cap à 10)
        price = float(a.get("price", 0))
        price_norm = math.log(max(price, 0.01) + 1) / 10.0
        seq_tensor[pos, action_dim + N_CATEGORIES] = min(price_norm, 1.0)

        # Dwell time normalized (max 30s = 30000ms)
        dwell = float(a.get("dwell_ms", 0))
        seq_tensor[pos, action_dim + N_CATEGORIES + 1] = min(dwell / 30000.0, 1.0)

        # Watch percentage [0,1]
        watch = float(a.get("watch_pct", 0))
        seq_tensor[pos, action_dim + N_CATEGORIES + 2] = min(max(watch, 0.0), 1.0)

        # Action weight (signal strength)
        weight = ACTION_WEIGHTS_MAP.get(atype, 1.0)
        weight_norm = (weight + 8.0) / 20.0  # normalize [-8,12] → [0,1]
        seq_tensor[pos, action_dim + N_CATEGORIES + 3] = weight_norm

        # ── t.md §1: Micro-pause behavioral signals ──────────────

        # Scroll velocity: normalized inverse speed (slow = high interest)
        # Raw value expected in px/s from client; 0 = stationary, 5000 = fast flick
        scroll_vel = float(a.get("scroll_velocity", 2500))
        scroll_vel_norm = 1.0 - min(scroll_vel / 5000.0, 1.0)  # invert: slow→high
        seq_tensor[pos, action_dim + N_CATEGORIES + 4] = scroll_vel_norm

        # Pause pattern: binary flag for 50-500ms micro-hesitation
        pause_ms = float(a.get("pause_ms", 0))
        pause_flag = 1.0 if 50.0 <= pause_ms <= 500.0 else 0.0
        seq_tensor[pos, action_dim + N_CATEGORIES + 5] = pause_flag

        # Touch pressure: normalized [0,1] from client force-touch / 3D touch
        # Defaults to 0.5 (neutral) when sensor data is unavailable
        pressure = float(a.get("touch_pressure", 0.5))
        seq_tensor[pos, action_dim + N_CATEGORIES + 6] = min(max(pressure, 0.0), 1.0)

    return seq_tensor
