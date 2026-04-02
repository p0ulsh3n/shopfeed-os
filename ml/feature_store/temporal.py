"""
Circadian Temporal Features — Vulnerability Scoring (t.md §2)
==============================================================
Adds time-of-day awareness to the feed pipeline. Research confirms
impulse buying increases ~30% between 10PM–2AM due to reduced
prefrontal cortex activity (decision fatigue + lowered inhibition).

This module is ADDITIVE: it produces features that the existing
pipeline multiplies into the final score — it never replaces or
overrides existing scoring logic.

Features produced:
    compute_temporal_features(hour, minute, day_of_week, session_duration_s)
        → torch.Tensor[6]

    get_vulnerability_multiplier(hour, session_duration_s)
        → float (1.0 baseline, up to 1.4 at peak vulnerability)
"""

from __future__ import annotations

import math
from typing import Optional

import torch


# ── Circadian vulnerability curve ─────────────────────────────────
# Based on research: impulsivity peaks 22h–2h, dips 6h–10h.
# Shape: smooth sinusoidal envelope peaking at ~1:00 AM.

# Hour → vulnerability weight (0.0 = low impulsivity, 1.0 = peak)
_HOURLY_VULNERABILITY: list[float] = [
    # 0h   1h   2h   3h   4h   5h   6h   7h
    0.85, 0.95, 0.90, 0.75, 0.55, 0.30, 0.10, 0.05,
    # 8h   9h  10h  11h  12h  13h  14h  15h
    0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.40, 0.35,
    # 16h  17h  18h  19h  20h  21h  22h  23h
    0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85,
]


def compute_temporal_features(
    hour: int = 12,
    minute: int = 0,
    day_of_week: int = 0,
    session_duration_s: float = 0.0,
) -> torch.Tensor:
    """Compute 6-dimensional temporal feature vector.

    Dimensions:
        [0] hour_sin         — cyclical hour encoding (sin component)
        [1] hour_cos         — cyclical hour encoding (cos component)
        [2] dow_sin          — cyclical day-of-week encoding (sin)
        [3] dow_cos          — cyclical day-of-week encoding (cos)
        [4] vulnerability    — circadian vulnerability score [0, 1]
        [5] session_fatigue  — decision fatigue from session length [0, 1]

    All values are in [−1, 1] or [0, 1] — safe for neural nets.
    """
    # Cyclical encoding: hour (24h cycle)
    hour_frac = (hour + minute / 60.0) / 24.0
    hour_sin = math.sin(2 * math.pi * hour_frac)
    hour_cos = math.cos(2 * math.pi * hour_frac)

    # Cyclical encoding: day of week (7-day cycle)
    dow_frac = day_of_week / 7.0
    dow_sin = math.sin(2 * math.pi * dow_frac)
    dow_cos = math.cos(2 * math.pi * dow_frac)

    # Vulnerability: lookup + interpolation
    h = hour % 24
    h_next = (h + 1) % 24
    frac = minute / 60.0
    vulnerability = _HOURLY_VULNERABILITY[h] * (1 - frac) + _HOURLY_VULNERABILITY[h_next] * frac

    # Session fatigue: log-saturating curve (plateaus around 30 min)
    # After 30 min of browsing, decision quality drops ~25%
    session_fatigue = min(1.0, math.log1p(session_duration_s / 300.0) / math.log1p(6.0))

    return torch.tensor(
        [hour_sin, hour_cos, dow_sin, dow_cos, vulnerability, session_fatigue],
        dtype=torch.float32,
    )


def get_vulnerability_multiplier(
    hour: int = 12,
    session_duration_s: float = 0.0,
) -> float:
    """Returns a multiplier for urgency/impulse signals in the feed.

    Range: [1.0, 1.4]
        - 1.0  at low vulnerability (morning, fresh session)
        - 1.4  at peak vulnerability (1AM, long session)

    This multiplier is applied in the re-ranking stage to slightly boost
    impulse-friendly content during high-vulnerability windows. It does
    NOT replace the existing scoring — it's multiplied on top.
    """
    h = hour % 24
    vuln = _HOURLY_VULNERABILITY[h]

    # Session fatigue adds up to 0.15 extra boost
    fatigue_bonus = min(0.15, session_duration_s / 7200.0)  # caps at 2h

    # Total multiplier: 1.0 + (vulnerability * 0.3) + fatigue_bonus
    # At 1AM (vuln=0.95): 1.0 + 0.285 + fatigue → ~1.3–1.4
    # At 9AM (vuln=0.10): 1.0 + 0.03 + fatigue → ~1.03
    return 1.0 + (vuln * 0.3) + fatigue_bonus
