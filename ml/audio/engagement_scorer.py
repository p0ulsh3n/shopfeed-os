"""
Engagement Scorer Audio — Scoring prédictif d'engagement des contenus audio.

Combine les features vocales Wav2Vec2 en un score composite [0,1]
prédisant l'engagement estimé (watch_time, completion_rate).

Règles empiriques basées sur la littérature TikTok/live commerce:
  - silence_ratio < 15% → +15 pts
  - speech_rate_wpm 100-160 → optimal engagement
  - pitch_variance > 0.3 → voix dynamique
  - energy_ratio > 0.75 → présence vocale forte
"""

from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


def predict_engagement(features: dict) -> dict:
    """
    Prédit le score d'engagement audio d'un contenu.

    Args:
        features: dict retourné par wav2vec_features.extract_speech_features()

    Returns:
        {
            score: float [0,1],
            grade: str (A/B/C/D),
            recommendations: [str],
        }
    """
    score = 0.0
    recommendations = list(features.get("recommendations", []))

    silence_ratio = features.get("silence_ratio", 0.25)
    speech_rate = features.get("speech_rate_wpm", 120)
    pitch_var = features.get("pitch_variance", 0.3)
    energy_ratio = features.get("energy_ratio", 0.75)

    # ── Silence ratio (max 25 pts) ───────────────────────────────────────────
    if silence_ratio < 0.10:
        score += 25.0
    elif silence_ratio < 0.15:
        score += 20.0
    elif silence_ratio < 0.20:
        score += 12.0
    elif silence_ratio < 0.30:
        score += 5.0
    else:
        score += 0.0

    # ── Speech rate (max 25 pts) ─────────────────────────────────────────────
    if 110 <= speech_rate <= 160:
        score += 25.0  # optimal
    elif 80 <= speech_rate < 110:
        score += 15.0  # un peu lent
    elif 160 < speech_rate <= 200:
        score += 15.0  # un peu rapide
    elif 60 <= speech_rate < 80:
        score += 5.0
    else:
        score += 0.0

    # ── Pitch variance (max 25 pts) ──────────────────────────────────────────
    if pitch_var > 0.5:
        score += 25.0  # très expressif
    elif pitch_var > 0.3:
        score += 18.0
    elif pitch_var > 0.1:
        score += 10.0
    else:
        score += 3.0   # monotone

    # ── Energy ratio (max 25 pts) ─────────────────────────────────────────────
    if energy_ratio > 0.85:
        score += 25.0
    elif energy_ratio > 0.75:
        score += 20.0
    elif energy_ratio > 0.60:
        score += 12.0
    else:
        score += 5.0

    # Normaliser en [0, 1]
    score_normalized = min(score / 100.0, 1.0)

    # Grade
    if score_normalized >= 0.80:
        grade = "A"
    elif score_normalized >= 0.65:
        grade = "B"
    elif score_normalized >= 0.45:
        grade = "C"
    else:
        grade = "D"

    # Recommandations supplémentaires si grade faible
    if grade in ("C", "D"):
        if not any("silence" in r.lower() for r in recommendations):
            recommendations.append("Active ton énergie vocale dès les premières secondes")
        recommendations.append("Écoute tes contenus les plus performants pour benchmarker ton audio")

    return {
        "score": round(score_normalized, 4),
        "grade": grade,
        "breakdown": {
            "silence_ratio_score": round(25 - (silence_ratio * 100), 1),
            "speech_rate_score": round(score / 4, 1),  # approximate
            "pitch_variance_score": round(pitch_var * 50, 1),
            "energy_ratio_score": round(energy_ratio * 25, 1),
        },
        "recommendations": recommendations,
    }
