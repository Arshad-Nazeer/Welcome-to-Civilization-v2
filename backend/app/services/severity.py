"""
backend/app/services/severity.py
--------------------------------
Computes the 0-10 toxicity severity score and assigns severity bands.
Formula:
  Score = 10 * max_over_labels(tier_weight * min(normalized_score, 1.0))
Tier Weights:
  - Critical (threat, identity_hate, severe_toxic): 1.0
  - Moderate (insult, obscene): 0.7
  - Base (toxic): 0.5
Bands:
  - 0.0 - 2.9: Low
  - 3.0 - 5.9: Medium
  - 6.0 - 8.4: High
  - 8.5 - 10.0: Critical
"""

from __future__ import annotations
from typing import Any

TIER_WEIGHTS: dict[str, float] = {
    "threat": 1.0,
    "identity_hate": 1.0,
    "severe_toxic": 1.0,
    "insult": 0.7,
    "obscene": 0.7,
    "toxic": 0.5,
}


def calculate_severity(normalized_scores: dict[str, float]) -> dict[str, Any]:
    """
    Calculate severity score from normalized scores dictionary.
    Returns: {"score": float (0.0-10.0), "band": str}
    """
    if not normalized_scores:
        return {"score": 0.0, "band": "Low"}

    weighted_values: list[float] = []
    for label, n_score in normalized_scores.items():
        weight = TIER_WEIGHTS.get(label, 0.5)
        # Cap normalized score to 1.0
        n_clipped = min(max(float(n_score), 0.0), 1.0)
        weighted_values.append(weight * n_clipped)

    max_weighted = max(weighted_values) if weighted_values else 0.0
    raw_score = 10.0 * max_weighted
    score = round(min(max(raw_score, 0.0), 10.0), 1)

    # Assign band
    if score < 3.0:
        band = "Low"
    elif score < 6.0:
        band = "Medium"
    elif score < 8.5:
        band = "High"
    else:
        band = "Critical"

    return {
        "score": score,
        "band": band,
    }
