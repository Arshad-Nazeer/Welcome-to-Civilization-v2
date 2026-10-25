#!/usr/bin/env python
"""
backend/scripts/calibrate_policy.py
-----------------------------------
Calibrates the moderation policy remove cuts on the validation set.
For each model and critical label (threat, identity_hate, severe_toxic):
- Loads validation probabilities from predictions/<model>/probs_val.npy.
- Loads validation ground truth labels from data/processed + data/splits/val_idx.json.
- Finds lowest probability threshold where validation precision >= 0.90 (min 10 predicted positives).
- If never reached, sets remove_cut to null (meaning that label alone can never trigger Remove).
- Saves calibrated settings to backend/policy.yaml.
- NEVER uses test data for calibration.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

# Ensure repository root src is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import yaml

from wtc.config import LABELS, get_predictions_dir, get_model_dir
from wtc.data import load_split

CRITICAL_LABELS = ["threat", "identity_hate", "severe_toxic"]
MODERATE_LABELS = ["insult", "obscene"]
BASE_LABELS = ["toxic"]
MODELS = ["1_tfidf_lr", "2b_bigru_attn_v2", "3_distilroberta", "detoxify"]


def calibrate_model_cuts(
    model_id: str,
    probs_val: np.ndarray,
    Y_val: np.ndarray,
    tuned_thresholds: dict[str, float],
) -> dict[str, float | None]:
    """Find the lowest threshold where validation precision >= 0.90 with >= 10 positives."""
    cuts: dict[str, float | None] = {}
    
    for lbl in CRITICAL_LABELS:
        idx = LABELS.index(lbl)
        p_col = probs_val[:, idx]
        y_col = Y_val[:, idx]

        # Search over candidate thresholds from 0.05 to 0.99
        candidates = np.linspace(0.05, 0.99, 189)
        best_cut: float | None = None

        for t in candidates:
            preds = (p_col >= t).astype(int)
            n_pos = int(np.sum(preds))
            if n_pos < 10:
                continue
            tp = int(np.sum((preds == 1) & (y_col == 1)))
            precision = tp / n_pos if n_pos > 0 else 0.0
            if precision >= 0.90:
                best_cut = float(round(t, 4))
                break  # First match is the lowest threshold satisfying precision >= 0.90

        cuts[lbl] = best_cut
    return cuts


def get_model_thresholds(model_id: str) -> dict[str, float]:
    """Load tuned thresholds for model from models/<id>/ or predictions/<id>/."""
    model_dir = get_model_dir(model_id)
    preds_dir = get_predictions_dir(model_id)
    t_file = model_dir / "thresholds.json"
    if not t_file.exists():
        t_file = preds_dir / "thresholds.json"
    
    if t_file.exists():
        with open(t_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("thresholds", {l: 0.5 for l in LABELS})
    return {l: 0.5 for l in LABELS}


def main() -> None:
    print("=" * 70)
    print("CALIBRATING ASYMMETRIC MODERATION POLICY (VALIDATION SET ONLY)")
    print("=" * 70)

    try:
        df_val = load_split("val")
        Y_val = df_val[LABELS].values.astype(np.float32)
        print(f"Loaded validation set: {len(df_val):,} comments.")
    except Exception as e:
        print(f"Warning: Could not load validation set ({e}). Using fallback defaults.")
        Y_val = None

    policy_data = {
        "policy_version": "1.0.0",
        "last_calibrated_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Imbalance-aware multi-model moderation policy with asymmetric remove cuts and corroboration.",
        "tiers": {
            "critical": CRITICAL_LABELS,
            "moderate": MODERATE_LABELS,
            "base": BASE_LABELS,
        },
        "tier_weights": {
            "critical": 1.0,
            "moderate": 0.7,
            "base": 0.5,
        },
        "watch_zone": {
            "min_normalized": 0.30,
            "max_normalized": 0.50,
            "target_labels": CRITICAL_LABELS,
        },
        "action_rules": {
            "remove_precision_target": 0.90,
            "remove_min_positives": 10,
            "remove_corroboration_required": True,
            "hide_warning_moderate_n_threshold": 0.75,
            "hide_warning_min_fired_labels": 3,
            "flag_review_moderate_n_threshold": 0.75,
        },
        "models": {},
    }

    for model_id in MODELS:
        thresholds = get_model_thresholds(model_id)
        preds_dir = get_predictions_dir(model_id)
        val_probs_file = preds_dir / "probs_val.npy"

        if val_probs_file.exists() and Y_val is not None:
            probs_val = np.load(val_probs_file)
            remove_cuts = calibrate_model_cuts(model_id, probs_val, Y_val, thresholds)
            print(f"[{model_id}] Calibrated remove cuts (precision >= 90%, min 10 positives):")
            for lbl, cut in remove_cuts.items():
                print(f"   - {lbl:<15}: remove_cut = {cut if cut is not None else 'null (never reached 90% precision)'}")
        else:
            print(f"[{model_id}] Validation file not found. Setting default fallback remove cuts.")
            remove_cuts = {
                "threat": 0.85,
                "identity_hate": 0.85,
                "severe_toxic": 0.85,
            }

        policy_data["models"][model_id] = {
            "thresholds": thresholds,
            "remove_cuts": remove_cuts,
        }

    output_path = Path(__file__).resolve().parent.parent / "policy.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(policy_data, f, sort_keys=False, default_flow_style=False)

    print("-" * 70)
    print(f"Successfully generated policy configuration: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
