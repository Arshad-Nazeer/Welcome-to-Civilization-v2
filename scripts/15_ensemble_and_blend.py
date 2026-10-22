#!/usr/bin/env python
"""
scripts/15_ensemble_and_blend.py
---------------------------------
Evaluate Multi-Seed Runs (42, 43, 44), Mean Ensemble, and TF-IDF Blend:
1. Loads probs_val_seed{42,43,44}.npy and evaluates Val Macro PR-AUC per seed.
2. Constructs 3-seed mean ensemble (3b_distilroberta_ens).
3. If ensemble beats best single seed by >= 0.01 on val macro PR-AUC,
   marks ensemble as serving choice; otherwise marks best single seed.
4. Performs grid-search blend with Model 1 (TF-IDF) on validation set only.
   If blend beats serving choice by >= 0.005 on val macro PR-AUC, recommends blend.
5. Saves predictions and summary json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

# Ensure project root src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from wtc.config import CFG, LABELS, PROJECT_ROOT, get_predictions_dir, get_reports_dir
from wtc.data import load_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multi-seed ensemble and blend")
    parser.add_argument("--smoke", action="store_true", help="Smoke run evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    smoke = args.smoke
    model_id = "3_distilroberta"
    preds_dir = get_predictions_dir(model_id, smoke=smoke)
    reports_dir = get_reports_dir(model_id, smoke=smoke)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df_val = load_split("val")
    df_test = load_split("test")
    if smoke:
        df_val = df_val.iloc[:500].reset_index(drop=True)
        df_test = df_test.iloc[:500].reset_index(drop=True)

    Y_val = df_val[LABELS].values.astype(np.float32)
    Y_test = df_test[LABELS].values.astype(np.float32)

    from wtc.config import load_models_config
    models_cfg = load_models_config().get("3_distilroberta", {})
    seeds = models_cfg.get("final", {}).get("seeds", [42])
    val_probs_by_seed: dict[int, np.ndarray] = {}
    test_probs_by_seed: dict[int, np.ndarray] = {}

    print("=" * 75)
    print("EVALUATING DISTILROBERTA RUNS, ENSEMBLE & BLEND")
    print(f"Configured Seeds: {seeds}")
    print("=" * 75)

    # 1. Load seed predictions
    for s in seeds:
        f_val = preds_dir / f"probs_val_seed{s}.npy"
        f_test = preds_dir / f"probs_test_seed{s}.npy"
        if not f_val.exists() or not f_test.exists():
            # If s == 42, try loading default probs_val.npy
            if s == 42 and (preds_dir / "probs_val.npy").exists() and (preds_dir / "probs_test.npy").exists():
                val_probs_by_seed[42] = np.load(preds_dir / "probs_val.npy")
                test_probs_by_seed[42] = np.load(preds_dir / "probs_test.npy")
                continue
            print(f"Warning: Seed {s} prediction files not found in {preds_dir}. Skipping seed {s}.")
            continue
        val_probs_by_seed[s] = np.load(f_val)
        test_probs_by_seed[s] = np.load(f_test)

    if not val_probs_by_seed:
        f_val_main = preds_dir / "probs_val.npy"
        f_test_main = preds_dir / "probs_test.npy"
        if f_val_main.exists() and f_test_main.exists():
            val_probs_by_seed[42] = np.load(f_val_main)
            test_probs_by_seed[42] = np.load(f_test_main)

    available_seeds = sorted(val_probs_by_seed.keys())
    print(f"Loaded predictions for seeds: {available_seeds}")

    # Evaluate each seed on validation set
    seed_val_metrics: dict[int, dict[str, float]] = {}
    best_single_seed = 42
    best_single_score = -1.0

    print("\n--- Single Seed Validation Performance ---")
    for s in available_seeds:
        p_val = val_probs_by_seed[s]
        pr_aucs = [average_precision_score(Y_val[:, i], p_val[:, i]) for i in range(len(LABELS))]
        macro_pr = float(np.mean(pr_aucs))
        seed_val_metrics[s] = {
            "val_macro_pr_auc": round(macro_pr, 4),
            "per_label_pr_auc": {l: round(v, 4) for l, v in zip(LABELS, pr_aucs)},
        }
        print(f"  Seed {s}: Val Macro PR-AUC = {macro_pr:.4f}")
        if macro_pr > best_single_score:
            best_single_score = macro_pr
            best_single_seed = s

    print(f"\nBest Single Seed: Seed {best_single_seed} (Val Macro PR-AUC = {best_single_score:.4f})")

    # 2. Mean Ensemble across available seeds
    ens_preds_dir = get_predictions_dir("3b_distilroberta_ens", smoke=smoke)
    ens_preds_dir.mkdir(parents=True, exist_ok=True)

    val_ens = np.mean([val_probs_by_seed[s] for s in available_seeds], axis=0)
    test_ens = np.mean([test_probs_by_seed[s] for s in available_seeds], axis=0)

    np.save(ens_preds_dir / "probs_val.npy", val_ens.astype(np.float32))
    np.save(ens_preds_dir / "probs_test.npy", test_ens.astype(np.float32))

    ens_pr_aucs = [average_precision_score(Y_val[:, i], val_ens[:, i]) for i in range(len(LABELS))]
    ens_val_macro_pr = float(np.mean(ens_pr_aucs))
    ens_gain = ens_val_macro_pr - best_single_score

    print("\n--- 3-Seed Mean Probability Ensemble ---")
    print(f"  Ensemble Val Macro PR-AUC: {ens_val_macro_pr:.4f} (Delta vs best single seed: {ens_gain:+.4f})")

    # Serving choice determination (threshold: +0.01)
    if ens_gain >= 0.01 and len(available_seeds) > 1:
        serving_choice = "ensemble"
        serving_val_p = val_ens
        serving_test_p = test_ens
        serving_score = ens_val_macro_pr
        print(f"  -> Serving Decision: ENSEMBLE selected (gain {ens_gain:+.4f} >= 0.01 requirement).")
    else:
        serving_choice = f"seed_{best_single_seed}"
        serving_val_p = val_probs_by_seed[best_single_seed]
        serving_test_p = test_probs_by_seed[best_single_seed]
        serving_score = best_single_score
        print(f"  -> Serving Decision: SINGLE SEED {best_single_seed} selected (ensemble gain {ens_gain:+.4f} < 0.01).")

    # Copy serving choice to main predictions/3_distilroberta/
    np.save(preds_dir / "probs_val.npy", serving_val_p.astype(np.float32))
    np.save(preds_dir / "probs_test.npy", serving_test_p.astype(np.float32))

    # 3. Model 1 (TF-IDF) Blend Check
    tfidf_preds_dir = get_predictions_dir("1_tfidf_lr", smoke=smoke)
    blend_results: dict[str, Any] = {}
    recommend_blend = False
    best_blend_alpha = 1.0
    best_blend_score = serving_score

    if tfidf_preds_dir.exists() and (tfidf_preds_dir / "probs_val.npy").exists():
        tfidf_val_p = np.load(tfidf_preds_dir / "probs_val.npy")
        tfidf_test_p = np.load(tfidf_preds_dir / "probs_test.npy")

        # Grid search blend weight alpha (weight of DistilRoBERTa)
        alphas = np.linspace(0.50, 0.95, 10)
        alpha_scores = []

        for a in alphas:
            p_blend_val = a * serving_val_p + (1.0 - a) * tfidf_val_p
            score = float(np.mean([average_precision_score(Y_val[:, i], p_blend_val[:, i]) for i in range(len(LABELS))]))
            alpha_scores.append((round(a, 3), round(score, 4)))
            if score > best_blend_score:
                best_blend_score = score
                best_blend_alpha = a

        blend_gain = best_blend_score - serving_score
        recommend_blend = (blend_gain >= 0.005)

        print("\n--- Blend Check with Model 1 (TF-IDF + LR) ---")
        print(f"  Grid Search Best Weight: {best_blend_alpha:.2f} DistilRoBERTa + {1.0-best_blend_alpha:.2f} TF-IDF")
        print(f"  Blend Val Macro PR-AUC: {best_blend_score:.4f} (Delta vs serving choice: {blend_gain:+.4f})")
        print(f"  -> Blend Recommendation: {'RECOMMENDED (gain >= 0.005)' if recommend_blend else 'NOT recommended (gain < 0.005)'}")

        blend_results = {
            "tfidf_preds_dir": str(tfidf_preds_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "grid_alphas": [float(x[0]) for x in alpha_scores],
            "grid_scores": [float(x[1]) for x in alpha_scores],
            "best_alpha": round(float(best_blend_alpha), 3),
            "best_blend_val_score": round(float(best_blend_score), 4),
            "blend_gain": round(float(blend_gain), 4),
            "recommended": bool(recommend_blend),
        }

        if recommend_blend:
            # Save blend probabilities as candidate
            blend_preds_dir = get_predictions_dir("3c_distilroberta_blend", smoke=smoke)
            blend_preds_dir.mkdir(parents=True, exist_ok=True)
            p_blend_test = best_blend_alpha * serving_test_p + (1.0 - best_blend_alpha) * tfidf_test_p
            np.save(blend_preds_dir / "probs_val.npy", (best_blend_alpha * serving_val_p + (1.0 - best_blend_alpha) * tfidf_val_p).astype(np.float32))
            np.save(blend_preds_dir / "probs_test.npy", p_blend_test.astype(np.float32))

    # 4. Save summary payload
    summary = {
        "seeds": available_seeds,
        "seed_metrics": seed_val_metrics,
        "best_single_seed": best_single_seed,
        "best_single_val_macro_pr_auc": round(best_single_score, 4),
        "ensemble_val_macro_pr_auc": round(ens_val_macro_pr, 4),
        "ensemble_gain_vs_best_seed": round(ens_gain, 4),
        "serving_choice": serving_choice,
        "blend_check": blend_results,
    }

    summary_file = reports_dir / "ensemble_blend_summary.json"
    with open(summary_file, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved ensemble and blend summary to {summary_file}")


if __name__ == "__main__":
    main()
