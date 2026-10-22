#!/usr/bin/env python
"""
scripts/22_paired_bootstrap.py
--------------------------------
Paired Bootstrap Hypothesis Testing:
Evaluates whether Model 3 (DistilRoBERTa-base) achieves statistically
significant improvements over the best earlier baseline (Model 2b / Model 2).
Uses 1,000 paired bootstrap resamples on the test split (seed 42).
Outputs:
  - reports/3_distilroberta/tables/paired_bootstrap_vs_model2b.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Ensure project root src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from wtc.config import CFG, LABELS, PROJECT_ROOT, get_predictions_dir, get_reports_dir
from wtc.data import load_split
from wtc.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired bootstrap hypothesis testing")
    parser.add_argument("--model-3", type=str, default="3_distilroberta", help="Model 3 ID")
    parser.add_argument("--baseline", type=str, default="2b_bigru_attn_v2", help="Baseline model ID")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Bootstrap resamples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--smoke", action="store_true", help="Smoke run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed = args.seed
    n_bootstrap = args.n_bootstrap
    smoke = args.smoke
    set_seed(seed)

    m3_id = args.model_3
    base_id = args.baseline

    p_m3 = get_predictions_dir(m3_id, smoke=smoke)
    p_base = get_predictions_dir(base_id, smoke=smoke)
    if not p_base.exists() or not (p_base / "probs_test.npy").exists():
        p_base = get_predictions_dir("2_bigru_attn", smoke=smoke)
        base_id = "2_bigru_attn"

    print("=" * 75)
    print(f"PAIRED BOOTSTRAP SIGNIFICANCE TESTING: {m3_id} vs {base_id}")
    print(f"Resamples: {n_bootstrap:,} | Seed: {seed} | Smoke: {smoke}")
    print("=" * 75)

    df_test = load_split("test")
    if smoke:
        df_test = df_test.iloc[:500].reset_index(drop=True)

    Y_test = df_test[LABELS].values.astype(np.int32)
    P_m3 = np.load(p_m3 / "probs_test.npy")
    P_base = np.load(p_base / "probs_test.npy")

    # Load tuned thresholds
    def load_th(p_dir: Path) -> np.ndarray:
        th_f = p_dir / "thresholds.json"
        if th_f.exists():
            with open(th_f, "r", encoding="utf-8") as fh:
                d = json.load(fh)
                th_dict = d.get("thresholds", {l: 0.5 for l in LABELS})
                return np.array([th_dict[l] for l in LABELS], dtype=np.float32)
        return np.full(len(LABELS), 0.5, dtype=np.float32)

    th_m3 = load_th(p_m3)
    th_base = load_th(p_base)

    Pred_m3 = (P_m3 >= th_m3).astype(np.int32)
    Pred_base = (P_base >= th_base).astype(np.int32)

    n_samples = len(Y_test)
    rng = np.random.RandomState(seed)

    # Store paired bootstrap deltas
    macro_f1_deltas = np.zeros(n_bootstrap, dtype=np.float64)
    macro_pr_deltas = np.zeros(n_bootstrap, dtype=np.float64)
    mean_roc_deltas = np.zeros(n_bootstrap, dtype=np.float64)

    per_label_f1_deltas: dict[str, np.ndarray] = {l: np.zeros(n_bootstrap, dtype=np.float64) for l in LABELS}
    per_label_pr_deltas: dict[str, np.ndarray] = {l: np.zeros(n_bootstrap, dtype=np.float64) for l in LABELS}

    print("\nRunning 1,000 paired bootstrap iterations...")
    for b in range(n_bootstrap):
        boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
        y_b = Y_test[boot_idx]
        pred_m3_b = Pred_m3[boot_idx]
        pred_base_b = Pred_base[boot_idx]
        p_m3_b = P_m3[boot_idx]
        p_base_b = P_base[boot_idx]

        f1_m3_list, f1_base_list = [], []
        pr_m3_list, pr_base_list = [], []
        roc_m3_list, roc_base_list = [], []

        for i, l in enumerate(LABELS):
            y_col = y_b[:, i]
            # F1
            f1_m = f1_score(y_col, pred_m3_b[:, i], zero_division=0)
            f1_b = f1_score(y_col, pred_base_b[:, i], zero_division=0)
            f1_m3_list.append(f1_m)
            f1_base_list.append(f1_b)
            per_label_f1_deltas[l][b] = f1_m - f1_b

            # PR-AUC
            if np.sum(y_col) > 0:
                pr_m = average_precision_score(y_col, p_m3_b[:, i])
                pr_b = average_precision_score(y_col, p_base_b[:, i])
            else:
                pr_m, pr_b = 0.0, 0.0
            pr_m3_list.append(pr_m)
            pr_base_list.append(pr_b)
            per_label_pr_deltas[l][b] = pr_m - pr_b

            # ROC-AUC
            if np.sum(y_col) > 0 and np.sum(y_col) < len(y_col):
                roc_m = roc_auc_score(y_col, p_m3_b[:, i])
                roc_b = roc_auc_score(y_col, p_base_b[:, i])
            else:
                roc_m, roc_b = 0.5, 0.5
            roc_m3_list.append(roc_m)
            roc_base_list.append(roc_b)

        macro_f1_deltas[b] = np.mean(f1_m3_list) - np.mean(f1_base_list)
        macro_pr_deltas[b] = np.mean(pr_m3_list) - np.mean(pr_base_list)
        mean_roc_deltas[b] = np.mean(roc_m3_list) - np.mean(roc_base_list)

    # Summarize results
    def summarize_delta(deltas: np.ndarray) -> tuple[float, float, float, float]:
        mean_d = float(np.mean(deltas))
        ci_low = float(np.percentile(deltas, 2.5))
        ci_high = float(np.percentile(deltas, 97.5))
        # Empirical one-sided p-value: P(delta <= 0)
        p_val = float(np.mean(deltas <= 0.0))
        return mean_d, ci_low, ci_high, p_val

    rows = []
    # Overall metrics
    for name, deltas in [
        ("Macro F1", macro_f1_deltas),
        ("Macro PR-AUC", macro_pr_deltas),
        ("Mean ROC-AUC", mean_roc_deltas),
    ]:
        m_d, low, high, p_v = summarize_delta(deltas)
        sig = "***" if p_v < 0.001 else "**" if p_v < 0.01 else "*" if p_v < 0.05 else "ns"
        rows.append({
            "category": "Overall",
            "metric": name,
            "mean_delta": round(m_d, 4),
            "ci_95_low": round(low, 4),
            "ci_95_high": round(high, 4),
            "p_value": round(p_v, 4),
            "significance": sig,
        })

    # Per-label metrics
    for l in LABELS:
        m_d, low, high, p_v = summarize_delta(per_label_f1_deltas[l])
        sig = "***" if p_v < 0.001 else "**" if p_v < 0.01 else "*" if p_v < 0.05 else "ns"
        rows.append({
            "category": f"Label: {l}",
            "metric": "F1 Score",
            "mean_delta": round(m_d, 4),
            "ci_95_low": round(low, 4),
            "ci_95_high": round(high, 4),
            "p_value": round(p_v, 4),
            "significance": sig,
        })
        m_d_pr, low_pr, high_pr, p_v_pr = summarize_delta(per_label_pr_deltas[l])
        sig_pr = "***" if p_v_pr < 0.001 else "**" if p_v_pr < 0.01 else "*" if p_v_pr < 0.05 else "ns"
        rows.append({
            "category": f"Label: {l}",
            "metric": "PR-AUC",
            "mean_delta": round(m_d_pr, 4),
            "ci_95_low": round(low_pr, 4),
            "ci_95_high": round(high_pr, 4),
            "p_value": round(p_v_pr, 4),
            "significance": sig_pr,
        })

    df_res = pd.DataFrame(rows)
    tables_dir = get_reports_dir(m3_id, smoke=smoke) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_csv = tables_dir / f"paired_bootstrap_vs_{base_id}.csv"
    df_res.to_csv(out_csv, index=False)

    print("\n" + "=" * 75)
    print(f"PAIRED BOOTSTRAP RESULTS: {m3_id} vs {base_id}")
    print("=" * 75)
    print(f"{'Category':20s} | {'Metric':14s} | {'Delta':>8s} | {'95% CI':>18s} | {'p-value':>8s} | {'Sig':>4s}")
    print("-" * 75)
    for r in rows:
        ci_str = f"[{r['ci_95_low']:+.4f}, {r['ci_95_high']:+.4f}]"
        print(f"{r['category']:20s} | {r['metric']:14s} | {r['mean_delta']:+8.4f} | {ci_str:>18s} | {r['p_value']:8.4f} | {r['significance']:>4s}")
    print("=" * 75)
    print(f"Saved paired bootstrap table to: {out_csv}")


if __name__ == "__main__":
    main()
