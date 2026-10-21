"""
src/wtc/evaluation.py
---------------------
Reusable evaluation module shared across all models (TF-IDF, BiGRU, DistilRoBERTa).
Strictly model-agnostic. Operates purely on probability matrices [N, 6] and true labels [N, 6].
Contract: reads probabilities, true labels, and metadata. Computes:
- Val threshold tuning (grid search + 200 bootstrap resamples, median threshold)
- Test metrics (tuned thresholds, 0.5 baseline, hierarchical severe_toxic rule)
- 1000 bootstrap resamples on test for 95% CIs on headline metrics
- Calibration metrics (Brier score, 15-bin ECE)
- Export of metrics.json and CSV tables
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    hamming_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import LABELS


# ------------------------------------------------------------------ #
#  Calibration: ECE (Expected Calibration Error)                     #
# ------------------------------------------------------------------ #

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error (ECE) with uniform bins in [0, 1]."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    ece = 0.0
    n_total = len(y_true)
    for b in range(n_bins):
        mask = bin_indices == b
        count = int(np.sum(mask))
        if count > 0:
            acc = float(np.mean(y_true[mask]))
            conf = float(np.mean(y_prob[mask]))
            ece += (count / n_total) * abs(acc - conf)
    return float(ece)


# ------------------------------------------------------------------ #
#  Threshold Optimization (VAL Only)                                  #
# ------------------------------------------------------------------ #

def _f1_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, th: float) -> float:
    pred = (y_prob >= th).astype(np.int32)
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    denom = 2 * tp + fp + fn
    return (2.0 * tp / denom) if denom > 0 else 0.0


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    grid: np.ndarray | None = None,
) -> float:
    """Find threshold in grid that maximizes F1 score."""
    if grid is None:
        grid = np.round(np.arange(0.01, 1.00, 0.01), 2)
    best_f1 = -1.0
    best_th = 0.5
    for th in grid:
        f1 = _f1_at_threshold(y_true, y_prob, float(th))
        if f1 > best_f1:
            best_f1 = f1
            best_th = float(th)
    return float(best_th)


def tune_thresholds_bootstrap(
    Y_val: np.ndarray,
    P_val: np.ndarray,
    labels: list[str] = LABELS,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Tune thresholds on VAL split:
    Per label: grid 0.01..0.99 step 0.01 maximising F1.
    Repeat on 200 bootstrap resamples of val (skip resamples without positives),
    take the MEDIAN threshold.
    """
    rng = np.random.default_rng(seed)
    n_samples = len(Y_val)
    grid = np.round(np.arange(0.01, 1.00, 0.01), 2)

    thresholds: dict[str, float] = {}
    single_run: dict[str, float] = {}
    details: dict[str, Any] = {}

    for idx, label in enumerate(labels):
        y_col = Y_val[:, idx].astype(np.int32)
        p_col = P_val[:, idx].astype(np.float64)

        single_th = find_optimal_threshold(y_col, p_col, grid)
        single_run[label] = round(single_th, 4)

        boot_ths = []
        for _ in range(n_bootstrap):
            b_idx = rng.integers(0, n_samples, size=n_samples)
            y_b = y_col[b_idx]
            if np.sum(y_b) == 0:
                continue
            p_b = p_col[b_idx]
            th_b = find_optimal_threshold(y_b, p_b, grid)
            boot_ths.append(th_b)

        med_th = float(np.median(boot_ths)) if boot_ths else single_th
        # Round to 2 decimal places to match grid resolution
        med_th = round(med_th, 2)
        thresholds[label] = med_th

        details[label] = {
            "n_resamples_evaluated": len(boot_ths),
            "median_threshold": med_th,
            "single_run_threshold": round(single_th, 2),
            "mean": round(float(np.mean(boot_ths)), 4) if boot_ths else med_th,
            "std": round(float(np.std(boot_ths)), 4) if boot_ths else 0.0,
            "q25": round(float(np.percentile(boot_ths, 25)), 4) if boot_ths else med_th,
            "q75": round(float(np.percentile(boot_ths, 75)), 4) if boot_ths else med_th,
        }

    return {
        "method": "bootstrap_median_f1",
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "grid_range": [0.01, 0.99, 0.01],
        "thresholds": thresholds,
        "val_optimal_single_run": single_run,
        "bootstrap_details": details,
    }


# ------------------------------------------------------------------ #
#  Core Metrics Calculation                                           #
# ------------------------------------------------------------------ #

def compute_label_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute complete suite of metrics for a single binary label."""
    y_true = y_true.astype(np.int32)
    y_prob = y_prob.astype(np.float64)
    y_pred = (y_prob >= threshold).astype(np.int32)

    support = int(np.sum(y_true))
    n = len(y_true)
    prevalence = float(support / n) if n > 0 else 0.0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        pr_auc = prevalence

    brier = float(brier_score_loss(y_true, y_prob))
    ece = compute_ece(y_true, y_prob, n_bins=15)

    return {
        "threshold": float(threshold),
        "support": support,
        "prevalence": prevalence,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": brier,
        "ece": ece,
    }


def compute_overall_metrics(
    Y_true: np.ndarray,
    P_prob: np.ndarray,
    thresholds: dict[str, float] | list[float] | np.ndarray,
    labels: list[str] = LABELS,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """
    Compute headline metrics (overall and per-label) given probability matrix and thresholds.
    """
    if isinstance(thresholds, dict):
        thresh_arr = np.array([thresholds[l] for l in labels], dtype=np.float64)
    else:
        thresh_arr = np.asarray(thresholds, dtype=np.float64)

    Y_pred = (P_prob >= thresh_arr).astype(np.int32)

    per_label = {}
    f1s = []
    precs = []
    recs = []
    rocs = []
    prs = []
    briers = []
    eces = []

    for idx, label in enumerate(labels):
        m = compute_label_metrics(Y_true[:, idx], P_prob[:, idx], threshold=float(thresh_arr[idx]))
        per_label[label] = m
        f1s.append(m["f1"])
        precs.append(m["precision"])
        recs.append(m["recall"])
        rocs.append(m["roc_auc"])
        prs.append(m["pr_auc"])
        briers.append(m["brier"])
        eces.append(m["ece"])

    # Headline micro-F1
    micro_f1 = float(f1_score(Y_true, Y_pred, average="micro", zero_division=0))
    hamming = float(hamming_loss(Y_true, Y_pred))
    exact_match = float(accuracy_score(Y_true, Y_pred))

    overall = {
        "macro_f1": float(np.mean(f1s)),
        "micro_f1": micro_f1,
        "macro_pr_auc": float(np.mean(prs)),
        "mean_roc_auc": float(np.mean(rocs)),
        "macro_precision": float(np.mean(precs)),
        "macro_recall": float(np.mean(recs)),
        "hamming_loss": hamming,
        "exact_match_accuracy": exact_match,
        "mean_brier": float(np.mean(briers)),
        "mean_ece": float(np.mean(eces)),
    }

    return overall, per_label


# ------------------------------------------------------------------ #
#  Bootstrap Confidence Intervals (TEST)                              #
# ------------------------------------------------------------------ #

def compute_bootstrap_cis(
    Y_true: np.ndarray,
    P_prob: np.ndarray,
    thresholds: dict[str, float] | np.ndarray,
    labels: list[str] = LABELS,
    n_bootstrap: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Run 1000 bootstrap resamples on TEST split for 95% CIs on every headline metric.
    Stores the resample seed so paired comparisons across models can use identical resamples.
    """
    rng = np.random.default_rng(seed)
    n_samples = len(Y_true)

    if isinstance(thresholds, dict):
        thresh_arr = np.array([thresholds[l] for l in labels], dtype=np.float64)
    else:
        thresh_arr = np.asarray(thresholds, dtype=np.float64)

    Y_pred = (P_prob >= thresh_arr).astype(np.int32)

    # Accumulators
    macro_f1_boots = []
    micro_f1_boots = []
    macro_pr_boots = []
    mean_roc_boots = []
    macro_prec_boots = []
    macro_rec_boots = []
    exact_boots = []
    hamming_boots = []

    label_boots = {
        label: {"f1": [], "precision": [], "recall": [], "pr_auc": [], "roc_auc": []}
        for label in labels
    }

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_samples, size=n_samples)
        y_b = Y_true[idx]
        p_b = P_prob[idx]
        pred_b = Y_pred[idx]

        # Overall
        exact_boots.append(float(accuracy_score(y_b, pred_b)))
        hamming_boots.append(float(hamming_loss(y_b, pred_b)))
        micro_f1_boots.append(float(f1_score(y_b, pred_b, average="micro", zero_division=0)))

        f1_list, prec_list, rec_list, pr_list, roc_list = [], [], [], [], []
        for l_idx, label in enumerate(labels):
            y_col = y_b[:, l_idx]
            p_col = p_b[:, l_idx]
            pred_col = pred_b[:, l_idx]

            prec = float(precision_score(y_col, pred_col, zero_division=0))
            rec = float(recall_score(y_col, pred_col, zero_division=0))
            f1 = float(f1_score(y_col, pred_col, zero_division=0))

            try:
                pr = float(average_precision_score(y_col, p_col))
            except ValueError:
                pr = float(np.mean(y_col))

            try:
                roc = float(roc_auc_score(y_col, p_col))
            except ValueError:
                roc = 0.5

            label_boots[label]["f1"].append(f1)
            label_boots[label]["precision"].append(prec)
            label_boots[label]["recall"].append(rec)
            label_boots[label]["pr_auc"].append(pr)
            label_boots[label]["roc_auc"].append(roc)

            f1_list.append(f1)
            prec_list.append(prec)
            rec_list.append(rec)
            pr_list.append(pr)
            roc_list.append(roc)

        macro_f1_boots.append(float(np.mean(f1_list)))
        macro_prec_boots.append(float(np.mean(prec_list)))
        macro_rec_boots.append(float(np.mean(rec_list)))
        macro_pr_boots.append(float(np.mean(pr_list)))
        mean_roc_boots.append(float(np.mean(roc_list)))

    lower_p = 100 * (alpha / 2.0)
    upper_p = 100 * (1.0 - alpha / 2.0)

    def ci_tuple(arr: list[float]) -> list[float]:
        return [round(float(np.percentile(arr, lower_p)), 4), round(float(np.percentile(arr, upper_p)), 4)]

    overall_cis = {
        "macro_f1_ci": ci_tuple(macro_f1_boots),
        "micro_f1_ci": ci_tuple(micro_f1_boots),
        "macro_pr_auc_ci": ci_tuple(macro_pr_boots),
        "mean_roc_auc_ci": ci_tuple(mean_roc_boots),
        "macro_precision_ci": ci_tuple(macro_prec_boots),
        "macro_recall_ci": ci_tuple(macro_rec_boots),
        "exact_match_accuracy_ci": ci_tuple(exact_boots),
        "hamming_loss_ci": ci_tuple(hamming_boots),
    }

    per_label_cis = {}
    for label in labels:
        per_label_cis[label] = {
            "f1_ci": ci_tuple(label_boots[label]["f1"]),
            "precision_ci": ci_tuple(label_boots[label]["precision"]),
            "recall_ci": ci_tuple(label_boots[label]["recall"]),
            "pr_auc_ci": ci_tuple(label_boots[label]["pr_auc"]),
            "roc_auc_ci": ci_tuple(label_boots[label]["roc_auc"]),
        }

    return {
        "seed": seed,
        "n_bootstrap": n_bootstrap,
        "alpha": alpha,
        "overall_cis": overall_cis,
        "per_label_cis": per_label_cis,
    }


# ------------------------------------------------------------------ #
#  Complete Evaluation Runner                                         #
# ------------------------------------------------------------------ #

def run_evaluation(
    P_val: np.ndarray,
    Y_val: np.ndarray,
    P_test: np.ndarray,
    Y_test: np.ndarray,
    model_id: str,
    meta: dict[str, Any] | None = None,
    labels: list[str] = LABELS,
    n_bootstrap_ci: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Execute standard evaluation pipeline:
    1. Tune thresholds on Val (bootstrap median)
    2. Evaluate Val (reference) and Test (with tuned thresholds)
    3. Evaluate Test with fixed 0.5 thresholds
    4. Evaluate Test with hierarchical severe_toxic rule
    5. Compute 1000-bootstrap 95% CIs on Test
    6. Return fully structured metrics dict
    """
    # 1. Tune thresholds
    thresh_res = tune_thresholds_bootstrap(Y_val, P_val, labels=labels, n_bootstrap=200, seed=seed)
    tuned_thresholds = thresh_res["thresholds"]

    # 2. Val reference
    val_overall, val_per_label = compute_overall_metrics(Y_val, P_val, tuned_thresholds, labels=labels)

    # 3. Test with tuned thresholds
    test_overall, test_per_label = compute_overall_metrics(Y_test, P_test, tuned_thresholds, labels=labels)

    # 4. Fixed 0.5 thresholds on Test
    fixed_05_thresh = {l: 0.5 for l in labels}
    test_05_overall, test_05_per_label = compute_overall_metrics(Y_test, P_test, fixed_05_thresh, labels=labels)

    # 5. Hierarchical rule on Test: severe_toxic positive only if toxic positive
    # (pred_hier_severe_toxic = pred_severe_toxic & pred_toxic)
    thresh_arr = np.array([tuned_thresholds[l] for l in labels], dtype=np.float64)
    Y_test_pred_tuned = (P_test >= thresh_arr).astype(np.int32)
    toxic_idx = labels.index("toxic")
    sev_idx = labels.index("severe_toxic")

    Y_test_pred_hier = Y_test_pred_tuned.copy()
    Y_test_pred_hier[:, sev_idx] = Y_test_pred_tuned[:, sev_idx] & Y_test_pred_tuned[:, toxic_idx]

    hier_sev_f1 = float(f1_score(Y_test[:, sev_idx], Y_test_pred_hier[:, sev_idx], zero_division=0))
    hier_sev_prec = float(precision_score(Y_test[:, sev_idx], Y_test_pred_hier[:, sev_idx], zero_division=0))
    hier_sev_rec = float(recall_score(Y_test[:, sev_idx], Y_test_pred_hier[:, sev_idx], zero_division=0))
    hier_micro_f1 = float(f1_score(Y_test, Y_test_pred_hier, average="micro", zero_division=0))
    hier_exact = float(accuracy_score(Y_test, Y_test_pred_hier))

    # 6. Bootstrap CIs on Test
    ci_res = compute_bootstrap_cis(
        Y_test,
        P_test,
        tuned_thresholds,
        labels=labels,
        n_bootstrap=n_bootstrap_ci,
        seed=seed,
    )

    # Merge CIs into overall and per_label
    for k, v in ci_res["overall_cis"].items():
        test_overall[k] = v

    for label in labels:
        test_per_label[label].update(ci_res["per_label_cis"][label])
        test_per_label[label]["threshold_fixed_05"] = 0.5
        test_per_label[label]["f1_at_05"] = test_05_per_label[label]["f1"]
        test_per_label[label]["precision_at_05"] = test_05_per_label[label]["precision"]
        test_per_label[label]["recall_at_05"] = test_05_per_label[label]["recall"]

    hierarchical_summary = {
        "severe_toxic_standard_f1": test_per_label["severe_toxic"]["f1"],
        "severe_toxic_standard_precision": test_per_label["severe_toxic"]["precision"],
        "severe_toxic_standard_recall": test_per_label["severe_toxic"]["recall"],
        "severe_toxic_hierarchical_f1": hier_sev_f1,
        "severe_toxic_hierarchical_precision": hier_sev_prec,
        "severe_toxic_hierarchical_recall": hier_sev_rec,
        "hierarchical_micro_f1": hier_micro_f1,
        "hierarchical_exact_match": hier_exact,
    }

    full_metrics = {
        "model_id": model_id,
        "meta": meta or {},
        "labels": labels,
        "threshold_tuning": thresh_res,
        "test": {
            "overall": test_overall,
            "per_label": test_per_label,
            "hierarchical": hierarchical_summary,
            "overall_fixed_05": test_05_overall,
            "per_label_fixed_05": test_05_per_label,
        },
        "val_reference": {
            "overall": val_overall,
            "per_label": val_per_label,
        },
        "bootstrap_info": {
            "seed": seed,
            "n_resamples": n_bootstrap_ci,
            "ci_level": 0.95,
        },
    }

    return full_metrics


# ------------------------------------------------------------------ #
#  Export Helpers (JSON and CSV tables)                              #
# ------------------------------------------------------------------ #

def export_evaluation_tables(metrics: dict[str, Any], output_tables_dir: Path) -> dict[str, Path]:
    """Save all evaluation tables as cleanly formatted CSV files."""
    output_tables_dir.mkdir(parents=True, exist_ok=True)
    labels = metrics["labels"]
    test_per_label = metrics["test"]["per_label"]
    val_per_label = metrics["val_reference"]["per_label"]
    test_overall = metrics["test"]["overall"]

    # Table 1: Test per-label detailed metrics
    rows_test = []
    for l in labels:
        m = test_per_label[l]
        rows_test.append({
            "label": l,
            "threshold": m["threshold"],
            "support": m["support"],
            "prevalence": round(m["prevalence"], 6),
            "precision": round(m["precision"], 4),
            "precision_ci_lower": m["precision_ci"][0],
            "precision_ci_upper": m["precision_ci"][1],
            "recall": round(m["recall"], 4),
            "recall_ci_lower": m["recall_ci"][0],
            "recall_ci_upper": m["recall_ci"][1],
            "f1": round(m["f1"], 4),
            "f1_ci_lower": m["f1_ci"][0],
            "f1_ci_upper": m["f1_ci"][1],
            "pr_auc": round(m["pr_auc"], 4),
            "pr_auc_ci_lower": m["pr_auc_ci"][0],
            "pr_auc_ci_upper": m["pr_auc_ci"][1],
            "roc_auc": round(m["roc_auc"], 4),
            "roc_auc_ci_lower": m["roc_auc_ci"][0],
            "roc_auc_ci_upper": m["roc_auc_ci"][1],
            "mcc": round(m["mcc"], 4),
            "brier": round(m["brier"], 4),
            "ece": round(m["ece"], 4),
            "tp": m["tp"],
            "fp": m["fp"],
            "fn": m["fn"],
            "tn": m["tn"],
        })
    df_test = pd.DataFrame(rows_test)
    p_test = output_tables_dir / "metrics_test_per_label.csv"
    df_test.to_csv(p_test, index=False)

    # Table 2: Val per-label reference
    rows_val = []
    for l in labels:
        m = val_per_label[l]
        rows_val.append({
            "label": l,
            "threshold": m["threshold"],
            "support": m["support"],
            "prevalence": round(m["prevalence"], 6),
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1": round(m["f1"], 4),
            "pr_auc": round(m["pr_auc"], 4),
            "roc_auc": round(m["roc_auc"], 4),
            "mcc": round(m["mcc"], 4),
            "brier": round(m["brier"], 4),
            "ece": round(m["ece"], 4),
            "tp": m["tp"],
            "fp": m["fp"],
            "fn": m["fn"],
            "tn": m["tn"],
        })
    df_val = pd.DataFrame(rows_val)
    p_val = output_tables_dir / "metrics_val_per_label.csv"
    df_val.to_csv(p_val, index=False)

    # Table 3: Test overall summary
    rows_summary = [
        {"metric": "Macro F1", "value": round(test_overall["macro_f1"], 4), "ci_95": f"[{test_overall['macro_f1_ci'][0]:.4f}, {test_overall['macro_f1_ci'][1]:.4f}]"},
        {"metric": "Micro F1", "value": round(test_overall["micro_f1"], 4), "ci_95": f"[{test_overall['micro_f1_ci'][0]:.4f}, {test_overall['micro_f1_ci'][1]:.4f}]"},
        {"metric": "Macro PR-AUC", "value": round(test_overall["macro_pr_auc"], 4), "ci_95": f"[{test_overall['macro_pr_auc_ci'][0]:.4f}, {test_overall['macro_pr_auc_ci'][1]:.4f}]"},
        {"metric": "Mean ROC-AUC", "value": round(test_overall["mean_roc_auc"], 4), "ci_95": f"[{test_overall['mean_roc_auc_ci'][0]:.4f}, {test_overall['mean_roc_auc_ci'][1]:.4f}]"},
        {"metric": "Macro Precision", "value": round(test_overall["macro_precision"], 4), "ci_95": f"[{test_overall['macro_precision_ci'][0]:.4f}, {test_overall['macro_precision_ci'][1]:.4f}]"},
        {"metric": "Macro Recall", "value": round(test_overall["macro_recall"], 4), "ci_95": f"[{test_overall['macro_recall_ci'][0]:.4f}, {test_overall['macro_recall_ci'][1]:.4f}]"},
        {"metric": "Exact Match Accuracy", "value": round(test_overall["exact_match_accuracy"], 4), "ci_95": f"[{test_overall['exact_match_accuracy_ci'][0]:.4f}, {test_overall['exact_match_accuracy_ci'][1]:.4f}]"},
        {"metric": "Hamming Loss", "value": round(test_overall["hamming_loss"], 4), "ci_95": f"[{test_overall['hamming_loss_ci'][0]:.4f}, {test_overall['hamming_loss_ci'][1]:.4f}]"},
        {"metric": "Mean Brier Score", "value": round(test_overall["mean_brier"], 4), "ci_95": "N/A"},
        {"metric": "Mean ECE (15 bins)", "value": round(test_overall["mean_ece"], 4), "ci_95": "N/A"},
    ]
    df_summary = pd.DataFrame(rows_summary)
    p_summary = output_tables_dir / "metrics_test_summary.csv"
    df_summary.to_csv(p_summary, index=False)

    # Table 4: Threshold comparison (Tuned vs 0.5)
    rows_comp = []
    for l in labels:
        m = test_per_label[l]
        rows_comp.append({
            "label": l,
            "tuned_threshold": m["threshold"],
            "f1_tuned": round(m["f1"], 4),
            "precision_tuned": round(m["precision"], 4),
            "recall_tuned": round(m["recall"], 4),
            "f1_fixed_05": round(m["f1_at_05"], 4),
            "precision_fixed_05": round(m["precision_at_05"], 4),
            "recall_fixed_05": round(m["recall_at_05"], 4),
            "f1_delta_tuned_minus_05": round(m["f1"] - m["f1_at_05"], 4),
        })
    df_comp = pd.DataFrame(rows_comp)
    p_comp = output_tables_dir / "thresholds_comparison.csv"
    df_comp.to_csv(p_comp, index=False)

    # Table 5: Hierarchical rule comparison
    hier = metrics["test"]["hierarchical"]
    rows_hier = [
        {"model_id": metrics["model_id"], "label": "severe_toxic (standard)", "precision": round(hier["severe_toxic_standard_precision"], 4), "recall": round(hier["severe_toxic_standard_recall"], 4), "f1": round(hier["severe_toxic_standard_f1"], 4)},
        {"model_id": metrics["model_id"], "label": "severe_toxic (hierarchical)", "precision": round(hier["severe_toxic_hierarchical_precision"], 4), "recall": round(hier["severe_toxic_hierarchical_recall"], 4), "f1": round(hier["severe_toxic_hierarchical_f1"], 4)},
    ]
    df_hier = pd.DataFrame(rows_hier)
    p_hier = output_tables_dir / "hierarchical_comparison.csv"
    df_hier.to_csv(p_hier, index=False)

    # Table 6: Calibration per label
    rows_calib = []
    for l in labels:
        m = test_per_label[l]
        rows_calib.append({
            "label": l,
            "brier_score": round(m["brier"], 4),
            "ece_15_bins": round(m["ece"], 4),
        })
    df_calib = pd.DataFrame(rows_calib)
    p_calib = output_tables_dir / "calibration_per_label.csv"
    df_calib.to_csv(p_calib, index=False)

    return {
        "test_per_label": p_test,
        "val_per_label": p_val,
        "test_summary": p_summary,
        "thresholds_comparison": p_comp,
        "hierarchical_comparison": p_hier,
        "calibration": p_calib,
    }
