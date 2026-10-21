"""
src/wtc/plotting.py
-------------------
Reusable plotting module for multi-label toxicity models.
Produces 200 DPI publication-quality figures with fixed per-model colors.
Visualizations:
1. per_label_metrics.png
2. headline_summary.png
3. pr_curves.png
4. roc_curves.png
5. threshold_vs_f1.png
6. confusion_matrices.png
7. calibration_reliability.png
8. probability_histograms.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve

from .config import LABELS

MODEL_COLORS: dict[str, str] = {
    "1_tfidf_lr": "#1f77b4",        # Muted blue
    "2_bigru_attn": "#2ca02c",      # Muted green
    "2b_bigru_attn_v2": "#00897b",  # Rich teal
    "3_distilroberta": "#d62728",   # Crimson red
}


def get_model_color(model_id: str) -> str:
    for k, v in MODEL_COLORS.items():
        if k in model_id:
            return v
    return "#1f77b4"


def plot_headline_summary(
    metrics: dict[str, Any],
    out_path: Path,
    model_id: str,
) -> None:
    """Plot macro/micro F1, macro PR-AUC, mean ROC-AUC with 95% CIs."""
    color = get_model_color(model_id)
    overall = metrics["test"]["overall"]

    metric_names = ["Macro F1", "Micro F1", "Macro PR-AUC", "Mean ROC-AUC"]
    vals = [
        overall["macro_f1"],
        overall["micro_f1"],
        overall["macro_pr_auc"],
        overall["mean_roc_auc"],
    ]
    cis = [
        overall["macro_f1_ci"],
        overall["micro_f1_ci"],
        overall["macro_pr_auc_ci"],
        overall["mean_roc_auc_ci"],
    ]

    err_low = [vals[i] - cis[i][0] for i in range(4)]
    err_high = [cis[i][1] - vals[i] for i in range(4)]
    yerr = np.array([err_low, err_high])

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    x = np.arange(len(metric_names))
    bars = ax.bar(x, vals, yerr=yerr, capsize=6, color=color, alpha=0.85, edgecolor="black", linewidth=1.2)

    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Model Summary: Headline Metrics on Test Split ({model_id})\nwith 95% Bootstrap CIs", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, val in zip(bars, vals):
        height = bar.get_height()
        ax.annotate(
            f"{val:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height / 2),
            ha="center", va="center",
            color="white", fontweight="bold", fontsize=11,
        )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_per_label_bars(
    metrics: dict[str, Any],
    out_path: Path,
    model_id: str,
) -> None:
    """Plot precision, recall, F1, ROC-AUC, PR-AUC per label with CI error bars."""
    color = get_model_color(model_id)
    labels = metrics["labels"]
    test_per_label = metrics["test"]["per_label"]

    metric_keys = ["precision", "recall", "f1", "pr_auc", "roc_auc"]
    metric_titles = ["Precision", "Recall", "F1 Score", "PR-AUC", "ROC-AUC"]

    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True, dpi=200)

    for idx, (m_key, m_title) in enumerate(zip(metric_keys, metric_titles)):
        ax = axes[idx]
        vals = [test_per_label[l][m_key] for l in labels]
        cis = [test_per_label[l][f"{m_key}_ci"] for l in labels]

        err_low = [vals[i] - cis[i][0] for i in range(len(labels))]
        err_high = [cis[i][1] - vals[i] for i in range(len(labels))]
        yerr = np.array([err_low, err_high])

        x = np.arange(len(labels))
        bars = ax.bar(x, vals, yerr=yerr, capsize=4, color=color, alpha=0.8, edgecolor="black", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
        ax.set_title(m_title, fontsize=12, fontweight="bold")
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        for bar, val in zip(bars, vals):
            ax.annotate(
                f"{val:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03),
                ha="center", va="bottom", fontsize=8,
            )

    axes[0].set_ylabel("Score", fontsize=12)
    fig.suptitle(f"Per-Label Performance Metrics ({model_id}) with 95% Bootstrap CIs", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curves(
    Y_test: np.ndarray,
    P_test: np.ndarray,
    thresholds: dict[str, float],
    labels: list[str],
    out_path: Path,
    model_id: str,
) -> None:
    """Plot Precision-Recall curves per label (2x3 subplots) with tuned threshold point."""
    color = get_model_color(model_id)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=200)
    axes = axes.flatten()

    for idx, label in enumerate(labels):
        ax = axes[idx]
        y_col = Y_test[:, idx]
        p_col = P_test[:, idx]
        th = thresholds[label]

        prec, rec, _ = precision_recall_curve(y_col, p_col)
        from sklearn.metrics import average_precision_score
        pr_auc = average_precision_score(y_col, p_col)
        prev = float(np.mean(y_col))

        ax.plot(rec, prec, color=color, lw=2.2, label=f"PR curve (AUC = {pr_auc:.3f})")
        ax.axhline(prev, color="gray", linestyle="--", alpha=0.7, label=f"Prevalence ({prev:.3f})")

        # Operating point for tuned threshold
        y_pred = (p_col >= th).astype(int)
        tp = np.sum((y_pred == 1) & (y_col == 1))
        fp = np.sum((y_pred == 1) & (y_col == 0))
        fn = np.sum((y_pred == 0) & (y_col == 1))
        pt_rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        pt_prec = tp / (tp + fp) if (tp + fp) > 0 else 0

        ax.plot(pt_rec, pt_prec, marker="o", markersize=8, color="#d62728", label=f"Thresh={th:.2f} (F1 pt)")

        ax.set_xlim([0.0, 1.02])
        ax.set_ylim([0.0, 1.02])
        ax.set_xlabel("Recall", fontsize=10)
        ax.set_ylabel("Precision", fontsize=10)
        ax.set_title(f"{label} (PR-AUC: {pr_auc:.4f})", fontsize=11, fontweight="bold")
        ax.legend(loc="lower left", fontsize=8.5)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Precision-Recall Curves per Label on Test Split ({model_id})", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_roc_curves(
    Y_test: np.ndarray,
    P_test: np.ndarray,
    thresholds: dict[str, float],
    labels: list[str],
    out_path: Path,
    model_id: str,
) -> None:
    """Plot ROC curves per label (2x3 subplots) with tuned threshold operating point."""
    color = get_model_color(model_id)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=200)
    axes = axes.flatten()

    for idx, label in enumerate(labels):
        ax = axes[idx]
        y_col = Y_test[:, idx]
        p_col = P_test[:, idx]
        th = thresholds[label]

        fpr, tpr, _ = roc_curve(y_col, p_col)
        from sklearn.metrics import roc_auc_score
        roc_auc = roc_auc_score(y_col, p_col)

        ax.plot(fpr, tpr, color=color, lw=2.2, label=f"ROC curve (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.7, label="Chance")

        # Operating point
        y_pred = (p_col >= th).astype(int)
        tp = np.sum((y_pred == 1) & (y_col == 1))
        fp = np.sum((y_pred == 1) & (y_col == 0))
        fn = np.sum((y_pred == 0) & (y_col == 1))
        tn = np.sum((y_pred == 0) & (y_col == 0))
        pt_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        pt_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        ax.plot(pt_fpr, pt_tpr, marker="o", markersize=8, color="#d62728", label=f"Thresh={th:.2f} (Oper pt)")

        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.set_ylabel("True Positive Rate", fontsize=10)
        ax.set_title(f"{label} (ROC-AUC: {roc_auc:.4f})", fontsize=11, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8.5)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Receiver Operating Characteristic (ROC) Curves ({model_id})", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_threshold_vs_f1(
    Y_val: np.ndarray,
    P_val: np.ndarray,
    thresholds: dict[str, float],
    labels: list[str],
    out_path: Path,
    model_id: str,
) -> None:
    """Plot F1 score vs threshold curve (2x3 subplots) with tuned threshold marked."""
    color = get_model_color(model_id)
    grid = np.round(np.arange(0.01, 1.00, 0.01), 2)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=200)
    axes = axes.flatten()

    for idx, label in enumerate(labels):
        ax = axes[idx]
        y_col = Y_val[:, idx]
        p_col = P_val[:, idx]
        th_chosen = thresholds[label]

        f1_scores = []
        for th in grid:
            pred = (p_col >= th).astype(int)
            tp = np.sum((pred == 1) & (y_col == 1))
            fp = np.sum((pred == 1) & (y_col == 0))
            fn = np.sum((pred == 0) & (y_col == 1))
            denom = 2 * tp + fp + fn
            f1_scores.append((2 * tp / denom) if denom > 0 else 0.0)

        f1_scores = np.array(f1_scores)
        best_f1_idx = np.argmax(f1_scores)

        ax.plot(grid, f1_scores, color=color, lw=2.2, label="Validation F1 vs Threshold")
        ax.axvline(th_chosen, color="#d62728", linestyle="--", lw=1.8, label=f"Tuned Threshold ({th_chosen:.2f})")
        ax.axvline(0.5, color="gray", linestyle=":", lw=1.5, label="Default 0.5")

        # Mark chosen threshold value
        chosen_idx = int(round(th_chosen * 100)) - 1
        chosen_idx = min(max(chosen_idx, 0), len(f1_scores) - 1)
        ax.plot(th_chosen, f1_scores[chosen_idx], marker="o", markersize=7, color="#d62728")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.02])
        ax.set_xlabel("Probability Threshold", fontsize=10)
        ax.set_ylabel("F1 Score", fontsize=10)
        ax.set_title(f"{label} (Opt Thresh: {th_chosen:.2f}, F1: {f1_scores[chosen_idx]:.3f})", fontsize=11, fontweight="bold")
        ax.legend(loc="lower center", fontsize=8.5)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Threshold Optimization Curves on Validation Split ({model_id})", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrices(
    metrics: dict[str, Any],
    out_path: Path,
    model_id: str,
) -> None:
    """Plot 2x3 confusion matrix heatmaps with counts and percentages."""
    labels = metrics["labels"]
    test_per_label = metrics["test"]["per_label"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9), dpi=200)
    axes = axes.flatten()

    for idx, label in enumerate(labels):
        ax = axes[idx]
        m = test_per_label[label]
        tp, fp, fn, tn = m["tp"], m["fp"], m["fn"], m["tn"]
        total = tp + fp + fn + tn

        cm = np.array([[tn, fp], [fn, tp]])
        cm_pct = cm / total * 100

        annot = np.array([
            [f"{tn:,}\n({cm_pct[0,0]:.1f}%)", f"{fp:,}\n({cm_pct[0,1]:.1f}%)"],
            [f"{fn:,}\n({cm_pct[1,0]:.1f}%)", f"{tp:,}\n({cm_pct[1,1]:.1f}%)"],
        ])

        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Neg (0)", "Pos (1)"],
            yticklabels=["Neg (0)", "Pos (1)"],
            annot_kws={"fontsize": 11, "fontweight": "bold"},
        )
        ax.set_title(f"{label} (th={m['threshold']:.2f}, F1={m['f1']:.3f})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)

    fig.suptitle(f"Test Split Confusion Matrices ({model_id})", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_calibration_curves(
    Y_test: np.ndarray,
    P_test: np.ndarray,
    labels: list[str],
    out_path: Path,
    model_id: str,
    n_bins: int = 15,
) -> None:
    """Plot reliability curves (calibration) per label (2x3 subplots) with 15 bins."""
    color = get_model_color(model_id)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=200)
    axes = axes.flatten()

    bins = np.linspace(0.0, 1.0, n_bins + 1)

    for idx, label in enumerate(labels):
        ax = axes[idx]
        y_col = Y_test[:, idx]
        p_col = P_test[:, idx]

        bin_indices = np.digitize(p_col, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        bin_confs = []
        bin_accs = []
        bin_counts = []
        ece = 0.0

        for b in range(n_bins):
            mask = bin_indices == b
            count = np.sum(mask)
            if count > 0:
                acc = np.mean(y_col[mask])
                conf = np.mean(p_col[mask])
                bin_accs.append(acc)
                bin_confs.append(conf)
                bin_counts.append(count)
                ece += (count / len(y_col)) * abs(acc - conf)

        from sklearn.metrics import brier_score_loss
        brier = brier_score_loss(y_col, p_col)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.7, label="Perfect calibration")
        ax.plot(bin_confs, bin_accs, marker="s", color=color, lw=2.0, label="Reliability curve")

        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel("Mean Predicted Probability", fontsize=10)
        ax.set_ylabel("Fraction of Positives", fontsize=10)
        ax.set_title(f"{label} (ECE: {ece:.4f}, Brier: {brier:.4f})", fontsize=11, fontweight="bold")
        ax.legend(loc="upper left", fontsize=8.5)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Reliability Diagrams (15-Bin Calibration) on Test Split ({model_id})", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_probability_histograms(
    Y_test: np.ndarray,
    P_test: np.ndarray,
    thresholds: dict[str, float],
    labels: list[str],
    out_path: Path,
    model_id: str,
) -> None:
    """Plot distribution of predicted probabilities by true class per label."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=200)
    axes = axes.flatten()

    for idx, label in enumerate(labels):
        ax = axes[idx]
        y_col = Y_test[:, idx]
        p_col = P_test[:, idx]
        th = thresholds[label]

        p_neg = p_col[y_col == 0]
        p_pos = p_col[y_col == 1]

        bins = np.linspace(0.0, 1.0, 50)
        ax.hist(p_neg, bins=bins, alpha=0.55, color="#1f77b4", label="True Negative (y=0)", density=True)
        ax.hist(p_pos, bins=bins, alpha=0.55, color="#d62728", label="True Positive (y=1)", density=True)
        ax.axvline(th, color="black", linestyle="--", lw=1.8, label=f"Tuned Thresh ({th:.2f})")

        ax.set_yscale("log")
        ax.set_xlim([-0.02, 1.02])
        ax.set_xlabel("Predicted Probability", fontsize=10)
        ax.set_ylabel("Log Density", fontsize=10)
        ax.set_title(f"{label} Prob Distribution", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8.5)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Predicted Probability Distributions by True Class ({model_id})", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def generate_all_visualizations(
    metrics: dict[str, Any],
    Y_val: np.ndarray,
    P_val: np.ndarray,
    Y_test: np.ndarray,
    P_test: np.ndarray,
    output_viz_dir: Path,
    model_id: str,
) -> dict[str, Path]:
    """Generate and save all 7 standard evaluation figures."""
    output_viz_dir.mkdir(parents=True, exist_ok=True)
    labels = metrics["labels"]
    thresholds = metrics["threshold_tuning"]["thresholds"]

    p_summary = output_viz_dir / "headline_summary.png"
    plot_headline_summary(metrics, p_summary, model_id)

    p_bars = output_viz_dir / "per_label_metrics.png"
    plot_per_label_bars(metrics, p_bars, model_id)

    p_pr = output_viz_dir / "pr_curves.png"
    plot_pr_curves(Y_test, P_test, thresholds, labels, p_pr, model_id)

    p_roc = output_viz_dir / "roc_curves.png"
    plot_roc_curves(Y_test, P_test, thresholds, labels, p_roc, model_id)

    p_thresh = output_viz_dir / "threshold_vs_f1.png"
    plot_threshold_vs_f1(Y_val, P_val, thresholds, labels, p_thresh, model_id)

    p_cm = output_viz_dir / "confusion_matrices.png"
    plot_confusion_matrices(metrics, p_cm, model_id)

    p_calib = output_viz_dir / "calibration_reliability.png"
    plot_calibration_curves(Y_test, P_test, labels, p_calib, model_id, n_bins=15)

    p_hist = output_viz_dir / "probability_histograms.png"
    plot_probability_histograms(Y_test, P_test, thresholds, labels, p_hist, model_id)

    return {
        "headline_summary": p_summary,
        "per_label_metrics": p_bars,
        "pr_curves": p_pr,
        "roc_curves": p_roc,
        "threshold_vs_f1": p_thresh,
        "confusion_matrices": p_cm,
        "calibration_reliability": p_calib,
        "probability_histograms": p_hist,
    }
