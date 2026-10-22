#!/usr/bin/env python
"""
scripts/20_evaluate.py
----------------------
Reusable model evaluation driver for Welcome to Civilization.
Shared across all three models: TF-IDF, BiGRU, DistilRoBERTa.
Contract:
- Reads predictions/<id>/probs_val.npy, probs_test.npy
- Reads models/<id>/meta.json
- Reads split labels
Produces:
1. predictions/<id>/thresholds.json (val bootstrap median F1)
2. reports/<id>/metrics.json (schema ready for cross-model comparison)
3. reports/<id>/tables/*.csv (test & val metrics, threshold comp, calibration)
4. predictions/<id>/{val,test}_predictions.csv (id, probs, preds, hier, ground truth)
5. visualizations/<id>/*.png (200 DPI figures with fixed model color)
6. Model-specific routines dispatched via adapter_type:
   - tfidf_lr: top 20 positive n-grams per label
   - bigru_attn: training curves, attention weights, dual-device benchmark.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
import warnings

# Ensure project root src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wtc.config import (
    CFG,
    ID_COL,
    LABELS,
    PROJECT_ROOT,
    TEXT_COL,
    get_model_dir,
    get_predictions_dir,
    get_reports_dir,
    get_visualizations_dir,
)
from wtc.data import load_split
from wtc.evaluation import export_evaluation_tables, run_evaluation
from wtc.inference import Moderator
from wtc.model_specific import run_model_specific_bigru, run_model_specific_distilroberta
from wtc.plotting import generate_all_visualizations, get_model_color
from wtc.utils import get_logger, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate toxicity model")
    parser.add_argument("--model-id", type=str, default="1_tfidf_lr", help="Model ID directory (e.g. 1_tfidf_lr)")
    parser.add_argument("--smoke", action="store_true", help="Evaluate smoke run outputs")
    parser.add_argument("--compare", action="store_true", help="Compare all available models")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Number of bootstrap resamples for CIs")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation random seed")
    return parser.parse_args()


def run_model_specific_tfidf(
    model_id: str,
    smoke: bool,
    reports_dir: Path,
    viz_dir: Path,
    labels: list[str],
) -> dict[str, Any]:
    """Extract top 20 positive word and char n-grams per label for Model 1."""
    model_dir = get_model_dir(model_id, smoke=smoke)
    f_vec_word = model_dir / "vectorizer_word.joblib"
    f_vec_char = model_dir / "vectorizer_char.joblib"
    f_clfs = model_dir / "classifiers.joblib"

    if not (f_vec_word.exists() and f_vec_char.exists() and f_clfs.exists()):
        return {}

    vec_word = joblib.load(f_vec_word)
    vec_char = joblib.load(f_vec_char)
    classifiers = joblib.load(f_clfs)

    word_feature_names = vec_word.get_feature_names_out()
    char_feature_names = vec_char.get_feature_names_out()
    n_word = len(word_feature_names)

    top_ngrams_rows = []
    top_per_label: dict[str, Any] = {}

    for label in labels:
        clf = classifiers[label]
        coefs = clf.coef_[0]

        word_coefs = coefs[:n_word]
        char_coefs = coefs[n_word:]

        # Top 20 word features
        top_word_idx = np.argsort(word_coefs)[::-1][:20]
        top_words = []
        for rank, idx in enumerate(top_word_idx, 1):
            top_words.append({"rank": rank, "feature": str(word_feature_names[idx]), "coef": round(float(word_coefs[idx]), 4)})
            top_ngrams_rows.append({
                "label": label,
                "type": "word",
                "rank": rank,
                "ngram": str(word_feature_names[idx]),
                "coefficient": round(float(word_coefs[idx]), 4),
            })

        # Top 20 char features
        top_char_idx = np.argsort(char_coefs)[::-1][:20]
        top_chars = []
        for rank, idx in enumerate(top_char_idx, 1):
            top_chars.append({"rank": rank, "feature": str(char_feature_names[idx]), "coef": round(float(char_coefs[idx]), 4)})
            top_ngrams_rows.append({
                "label": label,
                "type": "char",
                "rank": rank,
                "ngram": str(char_feature_names[idx]),
                "coefficient": round(float(char_coefs[idx]), 4),
            })

        top_per_label[label] = {
            "top_words": top_words,
            "top_chars": top_chars,
        }

    # Save CSV table
    tables_dir = reports_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df_ngrams = pd.DataFrame(top_ngrams_rows)
    p_ngrams_csv = tables_dir / "top_ngrams_per_label.csv"
    df_ngrams.to_csv(p_ngrams_csv, index=False)

    # Plot top features bar chart (6 subplots, top 10 word + top 10 char)
    color = get_model_color(model_id)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=200)
    axes = axes.flatten()

    for idx, label in enumerate(labels):
        ax = axes[idx]
        w_top = top_per_label[label]["top_words"][:8]
        c_top = top_per_label[label]["top_chars"][:8]

        feats = [f"w: {item['feature']}" for item in reversed(w_top)] + [f"c: {item['feature']}" for item in reversed(c_top)]
        weights = [item["coef"] for item in reversed(w_top)] + [item["coef"] for item in reversed(c_top)]
        colors = ["#1f77b4"] * len(w_top) + ["#ff7f0e"] * len(c_top)

        y_pos = np.arange(len(feats))
        ax.barh(y_pos, weights, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feats, fontsize=8)
        ax.set_xlabel("Logistic Regression Coefficient", fontsize=9)
        ax.set_title(f"{label} Top Features", fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(f"Top Positive Predictors per Label ({model_id})\nBlue: Word n-gram | Orange: Char n-gram", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    p_ngrams_png = viz_dir / "top_positive_ngrams.png"
    fig.savefig(p_ngrams_png, dpi=200)
    plt.close(fig)

    return {
        "top_ngrams_table": str(p_ngrams_csv.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "top_ngrams_figure": str(p_ngrams_png.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "top_per_label": top_per_label,
    }


# Model-specific evaluation dispatch table
MODEL_SPECIFIC_DISPATCH: dict[str, Any] = {
    "tfidf_lr": lambda mid, sm, r_dir, v_dir, lbls, df_t, P_t, Y_t: run_model_specific_tfidf(mid, sm, r_dir, v_dir, lbls),
    "1_tfidf_lr": lambda mid, sm, r_dir, v_dir, lbls, df_t, P_t, Y_t: run_model_specific_tfidf(mid, sm, r_dir, v_dir, lbls),
    "bigru_attn": run_model_specific_bigru,
    "2_bigru_attn": run_model_specific_bigru,
    "bigru_attn_v2": run_model_specific_bigru,
    "2b_bigru_attn_v2": run_model_specific_bigru,
    "distilroberta": run_model_specific_distilroberta,
    "3_distilroberta": run_model_specific_distilroberta,
    "transformer": run_model_specific_distilroberta,
    "detoxify": lambda **kwargs: {
        "model_source": "unitaryai/detoxify",
        "checkpoint": "original",
        "backbone": "bert-base-uncased",
        "description": "External pretrained BERT-base baseline from Unitary AI",
    },
}


def measure_inference_speed(
    model_id: str,
    smoke: bool,
    test_texts: list[str],
    meta: dict[str, Any],
) -> dict[str, Any]:
    """
    Adapter-aware inference benchmark:
    - Warm-up first
    - Median of 5 repeats
    - Batch sizes 1 and 1000
    - Measured on GPU and CPU (state threads used)
    - Records same headline keys as Model 1 for batch size 1000.
    """
    sample_size = min(1000, len(test_texts))
    test_sample = test_texts[:sample_size]
    adapter_type = meta.get("adapter_type", model_id)

    details = {}
    headline_info = {}

    def _benchmark_runner(mod_obj, batch_size: int, dev_name: str) -> dict[str, Any]:
        limit = 100 if (batch_size == 1 or (dev_name == "CPU" and torch.cuda.is_available())) else len(test_sample)
        curr_sample = test_sample[:min(limit, len(test_sample))]
        curr_n = len(curr_sample)
        warmup_n = min(10, curr_n)
        n_repeats = 3 if (dev_name == "CPU" and torch.cuda.is_available()) else 5

        if batch_size == 1:
            for txt in curr_sample[:warmup_n]:
                _ = mod_obj.predict_proba(txt)
        else:
            _ = mod_obj.predict_proba(curr_sample[:warmup_n])

        repeats = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            if batch_size == 1:
                for txt in curr_sample:
                    _ = mod_obj.predict_proba(txt)
            else:
                _ = mod_obj.predict_proba(curr_sample)
            repeats.append(time.perf_counter() - t0)

        med_sec = float(np.median(repeats))
        latency_ms = (med_sec / curr_n) * 1000.0
        throughput = curr_n / med_sec if med_sec > 0 else 0.0
        time_1000 = (med_sec / curr_n) * 1000.0

        return {
            "device": dev_name,
            "batch_size": batch_size,
            "sample_size": curr_n,
            "median_seconds": round(med_sec, 4),
            "latency_ms_per_comment": round(latency_ms, 3),
            "throughput_comments_per_sec": round(throughput, 1),
            "time_per_1000_comments_sec": round(time_1000, 3),
            "repeats_seconds": [round(r, 4) for r in repeats],
        }

    if "tfidf" in adapter_type:
        mod = Moderator(model_id=model_id, smoke=smoke)
        b1000 = _benchmark_runner(mod, 1000, "CPU")
        b1 = _benchmark_runner(mod, 1, "CPU")
        details["cpu_batch_1000"] = b1000
        details["cpu_batch_1"] = b1
        headline_info = b1000
        headline_info["device"] = "CPU"
        headline_info["batch_size"] = 1000
    else:
        # Neural model: GPU (if available) AND CPU
        import torch
        cpu_threads = torch.get_num_threads()
        has_gpu = torch.cuda.is_available()

        if has_gpu:
            mod_gpu = Moderator(model_id=model_id, smoke=smoke)
            gpu_b1000 = _benchmark_runner(mod_gpu, 1000, "GPU")
            gpu_b1 = _benchmark_runner(mod_gpu, 1, "GPU")
            details["gpu_batch_1000"] = gpu_b1000
            details["gpu_batch_1"] = gpu_b1
            headline_info = gpu_b1000
            headline_info["device"] = "GPU"
            headline_info["gpu_name"] = torch.cuda.get_device_name(0)

        # CPU benchmark
        mod_cpu = Moderator(model_id=model_id, smoke=smoke)
        if hasattr(mod_cpu.adapter, "to"):
            mod_cpu.adapter.to("cpu")
        elif hasattr(mod_cpu.adapter, "device"):
            mod_cpu.adapter.device = torch.device("cpu")
            if hasattr(mod_cpu.adapter, "model") and mod_cpu.adapter.model is not None:
                if hasattr(mod_cpu.adapter.model, "to"):
                    mod_cpu.adapter.model.to("cpu")
                elif hasattr(mod_cpu.adapter.model, "model") and hasattr(mod_cpu.adapter.model.model, "to"):
                    mod_cpu.adapter.model.model.to("cpu")

        cpu_b1000 = _benchmark_runner(mod_cpu, 1000, "CPU")
        cpu_b1 = _benchmark_runner(mod_cpu, 1, "CPU")
        details["cpu_batch_1000"] = cpu_b1000
        details["cpu_batch_1"] = cpu_b1

        if not has_gpu:
            headline_info = cpu_b1000
            headline_info["device"] = "CPU"
        headline_info["cpu_threads"] = cpu_threads

    res = {
        "benchmark_sample_size": sample_size,
        "total_time_seconds": headline_info["median_seconds"],
        "latency_ms_per_comment": headline_info["latency_ms_per_comment"],
        "throughput_comments_per_sec": headline_info["throughput_comments_per_sec"],
        "time_per_1000_comments_sec": headline_info["time_per_1000_comments_sec"],
        "device": headline_info.get("device", "CPU"),
        "batch_size": headline_info.get("batch_size", 1000),
        "details": details,
    }
    if "cpu_threads" in headline_info:
        res["cpu_threads"] = headline_info["cpu_threads"]
    if "gpu_name" in headline_info:
        res["gpu_name"] = headline_info["gpu_name"]
    return res


def compare_models() -> None:
    """Compare all evaluated models saved in reports/ with generic & per-label metrics."""
    model_ids_map = CFG.get("model_ids", {})
    available_metrics = {}

    for name, mid in model_ids_map.items():
        m_file = PROJECT_ROOT / f"reports/{mid}/metrics.json"
        if m_file.exists():
            with open(m_file, "r", encoding="utf-8") as fh:
                available_metrics[mid] = json.load(fh)

    if not available_metrics:
        print("No evaluated models found for comparison.")
        return

    print(f"\nFound {len(available_metrics)} models for comparison: {list(available_metrics.keys())}")
    
    # 1. Overall Generics Metrics Table
    rows_overall = []
    rows_per_label = []

    for mid, data in available_metrics.items():
        ov = data["test"]["overall"]
        per_l = data["test"]["per_label"]
        bench = data.get("inference_benchmark", {})

        tot_tp = sum(m.get("tp", 0) for m in per_l.values())
        tot_fp = sum(m.get("fp", 0) for m in per_l.values())
        tot_fn = sum(m.get("fn", 0) for m in per_l.values())
        calc_micro_prec = (tot_tp / (tot_tp + tot_fp)) if (tot_tp + tot_fp) > 0 else 0.0
        calc_micro_rec = (tot_tp / (tot_tp + tot_fn)) if (tot_tp + tot_fn) > 0 else 0.0

        rows_overall.append({
            "model_id": mid,
            "overall_macro_f1": round(ov["macro_f1"], 4),
            "overall_macro_precision": round(ov.get("macro_precision", np.nan), 4),
            "overall_macro_recall": round(ov.get("macro_recall", np.nan), 4),
            "overall_micro_f1": round(ov["micro_f1"], 4),
            "overall_micro_precision": round(ov.get("micro_precision", calc_micro_prec), 4),
            "overall_micro_recall": round(ov.get("micro_recall", calc_micro_rec), 4),
            "overall_macro_pr_auc": round(ov["macro_pr_auc"], 4),
            "overall_mean_roc_auc": round(ov["mean_roc_auc"], 4),
            "exact_match_acc": round(ov["exact_match_accuracy"], 4),
            "hamming_loss": round(ov["hamming_loss"], 4),
            "latency_ms": bench.get("latency_ms_per_comment", np.nan),
            "throughput_cps": bench.get("throughput_comments_per_sec", np.nan),
        })

        for lbl, m in per_l.items():
            rows_per_label.append({
                "model_id": mid,
                "label": lbl,
                "support": m.get("support", 0),
                "threshold": round(m.get("threshold", 0.5), 2),
                "f1": round(m.get("f1", 0.0), 4),
                "precision": round(m.get("precision", 0.0), 4),
                "recall": round(m.get("recall", 0.0), 4),
                "pr_auc": round(m.get("pr_auc", 0.0), 4),
                "roc_auc": round(m.get("roc_auc", 0.0), 4),
                "tp": m.get("tp", 0),
                "fp": m.get("fp", 0),
                "fn": m.get("fn", 0),
                "tn": m.get("tn", 0),
            })

    # Save overall comparison CSV
    df_comp = pd.DataFrame(rows_overall)
    comp_file = PROJECT_ROOT / "reports/model_comparison.csv"
    df_comp.to_csv(comp_file, index=False)
    print(f"\n[Saved] Overall comparison table: {comp_file}")
    print(df_comp.to_string(index=False))

    # Save per-label comparison CSV
    df_per_l = pd.DataFrame(rows_per_label)
    per_l_file = PROJECT_ROOT / "reports/model_comparison_per_label.csv"
    df_per_l.to_csv(per_l_file, index=False)
    print(f"\n[Saved] Per-label comparison table: {per_l_file}")

    # Generate Markdown Summary Report: reports/model_comparison.md
    md_file = PROJECT_ROOT / "reports/model_comparison.md"
    md_lines = [
        "# Model Comparison & Generic Evaluation Metrics",
        "",
        f"**Generated UTC**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
        f"**Dataset Split**: Test Split (15,841 comments, multi-label toxicity)  ",
        "",
        "---",
        "",
        "## 1. Overall Generics Metrics Comparison",
        "",
        "| Model ID | Macro F1 | Macro Precision | Macro Recall | Micro F1 | Micro Precision | Micro Recall | Macro PR-AUC | Mean ROC-AUC | Exact Match | Throughput (cps) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows_overall:
        md_lines.append(
            f"| `{r['model_id']}` | **{r['overall_macro_f1']:.4f}** | {r['overall_macro_precision']:.4f} | {r['overall_macro_recall']:.4f} | "
            f"{r['overall_micro_f1']:.4f} | {r['overall_micro_precision']:.4f} | {r['overall_micro_recall']:.4f} | "
            f"**{r['overall_macro_pr_auc']:.4f}** | {r['overall_mean_roc_auc']:.4f} | {r['exact_match_acc']:.4f} | {r['throughput_cps']} |"
        )

    md_lines.extend([
        "",
        "## 2. Per-Label Metrics Breakdown Across Models",
        "",
        "| Label | Model ID | Tuned Thresh | F1 Score | Precision | Recall | PR-AUC | ROC-AUC | TP | FP | FN |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ])
    for r in rows_per_label:
        md_lines.append(
            f"| `{r['label']}` | `{r['model_id']}` | {r['threshold']:.2f} | **{r['f1']:.4f}** | {r['precision']:.4f} | {r['recall']:.4f} | "
            f"**{r['pr_auc']:.4f}** | {r['roc_auc']:.4f} | {r['tp']} | {r['fp']} | {r['fn']} |"
        )

    with open(md_file, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines) + "\n")
    print(f"\n[Saved] Comparison Markdown report: {md_file}")


def main() -> None:
    args = parse_args()
    if args.compare:
        compare_models()
        return

    model_id = args.model_id
    smoke = args.smoke
    seed = args.seed
    n_bootstrap = args.n_bootstrap
    set_seed(seed)

    model_dir = get_model_dir(model_id, smoke=smoke)
    preds_dir = get_predictions_dir(model_id, smoke=smoke)
    reports_dir = get_reports_dir(model_id, smoke=smoke)
    viz_dir = get_visualizations_dir(model_id, smoke=smoke)

    logger = get_logger(f"evaluate_{model_id}{'_smoke' if smoke else ''}")
    logger.info(f"Running evaluation for model: {model_id} (smoke={smoke})")

    # 1. Load predictions and metadata
    f_probs_val = preds_dir / "probs_val.npy"
    f_probs_test = preds_dir / "probs_test.npy"
    f_meta = model_dir / "meta.json"

    if not f_probs_val.exists() or not f_probs_test.exists():
        raise FileNotFoundError(f"Missing probability predictions in {preds_dir}. Run training first.")

    P_val = np.load(f_probs_val)
    P_test = np.load(f_probs_test)

    meta = {}
    if f_meta.exists():
        with open(f_meta, "r", encoding="utf-8") as fh:
            meta = json.load(fh)

    labels = meta.get("labels", LABELS)
    adapter_type = meta.get("adapter_type", model_id)

    # 2. Load split ground truth
    df_val = load_split("val")
    df_test = load_split("test")

    if smoke:
        df_val = df_val.iloc[:500].reset_index(drop=True)
        df_test = df_test.iloc[:500].reset_index(drop=True)

    Y_val = df_val[labels].values.astype(np.int32)
    Y_test = df_test[labels].values.astype(np.int32)

    logger.info(f"Loaded ground truth: val shape {Y_val.shape}, test shape {Y_test.shape}")
    logger.info(f"Loaded probabilities: val shape {P_val.shape}, test shape {P_test.shape}")

    # 3. Execute evaluation pipeline
    print("\n[Evaluation] Tuning thresholds on Val & evaluating Test metrics...")
    metrics = run_evaluation(
        P_val=P_val,
        Y_val=Y_val,
        P_test=P_test,
        Y_test=Y_test,
        model_id=model_id,
        meta=meta,
        labels=labels,
        n_bootstrap_ci=n_bootstrap,
        seed=seed,
    )

    # Save thresholds.json to predictions/ and copy to models/
    f_thresh = preds_dir / "thresholds.json"
    with open(f_thresh, "w", encoding="utf-8") as fh:
        json.dump(metrics["threshold_tuning"], fh, indent=2)
    shutil.copy2(f_thresh, model_dir / "thresholds.json")
    print(f"  Saved thresholds to {f_thresh.relative_to(PROJECT_ROOT)} and {model_dir.relative_to(PROJECT_ROOT)}")

    # Add bootstrap info enhancements (reproducibility)
    metrics["bootstrap_info"]["n_test"] = len(Y_test)
    metrics["bootstrap_info"]["n_val"] = len(Y_val)
    metrics["bootstrap_info"]["rng"] = "np.random.default_rng(seed)"

    # 4. Generate prediction CSVs
    print("\n[Predictions] Generating val and test prediction CSV files...")
    tuned_thresholds = metrics["threshold_tuning"]["thresholds"]
    thresh_arr = np.array([tuned_thresholds[l] for l in labels], dtype=np.float64)

    tox_idx = labels.index("toxic")
    sev_idx = labels.index("severe_toxic")

    for split_name, df_split, P_split, Y_split in [
        ("val", df_val, P_val, Y_val),
        ("test", df_test, P_test, Y_test),
    ]:
        pred_bin = (P_split >= thresh_arr).astype(np.int32)
        hier_sev = pred_bin[:, sev_idx] & pred_bin[:, tox_idx]

        out_df = pd.DataFrame({ID_COL: df_split[ID_COL].values})
        # Probabilities
        for idx, l in enumerate(labels):
            out_df[f"p_{l}"] = np.round(P_split[:, idx], 6)
        # Predictions (tuned)
        for idx, l in enumerate(labels):
            out_df[f"pred_{l}"] = pred_bin[:, idx]
        out_df["pred_hier_severe_toxic"] = hier_sev
        # True labels
        for idx, l in enumerate(labels):
            out_df[f"y_{l}"] = Y_split[:, idx]

        p_csv = preds_dir / f"{split_name}_predictions.csv"
        out_df.to_csv(p_csv, index=False)
        print(f"  Saved {len(out_df):,} rows to {p_csv.relative_to(PROJECT_ROOT)}")

    # 5. Export tables
    print("\n[Tables] Exporting CSV tables...")
    table_paths = export_evaluation_tables(metrics, reports_dir / "tables")
    for t_name, t_p in table_paths.items():
        print(f"  Saved {t_name} table: {t_p.relative_to(PROJECT_ROOT)}")

    # 6. Generate visualizations
    print("\n[Visualizations] Generating 200 DPI evaluation figures...")
    viz_paths = generate_all_visualizations(
        metrics=metrics,
        Y_val=Y_val,
        P_val=P_val,
        Y_test=Y_test,
        P_test=P_test,
        output_viz_dir=viz_dir,
        model_id=model_id,
    )
    for v_name, v_p in viz_paths.items():
        print(f"  Saved figure: {v_p.relative_to(PROJECT_ROOT)}")

    # 7. Model-specific evaluation (via dispatch)
    print("\n[Model-Specific] Running adapter-specific evaluation routines...")
    handler = None
    for k in (adapter_type, model_id):
        if k in MODEL_SPECIFIC_DISPATCH:
            handler = MODEL_SPECIFIC_DISPATCH[k]
            break

    if handler is not None:
        try:
            model_spec_info = handler(
                model_id=model_id,
                smoke=smoke,
                reports_dir=reports_dir,
                viz_dir=viz_dir,
                labels=labels,
                df_test=df_test,
                P_test=P_test,
                Y_test=Y_test,
            )
            metrics["model_specific"] = model_spec_info
        except Exception as e:
            warnings.warn(f"Model-specific evaluation failed: {e}", UserWarning)
            metrics["model_specific"] = {}
    else:
        warnings.warn(f"No model-specific routine registered for adapter '{adapter_type}'. Skipping.", UserWarning)
        metrics["model_specific"] = {}

    # Measure inference benchmark (adapter-aware)
    print("\n[Benchmark] Measuring inference latency and throughput...")
    bench_info = measure_inference_speed(
        model_id=model_id,
        smoke=smoke,
        test_texts=df_test[CFG.get("text_column", "comment_text")].fillna("").tolist(),
        meta=meta,
    )
    metrics["inference_benchmark"] = bench_info
    print(f"  Inference Benchmark: {bench_info['throughput_comments_per_sec']} comments/sec ({bench_info['latency_ms_per_comment']} ms/comment) on {bench_info.get('device', 'CPU')}")

    # Model disk size
    if "model_size_mb" in meta:
        metrics["model_size_mb"] = meta["model_size_mb"]

    # 8. Save final metrics.json
    metrics_file = reports_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\n[Metrics] Saved complete metrics to {metrics_file.relative_to(PROJECT_ROOT)}")

    # Print summary table
    ov = metrics["test"]["overall"]
    print("\n" + "=" * 70)
    print(f"EVALUATION COMPLETE: {model_id} (smoke={smoke})")
    print("=" * 70)
    print(f"  Macro F1:     {ov['macro_f1']:.4f}  95% CI: [{ov['macro_f1_ci'][0]:.4f}, {ov['macro_f1_ci'][1]:.4f}]")
    print(f"  Micro F1:     {ov['micro_f1']:.4f}  95% CI: [{ov['micro_f1_ci'][0]:.4f}, {ov['micro_f1_ci'][1]:.4f}]")
    print(f"  Macro PR-AUC: {ov['macro_pr_auc']:.4f}  95% CI: [{ov['macro_pr_auc_ci'][0]:.4f}, {ov['macro_pr_auc_ci'][1]:.4f}]")
    print(f"  Mean ROC-AUC: {ov['mean_roc_auc']:.4f}  95% CI: [{ov['mean_roc_auc_ci'][0]:.4f}, {ov['mean_roc_auc_ci'][1]:.4f}]")
    print("\n  Per-Label Performance (Tuned Thresholds):")
    for l in labels:
        lm = metrics["test"]["per_label"][l]
        print(f"    - {l:14s}: Thresh={lm['threshold']:.2f} | F1={lm['f1']:.4f} | PR-AUC={lm['pr_auc']:.4f} | ROC-AUC={lm['roc_auc']:.4f} | Prec={lm['precision']:.4f} | Rec={lm['recall']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
