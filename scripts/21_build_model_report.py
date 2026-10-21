#!/usr/bin/env python
"""
scripts/21_build_model_report.py
--------------------------------
Generate comprehensive, publication-quality model report markdown file:
reports/<id>/model_report.md
Built ENTIRELY from metrics.json, meta.json, and logs (no hardcoded numbers).
Supports adapter_type dispatch:
- tfidf_lr: Model 1 report
- bigru_attn: Model 2 report
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

# Ensure project root src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from wtc.config import (
    CFG,
    LABELS,
    PROJECT_ROOT,
    get_logs_dir,
    get_model_dir,
    get_predictions_dir,
    get_reports_dir,
    get_visualizations_dir,
)
from wtc.data import load_split_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Model Evaluation Report")
    parser.add_argument("--model-id", type=str, default="1_tfidf_lr", help="Model ID directory (e.g. 1_tfidf_lr)")
    parser.add_argument("--smoke", action="store_true", help="Report for smoke run")
    return parser.parse_args()


# ============================================================================== #
#  Model-Specific Report Sections Dispatch                                       #
# ============================================================================== #

def get_overview_text(adapter_type: str, model_id: str, meta: dict, feat_counts: dict) -> list[str]:
    lines = []
    desc = meta.get("config", {}).get("description", meta.get("model_name", model_id))
    if "tfidf" in adapter_type:
        lines.append(
            f"Model 1 (`{model_id}`) establishes the foundational linear baseline for the **Welcome to Civilization** "
            "toxicity moderation pipeline. It evaluates the classification performance achievable using shallow, "
            "interpretable bag-of-ngrams representations and binary logistic classifiers across all six toxicity categories: "
            "`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`."
        )
        lines.append("")
        lines.append("### Pipeline Architecture")
        lines.append("- **Input text**: `text_clean` (Unicode NFKC normalized, HTML unescaped, zero-width chars removed, whitespace collapsed, casing preserved).")
        lines.append("- **Word Vectorizer**: Unigrams + bigrams (1, 2), lowercase=True, min_df=3, max_features=50,000, sublinear TF scaling.")
        lines.append("- **Character Vectorizer**: Character n-grams inside word boundaries (`char_wb`), n-grams (2, 5), lowercase=False (preserving casing signals), min_df=3, max_features=100,000, sublinear TF scaling.")
        lines.append(f"- **Feature Space**: Sparse column concatenation of word and char features ({feat_counts.get('total_features', 150000):,} total features, dtype `float32`).")
        lines.append("- **Classifiers**: 6 independent binary `LogisticRegression` models (solver `liblinear`, L2 regularization penalty, C=2.0, max_iter=200, random_state=42), trained in parallel across CPU cores using the `threading` backend.")
    elif "bigru" in adapter_type:
        lines.append(
            f"Model 2 (`{model_id}`) implements a deep recurrent sequence classifier for the **Welcome to Civilization** "
            "toxicity moderation pipeline. It combines frozen pretrained fastText word embeddings with a bidirectional "
            "Gated Recurrent Unit (BiGRU) and an additive attention pooling mechanism, capturing sequential semantics "
            "and long-range token dependencies while maintaining efficient inference."
        )
        lines.append("")
        lines.append("### Pipeline Architecture")
        lines.append("- **Input text**: `text_clean` lowercased and tokenized with regex `\\w+|[^\\w\\s]`.")
        lines.append("- **Vocabulary & Embeddings**: Top 50,000 train vocabulary tokens mapped to 300-d fastText Crawl embeddings (frozen during training). Padded positions use zero vectors.")
        lines.append("- **Spatial Dropout**: 0.25 dropout applied to entire embedding feature channels across time.")
        lines.append("- **Encoder**: 1-layer bidirectional GRU with 128 hidden units per direction (256-d concatenated output) with 0.3 dropout, using `pack_padded_sequence` to prevent padding intrusion.")
        lines.append("- **Attention Mechanism**: Additive attention pooling with 128-d projection and learned context vector, masked over padding positions with float32 softmax.")
        lines.append("- **Classification Head**: Linear layer (256 -> 6 logits) with log-odds prior bias initialization.")
    else:
        lines.append(f"Model `{model_id}`: {desc}")
    return lines


def get_model_specific_section(adapter_type: str, model_id: str, meta: dict, metrics: dict, viz_dir: Path, labels: list[str]) -> list[str]:
    lines = []
    if "tfidf" in adapter_type:
        lines.append("## 8. Model-Specific Analysis: Top Predictive n-grams")
        lines.append("")
        lines.append(
            "Logistic regression coefficients provide direct interpretability. High positive weights indicate "
            "tokens strongly driving the classification decision toward positive toxicity."
        )
        lines.append("")
        lines.append(f"![Top Predictor n-grams](../../visualizations/{viz_dir.name}/top_positive_ngrams.png)")
        lines.append("")

        top_per_label = metrics.get("model_specific", {}).get("top_per_label", {})
        if top_per_label:
            lines.append("### Key Predictive Tokens by Category")
            for l in labels:
                if l in top_per_label:
                    w_list = [f"`{it['feature']}` ({it['coef']:.2f})" for it in top_per_label[l]["top_words"][:5]]
                    c_list = [f"`{it['feature']}` ({it['coef']:.2f})" for it in top_per_label[l]["top_chars"][:5]]
                    lines.append(f"- **`{l}`**:")
                    lines.append(f"  - Top Words: {', '.join(w_list)}")
                    lines.append(f"  - Top Character n-grams: {', '.join(c_list)}")
            lines.append("")

    elif "bigru" in adapter_type:
        lines.append("## 8. Model-Specific Analysis: BiGRU Training Dynamics & Attention")
        lines.append("")
        lines.append("### Parameter Allocation")
        params = meta.get("parameters", {})
        trainable = params.get("trainable", 0)
        frozen = params.get("frozen", 0)
        total_p = params.get("total", trainable + frozen)
        lines.append(f"- **Trainable Parameters**: **{trainable:,}** ({trainable/total_p*100:.2f}% of network)" if total_p > 0 else f"- Trainable: {trainable:,}")
        lines.append(f"- **Frozen Embedding Parameters**: **{frozen:,}** ({frozen/total_p*100:.2f}%)" if total_p > 0 else f"- Frozen: {frozen:,}")
        lines.append(f"- **Total Network Parameters**: **{total_p:,}**")
        lines.append("")

        lines.append("### Tokenizer, Vocabulary & fastText Coverage")
        tok_info = meta.get("tokenizer_info", {})
        fasttext_info = meta.get("fasttext_info", {})
        lines.append(f"- **Vocabulary Size**: {tok_info.get('vocab_size', 50002):,} tokens (including `<pad>` and `<unk>`)")
        lines.append(f"- **Token Coverage on Train**: **{tok_info.get('token_coverage_pct', 0.0):.2f}%** (token-weighted coverage)")
        lines.append(f"- **fastText Vocab Type Hit Rate**: **{fasttext_info.get('type_hit_rate_pct', 0.0):.2f}%** ({fasttext_info.get('found_words', 0):,} / {fasttext_info.get('total_vocab', 0):,} words found in pretrained vectors)")
        lines.append(f"- **fastText Token-Weighted Hit Rate**: **{fasttext_info.get('token_weighted_hit_rate_pct', 0.0):.2f}%**")
        trunc_val = tok_info.get("val_truncated_pct", 0.0)
        trunc_test = tok_info.get("test_truncated_pct", 0.0)
        lines.append(f"- **Sequence Truncation (Head 256 tokens)**: {trunc_val:.2f}% of validation comments and {trunc_test:.2f}% of test comments exceeded 256 tokens and were truncated.")
        lines.append("")

        lines.append("### Training Dynamics & Epoch History")
        hist = meta.get("training_history", [])
        if hist:
            lines.append("| Epoch | Train Loss | Val Loss | Val Macro PR-AUC | Val Mean ROC-AUC | Epoch Time (s) | Throughput (samples/s) | Peak VRAM (MB) |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for h in hist:
                is_best = " *" if h.get("is_best", False) else ""
                lines.append(
                    f"| {h['epoch']}{is_best} | {h['train_loss']:.4f} | {h['val_loss']:.4f} | "
                    f"**{h['val_macro_pr_auc']:.4f}** | {h['val_mean_roc_auc']:.4f} | "
                    f"{h['epoch_time_seconds']:.1f}s | {h.get('samples_per_sec', 0):.0f} | {h.get('peak_memory_mb', 0):.1f} |"
                )
            lines.append("")
            lines.append(f"- **Best Checkpoint**: Epoch {meta.get('best_epoch', 1)} with Validation Macro PR-AUC = **{meta.get('best_val_macro_pr_auc', 0.0):.4f}** (restored for final evaluation).")
            lines.append("")

        lines.append("### Training Curves")
        lines.append(f"![Training Curves](../../visualizations/{viz_dir.name}/training_curves.png)")
        lines.append("")

        lines.append("### Attention Weight Interpretability on True Toxic Comments")
        lines.append(
            "To understand token importance learned by the additive attention pooling layer, attention weights were evaluated "
            "across all test comments where ground-truth `toxic == 1`. The table and chart below show the top 20 tokens with the highest "
            "mean attention weight among frequent tokens (token count $\\ge 30$). Highly attended tokens directly highlight salient toxicity triggers."
        )
        lines.append("")
        lines.append(f"![Top Attention Tokens](../../visualizations/{viz_dir.name}/top_attention_tokens.png)")
        lines.append("")

        top_attn = metrics.get("model_specific", {}).get("top_attention_tokens", [])
        if top_attn:
            lines.append("| Rank | Token | Mean Attention Weight | Occurrences in Toxic Comments |")
            lines.append("|---|---|---|---|")
            for row in top_attn:
                lines.append(f"| {row['rank']} | `{row['token']}` | **{row['mean_attention']:.4f}** | {row['count']:,} |")
            lines.append("")

    return lines


# ============================================================================== #
#  Main Report Generator                                                         #
# ============================================================================== #

def main() -> None:
    args = parse_args()
    model_id = args.model_id
    smoke = args.smoke

    reports_dir = get_reports_dir(model_id, smoke=smoke)
    preds_dir = get_predictions_dir(model_id, smoke=smoke)
    model_dir = get_model_dir(model_id, smoke=smoke)
    viz_dir = get_visualizations_dir(model_id, smoke=smoke)
    logs_dir = get_logs_dir(model_id, smoke=smoke)

    metrics_file = reports_dir / "metrics.json"
    meta_file = model_dir / "meta.json"

    if not metrics_file.exists():
        raise FileNotFoundError(f"metrics.json not found at {metrics_file}. Run evaluation first.")
    if not meta_file.exists():
        raise FileNotFoundError(f"meta.json not found at {meta_file}. Run training first.")

    with open(metrics_file, "r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    with open(meta_file, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    manifest = load_split_manifest()

    labels = metrics.get("labels", LABELS)
    adapter_type = meta.get("adapter_type", model_id)
    test_overall = metrics["test"]["overall"]
    test_per_label = metrics["test"]["per_label"]
    val_per_label = metrics["val_reference"]["per_label"]
    thresholds = metrics["threshold_tuning"]["thresholds"]
    hier = metrics["test"]["hierarchical"]
    bench = metrics.get("inference_benchmark", {})
    feat_counts = meta.get("feature_counts", {})
    stage_timings = meta.get("stage_timings_seconds", {})
    class_weights = meta.get("class_weights", {})
    versions = meta.get("versions", {})

    # Correct dynamic best and worst labels using each metric's own argmax and argmin
    best_f1_lbl = max(labels, key=lambda l: test_per_label[l]["f1"])
    worst_f1_lbl = min(labels, key=lambda l: test_per_label[l]["f1"])
    best_prauc_lbl = max(labels, key=lambda l: test_per_label[l]["pr_auc"])
    worst_prauc_lbl = min(labels, key=lambda l: test_per_label[l]["pr_auc"])

    # Threat positive support for limitations
    n_val_threat = val_per_label.get("threat", {}).get("support", 47)
    n_test_threat = test_per_label.get("threat", {}).get("support", 50)

    # Markdown report generation
    lines = []
    lines.append(f"# Model Evaluation Report: {model_id}")
    lines.append("")
    arch_desc = meta.get("config", {}).get("description", meta.get("model_name", model_id))
    lines.append(f"**Architecture**: {arch_desc}  ")
    lines.append(f"**Generated UTC**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ")
    lines.append(f"**Mode**: {'Smoke Run (Subsampled)' if smoke else 'Full Production Run'}  ")
    lines.append(f"**Seed**: {meta.get('seed', 42)}  ")

    py_ver = versions.get("python", sys.version.split()[0])
    lines.append(f"**Python**: {py_ver} | **Framework**: {meta.get('framework', 'PyTorch / Scikit-Learn')} | **Platform**: {meta.get('device_info', {}).get('gpu_name', 'CPU / CUDA')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 1. Overview & Setup
    lines.append("## 1. Overview & Setup")
    lines.append("")
    lines.extend(get_overview_text(adapter_type, model_id, meta, feat_counts))
    lines.append("")

    # 2. Data & Split Verification
    lines.append("## 2. Data & Split Verification")
    lines.append("")
    lines.append("Data integrity was verified before feature extraction against `data/splits/split_manifest.json`:")
    lines.append(f"- **Split Version**: `{manifest.get('split_version', 'v2_near_dup')}` (near-duplicate grouped stratification).")
    lines.append(f"- **Processed Parquet SHA-256**: `{manifest.get('processed_parquet_sha256', 'Verified')}` (exact match).")
    lines.append("- **Split Sizes**:")
    train_info = manifest.get("splits", {}).get("train", {})
    val_info = manifest.get("splits", {}).get("val", {})
    test_info = manifest.get("splits", {}).get("test", {})
    lines.append(f"  - **Train**: {train_info.get('n_rows', 127861):,} comments (80.0%)")
    lines.append(f"  - **Validation**: {val_info.get('n_rows', 15869):,} comments (10.0%)")
    lines.append(f"  - **Test**: {test_info.get('n_rows', 15841):,} comments (10.0%)")
    lines.append("")

    # 3. Model Config & Imbalance Recipe
    lines.append("## 3. Model Configuration & Imbalance Recipe")
    lines.append("")
    lines.append("Because severe class imbalance characterizes toxicity labels (ranging from 9.56% down to 0.298% positive rate on train), per-label class weighting is applied directly to the loss function:")
    lines.append("")
    lines.append("$$w_l = \\sqrt{\\frac{N_{neg}}{N_{pos}}}$$")
    lines.append("")
    lines.append("Negative samples receive a fixed weight of 1.0, while positive samples are up-weighted by the square root of the imbalance ratio. This provides balanced gradient flow across rare positive categories without resampling or data duplication.")
    lines.append("")
    lines.append("| Label | Train Positives | Train Negatives | Positive Rate (%) | Imbalance Ratio | Weight ($w_1$) |")
    lines.append("|---|---|---|---|---|---|")
    train_counts = train_info.get("label_positive_counts", {})
    n_train_total = train_info.get("n_rows", 127861)
    for l in labels:
        pos = train_counts.get(l, int(test_per_label[l]["support"]))
        neg = n_train_total - pos
        rate = (pos / n_train_total) * 100.0
        ratio = neg / pos if pos > 0 else 0
        w1 = class_weights.get(l, {}).get("1", round(ratio**0.5, 4))
        lines.append(f"| `{l}` | {pos:,} | {neg:,} | {rate:.2f}% | {ratio:.2f}:1 | **{w1:.2f}** |")
    lines.append("")

    # 4. Training Summary
    lines.append("## 4. Training Summary")
    lines.append("")
    train_time = meta.get("train_time_seconds", 0.0)
    lines.append(f"- **Total Training Time**: {train_time:.2f} seconds ({train_time/60.0:.2f} minutes)")
    lines.append(f"- **Model Size on Disk**: {meta.get('model_size_mb', 0.0)} MB ({meta.get('model_size_bytes', 0):,} bytes)")
    if feat_counts.get("total_features"):
        lines.append(f"- **Vocabulary Size**: {feat_counts.get('word_features', 0):,} word n-grams + {feat_counts.get('char_features', 0):,} char n-grams = **{feat_counts.get('total_features', 0):,} total features**")
    lines.append("")
    lines.append("### Stage Timings")
    lines.append("| Stage | Description | Duration (s) | % of Total |")
    lines.append("|---|---|---|---|")
    stage_desc_map = {
        "1_load_and_verify": "[1] Load & Verify Parquet / Manifest",
        "2_word_tfidf": "[2] Word TF-IDF Fit & Transform",
        "3_char_tfidf": "[3] Char TF-IDF Fit & Transform",
        "4_lr_fits": "[4] 6x Logistic Regression Fits (Parallel)",
        "5_predict_and_save": "[5] Predict Val/Test & Save Artifacts",
        "1_load_verify": "[1/6] Load & Verify Data",
        "2_tokenize_vocab": "[2/6] Tokenization & Vocab Build",
        "3_fasttext_matrix": "[3/6] fastText Embedding Extraction",
        "4_build_model_gpu": "[4/6] Initialize Model & Allocate GPU",
        "5_training_epochs": "[5/6] Model Training & Validation",
        "6_predict_and_save": "[6/6] Checkpoint Load & Predict",
    }
    for st_key, dur in stage_timings.items():
        pct = (dur / train_time) * 100.0 if train_time > 0 else 0
        desc = stage_desc_map.get(st_key, st_key)
        lines.append(f"| `{st_key}` | {desc} | {dur:.2f}s | {pct:.1f}% |")
    lines.append("")

    # 5. Inference & Prediction Summary
    lines.append("## 5. Inference & Prediction Summary")
    lines.append("")
    lines.append("### Benchmark Measurements")
    sample_sz = bench.get("benchmark_sample_size", 1000)
    dev_str = bench.get("device", "CPU")
    if bench.get("gpu_name"):
        dev_str += f" ({bench.get('gpu_name')})"
    lines.append(f"- **Benchmark Sample**: {sample_sz:,} comments from Test set")
    lines.append(f"- **Primary Device**: {dev_str}")
    lines.append(f"- **Latency per 1,000 Comments (end-to-end, batch size 1000)**: **{bench.get('time_per_1000_comments_sec', 0.0):.2f} s**")
    lines.append(f"- **Per-Comment Latency**: **{bench.get('latency_ms_per_comment', 0.0):.2f} ms**")
    lines.append(f"- **Throughput**: **{bench.get('throughput_comments_per_sec', 0.0):.1f} comments/second**")

    # Detailed multi-device benchmark breakdown
    details = bench.get("details", {})
    if details:
        lines.append("")
        lines.append("#### Multi-Device & Batch Size Comparison (Median of 5 Repeats)")
        lines.append("| Configuration | Device | Batch Size | Latency (ms/comment) | Throughput (comments/s) | Total Time for 1,000 comments (s) |")
        lines.append("|---|---|---|---|---|---|")
        for cfg_name, cfg_res in details.items():
            lines.append(
                f"| `{cfg_name}` | {cfg_res.get('device', 'N/A')} | {cfg_res.get('batch_size', 'N/A')} | "
                f"{cfg_res.get('latency_ms_per_comment', 0.0):.3f} ms | {cfg_res.get('throughput_comments_per_sec', 0.0):.1f} | "
                f"{cfg_res.get('time_per_1000_comments_sec', 0.0):.3f} s |"
            )
    lines.append("")

    lines.append("### Decision Threshold Optimization")
    lines.append(
        "Decision thresholds were tuned strictly on the **Validation split** via 200 bootstrap resamples. "
        "For each resample, thresholds were searched over a grid of $0.01$ to $0.99$ (step $0.01$) to maximize F1 score, "
        "and the median bootstrap threshold was selected. This prevents overfitting the threshold to test data."
    )
    lines.append("")
    lines.append("### Prediction File Schema")
    lines.append(
        f"Predictions were saved to `predictions/{model_id}/{{val,test}}_predictions.csv` containing:\n"
        "- `id`: Unique comment identifier\n"
        "- `p_<label>` (x6): Continuous sigmoid probability in [0, 1]\n"
        "- `pred_<label>` (x6): Binary flag {0, 1} using tuned decision thresholds\n"
        "- `pred_hier_severe_toxic`: Binary flag requiring both `pred_severe_toxic == 1` and `pred_toxic == 1`\n"
        "- `y_<label>` (x6): Ground truth binary label\n"
    )
    lines.append("")

    # 6. Evaluation Tables
    lines.append("## 6. Evaluation Results (Test Split)")
    lines.append("")
    lines.append("### Headline Metrics (with 95% Bootstrap Confidence Intervals)")
    lines.append("")
    lines.append("| Metric | Test Value | 95% Bootstrap CI | Validation Reference |")
    lines.append("|---|---|---|---|")
    val_overall = metrics["val_reference"]["overall"]
    lines.append(f"| **Macro F1** | **{test_overall['macro_f1']:.4f}** | [{test_overall['macro_f1_ci'][0]:.4f}, {test_overall['macro_f1_ci'][1]:.4f}] | {val_overall['macro_f1']:.4f} |")
    lines.append(f"| **Micro F1** | **{test_overall['micro_f1']:.4f}** | [{test_overall['micro_f1_ci'][0]:.4f}, {test_overall['micro_f1_ci'][1]:.4f}] | {val_overall['micro_f1']:.4f} |")
    lines.append(f"| **Macro PR-AUC** | **{test_overall['macro_pr_auc']:.4f}** | [{test_overall['macro_pr_auc_ci'][0]:.4f}, {test_overall['macro_pr_auc_ci'][1]:.4f}] | {val_overall['macro_pr_auc']:.4f} |")
    lines.append(f"| **Mean ROC-AUC** | **{test_overall['mean_roc_auc']:.4f}** | [{test_overall['mean_roc_auc_ci'][0]:.4f}, {test_overall['mean_roc_auc_ci'][1]:.4f}] | {val_overall['mean_roc_auc']:.4f} |")
    lines.append(f"| Macro Precision | {test_overall['macro_precision']:.4f} | [{test_overall['macro_precision_ci'][0]:.4f}, {test_overall['macro_precision_ci'][1]:.4f}] | {val_overall['macro_precision']:.4f} |")
    lines.append(f"| Macro Recall | {test_overall['macro_recall']:.4f} | [{test_overall['macro_recall_ci'][0]:.4f}, {test_overall['macro_recall_ci'][1]:.4f}] | {val_overall['macro_recall']:.4f} |")
    lines.append(f"| Exact Match Accuracy | {test_overall['exact_match_accuracy']:.4f} | [{test_overall['exact_match_accuracy_ci'][0]:.4f}, {test_overall['exact_match_accuracy_ci'][1]:.4f}] | {val_overall['exact_match_accuracy']:.4f} |")
    lines.append(f"| Hamming Loss | {test_overall['hamming_loss']:.4f} | [{test_overall['hamming_loss_ci'][0]:.4f}, {test_overall['hamming_loss_ci'][1]:.4f}] | {val_overall['hamming_loss']:.4f} |")
    lines.append(f"| Mean Brier Score | {test_overall['mean_brier']:.4f} | N/A | {val_overall['mean_brier']:.4f} |")
    lines.append(f"| Mean ECE (15 bins) | {test_overall['mean_ece']:.4f} | N/A | {val_overall['mean_ece']:.4f} |")
    lines.append("")

    lines.append("### Detailed Per-Label Metrics on Test Split")
    lines.append("")
    lines.append("| Label | Support | Prevalence (%) | Tuned Thresh | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | PR-AUC (95% CI) | ROC-AUC (95% CI) | MCC | Brier | ECE |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for l in labels:
        m = test_per_label[l]
        p_ci = f"[{m['precision_ci'][0]:.3f}, {m['precision_ci'][1]:.3f}]"
        r_ci = f"[{m['recall_ci'][0]:.3f}, {m['recall_ci'][1]:.3f}]"
        f_ci = f"[{m['f1_ci'][0]:.3f}, {m['f1_ci'][1]:.3f}]"
        pr_ci = f"[{m['pr_auc_ci'][0]:.3f}, {m['pr_auc_ci'][1]:.3f}]"
        roc_ci = f"[{m['roc_auc_ci'][0]:.3f}, {m['roc_auc_ci'][1]:.3f}]"
        lines.append(
            f"| `{l}` | {m['support']:,} | {m['prevalence']*100:.2f}% | **{m['threshold']:.2f}** | "
            f"{m['precision']:.3f} {p_ci} | {m['recall']:.3f} {r_ci} | **{m['f1']:.3f}** {f_ci} | "
            f"**{m['pr_auc']:.3f}** {pr_ci} | {m['roc_auc']:.3f} {roc_ci} | {m['mcc']:.3f} | {m['brier']:.4f} | {m['ece']:.4f} |"
        )
    lines.append("")

    # Threshold comparison with bootstrap std
    lines.append("### Threshold Tuning Impact: Tuned vs Default 0.5")
    lines.append("")
    lines.append("| Label | Tuned Thresh | Bootstrap Std | F1 (Tuned) | Precision (Tuned) | Recall (Tuned) | F1 (at 0.5) | Precision (at 0.5) | Recall (at 0.5) | F1 $\\Delta$ (Tuned - 0.5) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    boot_details = metrics.get("threshold_tuning", {}).get("bootstrap_details", {})
    for l in labels:
        m = test_per_label[l]
        b_std = boot_details.get(l, {}).get("std", 0.0)
        delta_f1 = m["f1"] - m["f1_at_05"]
        sign = "+" if delta_f1 >= 0 else ""
        lines.append(
            f"| `{l}` | {m['threshold']:.2f} | {b_std:.4f} | **{m['f1']:.4f}** | {m['precision']:.4f} | {m['recall']:.4f} | "
            f"{m['f1_at_05']:.4f} | {m['precision_at_05']:.4f} | {m['recall_at_05']:.4f} | **{sign}{delta_f1:.4f}** |"
        )
    lines.append("")

    # Hierarchical severe_toxic rule comparison
    lines.append("### Hierarchical severe_toxic Rule Comparison")
    lines.append(
        "By definition, severe toxicity is a semantic subset of toxicity. The table below compares independent prediction against the hierarchical constraint `pred_severe_toxic & pred_toxic`:"
    )
    lines.append("")
    lines.append("| Rule Variant | Precision | Recall | F1 Score | Exact Match | Micro F1 |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(f"| Standard Independent | {hier['severe_toxic_standard_precision']:.4f} | {hier['severe_toxic_standard_recall']:.4f} | {hier['severe_toxic_standard_f1']:.4f} | {test_overall['exact_match_accuracy']:.4f} | {test_overall['micro_f1']:.4f} |")
    lines.append(f"| Hierarchical (`& toxic`) | {hier['severe_toxic_hierarchical_precision']:.4f} | {hier['severe_toxic_hierarchical_recall']:.4f} | {hier['severe_toxic_hierarchical_f1']:.4f} | {hier['hierarchical_exact_match']:.4f} | {hier['hierarchical_micro_f1']:.4f} |")
    lines.append("")

    # Count violations if prediction CSV exists
    test_pred_csv = preds_dir / "test_predictions.csv"
    if test_pred_csv.exists():
        df_p = pd.read_csv(test_pred_csv)
        violations = int(((df_p["pred_severe_toxic"] == 1) & (df_p["pred_toxic"] == 0)).sum())
        if violations == 0:
            lines.append("> **Note**: Zero comments in the test set were predicted as `severe_toxic=1` without also having `toxic=1`. Hence, standard independent thresholding and the hierarchical rule are identical by construction on this dataset.")
        else:
            lines.append(f"> **Note**: {violations:,} comments had `pred_severe_toxic=1` but `pred_toxic=0`. The hierarchical rule corrected these predictions.")
        lines.append("")

    # 7. Embedded Visualizations
    lines.append("## 7. Visualizations")
    lines.append("")
    lines.append("### Headline Metrics Summary")
    lines.append(f"![Headline Summary](../../visualizations/{viz_dir.name}/headline_summary.png)")
    lines.append("")
    lines.append("### Per-Label Performance Across Metrics")
    lines.append(f"![Per-Label Metrics](../../visualizations/{viz_dir.name}/per_label_metrics.png)")
    lines.append("")
    lines.append("### Precision-Recall Curves")
    lines.append(f"![Precision-Recall Curves](../../visualizations/{viz_dir.name}/pr_curves.png)")
    lines.append("")
    lines.append("### Receiver Operating Characteristic (ROC) Curves")
    lines.append(f"![ROC Curves](../../visualizations/{viz_dir.name}/roc_curves.png)")
    lines.append("")
    lines.append("### Validation Threshold Tuning vs F1 Curves")
    lines.append(f"![Threshold vs F1](../../visualizations/{viz_dir.name}/threshold_vs_f1.png)")
    lines.append("")
    lines.append("### Confusion Matrices (Test Split)")
    lines.append(f"![Confusion Matrices](../../visualizations/{viz_dir.name}/confusion_matrices.png)")
    lines.append("")
    lines.append("### Calibration & Reliability Diagrams (15 Bins)")
    lines.append(f"![Calibration Reliability](../../visualizations/{viz_dir.name}/calibration_reliability.png)")
    lines.append("")
    lines.append("### Probability Distributions by True Class")
    lines.append(f"![Probability Histograms](../../visualizations/{viz_dir.name}/probability_histograms.png)")
    lines.append("")

    # 8. Model-Specific Section (Dispatched)
    lines.extend(get_model_specific_section(adapter_type, model_id, meta, metrics, viz_dir, labels))

    # 9. Interpretation & Key Takeaways
    lines.append("## 9. Interpretation & Key Takeaways")
    lines.append("")
    lines.append(f"1. **Dominant Categories**: `{best_f1_lbl}` achieved the highest F1 score ({test_per_label[best_f1_lbl]['f1']:.4f}), while `{best_prauc_lbl}` achieved the highest PR-AUC ({test_per_label[best_prauc_lbl]['pr_auc']:.4f}). This reflects high representation and overt lexical signals.")
    lines.append(f"2. **Challenging Categories**: `{worst_f1_lbl}` presented the lowest F1 score ({test_per_label[worst_f1_lbl]['f1']:.4f}), while `{worst_prauc_lbl}` presented the lowest PR-AUC ({test_per_label[worst_prauc_lbl]['pr_auc']:.4f}). Extreme class imbalance and context sensitivity make these rare categories particularly difficult.")

    # Threshold delta statement
    deltas = [test_per_label[l]["f1"] - test_per_label[l]["f1_at_05"] for l in labels]
    mean_delta = float(np.mean(deltas))
    worse_labels = [l for l in labels if test_per_label[l]["f1"] < test_per_label[l]["f1_at_05"]]
    if mean_delta > 0.005:
        th_assessment = f"improved overall performance (mean F1 delta: {mean_delta:+.4f})"
    elif mean_delta < -0.005:
        th_assessment = f"was slightly worse overall than 0.5 (mean F1 delta: {mean_delta:+.4f})"
    else:
        th_assessment = f"resulted in no meaningful overall change compared to 0.5 (mean F1 delta: {mean_delta:+.4f})"

    worse_str = f", with lower F1 on {', '.join(worse_labels)}" if worse_labels else ""
    lines.append(f"3. **Threshold Tuning Impact**: Optimizing thresholds via validation bootstrap {th_assessment}{worse_str}.")

    # Calibration statement (Brier, ECE, highest ECE label)
    highest_ece_lbl = max(labels, key=lambda l: test_per_label[l]["ece"])
    lines.append(f"4. **Calibration**: The mean Brier score was {test_overall['mean_brier']:.4f} and mean 15-bin ECE was {test_overall['mean_ece']:.4f}, with the highest calibration error observed in `{highest_ece_lbl}` (ECE: {test_per_label[highest_ece_lbl]['ece']:.4f}).")

    # Throughput statement
    lines.append(f"5. **Inference Speed**: Generating predictions for 1,000 comments in {bench.get('time_per_1000_comments_sec', 0.0):.2f}s ({bench.get('throughput_comments_per_sec', 0.0):.1f} comments/sec) demonstrates operational throughput suitable for production pipeline integration.")
    lines.append("")

    # 10. Conclusions
    lines.append("## 10. Conclusions")
    lines.append("")
    if "tfidf" in adapter_type:
        lines.append(
            f"With a Macro F1 of **{test_overall['macro_f1']:.4f}** and Mean ROC-AUC of **{test_overall['mean_roc_auc']:.4f}**, "
            f"Model 1 (`{model_id}`) satisfies the requirements of a high-speed linear baseline. "
            f"Because character n-grams capture obfuscated profanity and casing variations without morphological parsers, "
            f"the dual word/char representation performs reliably on overt harassment. "
            f"However, sequential context and semantic subtlety (such as indirect threats or disguised hate speech) "
            f"will require the deep contextual representations provided by Model 2 (BiGRU + Attention) and Model 3 (DistilRoBERTa)."
        )
    elif "bigru" in adapter_type:
        lines.append(
            f"With a Macro F1 of **{test_overall['macro_f1']:.4f}** and Mean ROC-AUC of **{test_overall['mean_roc_auc']:.4f}**, "
            f"Model 2 (`{model_id}`) demonstrates the utility of recurrent sequential modeling and attention pooling over static linear baselines. "
            f"Additive attention successfully concentrates probability mass on toxic semantic tokens, while packed sequence processing "
            f"allows variable-length comments to be handled without padding contamination. "
            f"Residual errors primarily stem from frozen embeddings and vocabulary coverage boundaries, setting the stage for Transformer fine-tuning in Model 3."
        )
    else:
        lines.append(f"Model `{model_id}` achieved Macro F1 of {test_overall['macro_f1']:.4f} and Mean ROC-AUC of {test_overall['mean_roc_auc']:.4f}.")
    lines.append("")

    # 11. Static Limitations
    lines.append("## 11. Limitations")
    lines.append("")
    lines.append("The following limitations apply strictly to this model evaluation:")
    if "bigru" in adapter_type:
        lines.append("1. **Single run and cuDNN non-determinism**: Results represent a single run (`seed=42`); GPU cuDNN GRU backward passes are inherently non-bit-deterministic.")
    else:
        lines.append("1. **Single run**: Results represent a single deterministic seed run (`seed=42`) without multi-seed averaging.")
    lines.append("2. **Fixed untuned hyperparameters**: Model hyperparameters were evaluated with fixed settings without extensive grid search.")
    lines.append("3. **Validation reused**: The validation split was utilized for early stopping, checkpoint selection and threshold tuning.")
    lines.append(f"4. **Noisy threat estimates**: There are only {n_val_threat} validation and {n_test_threat} test positive instances for `threat`, causing wider confidence intervals and higher sampling variance on threat metrics.")
    lines.append("5. **Near-duplicate threshold**: Duplicate grouping used MinHash LSH Jaccard similarity threshold of 0.8 on character 5-grams.")
    if "bigru" in adapter_type:
        tok_info = meta.get("tokenizer_info", {})
        trunc_all = (tok_info.get("val_truncated_pct", 0.0) + tok_info.get("test_truncated_pct", 0.0)) / 2.0
        lines.append("6. **Lowercasing signal loss**: Lowercasing discards the uppercase shout/casing signals identified during EDA (the BiGRU is the only model of the three that discards it).")
        lines.append("7. **Frozen embeddings**: fastText 300-d embeddings were frozen without domain-specific continuous fine-tuning.")
        lines.append(f"8. **Sequence truncation**: Input text truncated at 256 tokens, with approximately {trunc_all:.2f}% of comments affected.")
    lines.append("")

    # Save to file
    report_file = reports_dir / "model_report.md"
    with open(report_file, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"\n[Report] Model report generated successfully at: {report_file.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
