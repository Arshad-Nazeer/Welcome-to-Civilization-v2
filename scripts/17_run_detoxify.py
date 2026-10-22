#!/usr/bin/env python
"""
scripts/17_run_detoxify.py
--------------------------
Inference and evaluation driver for the external pretrained baseline:
Unitary AI Detoxify Original (BERT-base fine-tuned on Jigsaw 2018).

Pipeline Steps:
1. Loads validation split (15,869 comments) and test split (15,841 comments).
2. Runs Detoxify("original") batch inference.
3. Correctly maps output scores to repository label schema:
   - toxicity        -> toxic
   - severe_toxicity -> severe_toxic
   - obscene         -> obscene
   - threat          -> threat
   - insult          -> insult
   - identity_attack -> identity_hate
4. Saves predicted probability arrays:
   - predictions/detoxify/probs_val.npy
   - predictions/detoxify/probs_test.npy
5. Emits model metadata:
   - models/detoxify/meta.json
6. Invokes evaluation engine (scripts/20_evaluate.py --model-id detoxify).
7. Invokes model report generator (scripts/21_build_model_report.py --model-id detoxify).
8. Runs paired bootstrap hypothesis test against Model 3 and Model 1.
9. Updates cross-model comparison tables (scripts/20_evaluate.py --compare).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

# Ensure project root src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import torch

from wtc.config import (
    CFG,
    LABELS,
    PROJECT_ROOT,
    get_model_dir,
    get_predictions_dir,
    get_reports_dir,
    get_visualizations_dir,
)
from wtc.data import load_split
from wtc.preprocess import clean_text
from wtc.utils import get_logger, set_seed


LABEL_MAPPING = {
    "toxic": "toxicity",
    "severe_toxic": "severe_toxicity",
    "obscene": "obscene",
    "threat": "threat",
    "insult": "insult",
    "identity_hate": "identity_attack",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Detoxify Original External Baseline")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test with 500 samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-eval", action="store_true", help="Skip subsequent evaluation scripts")
    return parser.parse_args()


def predict_dataset_detoxify(
    model: Any,
    texts: list[str],
    labels: list[str],
    batch_size: int = 64,
    desc: str = "Inference",
) -> np.ndarray:
    """Run batch prediction with Detoxify using dynamic length-sorted batching."""
    n_samples = len(texts)
    if n_samples == 0:
        return np.empty((0, len(labels)), dtype=np.float32)

    t0 = time.time()
    print(f"[{desc}] Starting prediction for {n_samples:,} comments (batch_size={batch_size})...", flush=True)

    lengths = np.array([len(t) for t in texts], dtype=np.int64)
    sorted_indices = np.argsort(lengths)
    probs_ordered = np.zeros((n_samples, len(labels)), dtype=np.float32)

    for i in range(0, n_samples, batch_size):
        b_idx = sorted_indices[i : i + batch_size]
        b_texts = [texts[idx] for idx in b_idx]
        preds_dict = model.predict(b_texts)

        chunk_cols = []
        for l in labels:
            detox_key = LABEL_MAPPING[l]
            vals = preds_dict[detox_key]
            if isinstance(vals, (int, float)):
                vals = [vals]
            chunk_cols.append(np.array(vals, dtype=np.float32))

        chunk_arr = np.column_stack(chunk_cols)
        probs_ordered[b_idx] = chunk_arr

        done = min(i + batch_size, n_samples)
        if (i // batch_size) % 25 == 0 or done == n_samples:
            elapsed = time.time() - t0
            speed = done / elapsed if elapsed > 0 else 0.0
            print(f"  Processed {done:,}/{n_samples:,} comments ({done/n_samples*100:.1f}%) - {speed:.1f} comments/sec", flush=True)

    total_time = time.time() - t0
    print(f"[{desc}] Completed in {total_time:.2f}s ({n_samples / total_time:.1f} comments/sec). Shape: {probs_ordered.shape}", flush=True)
    return probs_ordered


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    smoke = args.smoke
    batch_size = args.batch_size
    model_id = "detoxify"

    logger = get_logger("run_detoxify")
    logger.info("Initializing Detoxify External Baseline Pipeline")

    model_dir = get_model_dir(model_id, smoke=smoke)
    preds_dir = get_predictions_dir(model_id, smoke=smoke)
    reports_dir = get_reports_dir(model_id, smoke=smoke)
    viz_dir = get_visualizations_dir(model_id, smoke=smoke)

    print("=" * 75)
    print("DETOXIFY ORIGINAL: EXTERNAL PRETRAINED BASELINE PIPELINE")
    print(f"Model ID: {model_id} | Smoke: {smoke} | Batch Size: {batch_size}")
    print("=" * 75)

    # 1. Load splits
    print("\n[Data] Loading validation and test splits...")
    df_val = load_split("val")
    df_test = load_split("test")

    if smoke:
        df_val = df_val.iloc[:500].reset_index(drop=True)
        df_test = df_test.iloc[:500].reset_index(drop=True)
        print(f"  [Smoke Mode] Trimmed to 500 samples each.")

    # We use text_clean as standard cleaned text input
    text_col = "text_clean" if "text_clean" in df_val.columns else CFG.get("text_column", "comment_text")
    val_texts = df_val[text_col].fillna("").tolist()
    test_texts = df_test[text_col].fillna("").tolist()

    print(f"  Validation comments: {len(val_texts):,}")
    print(f"  Test comments:       {len(test_texts):,}")

    # 2. Load Detoxify model
    print("\n[Model] Loading Detoxify('original')...")
    from detoxify import Detoxify

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Compute device: {device.upper()}")
    t_load_0 = time.time()
    model = Detoxify("original", device=device)
    load_time = time.time() - t_load_0
    print(f"  Model loaded successfully in {load_time:.2f}s.")

    # 3. Predict on validation set
    t_val_0 = time.time()
    P_val = predict_dataset_detoxify(
        model=model,
        texts=val_texts,
        labels=LABELS,
        batch_size=batch_size,
        desc="Validation Set",
    )
    val_time = time.time() - t_val_0

    # 4. Predict on test set
    t_test_0 = time.time()
    P_test = predict_dataset_detoxify(
        model=model,
        texts=test_texts,
        labels=LABELS,
        batch_size=batch_size,
        desc="Test Set",
    )
    test_time = time.time() - t_test_0

    # 5. Save probability arrays
    print("\n[Saving] Saving probability arrays to predictions/detoxify/...")
    p_val_file = preds_dir / "probs_val.npy"
    p_test_file = preds_dir / "probs_test.npy"
    np.save(p_val_file, P_val)
    np.save(p_test_file, P_test)
    print(f"  Saved: {p_val_file.relative_to(PROJECT_ROOT)} ({p_val_file.stat().st_size:,} bytes)")
    print(f"  Saved: {p_test_file.relative_to(PROJECT_ROOT)} ({p_test_file.stat().st_size:,} bytes)")

    # 6. Save model metadata
    print("\n[Metadata] Writing models/detoxify/meta.json...")
    meta = {
        "model_id": model_id,
        "model_name": "Detoxify Original (BERT-base Pretrained Baseline)",
        "adapter_type": "detoxify",
        "role": "External Pretrained Baseline",
        "labels": LABELS,
        "source": "unitaryai/detoxify",
        "repository_url": "https://github.com/unitaryai/detoxify",
        "checkpoint": "original",
        "backbone": "bert-base-uncased",
        "num_classes": len(LABELS),
        "parameters": {
            "total": 110_000_000,
            "trainable": 0,
            "frozen": 110_000_000,
        },
        "model_size_mb": 418.0,
        "model_size_bytes": 438_308_864,
        "label_mapping": LABEL_MAPPING,
        "training_source": "2018 Kaggle Jigsaw Toxic Comment Classification Challenge",
        "stage_timings_seconds": {
            "model_load": round(load_time, 2),
            "val_inference": round(val_time, 2),
            "test_inference": round(test_time, 2),
            "total": round(load_time + val_time + test_time, 2),
        },
        "device_info": {
            "device": device,
            "cpu_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    meta_file = model_dir / "meta.json"
    with open(meta_file, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  Saved metadata to {meta_file.relative_to(PROJECT_ROOT)}")

    if args.skip_eval:
        print("\nSkipping evaluation steps as requested.")
        return

    # 7. Run scripts/20_evaluate.py --model-id detoxify
    print("\n" + "=" * 75)
    print("RUNNING EVALUATION SUITE FOR DETOXIFY")
    print("=" * 75)
    cmd_eval = [sys.executable, "scripts/20_evaluate.py", "--model-id", model_id]
    if smoke:
        cmd_eval.append("--smoke")
    res_eval = subprocess.run(cmd_eval, check=True)

    # 8. Run scripts/21_build_model_report.py --model-id detoxify
    print("\n" + "=" * 75)
    print("GENERATING MODEL REPORT FOR DETOXIFY")
    print("=" * 75)
    cmd_rep = [sys.executable, "scripts/21_build_model_report.py", "--model-id", model_id]
    if smoke:
        cmd_rep.append("--smoke")
    subprocess.run(cmd_rep, check=True)

    # 9. Run paired bootstrap test: Model 3 vs Detoxify
    print("\n" + "=" * 75)
    print("RUNNING PAIRED BOOTSTRAP SIGNIFICANCE TEST: Model 3 vs Detoxify")
    print("=" * 75)
    cmd_boot = [
        sys.executable,
        "scripts/22_paired_bootstrap.py",
        "--model-3", "3_distilroberta",
        "--baseline", "detoxify",
    ]
    if smoke:
        cmd_boot.append("--smoke")
    try:
        subprocess.run(cmd_boot, check=True)
    except Exception as e:
        print(f"Paired bootstrap vs Model 3 failed: {e}")

    # 10. Run paired bootstrap test: Detoxify vs Model 1
    print("\n" + "=" * 75)
    print("RUNNING PAIRED BOOTSTRAP SIGNIFICANCE TEST: Detoxify vs Model 1")
    print("=" * 75)
    cmd_boot_m1 = [
        sys.executable,
        "scripts/22_paired_bootstrap.py",
        "--model-3", "detoxify",
        "--baseline", "1_tfidf_lr",
    ]
    if smoke:
        cmd_boot_m1.append("--smoke")
    try:
        subprocess.run(cmd_boot_m1, check=True)
    except Exception as e:
        print(f"Paired bootstrap vs Model 1 failed: {e}")

    # 11. Compare all models across the workspace
    print("\n" + "=" * 75)
    print("UPDATING COMPREHENSIVE MODEL COMPARISON (INCLUDING DETOXIFY BASELINE)")
    print("=" * 75)
    cmd_comp = [sys.executable, "scripts/20_evaluate.py", "--compare"]
    subprocess.run(cmd_comp, check=True)

    print("\n" + "=" * 75)
    print("DETOXIFY BASELINE EVALUATION PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    main()
