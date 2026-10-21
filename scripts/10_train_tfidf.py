#!/usr/bin/env python
"""
scripts/10_train_tfidf.py
-------------------------
Train Model 1: Word + Char TF-IDF with Logistic Regression.
Supports --smoke mode for rapid verification.
Stage-based live progress:
  [1/5] load+verify
  [2/5] word TF-IDF fit+transform
  [3/5] char TF-IDF fit+transform
  [4/5] per-label LR fits (liblinear, class_weight sqrt(neg/pos), threading backend)
  [5/5] predict val/test and save
Updates models/<id>/status.json and logs to logs/<id>/train_<date>.log.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time

# Ensure project root src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import joblib
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from wtc.config import (
    CFG,
    LABELS,
    PROJECT_ROOT,
    get_logs_dir,
    get_model_dir,
    get_model_id,
    get_predictions_dir,
    load_models_config,
)
from wtc.data import load_split, load_split_manifest
from wtc.utils import get_logger, set_seed, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Model 1: TF-IDF + Logistic Regression")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test with reduced data subset")
    parser.add_argument("--model-name", type=str, default="tfidf_lr", help="Model key in configs/models.yaml")
    return parser.parse_args()


def update_status(status_file: Path, state: str, stage: str, details: dict | None = None) -> None:
    payload = {
        "state": state,
        "stage": stage,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if details:
        payload.update(details)
    try:
        with open(status_file, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    except Exception:
        pass


def main() -> None:
    args = parse_args()
    smoke = args.smoke
    model_name = args.model_name
    model_id = get_model_id(model_name)

    models_cfg = load_models_config()
    if model_name not in models_cfg:
        raise KeyError(f"Model '{model_name}' not found in configs/models.yaml")
    m_cfg = models_cfg[model_name]

    seed = int(m_cfg.get("seed", CFG.get("seed", 42)))
    set_seed(seed)

    model_dir = get_model_dir(model_id, smoke=smoke)
    preds_dir = get_predictions_dir(model_id, smoke=smoke)
    logs_dir = get_logs_dir(model_id, smoke=smoke)
    status_file = model_dir / "status.json"

    logger = get_logger(f"train_{model_id}{'_smoke' if smoke else ''}", log_dir=logs_dir)
    logger.info("=" * 70)
    logger.info(f"Starting Training for Model: {model_id} (smoke={smoke})")
    logger.info("=" * 70)

    total_start_time = time.perf_counter()
    stage_timings: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    #  [1/5] Load and Verify Data                                         #
    # ------------------------------------------------------------------ #
    stage1_start = time.perf_counter()
    print("\n[1/5] Loading and verifying splits...")
    logger.info("[1/5] Loading and verifying splits...")
    update_status(status_file, "RUNNING", "1/5_load_and_verify")

    manifest = load_split_manifest()
    split_version = manifest.get("split_version", "")
    if split_version != "v2_near_dup":
        err_msg = f"Data verification failed: expected split_version 'v2_near_dup', got '{split_version}'"
        logger.error(err_msg)
        update_status(status_file, "FAILED", "1/5_load_and_verify", {"error": err_msg})
        raise ValueError(err_msg)

    processed_parquet_path = PROJECT_ROOT / "data" / "processed" / "processed.parquet"
    if not processed_parquet_path.exists():
        err_msg = f"Processed parquet file not found at {processed_parquet_path}"
        logger.error(err_msg)
        update_status(status_file, "FAILED", "1/5_load_and_verify", {"error": err_msg})
        raise FileNotFoundError(err_msg)

    parquet_sha = sha256_file(processed_parquet_path)
    expected_sha = manifest.get("processed_parquet_sha256", "")
    if parquet_sha != expected_sha:
        err_msg = f"SHA256 mismatch for processed.parquet!\nExpected: {expected_sha}\nFound:    {parquet_sha}"
        logger.error(err_msg)
        update_status(status_file, "FAILED", "1/5_load_and_verify", {"error": err_msg})
        raise ValueError(err_msg)

    logger.info(f"SHA256 verified: {parquet_sha} matches split_manifest.json")

    df_train = load_split("train")
    df_val = load_split("val")
    df_test = load_split("test")

    if smoke:
        logger.info("Smoke mode enabled: subsampling train to 2,000, val to 500, test to 500 rows")
        df_train = df_train.iloc[:2000].reset_index(drop=True)
        df_val = df_val.iloc[:500].reset_index(drop=True)
        df_test = df_test.iloc[:500].reset_index(drop=True)
    else:
        # Runtime size validation
        if len(df_train) != 127861:
            raise ValueError(f"Train row count mismatch: expected 127,861, got {len(df_train)}")
        if len(df_val) != 15869:
            raise ValueError(f"Val row count mismatch: expected 15,869, got {len(df_val)}")
        if len(df_test) != 15841:
            raise ValueError(f"Test row count mismatch: expected 15,841, got {len(df_test)}")

    input_col = m_cfg.get("input_col", "text_clean")
    logger.info(f"Using input column: {input_col}")
    logger.info(f"Loaded rows: train={len(df_train):,}, val={len(df_val):,}, test={len(df_test):,}")

    train_texts = df_train[input_col].fillna("").tolist()
    val_texts = df_val[input_col].fillna("").tolist()
    test_texts = df_test[input_col].fillna("").tolist()

    Y_train = df_train[LABELS].values.astype(np.int32)
    Y_val = df_val[LABELS].values.astype(np.int32)
    Y_test = df_test[LABELS].values.astype(np.int32)

    stage1_elapsed = time.perf_counter() - stage1_start
    stage_timings["1_load_and_verify"] = stage1_elapsed
    print(f"  Done in {stage1_elapsed:.2f}s (Train: {len(df_train):,}, Val: {len(df_val):,}, Test: {len(df_test):,})")

    # ------------------------------------------------------------------ #
    #  [2/5] Word TF-IDF Fit and Transform                                #
    # ------------------------------------------------------------------ #
    stage2_start = time.perf_counter()
    print("\n[2/5] Fitting and transforming Word TF-IDF (train only)...")
    logger.info("[2/5] Fitting and transforming Word TF-IDF (train only)...")
    update_status(status_file, "RUNNING", "2/5_word_tfidf", {"elapsed_so_far": time.perf_counter() - total_start_time})

    word_cfg = m_cfg["vectorizer_word"]
    vec_word = TfidfVectorizer(
        ngram_range=tuple(word_cfg.get("ngram_range", [1, 2])),
        lowercase=word_cfg.get("lowercase", True),
        min_df=word_cfg.get("min_df", 3),
        max_features=word_cfg.get("max_features", 50000),
        sublinear_tf=word_cfg.get("sublinear_tf", True),
        dtype=np.float32,
    )

    X_train_word = vec_word.fit_transform(train_texts)
    X_val_word = vec_word.transform(val_texts)
    X_test_word = vec_word.transform(test_texts)

    n_word_features = X_train_word.shape[1]
    stage2_elapsed = time.perf_counter() - stage2_start
    stage_timings["2_word_tfidf"] = stage2_elapsed
    print(f"  Done in {stage2_elapsed:.2f}s. Extracted {n_word_features:,} word features.")
    logger.info(f"Word TF-IDF fitted in {stage2_elapsed:.2f}s: {n_word_features} features, shape {X_train_word.shape}")

    # ------------------------------------------------------------------ #
    #  [3/5] Char TF-IDF Fit and Transform                                #
    # ------------------------------------------------------------------ #
    stage3_start = time.perf_counter()
    print("\n[3/5] Fitting and transforming Char TF-IDF (train only)...")
    logger.info("[3/5] Fitting and transforming Char TF-IDF (train only)...")
    update_status(status_file, "RUNNING", "3/5_char_tfidf", {"elapsed_so_far": time.perf_counter() - total_start_time})

    char_cfg = m_cfg["vectorizer_char"]
    vec_char = TfidfVectorizer(
        analyzer=char_cfg.get("analyzer", "char_wb"),
        ngram_range=tuple(char_cfg.get("ngram_range", [2, 5])),
        lowercase=char_cfg.get("lowercase", False),
        min_df=char_cfg.get("min_df", 3),
        max_features=char_cfg.get("max_features", 100000),
        sublinear_tf=char_cfg.get("sublinear_tf", True),
        dtype=np.float32,
    )

    X_train_char = vec_char.fit_transform(train_texts)
    X_val_char = vec_char.transform(val_texts)
    X_test_char = vec_char.transform(test_texts)

    n_char_features = X_train_char.shape[1]
    stage3_elapsed = time.perf_counter() - stage3_start
    stage_timings["3_char_tfidf"] = stage3_elapsed
    print(f"  Done in {stage3_elapsed:.2f}s. Extracted {n_char_features:,} char features.")
    logger.info(f"Char TF-IDF fitted in {stage3_elapsed:.2f}s: {n_char_features} features, shape {X_train_char.shape}")

    # Feature hstacking
    X_train = scipy.sparse.hstack([X_train_word, X_train_char], format="csr", dtype=np.float32)
    X_val = scipy.sparse.hstack([X_val_word, X_val_char], format="csr", dtype=np.float32)
    X_test = scipy.sparse.hstack([X_test_word, X_test_char], format="csr", dtype=np.float32)
    n_total_features = X_train.shape[1]
    logger.info(f"Combined feature matrix: {n_total_features:,} total features. Sparsity: {1.0 - (X_train.nnz / (X_train.shape[0] * X_train.shape[1])):.4%}")

    # ------------------------------------------------------------------ #
    #  [4/5] Per-Label Logistic Regression Fits                           #
    # ------------------------------------------------------------------ #
    stage4_start = time.perf_counter()
    print("\n[4/5] Computing class weights and training 6 Logistic Regression classifiers...")
    logger.info("[4/5] Computing class weights and training 6 Logistic Regression classifiers...")

    # Class weights w_l = (neg/pos)^0.5
    class_weights: dict[str, float] = {}
    print("  Calculated train split class weights:")
    for idx, label in enumerate(LABELS):
        pos_cnt = int(np.sum(Y_train[:, idx] == 1))
        neg_cnt = int(np.sum(Y_train[:, idx] == 0))
        if pos_cnt == 0:
            w_l = 1.0
        else:
            w_l = float(np.sqrt(neg_cnt / pos_cnt))
        class_weights[label] = round(w_l, 4)
        print(f"    - {label:14s}: pos={pos_cnt:,}, neg={neg_cnt:,}, w_1 = {w_l:.4f}")
        logger.info(f"Class weight {label}: pos={pos_cnt}, neg={neg_cnt}, weight={w_l:.4f}")

    clf_cfg = m_cfg.get("classifier", {})
    penalty = clf_cfg.get("penalty", "l2")
    solver = clf_cfg.get("solver", "liblinear")
    max_iter = int(clf_cfg.get("max_iter", 200))
    tune_c_grid = m_cfg.get("tune", {}).get("C", [2.0])

    classifiers: dict[str, LogisticRegression] = {}
    labels_done: list[str] = []

    # If tune_c_grid has multiple values, select best C by val macro PR-AUC
    chosen_c = float(tune_c_grid[0])
    if len(tune_c_grid) > 1:
        logger.info(f"Tune grid with {len(tune_c_grid)} C values: {tune_c_grid}. Searching best C on validation split...")
        best_val_prauc = -1.0
        best_c = chosen_c
        for cand_c in tune_c_grid:
            cand_pr_aucs = []
            for col_idx, label in enumerate(LABELS):
                cand_clf = LogisticRegression(
                    penalty=penalty,
                    solver=solver,
                    C=float(cand_c),
                    max_iter=max_iter,
                    random_state=seed,
                    class_weight={0: 1.0, 1: class_weights[label]},
                )
                cand_clf.fit(X_train, Y_train[:, col_idx])
                p_val_cand = cand_clf.predict_proba(X_val)[:, 1]
                cand_pr_aucs.append(average_precision_score(Y_val[:, col_idx], p_val_cand))
            mean_prauc = float(np.mean(cand_pr_aucs))
            logger.info(f"Tune candidate C={cand_c}: Val Macro PR-AUC = {mean_prauc:.4f}")
            if mean_prauc > best_val_prauc:
                best_val_prauc = mean_prauc
                best_c = float(cand_c)
        chosen_c = best_c
        logger.info(f"Selected optimal C = {chosen_c} (Val Macro PR-AUC: {best_val_prauc:.4f})")

    logger.info(f"Fitting final classifiers with C={chosen_c}, penalty={penalty}, solver={solver}, max_iter={max_iter}")

    # Fit classifiers in parallel using joblib threading backend (n_jobs=6) so X_train matrix is shared
    def _fit_single(label_name: str, col_idx: int) -> tuple[str, LogisticRegression, float]:
        t0 = time.perf_counter()
        clf = LogisticRegression(
            penalty=penalty,
            solver=solver,
            C=chosen_c,
            max_iter=max_iter,
            random_state=seed,
            class_weight={0: 1.0, 1: class_weights[label_name]},
        )
        clf.fit(X_train, Y_train[:, col_idx])
        dur = time.perf_counter() - t0
        return label_name, clf, dur

    with ThreadPoolExecutor(max_workers=6) as executor:
        future_map = {
            executor.submit(_fit_single, label, idx): (label, idx)
            for idx, label in enumerate(LABELS)
        }

        total_labels = len(LABELS)
        for future in as_completed(future_map):
            lbl, clf, dur = future.result()
            classifiers[lbl] = clf
            labels_done.append(lbl)

            elapsed_stage = time.perf_counter() - stage4_start
            n_done = len(labels_done)
            rate = elapsed_stage / n_done
            eta = (total_labels - n_done) * rate

            status_msg = f"  [{n_done}/{total_labels}] Label '{lbl}' completed in {dur:.2f}s (Elapsed: {elapsed_stage:.1f}s, ETA: {eta:.1f}s)"
            print(status_msg)
            logger.info(status_msg)

            update_status(
                status_file,
                "RUNNING",
                "4/5_logistic_regression",
                {
                    "labels_done": labels_done,
                    "elapsed_so_far": time.perf_counter() - total_start_time,
                    "stage_elapsed": elapsed_stage,
                    "eta_remaining": eta,
                },
            )

    stage4_elapsed = time.perf_counter() - stage4_start
    stage_timings["4_lr_fits"] = stage4_elapsed
    print(f"  All 6 classifiers trained in {stage4_elapsed:.2f}s.")

    # ------------------------------------------------------------------ #
    #  [5/5] Predict Val / Test and Save Artifacts                        #
    # ------------------------------------------------------------------ #
    stage5_start = time.perf_counter()
    print("\n[5/5] Generating predictions and saving model artifacts...")
    logger.info("[5/5] Generating predictions and saving model artifacts...")
    update_status(status_file, "RUNNING", "5/5_predict_and_save", {"elapsed_so_far": time.perf_counter() - total_start_time})

    val_prob_cols = []
    test_prob_cols = []
    for label in LABELS:
        clf = classifiers[label]
        val_prob_cols.append(clf.predict_proba(X_val)[:, 1])
        test_prob_cols.append(clf.predict_proba(X_test)[:, 1])

    probs_val = np.column_stack(val_prob_cols).astype(np.float32)
    probs_test = np.column_stack(test_prob_cols).astype(np.float32)

    # Save predictions
    p_val_file = preds_dir / "probs_val.npy"
    p_test_file = preds_dir / "probs_test.npy"
    np.save(p_val_file, probs_val)
    np.save(p_test_file, probs_test)
    logger.info(f"Saved probs_val.npy shape {probs_val.shape} to {p_val_file}")
    logger.info(f"Saved probs_test.npy shape {probs_test.shape} to {p_test_file}")

    # Save models
    f_vec_word = model_dir / "vectorizer_word.joblib"
    f_vec_char = model_dir / "vectorizer_char.joblib"
    f_clfs = model_dir / "classifiers.joblib"

    joblib.dump(vec_word, f_vec_word, compress=3)
    joblib.dump(vec_char, f_vec_char, compress=3)
    joblib.dump(classifiers, f_clfs, compress=3)

    size_word = f_vec_word.stat().st_size
    size_char = f_vec_char.stat().st_size
    size_clfs = f_clfs.stat().st_size
    total_model_bytes = size_word + size_char + size_clfs
    total_model_mb = round(total_model_bytes / (1024 * 1024), 2)

    total_elapsed = time.perf_counter() - total_start_time
    stage5_elapsed = time.perf_counter() - stage5_start
    stage_timings["5_predict_and_save"] = stage5_elapsed

    meta_payload = {
        "model_id": model_id,
        "model_name": model_name,
        "adapter_type": "tfidf_lr",
        "smoke": smoke,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "labels": LABELS,
        "input_column": input_col,
        "class_weights": {l: {0: 1.0, 1: class_weights[l]} for l in LABELS},
        "chosen_C": chosen_c,
        "config": m_cfg,
        "feature_counts": {
            "word_features": n_word_features,
            "char_features": n_char_features,
            "total_features": n_total_features,
        },
        "artifacts": {
            "vectorizer_word": str(f_vec_word.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "vectorizer_char": str(f_vec_char.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "classifiers": str(f_clfs.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "probs_val": str(p_val_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "probs_test": str(p_test_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
        "model_size_bytes": total_model_bytes,
        "model_size_mb": total_model_mb,
        "train_time_seconds": round(total_elapsed, 2),
        "stage_timings_seconds": {k: round(v, 2) for k, v in stage_timings.items()},
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": getattr(sys.modules.get("sklearn"), "__version__", "1.5.2"),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "joblib": joblib.__version__,
        },
    }

    meta_file = model_dir / "meta.json"
    with open(meta_file, "w", encoding="utf-8") as fh:
        json.dump(meta_payload, fh, indent=2)

    update_status(
        status_file,
        "COMPLETED",
        "done",
        {
            "total_train_seconds": round(total_elapsed, 2),
            "stage_timings": {k: round(v, 2) for k, v in stage_timings.items()},
            "model_size_mb": total_model_mb,
        },
    )

    # DONE Banner
    banner = f"""
================================================================================
  TRAINING COMPLETED SUCCESSFULLY: {model_id} (smoke={smoke})
================================================================================
  Total Time: {total_elapsed:.2f}s ({total_elapsed / 60:.2f} min)

  Stage Timings:
    [1/5] Load & Verify:      {stage_timings['1_load_and_verify']:.2f}s
    [2/5] Word TF-IDF:        {stage_timings['2_word_tfidf']:.2f}s ({n_word_features:,} features)
    [3/5] Char TF-IDF:        {stage_timings['3_char_tfidf']:.2f}s ({n_char_features:,} features)
    [4/5] LR Classifiers (x6):{stage_timings['4_lr_fits']:.2f}s
    [5/5] Predict & Save:     {stage_timings['5_predict_and_save']:.2f}s

  Model Size on Disk: {total_model_mb} MB ({total_model_bytes:,} bytes)
  Outputs:
    - Models:      {model_dir.relative_to(PROJECT_ROOT)}
    - Predictions: {preds_dir.relative_to(PROJECT_ROOT)}
    - Logs:        {logs_dir.relative_to(PROJECT_ROOT)}
================================================================================
"""
    print(banner)
    logger.info(banner)

    # Beep notification (winsound on Windows)
    try:
        import winsound
        winsound.Beep(1000, 350)
    except Exception:
        pass


if __name__ == "__main__":
    main()
