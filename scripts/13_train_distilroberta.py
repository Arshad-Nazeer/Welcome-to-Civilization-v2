#!/usr/bin/env python
"""
scripts/13_train_distilroberta.py
----------------------------------
Train Model 3: DistilRoBERTa-base fine-tuning for multi-label toxicity.
Stages:
  [1/6] load+verify
  [2/6] tokenize (byte-level BPE, case preserved)
  [3/6] setup model & optimizer (custom head, log-odds bias init, damped pos_weight, LLRD AdamW)
  [4/6] training loop (length-bucketed batches, FP16 AMP, live progress, resumable)
  [5/6] predict val & test with best model + hierarchy ablation
  [6/6] save backend-ready bundle & parity check
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

# Ensure project root src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from safetensors.torch import save_file
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from wtc.config import (
    CFG,
    LABELS,
    PROJECT_ROOT,
    get_logs_dir,
    get_model_dir,
    get_predictions_dir,
    load_models_config,
)
from wtc.data import load_split, load_split_manifest
from wtc.distilroberta_module import (
    DistilRobertaForMultilabel,
    LengthBucketedBatchSampler,
    MultilabelLoss,
    get_layerwise_optimizer_groups,
)
from wtc.utils import get_logger, set_seed, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Model 3: DistilRoBERTa-base")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test run")
    parser.add_argument("--resume", action="store_true", help="Resume from last.pt checkpoint")
    parser.add_argument("--params", type=str, default="", help="Path to tuned hyperparameters JSON")
    parser.add_argument("--save-seed-probs", action="store_true", help="Save probs_val_seed{seed}.npy and test")
    parser.add_argument("--epochs", type=int, default=0, help="Override number of training epochs")
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


def get_gpu_utilization() -> int | None:
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        return int(res.stdout.strip().split("\n")[0])
    except Exception:
        return None


class ToxicityDataset(Dataset):
    def __init__(self, input_ids_list: list[list[int]], labels: np.ndarray | None = None):
        self.input_ids_list = input_ids_list
        self.labels = labels

    def __len__(self) -> int:
        return len(self.input_ids_list)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {"input_ids": self.input_ids_list[idx]}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


class DynamicPaddingCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        lengths = [len(x["input_ids"]) for x in batch]
        max_len = max(lengths)

        batch_ids = []
        batch_masks = []
        for x in batch:
            ids = x["input_ids"]
            pad_len = max_len - len(ids)
            batch_ids.append(ids + [self.pad_token_id] * pad_len)
            batch_masks.append([1] * len(ids) + [0] * pad_len)

        out = {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_masks, dtype=torch.long),
        }
        if "labels" in batch[0]:
            labels = [x["labels"] for x in batch]
            out["labels"] = torch.tensor(np.array(labels), dtype=torch.float32)

        return out


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    labels: list[str],
) -> tuple[float, float, float, np.ndarray]:
    """Evaluate model on validation loader; returns val_loss, macro_pr_auc, mean_roc_auc, probs."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["labels"].to(device)

            with torch.amp.autocast("cuda"):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, targets)

            total_loss += loss.item()
            n_batches += 1

            probs = torch.sigmoid(logits.float()).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.cpu().numpy())

    val_loss = total_loss / max(1, n_batches)
    probs_np = np.vstack(all_probs)
    targets_np = np.vstack(all_targets)

    pr_aucs = []
    roc_aucs = []
    for i in range(len(labels)):
        y_c = targets_np[:, i]
        p_c = probs_np[:, i]
        if np.sum(y_c) > 0:
            pr_aucs.append(average_precision_score(y_c, p_c))
            try:
                roc_aucs.append(roc_auc_score(y_c, p_c))
            except Exception:
                roc_aucs.append(0.5)
        else:
            pr_aucs.append(0.0)
            roc_aucs.append(0.5)

    macro_pr = float(np.mean(pr_aucs))
    mean_roc = float(np.mean(roc_aucs))
    return val_loss, macro_pr, mean_roc, probs_np


def main() -> None:
    args = parse_args()
    seed = args.seed
    smoke = args.smoke
    resume = args.resume
    params_file = args.params
    set_seed(seed)

    # 1. Hardware validation
    if not torch.cuda.is_available():
        print("[FATAL ERROR] CUDA is NOT available! Aborting execution per spec.")
        sys.exit(1)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    torch_version = torch.__version__

    print("=" * 75)
    print("WELCOME TO CIVILIZATION - MODEL 3: DistilRoBERTa-base Fine-Tuning")
    print(f"Device: {gpu_name} ({vram_gb:.2f} GB VRAM) | PyTorch: {torch_version}")
    print(f"Seed: {seed} | Smoke: {smoke} | Resume: {resume}")
    print("=" * 75)

    model_id = "3_distilroberta"
    model_dir = get_model_dir(model_id, smoke=smoke)
    preds_dir = get_predictions_dir(model_id, smoke=smoke)
    logs_dir = get_logs_dir(model_id, smoke=smoke)

    logger = get_logger(f"train_{model_id}{'_smoke' if smoke else ''}_seed{seed}")
    logger.info(f"Starting training run for {model_id} (smoke={smoke}, seed={seed})")

    status_file = model_dir / "status.json"
    total_start_time = time.perf_counter()
    stage_timings: dict[str, float] = {}

    # Load configuration
    models_cfg = load_models_config()
    m_cfg = models_cfg.get("distilroberta", {})
    t_cfg = m_cfg.get("training", {})
    tok_cfg = m_cfg.get("tokenizer", {})
    imb_cfg = m_cfg.get("imbalance", {})
    head_cfg = m_cfg.get("head", {})

    # Apply tuned parameters if provided
    tuned_params: dict[str, Any] = {}
    if params_file and Path(params_file).exists():
        with open(params_file, "r", encoding="utf-8") as fh:
            tuned_params = json.load(fh)
        logger.info(f"Loaded tuned hyperparameters from {params_file}: {tuned_params}")

    base_lr = float(tuned_params.get("lr", t_cfg.get("lr", 3.0e-5)))
    pos_weight_power = float(tuned_params.get("pos_weight_power", imb_cfg.get("power", 0.45)))
    pos_weight_cap = float(tuned_params.get("pos_weight_cap", imb_cfg.get("max_weight", 14.0)))
    focal_gamma = float(tuned_params.get("focal_gamma", imb_cfg.get("focal_gamma", 0.0)))
    oversample_factor = int(tuned_params.get("oversample_factor", imb_cfg.get("oversample", {}).get("factor", 1)))
    warmup_ratio = float(tuned_params.get("warmup_ratio", t_cfg.get("warmup_ratio", 0.10)))
    weight_decay = float(tuned_params.get("weight_decay", t_cfg.get("weight_decay", 0.01)))
    layerwise_lr_decay = float(tuned_params.get("layerwise_lr_decay", t_cfg.get("layerwise_lr_decay", 0.9)))
    head_dropout = float(tuned_params.get("head_dropout", head_cfg.get("dropout", 0.1)))

    batch_size = int(t_cfg.get("batch_size", 32))
    eval_batch_size = int(t_cfg.get("eval_batch_size", 128))
    grad_clip = float(t_cfg.get("grad_clip", 1.0))
    max_epochs = args.epochs if args.epochs > 0 else int(t_cfg.get("epochs", 4))
    if smoke:
        max_epochs = 1

    max_len = int(tok_cfg.get("max_len", 128))
    pretrained_name = m_cfg.get("pretrained", "distilroberta-base")
    input_col = m_cfg.get("input_col", "text_clean")

    # ------------------------------------------------------------------ #
    #  [1/6] Load & Verify Data                                           #
    # ------------------------------------------------------------------ #
    stage1_start = time.perf_counter()
    print("\n[1/6] Loading and verifying splits...")
    update_status(status_file, "RUNNING", "1/6_load_and_verify")

    manifest = load_split_manifest()
    expected_train = manifest["splits"]["train"]["n_rows"]
    expected_val = manifest["splits"]["val"]["n_rows"]
    expected_test = manifest["splits"]["test"]["n_rows"]

    df_train = load_split("train")
    df_val = load_split("val")
    df_test = load_split("test")

    if not smoke:
        assert len(df_train) == expected_train, f"Train size mismatch: {len(df_train)} vs {expected_train}"
        assert len(df_val) == expected_val, f"Val size mismatch: {len(df_val)} vs {expected_val}"
        assert len(df_test) == expected_test, f"Test size mismatch: {len(df_test)} vs {expected_test}"
    else:
        df_train = df_train.iloc[:1000].reset_index(drop=True)
        df_val = df_val.iloc[:500].reset_index(drop=True)
        df_test = df_test.iloc[:500].reset_index(drop=True)

    print(f"  Verified splits: Train={len(df_train):,}, Val={len(df_val):,}, Test={len(df_test):,}")
    stage_timings["1_load_verify"] = time.perf_counter() - stage1_start

    # ------------------------------------------------------------------ #
    #  [2/6] Tokenize                                                     #
    # ------------------------------------------------------------------ #
    stage2_start = time.perf_counter()
    print(f"\n[2/6] Tokenizing with {pretrained_name} (max_len={max_len}, casing preserved)...")
    update_status(status_file, "RUNNING", "2/6_tokenize")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    train_texts = df_train[input_col].fillna("").tolist()
    val_texts = df_val[input_col].fillna("").tolist()
    test_texts = df_test[input_col].fillna("").tolist()

    t_enc_start = time.perf_counter()
    train_enc = tokenizer(train_texts, truncation=True, max_length=max_len, padding=False, return_attention_mask=False)
    val_enc = tokenizer(val_texts, truncation=True, max_length=max_len, padding=False, return_attention_mask=False)
    test_enc = tokenizer(test_texts, truncation=True, max_length=max_len, padding=False, return_attention_mask=False)

    train_ids = train_enc["input_ids"]
    val_ids = val_enc["input_ids"]
    test_ids = test_enc["input_ids"]

    train_lengths = [len(x) for x in train_ids]
    val_lengths = [len(x) for x in val_ids]
    test_lengths = [len(x) for x in test_ids]

    trunc_train = sum(1 for l in train_lengths if l >= max_len) / len(train_lengths) * 100
    trunc_val = sum(1 for l in val_lengths if l >= max_len) / len(val_lengths) * 100
    trunc_test = sum(1 for l in test_lengths if l >= max_len) / len(test_lengths) * 100

    print(f"  Tokenization complete in {time.perf_counter() - t_enc_start:.1f}s.")
    print(f"  Truncation rates (>= {max_len} tokens): Train={trunc_train:.1f}%, Val={trunc_val:.1f}%, Test={trunc_test:.1f}%")
    stage_timings["2_tokenize"] = time.perf_counter() - stage2_start

    # ------------------------------------------------------------------ #
    #  [3/6] Setup Model, Loss & Optimizer                                #
    # ------------------------------------------------------------------ #
    stage3_start = time.perf_counter()
    print("\n[3/6] Setting up model, damped pos_weight BCE, and LLRD AdamW...")
    update_status(status_file, "RUNNING", "3/6_setup_model")

    Y_train = df_train[LABELS].values.astype(np.float32)
    Y_val = df_val[LABELS].values.astype(np.float32)
    Y_test = df_test[LABELS].values.astype(np.float32)

    # Calculate damped positive class weights
    n_samples = len(Y_train)
    pos_counts = Y_train.sum(axis=0)
    neg_counts = n_samples - pos_counts

    pos_weights_arr = np.zeros(len(LABELS), dtype=np.float32)
    for i, label in enumerate(LABELS):
        pos = max(1, pos_counts[i])
        neg = max(1, neg_counts[i])
        # If oversampling rare labels, adjust pos counts to prevent double-counting
        if oversample_factor > 1 and label in ["threat", "severe_toxic", "identity_hate"]:
            eff_pos = pos * oversample_factor
            w = (neg / eff_pos) ** pos_weight_power
        else:
            w = (neg / pos) ** pos_weight_power
        pos_weights_arr[i] = min(w, pos_weight_cap)

    print(f"  Computed pos_weight per label (power={pos_weight_power}, cap={pos_weight_cap}):")
    for l, w in zip(LABELS, pos_weights_arr):
        print(f"    - {l:14s}: {w:.3f}")

    pos_weight_tensor = torch.tensor(pos_weights_arr, device=device)
    criterion = MultilabelLoss(pos_weight=pos_weight_tensor, focal_gamma=focal_gamma)

    # Initialize model
    model = DistilRobertaForMultilabel(
        pretrained_name=pretrained_name,
        num_classes=len(LABELS),
        dropout=head_dropout,
    )
    # Log-odds bias initialization on classifier head
    model.classifier.init_bias_log_odds(pos_counts, n_samples)
    model.to(device)

    # Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model Parameters: {trainable_params:,} trainable / {total_params:,} total (~{total_params/1e6:.1f}M)")

    # Optimizer with Layer-wise LR Decay (LLRD)
    optimizer_groups = get_layerwise_optimizer_groups(
        model=model,
        base_lr=base_lr,
        weight_decay=weight_decay,
        layerwise_lr_decay=layerwise_lr_decay,
    )
    optimizer = torch.optim.AdamW(optimizer_groups)

    # Data loaders
    collator = DynamicPaddingCollator(pad_token_id=tokenizer.pad_token_id)
    train_dataset = ToxicityDataset(train_ids, Y_train)
    val_dataset = ToxicityDataset(val_ids, Y_val)
    test_dataset = ToxicityDataset(test_ids, Y_test)

    # Length-bucketed train sampler
    train_sampler = LengthBucketedBatchSampler(
        lengths=train_lengths,
        batch_size=batch_size,
        shuffle=True,
        bucket_size_multiplier=20,
        seed=seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collator,
        num_workers=0,
    )

    # Validation & Test: length-sorted batches for fast evaluation
    val_sorted_idx = np.argsort(val_lengths)
    val_sorted_ids = [val_ids[i] for i in val_sorted_idx]
    val_sorted_y = Y_val[val_sorted_idx]
    val_sorted_dataset = ToxicityDataset(val_sorted_ids, val_sorted_y)
    val_loader = DataLoader(
        val_sorted_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    test_sorted_idx = np.argsort(test_lengths)
    test_sorted_ids = [test_ids[i] for i in test_sorted_idx]
    test_sorted_y = Y_test[test_sorted_idx]
    test_sorted_dataset = ToxicityDataset(test_sorted_ids, test_sorted_y)
    test_loader = DataLoader(
        test_sorted_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    total_training_steps = len(train_loader) * max_epochs
    warmup_steps = int(total_training_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )
    scaler = torch.amp.GradScaler("cuda")

    print(f"  Training Steps: {total_training_steps:,} total across {max_epochs} epochs ({warmup_steps} warmup steps).")
    stage_timings["3_setup_model"] = time.perf_counter() - stage3_start

    # ------------------------------------------------------------------ #
    #  [4/6] Training Loop                                                #
    # ------------------------------------------------------------------ #
    stage4_start = time.perf_counter()
    print("\n[4/6] Starting training loop with live progress by epochs...")
    update_status(status_file, "RUNNING", "4/6_training")

    best_pt = model_dir / "best_model.pt"
    last_pt = model_dir / "last.pt"

    start_epoch = 1
    best_val_metric = -1.0
    best_epoch = 1
    history: list[dict[str, Any]] = []

    if resume and last_pt.exists():
        checkpoint = torch.load(last_pt, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if "scheduler_state_dict" in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_metric = checkpoint["best_metric"]
        best_epoch = checkpoint["best_epoch"]
        history = checkpoint.get("history", [])
        print(f"  Resumed from epoch {checkpoint['epoch']} (Best Val Macro PR-AUC: {best_val_metric:.4f} at Ep {best_epoch})")

    steps_per_epoch = len(train_loader)
    log_every_n_steps = max(20, steps_per_epoch // 20)

    for epoch in range(start_epoch, max_epochs + 1):
        ep_t0 = time.perf_counter()
        train_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        n_samples_processed = 0

        print(f"\n--- Epoch {epoch}/{max_epochs} (Seed {seed}) ---")

        for step, batch in enumerate(train_loader, 1):
            t_step_start = time.perf_counter()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["labels"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            bs = input_ids.size(0)
            n_samples_processed += bs
            running_loss += loss.item() * bs

            if step % log_every_n_steps == 0 or step == steps_per_epoch:
                elapsed_ep = time.perf_counter() - ep_t0
                current_rate = n_samples_processed / max(0.1, elapsed_ep)
                eta_sec = (steps_per_epoch - step) * (elapsed_ep / step)
                avg_loss = running_loss / max(1, n_samples_processed)
                current_lr = scheduler.get_last_lr()[0]
                peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)

                sys.stdout.write(
                    f"\rEpoch {epoch}/{max_epochs} [{step:4d}/{steps_per_epoch:4d}] "
                    f"Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | "
                    f"Speed: {current_rate:.0f} spl/s | VRAM: {peak_vram:.0f}MB | "
                    f"ETA: {eta_sec/60:.1f}m"
                )
                sys.stdout.flush()

        print()  # Newline after progress bar
        train_loss = running_loss / max(1, len(df_train))

        # Evaluate on validation
        val_loss, val_macro_pr, val_mean_roc, _ = evaluate_model(model, val_loader, criterion, device, LABELS)

        ep_duration = time.perf_counter() - ep_t0
        samples_per_sec = len(df_train) / ep_duration
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        gpu_util = get_gpu_utilization()

        is_best = False
        if val_macro_pr > best_val_metric:
            best_val_metric = val_macro_pr
            best_epoch = epoch
            is_best = True
            torch.save(model.state_dict(), best_pt)

        ep_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_macro_pr_auc": round(val_macro_pr, 4),
            "val_mean_roc_auc": round(val_mean_roc, 4),
            "epoch_time_seconds": round(ep_duration, 2),
            "samples_per_sec": round(samples_per_sec, 1),
            "peak_memory_mb": round(peak_vram_mb, 1),
            "gpu_util_pct": gpu_util,
            "learning_rate": scheduler.get_last_lr()[0],
            "is_best": is_best,
        }
        history.append(ep_record)

        summary_line = (
            f"Epoch {epoch}/{max_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Macro PR-AUC: {val_macro_pr:.4f}{' (*BEST*)' if is_best else ''} | "
            f"Val ROC-AUC: {val_mean_roc:.4f} | Time: {ep_duration:.1f}s ({samples_per_sec:.0f} spl/s) | "
            f"VRAM: {peak_vram_mb:.0f}MB"
        )
        print(f"  {summary_line}")
        logger.info(summary_line)

        # Save last checkpoint for resumption
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_metric": best_val_metric,
                "best_epoch": best_epoch,
                "history": history,
            },
            last_pt,
        )

        update_status(
            status_file,
            "RUNNING",
            "4/6_training",
            {
                "current_epoch": epoch,
                "best_epoch": best_epoch,
                "best_val_macro_pr_auc": round(best_val_metric, 4),
                "elapsed_so_far": round(time.perf_counter() - total_start_time, 2),
            },
        )

    stage_timings["4_training"] = time.perf_counter() - stage4_start
    print(f"\nTraining completed. Restoring best model from Epoch {best_epoch} (Val Macro PR-AUC: {best_val_metric:.4f})...")
    model.load_state_dict(torch.load(best_pt, map_location=device, weights_only=True))

    # ------------------------------------------------------------------ #
    #  [5/6] Predict Val & Test with Best Model                           #
    # ------------------------------------------------------------------ #
    stage5_start = time.perf_counter()
    print("\n[5/6] Generating sigmoid probability predictions for Val & Test...")
    update_status(status_file, "RUNNING", "5/6_predict")

    # Evaluate validation
    val_loss, val_macro_pr, val_mean_roc, probs_val_sorted = evaluate_model(model, val_loader, criterion, device, LABELS)
    # Restore original unsorted order
    probs_val = np.zeros_like(probs_val_sorted)
    probs_val[val_sorted_idx] = probs_val_sorted

    # Evaluate test
    _, test_macro_pr, test_mean_roc, probs_test_sorted = evaluate_model(model, test_loader, criterion, device, LABELS)
    probs_test = np.zeros_like(probs_test_sorted)
    probs_test[test_sorted_idx] = probs_test_sorted

    # Save probabilities
    p_val_file = preds_dir / "probs_val.npy"
    p_test_file = preds_dir / "probs_test.npy"
    np.save(p_val_file, probs_val.astype(np.float32))
    np.save(p_test_file, probs_test.astype(np.float32))

    if args.save_seed_probs:
        np.save(preds_dir / f"probs_val_seed{seed}.npy", probs_val.astype(np.float32))
        np.save(preds_dir / f"probs_test_seed{seed}.npy", probs_test.astype(np.float32))

    print(f"  Saved probs_val.npy shape {probs_val.shape} and probs_test.npy shape {probs_test.shape}")

    # Hierarchy fix ablation on validation set
    tox_i = LABELS.index("toxic")
    sev_i = LABELS.index("severe_toxic")
    probs_val_hier = probs_val.copy()
    probs_val_hier[:, tox_i] = np.maximum(probs_val_hier[:, tox_i], probs_val_hier[:, sev_i])

    hier_pr_aucs = [average_precision_score(Y_val[:, i], probs_val_hier[:, i]) for i in range(len(LABELS))]
    val_hier_macro_pr = float(np.mean(hier_pr_aucs))

    print(f"  Hierarchy Fix Ablation on Val:")
    print(f"    - Standard:   Val Macro PR-AUC = {val_macro_pr:.4f}")
    print(f"    - Hierarchy:  Val Macro PR-AUC = {val_hier_macro_pr:.4f} (Delta: {val_hier_macro_pr - val_macro_pr:+.4f})")
    use_hierarchy_serving = (val_hier_macro_pr > val_macro_pr)
    print(f"    - Serving Choice on Val: {'ENABLE hierarchy fix' if use_hierarchy_serving else 'KEEP standard (no hierarchy fix)'}")

    stage_timings["5_predict"] = time.perf_counter() - stage5_start

    # ------------------------------------------------------------------ #
    #  [6/6] Save Backend-Ready Bundle & Parity Check                     #
    # ------------------------------------------------------------------ #
    stage6_start = time.perf_counter()
    print("\n[6/6] Saving backend-ready bundle (safetensors, tokenizer, meta.json)...")
    update_status(status_file, "RUNNING", "6/6_save_bundle")

    # 1. Save safetensors
    safetensors_file = model_dir / "model.safetensors"
    state_dict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    save_file(state_dict_cpu, str(safetensors_file))
    weights_sha256 = sha256_file(safetensors_file)
    weights_size_mb = round(safetensors_file.stat().st_size / (1024 * 1024), 2)
    print(f"  Saved {safetensors_file.name} ({weights_size_mb} MB, SHA-256: {weights_sha256[:16]}...)")

    # 2. Save tokenizer assets
    tokenizer.save_pretrained(str(model_dir))
    print(f"  Saved tokenizer assets to {model_dir}")

    # 3. Save history.json
    history_file = model_dir / "history.json"
    with open(history_file, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)

    # 4. Save meta.json
    meta_payload = {
        "model_id": model_id,
        "model_name": "DistilRoBERTa-base",
        "adapter_type": "transformer",
        "smoke": smoke,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "labels": LABELS,
        "input_column": input_col,
        "clean_text_version": "v1_nfkc_clean",
        "config": m_cfg,
        "parameters": {
            "trainable": trainable_params,
            "total": total_params,
        },
        "best_epoch": best_epoch,
        "best_val_macro_pr_auc": round(best_val_metric, 4),
        "weights_sha256": weights_sha256,
        "model_size_mb": weights_size_mb,
        "hierarchy_fix": use_hierarchy_serving,
        "hierarchy_ablation": {
            "val_macro_pr_auc_standard": round(val_macro_pr, 4),
            "val_macro_pr_auc_hierarchy": round(val_hier_macro_pr, 4),
            "delta": round(val_hier_macro_pr - val_macro_pr, 4),
            "serving_choice": "hierarchy" if use_hierarchy_serving else "standard",
        },
        "artifacts": {
            "weights": str(safetensors_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "probs_val": str(p_val_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "probs_test": str(p_test_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "history": str(history_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
        "training_history": history,
        "tuning_summary": {
            "params_file": params_file,
            "best_params": tuned_params,
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda if torch.cuda.is_available() else "None",
            "transformers": "4.45.2",
            "safetensors": "0.8.0",
        },
        "stage_timings": stage_timings,
        "total_train_time_sec": round(time.perf_counter() - total_start_time, 2),
    }

    meta_file = model_dir / "meta.json"
    with open(meta_file, "w", encoding="utf-8") as fh:
        json.dump(meta_payload, fh, indent=2)
    print(f"  Saved metadata to {meta_file}")

    # 5. Parity Check via Moderator
    print("\n[Parity Check] Verifying Moderator inference parity on test split...")
    from wtc.inference import Moderator

    moderator = Moderator(model_id=model_id, smoke=smoke)
    # Use raw un-hierarchy-fixed adapter check or moderator predict_proba
    moderator_preds = moderator.predict_proba(test_texts)
    # Compare with saved test probs (accounting for hierarchy fix if active in moderator)
    expected_test_probs = probs_test.copy()
    if use_hierarchy_serving:
        expected_test_probs[:, tox_i] = np.maximum(expected_test_probs[:, tox_i], expected_test_probs[:, sev_i])

    max_diff = float(np.max(np.abs(moderator_preds - expected_test_probs)))
    print(f"  Max absolute prediction difference: {max_diff:.2e}")
    if max_diff > 1e-4:
        print(f"[FATAL ERROR] Parity check failed! Max diff {max_diff:.2e} > 1e-4. Aborting!")
        sys.exit(1)
    print("  [SUCCESS] Parity check passed: Moderator output matches saved test probabilities within 1e-4.")

    stage_timings["6_save_bundle"] = time.perf_counter() - stage6_start
    total_elapsed = time.perf_counter() - total_start_time

    update_status(
        status_file,
        "COMPLETED",
        "DONE",
        {
            "best_epoch": best_epoch,
            "best_val_macro_pr_auc": round(best_val_metric, 4),
            "total_elapsed_seconds": round(total_elapsed, 2),
        },
    )

    print("\n" + "=" * 75)
    print(f"MODEL 3 TRAINING COMPLETE ({total_elapsed/60:.2f} min total)")
    print(f"Best Checkpoint: Epoch {best_epoch} (Val Macro PR-AUC = {best_val_metric:.4f})")
    print(f"Weights: {safetensors_file} ({weights_size_mb} MB)")
    print("=" * 75)


if __name__ == "__main__":
    main()
