#!/usr/bin/env python
"""
scripts/14_tune_distilroberta.py
---------------------------------
Hyperparameter tuning for Model 3: DistilRoBERTa-base fine-tuning.
Optuna TPE sampler + MedianPruner with SQLite storage under runs/.
Each trial: 40% multilabel-stratified train subsample, 2 epochs.
Objective: mean(val_macro_pr_auc, mean val PR-AUC of [threat, severe_toxic, identity_hate]).
Search space:
  - lr: 2e-5..6e-5 (log)
  - pos_weight_power: 0.3..0.6 (float)
  - pos_weight_cap: {8, 14, 20} (categorical)
  - focal_gamma: {0, 1, 2} (categorical)
  - oversample_factor: {1, 2, 4} (categorical)
  - warmup_ratio: 0.06..0.15 (float)
  - weight_decay: {0.01, 0.1} (categorical)
  - layerwise_lr_decay: {1.0, 0.9} (categorical)
  - head_dropout: {0.1, 0.2} (categorical)
Fixed: batch 32, max_len 128.
Writes best parameters to runs/distilroberta_best_params.json.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

# Ensure project root src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from wtc.config import CFG, LABELS, PROJECT_ROOT, load_models_config
from wtc.data import load_split
from wtc.distilroberta_module import (
    DistilRobertaForMultilabel,
    LengthBucketedBatchSampler,
    MultilabelLoss,
    get_layerwise_optimizer_groups,
)
from wtc.utils import get_logger, set_seed


class FastDataset(Dataset):
    def __init__(self, ids: list[list[int]], targets: np.ndarray):
        self.ids = ids
        self.targets = targets

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {"input_ids": self.ids[idx], "labels": self.targets[idx]}


class FastCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        lengths = [len(x["input_ids"]) for x in batch]
        max_l = max(lengths)
        batch_ids = []
        batch_masks = []
        for x in batch:
            ids = x["input_ids"]
            pad = max_l - len(ids)
            batch_ids.append(ids + [self.pad_id] * pad)
            batch_masks.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_masks, dtype=torch.long),
            "labels": torch.tensor(np.array([x["labels"] for x in batch]), dtype=torch.float32),
        }


def parse_args() -> argparse.Namespace:
    models_cfg = load_models_config().get("3_distilroberta", {})
    cfg_trials = models_cfg.get("tune", {}).get("n_trials", 5)
    parser = argparse.ArgumentParser(description="Tune Model 3 DistilRoBERTa hyperparameters")
    parser.add_argument("--n-trials", type=int, default=cfg_trials, help=f"Total tuning trials (default {cfg_trials} from config)")
    parser.add_argument("--quick", action="store_true", help="Quick mode (4 trials)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling and tuning")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed = args.seed
    models_cfg = load_models_config().get("3_distilroberta", {})
    cfg_trials = models_cfg.get("tune", {}).get("n_trials", 5)
    n_trials = 4 if args.quick else (args.n_trials or cfg_trials)
    set_seed(seed)

    if not torch.cuda.is_available():
        print("[FATAL ERROR] CUDA is required for tuning! Aborting.")
        sys.exit(1)

    device = torch.device("cuda")
    print("=" * 75)
    print("OPTUNA HYPERPARAMETER TUNING: DistilRoBERTa-base")
    print(f"Device: {torch.cuda.get_device_name(0)} | Trials: {n_trials} | Seed: {seed}")
    print("=" * 75)

    runs_dir = PROJECT_ROOT / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    db_path = runs_dir / "distilroberta_tune.db"
    storage_url = f"sqlite:///{db_path.resolve()}"

    # 1. Load splits
    df_train_full = load_split("train")
    df_val = load_split("val")

    # 40% multilabel-stratified subsample of train
    Y_full = df_train_full[LABELS].values.astype(np.float32)
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.6, train_size=0.4, random_state=seed)
    train_sub_idx, _ = next(msss.split(np.zeros(len(df_train_full)), Y_full))

    df_train = df_train_full.iloc[train_sub_idx].reset_index(drop=True)
    Y_train = df_train[LABELS].values.astype(np.float32)
    Y_val = df_val[LABELS].values.astype(np.float32)

    print(f"Tuning dataset: Train (40% subsample) = {len(df_train):,} rows | Val = {len(df_val):,} rows")

    # 2. Pre-tokenize
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    max_len = 128

    print("Pre-tokenizing train subsample and validation texts...")
    train_texts = df_train["text_clean"].fillna("").tolist()
    val_texts = df_val["text_clean"].fillna("").tolist()

    train_enc = tokenizer(train_texts, truncation=True, max_length=max_len, padding=False, return_attention_mask=False)
    val_enc = tokenizer(val_texts, truncation=True, max_length=max_len, padding=False, return_attention_mask=False)

    train_ids = train_enc["input_ids"]
    val_ids = val_enc["input_ids"]

    train_lengths = [len(x) for x in train_ids]
    val_lengths = [len(x) for x in val_ids]

    val_sorted_idx = np.argsort(val_lengths)
    val_sorted_ids = [val_ids[i] for i in val_sorted_idx]
    val_sorted_y = Y_val[val_sorted_idx]

    collator = FastCollator(pad_id=tokenizer.pad_token_id)
    val_dataset = FastDataset(val_sorted_ids, val_sorted_y)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=collator, num_workers=0)

    rare_labels = ["threat", "severe_toxic", "identity_hate"]
    rare_indices = [LABELS.index(l) for l in rare_labels]

    def objective(trial: optuna.Trial) -> float:
        # Sample hyperparameters from defined search space
        lr = trial.suggest_float("lr", 2.0e-5, 6.0e-5, log=True)
        pos_weight_power = trial.suggest_float("pos_weight_power", 0.3, 0.6)
        pos_weight_cap = trial.suggest_categorical("pos_weight_cap", [8.0, 14.0, 20.0])
        focal_gamma = trial.suggest_categorical("focal_gamma", [0.0, 1.0, 2.0])
        oversample_factor = trial.suggest_categorical("oversample_factor", [1, 2, 4])
        warmup_ratio = trial.suggest_float("warmup_ratio", 0.06, 0.15)
        weight_decay = trial.suggest_categorical("weight_decay", [0.01, 0.1])
        layerwise_lr_decay = trial.suggest_categorical("layerwise_lr_decay", [1.0, 0.9])
        head_dropout = trial.suggest_categorical("head_dropout", [0.1, 0.2])

        # Compute pos_weights
        pos_counts = Y_train.sum(axis=0)
        neg_counts = len(Y_train) - pos_counts
        pos_w = np.zeros(len(LABELS), dtype=np.float32)
        for i, l in enumerate(LABELS):
            pos = max(1, pos_counts[i])
            neg = max(1, neg_counts[i])
            if oversample_factor > 1 and l in rare_labels:
                eff_pos = pos * oversample_factor
                w = (neg / eff_pos) ** pos_weight_power
            else:
                w = (neg / pos) ** pos_weight_power
            pos_w[i] = min(w, pos_weight_cap)

        criterion = MultilabelLoss(
            pos_weight=torch.tensor(pos_w, device=device),
            focal_gamma=focal_gamma,
        )

        model = DistilRobertaForMultilabel(
            pretrained_name="distilroberta-base",
            num_classes=len(LABELS),
            dropout=head_dropout,
        )
        model.classifier.init_bias_log_odds(pos_counts, len(Y_train))
        model.to(device)

        opt_groups = get_layerwise_optimizer_groups(
            model=model,
            base_lr=lr,
            weight_decay=weight_decay,
            layerwise_lr_decay=layerwise_lr_decay,
        )
        optimizer = torch.optim.AdamW(opt_groups)

        batch_size = 32
        train_sampler = LengthBucketedBatchSampler(
            lengths=train_lengths,
            batch_size=batch_size,
            shuffle=True,
            bucket_size_multiplier=20,
            seed=seed + trial.number,
        )
        train_dataset = FastDataset(train_ids, Y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=collator,
            num_workers=0,
        )

        epochs = 2
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        scaler = torch.amp.GradScaler("cuda")

        best_score = -1.0

        for epoch in range(1, epochs + 1):
            train_sampler.set_epoch(epoch)
            model.train()

            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["labels"].to(device)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda"):
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, targets)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            # Validation
            model.eval()
            all_p, all_y = [], []
            with torch.no_grad():
                for batch in val_loader:
                    ids = batch["input_ids"].to(device)
                    mask = batch["attention_mask"].to(device)
                    y = batch["labels"]
                    with torch.amp.autocast("cuda"):
                        lg = model(ids, mask)
                    all_p.append(torch.sigmoid(lg.float()).cpu().numpy())
                    all_y.append(y.numpy())

            p_val = np.vstack(all_p)
            y_val = np.vstack(all_y)

            pr_aucs = [average_precision_score(y_val[:, i], p_val[:, i]) for i in range(len(LABELS))]
            macro_pr = float(np.mean(pr_aucs))
            rare_pr = float(np.mean([pr_aucs[i] for i in rare_indices]))
            trial_score = (macro_pr + rare_pr) / 2.0

            trial.report(trial_score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if trial_score > best_score:
                best_score = trial_score

        # Cleanup memory
        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()
        return best_score

    # Setup study
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
    study = optuna.create_study(
        study_name="distilroberta_hparam_tune",
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining_trials = max(0, n_trials - completed_trials)
    print(f"Study loaded from {db_path}: {completed_trials} already completed, {remaining_trials} remaining.")

    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials, show_progress_bar=True)

    print("\n" + "=" * 75)
    print("TUNING COMPLETE")
    print(f"Best Trial #{study.best_trial.number}: Objective = {study.best_value:.4f}")
    print("Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  - {k}: {v}")
    print("=" * 75)

    # Save best parameters to JSON
    best_params_file = runs_dir / "distilroberta_best_params.json"
    with open(best_params_file, "w", encoding="utf-8") as fh:
        json.dump(study.best_params, fh, indent=2)
    print(f"Saved best parameters to {best_params_file}")

    # Export all trials to CSV
    df_trials = study.trials_dataframe()
    df_trials.to_csv(runs_dir / "distilroberta_tune_results.csv", index=False)
    print(f"Saved tuning trials log to {runs_dir / 'distilroberta_tune_results.csv'}")


if __name__ == "__main__":
    main()
