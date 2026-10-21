#!/usr/bin/env python
"""
scripts/11_train_bigru.py
-------------------------
Train Model 2: BiGRU + Additive Attention with frozen fastText embeddings.
Supports --smoke and --resume.
Stages:
  [1/6] load+verify
  [2/6] tokenize+vocab
  [3/6] fastText matrix (cached)
  [4/6] move data to GPU and build the model
  [5/6] training epochs (length-bucketed batches, AMP, early stopping on val macro PR-AUC)
  [6/6] load best, predict val/test, save
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import zipfile

# Ensure project root src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn as nn
from tqdm import tqdm

from wtc.bigru_module import BiGruAttentionModel
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
    parser = argparse.ArgumentParser(description="Train Model 2: BiGRU + Additive Attention")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test with reduced data subset")
    parser.add_argument("--resume", action="store_true", help="Resume training from last.pt checkpoint")
    parser.add_argument("--model-name", type=str, default="bigru_attn", help="Model key in configs/models.yaml")
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
    """Query current GPU utilization percentage via nvidia-smi."""
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


def build_vocab(train_texts: list[str], max_vocab: int = 50000, min_freq: int = 2) -> tuple[dict[str, int], dict[str, Any]]:
    """Build vocabulary from lowercased train comments."""
    tokenizer_regex = re.compile(r"\w+|[^\w\s]")
    counts: dict[str, int] = {}
    total_tokens = 0

    for text in train_texts:
        tokens = tokenizer_regex.findall(text.lower())
        total_tokens += len(tokens)
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1

    # Filter by min_freq and sort by frequency descending
    filtered = [(tok, cnt) for tok, cnt in counts.items() if cnt >= min_freq]
    filtered.sort(key=lambda x: x[1], reverse=True)

    top_tokens = filtered[:max_vocab]
    vocab_tokens_covered = sum(cnt for _, cnt in top_tokens)
    token_coverage = (vocab_tokens_covered / total_tokens) * 100.0 if total_tokens > 0 else 0.0

    word2idx: dict[str, int] = {"<pad>": 0, "<unk>": 1}
    for idx, (tok, _) in enumerate(top_tokens, start=2):
        word2idx[tok] = idx

    stats = {
        "total_unique_tokens": len(counts),
        "total_token_occurrences": total_tokens,
        "vocab_size": len(word2idx),
        "token_coverage_pct": round(token_coverage, 2),
        "pad_idx": 0,
        "unk_idx": 1,
    }
    return word2idx, stats


def get_fasttext_embeddings(
    word2idx: dict[str, int],
    cache_path: Path,
    fasttext_zip: Path,
    embedding_dim: int = 300,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load or build fastText embedding matrix."""
    vocab_hash = hashlib.sha256(json.dumps(word2idx, sort_keys=True).encode("utf-8")).hexdigest()
    meta_cache_path = cache_path.parent / "embedding_meta.json"

    # Check cache
    if cache_path.exists() and meta_cache_path.exists():
        try:
            with open(meta_cache_path, "r", encoding="utf-8") as fh:
                cached_meta = json.load(fh)
            if cached_meta.get("vocab_hash") == vocab_hash:
                matrix = np.load(cache_path)
                return matrix, cached_meta
        except Exception:
            pass

    if not fasttext_zip.exists():
        err_msg = (
            f"fastText zip file not found at {fasttext_zip}!\n"
            "Please download crawl-300d-2M.vec.zip from:\n"
            "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip\n"
            "and place it under external/fasttext/"
        )
        raise FileNotFoundError(err_msg)

    print(f"  Streaming fastText vectors directly from {fasttext_zip.name}...")
    vocab_set = set(word2idx.keys())
    matrix = np.zeros((len(word2idx), embedding_dim), dtype=np.float32)
    found_words = set()

    with zipfile.ZipFile(fasttext_zip, "r") as zf:
        vec_names = [n for n in zf.namelist() if n.endswith(".vec")]
        if not vec_names:
            raise ValueError(f"No .vec file found inside {fasttext_zip}")
        with zf.open(vec_names[0], "r") as fh:
            # Read header
            header = fh.readline().decode("utf-8").strip().split()
            total_lines = int(header[0])

            with tqdm(total=total_lines, desc="  fastText stream", unit="vecs", leave=False) as pbar:
                for line in fh:
                    pbar.update(1)
                    line_str = line.decode("utf-8", errors="ignore").rstrip()
                    space_idx = line_str.find(" ")
                    if space_idx == -1:
                        continue
                    word = line_str[:space_idx]
                    if word in vocab_set:
                        vec_parts = line_str[space_idx + 1 :].split(" ")
                        if len(vec_parts) == embedding_dim:
                            vec = np.array(vec_parts, dtype=np.float32)
                            matrix[word2idx[word]] = vec
                            found_words.add(word)

    # Impute missing vectors and <unk>
    found_indices = [word2idx[w] for w in found_words]
    if found_indices:
        found_vecs = matrix[found_indices]
        mean_vec = np.mean(found_vecs, axis=0)
        std_vec = np.std(found_vecs, axis=0)
    else:
        mean_vec = np.zeros(embedding_dim, dtype=np.float32)
        std_vec = np.ones(embedding_dim, dtype=np.float32)

    rng = np.random.default_rng(seed)
    missing_count = 0
    for word, idx in word2idx.items():
        if idx == 0:  # <pad>
            matrix[idx] = 0.0
        elif word not in found_words:
            matrix[idx] = rng.normal(mean_vec, std_vec).astype(np.float32)
            missing_count += 1

    type_hit_rate = (len(found_words) / (len(word2idx) - 2)) * 100.0 if len(word2idx) > 2 else 0.0

    fasttext_meta = {
        "vocab_hash": vocab_hash,
        "fasttext_source": str(fasttext_zip.name),
        "total_vocab": len(word2idx),
        "found_words": len(found_words),
        "missing_words": missing_count,
        "type_hit_rate_pct": round(type_hit_rate, 2),
        "embedding_dim": embedding_dim,
    }

    # Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, matrix)
    with open(meta_cache_path, "w", encoding="utf-8") as fh:
        json.dump(fasttext_meta, fh, indent=2)

    return matrix, fasttext_meta


def main() -> None:
    args = parse_args()
    smoke = args.smoke
    resume = args.resume
    model_name = args.model_name
    model_id = get_model_id(model_name)

    models_cfg = load_models_config()
    if model_name not in models_cfg:
        raise KeyError(f"Model '{model_name}' not found in configs/models.yaml")
    m_cfg = models_cfg[model_name]

    seed = int(m_cfg.get("seed", 42))
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if not torch.cuda.is_available():
        print("\nERROR: CUDA GPU is not available! Model 2 requires an NVIDIA GPU for execution.")
        sys.exit(1)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    model_dir = get_model_dir(model_id, smoke=smoke)
    preds_dir = get_predictions_dir(model_id, smoke=smoke)
    logs_dir = get_logs_dir(model_id, smoke=smoke)
    status_file = model_dir / "status.json"

    logger = get_logger(f"train_{model_id}{'_smoke' if smoke else ''}", log_dir=logs_dir)
    logger.info("=" * 70)
    logger.info(f"Starting Training for Model: {model_id} (smoke={smoke}, resume={resume})")
    logger.info(f"Device: {gpu_name} ({total_vram_gb:.2f} GB VRAM) | PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")
    logger.info("=" * 70)

    total_start_time = time.perf_counter()
    stage_timings: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    #  [1/6] Load and Verify Data                                         #
    # ------------------------------------------------------------------ #
    stage1_start = time.perf_counter()
    print("\n[1/6] Loading and verifying data splits...")
    logger.info("[1/6] Loading and verifying data splits...")
    update_status(status_file, "RUNNING", "1/6_load_verify")

    manifest = load_split_manifest()
    split_version = manifest.get("split_version", "")
    if split_version != "v2_near_dup":
        raise ValueError(f"Data verification failed: expected split_version 'v2_near_dup', got '{split_version}'")

    processed_parquet_path = PROJECT_ROOT / "data" / "processed" / "processed.parquet"
    parquet_sha = sha256_file(processed_parquet_path)
    expected_sha = manifest.get("processed_parquet_sha256", "")
    if parquet_sha != expected_sha:
        raise ValueError(f"SHA256 mismatch for processed.parquet!\nExpected: {expected_sha}\nFound: {parquet_sha}")

    df_train_full = load_split("train")
    df_val = load_split("val")
    df_test = load_split("test")

    input_col = m_cfg.get("input_col", "text_clean")

    # In smoke mode, train is subsampled for training, but vocab is built from full train split
    if smoke:
        logger.info("Smoke mode enabled: subsampling train to 2,000, val to 500, test to 500 rows for training")
        df_train_sub = df_train_full.iloc[:2000].reset_index(drop=True)
        df_val = df_val.iloc[:500].reset_index(drop=True)
        df_test = df_test.iloc[:500].reset_index(drop=True)
    else:
        df_train_sub = df_train_full

    stage1_elapsed = time.perf_counter() - stage1_start
    stage_timings["1_load_verify"] = stage1_elapsed
    print(f"  Done in {stage1_elapsed:.2f}s (Train: {len(df_train_sub):,}, Val: {len(df_val):,}, Test: {len(df_test):,})")

    # ------------------------------------------------------------------ #
    #  [2/6] Tokenize and Build Vocabulary                                #
    # ------------------------------------------------------------------ #
    stage2_start = time.perf_counter()
    print("\n[2/6] Tokenizing text and building vocabulary (from full train split)...")
    logger.info("[2/6] Tokenizing text and building vocabulary (from full train split)...")
    update_status(status_file, "RUNNING", "2/6_tokenize_vocab")

    v_cfg = m_cfg.get("vocab", {})
    max_vocab = int(v_cfg.get("max_vocab", 50000))
    min_freq = int(v_cfg.get("min_freq", 2))
    max_len = int(m_cfg.get("tokenizer", {}).get("max_len", 256))

    full_train_texts = df_train_full[input_col].fillna("").tolist()
    word2idx, vocab_stats = build_vocab(full_train_texts, max_vocab=max_vocab, min_freq=min_freq)
    pad_idx = vocab_stats["pad_idx"]
    unk_idx = vocab_stats["unk_idx"]

    coverage = vocab_stats["token_coverage_pct"]
    logger.info(f"Vocab built: {vocab_stats['vocab_size']:,} tokens. Token coverage on train: {coverage:.2f}%")
    if abs(coverage - 98.6) > 2.0:
        logger.warning(f"Token coverage ({coverage:.2f}%) differs somewhat from expected ~98.6%.")

    tokenizer_regex = re.compile(r"\w+|[^\w\s]")

    def tokenize_and_pad(texts: list[str]) -> tuple[np.ndarray, np.ndarray, float]:
        n_samples = len(texts)
        padded_ids = np.full((n_samples, max_len), pad_idx, dtype=np.int64)
        lengths = np.zeros(n_samples, dtype=np.int64)
        truncated_count = 0

        for i, txt in enumerate(texts):
            tokens = tokenizer_regex.findall(txt.lower())
            if len(tokens) > max_len:
                truncated_count += 1
                tokens = tokens[:max_len]
            ids = [word2idx.get(t, unk_idx) for t in tokens]
            if not ids:
                ids = [unk_idx]
            l = len(ids)
            padded_ids[i, :l] = ids
            lengths[i] = l

        trunc_pct = (truncated_count / n_samples) * 100.0 if n_samples > 0 else 0.0
        return padded_ids, lengths, trunc_pct

    train_ids_np, train_lengths_np, train_trunc_pct = tokenize_and_pad(df_train_sub[input_col].fillna("").tolist())
    val_ids_np, val_lengths_np, val_trunc_pct = tokenize_and_pad(df_val[input_col].fillna("").tolist())
    test_ids_np, test_lengths_np, test_trunc_pct = tokenize_and_pad(df_test[input_col].fillna("").tolist())

    logger.info(f"Truncation rates (> {max_len} tokens): Val={val_trunc_pct:.2f}%, Test={test_trunc_pct:.2f}%")
    vocab_stats["val_truncated_pct"] = round(val_trunc_pct, 2)
    vocab_stats["test_truncated_pct"] = round(test_trunc_pct, 2)

    # Save vocab.json
    vocab_file = model_dir / "vocab.json"
    with open(vocab_file, "w", encoding="utf-8") as fh:
        json.dump({"word2idx": word2idx, "stats": vocab_stats}, fh)

    stage2_elapsed = time.perf_counter() - stage2_start
    stage_timings["2_tokenize_vocab"] = stage2_elapsed
    print(f"  Done in {stage2_elapsed:.2f}s. Vocab size: {len(word2idx):,}, Coverage: {coverage:.2f}%")

    # ------------------------------------------------------------------ #
    #  [3/6] Load or Build fastText Embedding Matrix                     #
    # ------------------------------------------------------------------ #
    stage3_start = time.perf_counter()
    print("\n[3/6] Loading or building fastText embedding matrix...")
    logger.info("[3/6] Loading or building fastText embedding matrix...")
    update_status(status_file, "RUNNING", "3/6_fasttext_matrix")

    fasttext_zip = PROJECT_ROOT / "external" / "fasttext" / "crawl-300d-2M.vec.zip"
    cache_dir = PROJECT_ROOT / "models" / "2_bigru_attn"
    cache_path = cache_dir / "embedding_cache.npy"

    embedding_matrix, fasttext_meta = get_fasttext_embeddings(
        word2idx=word2idx,
        cache_path=cache_path,
        fasttext_zip=fasttext_zip,
        embedding_dim=int(m_cfg.get("embedding", {}).get("dim", 300)),
        seed=seed,
    )

    # Token-weighted hit rate
    tok_weighted_hits = 0
    tok_total = 0
    sample_text = " ".join(full_train_texts[:5000]).lower()
    for tok in tokenizer_regex.findall(sample_text):
        tok_total += 1
        if tok in word2idx and not np.all(embedding_matrix[word2idx[tok]] == 0.0):
            tok_weighted_hits += 1
    hit_rate_weighted = (tok_weighted_hits / tok_total) * 100.0 if tok_total > 0 else fasttext_meta["type_hit_rate_pct"]
    fasttext_meta["token_weighted_hit_rate_pct"] = round(hit_rate_weighted, 2)

    logger.info(f"fastText hit rate: {fasttext_meta['type_hit_rate_pct']:.2f}% (types) | {fasttext_meta['found_words']:,} words loaded")

    stage3_elapsed = time.perf_counter() - stage3_start
    stage_timings["3_fasttext_matrix"] = stage3_elapsed
    print(f"  Done in {stage3_elapsed:.2f}s. Hit rate: {fasttext_meta['type_hit_rate_pct']:.2f}%")

    # ------------------------------------------------------------------ #
    #  [4/6] Move Data to GPU & Build Model                              #
    # ------------------------------------------------------------------ #
    stage4_start = time.perf_counter()
    print("\n[4/6] Allocating tensors to GPU and constructing BiGRU model...")
    logger.info("[4/6] Allocating tensors to GPU and constructing BiGRU model...")
    update_status(status_file, "RUNNING", "4/6_build_model_gpu")

    # Compute train split class weights: w_l = (neg/pos)^0.5
    Y_train_np = df_train_sub[LABELS].values.astype(np.float32)
    Y_val_np = df_val[LABELS].values.astype(np.float32)
    Y_test_np = df_test[LABELS].values.astype(np.float32)

    pos_weights_list = []
    pos_rates_list = []
    class_weights_dict: dict[str, Any] = {}

    for idx, l in enumerate(LABELS):
        pos_cnt = int(np.sum(Y_train_np[:, idx] == 1))
        neg_cnt = int(np.sum(Y_train_np[:, idx] == 0))
        n_tot = len(Y_train_np)
        pos_rate = pos_cnt / n_tot if n_tot > 0 else 0.01
        w_l = math.sqrt(neg_cnt / pos_cnt) if pos_cnt > 0 else 1.0

        pos_weights_list.append(w_l)
        pos_rates_list.append(pos_rate)
        class_weights_dict[l] = {"0": 1.0, "1": round(w_l, 4)}
        logger.info(f"Label '{l}': pos={pos_cnt:,}, neg={neg_cnt:,}, w_1={w_l:.4f}")

    pos_weights_tensor = torch.tensor(pos_weights_list, dtype=torch.float32, device=device)

    # Move entire datasets to GPU tensors
    X_train_gpu = torch.tensor(train_ids_np, dtype=torch.long, device=device)
    Y_train_gpu = torch.tensor(Y_train_np, dtype=torch.float32, device=device)
    X_val_gpu = torch.tensor(val_ids_np, dtype=torch.long, device=device)
    Y_val_gpu = torch.tensor(Y_val_np, dtype=torch.float32, device=device)
    X_test_gpu = torch.tensor(test_ids_np, dtype=torch.long, device=device)

    # Construct model
    cfg_enc = m_cfg.get("encoder", {})
    cfg_attn = m_cfg.get("attention", {})
    cfg_embed = m_cfg.get("embedding", {})

    model = BiGruAttentionModel(
        vocab_size=len(word2idx),
        embedding_dim=int(cfg_embed.get("dim", 300)),
        hidden_dim=int(cfg_enc.get("hidden_dim", 128)),
        attention_dim=int(cfg_attn.get("dim", 128)),
        num_classes=len(LABELS),
        pad_idx=pad_idx,
        dropout_gru=float(cfg_enc.get("output_dropout", 0.3)),
        spatial_dropout=float(cfg_embed.get("spatial_dropout", 0.25)),
        pretrained_embeddings=embedding_matrix,
    )
    model.init_head_bias(pos_rates_list)
    model.to(device)

    param_counts = model.count_parameters()
    logger.info(f"Model parameters: Trainable = {param_counts['trainable']:,} | Frozen = {param_counts['frozen']:,} | Total = {param_counts['total']:,}")

    # Optimizer on trainable parameters only
    t_cfg = m_cfg.get("training", {})
    lr = float(t_cfg.get("lr", 1e-3))
    weight_decay = float(t_cfg.get("weight_decay", 0.01))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    # Loss function with pos_weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights_tensor)

    # Mixed precision & TF32 settings
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    scaler = torch.amp.GradScaler("cuda", enabled=bool(t_cfg.get("mixed_precision", True)))

    stage4_elapsed = time.perf_counter() - stage4_start
    stage_timings["4_build_model_gpu"] = stage4_elapsed
    print(f"  Done in {stage4_elapsed:.2f}s. Trainable params: {param_counts['trainable']:,}")

    # ------------------------------------------------------------------ #
    #  [5/6] Training Loop with Length-Bucketed Batches & Early Stopping  #
    # ------------------------------------------------------------------ #
    stage5_start = time.perf_counter()
    max_epochs = 2 if smoke else int(t_cfg.get("epochs", 6))
    batch_size = int(t_cfg.get("batch_size", 128))
    patience = int(t_cfg.get("patience", 2))
    grad_clip = float(t_cfg.get("grad_clip", 1.0))

    start_epoch = 1
    best_val_metric = -1.0
    best_epoch = 1
    patience_counter = 0
    history: list[dict[str, Any]] = []

    last_pt = model_dir / "last.pt"
    best_pt = model_dir / "best.pt"

    if resume and last_pt.exists():
        logger.info(f"Resuming training from checkpoint: {last_pt}")
        checkpoint = torch.load(last_pt, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_metric = checkpoint["best_metric"]
        best_epoch = checkpoint["best_epoch"]
        patience_counter = checkpoint["patience_counter"]
        history = checkpoint["history"]
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"].cpu())
        print(f"  Resumed from epoch {checkpoint['epoch']} (best val PR-AUC: {best_val_metric:.4f})")

    print(f"\n[5/6] Training BiGRU for up to {max_epochs} epochs (batch size {batch_size}, patience {patience})...")
    logger.info(f"[5/6] Training BiGRU for up to {max_epochs} epochs...")
    update_status(status_file, "RUNNING", "5/6_training_epochs")

    n_train = len(train_lengths_np)
    epoch_times = []

    # Validation evaluation helper
    def evaluate_validation(eval_x_gpu, eval_lengths_np, eval_y_np):
        model.eval()
        n_eval = len(eval_lengths_np)
        eval_batch_size = 1024
        sorted_eval_idx = np.argsort(eval_lengths_np)
        probs_all = np.zeros((n_eval, len(LABELS)), dtype=np.float32)
        total_eval_loss = 0.0
        n_eval_batches = 0

        with torch.no_grad():
            for b_start in range(0, n_eval, eval_batch_size):
                b_indices = sorted_eval_idx[b_start : b_start + eval_batch_size]
                b_lengths = eval_lengths_np[b_indices]
                max_len_b = max(8, int(math.ceil(max(b_lengths) / 8.0) * 8))
                max_len_b = min(max_len_b, max_len)

                b_ids = eval_x_gpu[b_indices, :max_len_b]
                b_logits = model(b_ids, b_lengths)
                b_y = eval_y_np[b_indices]

                loss_b = criterion(b_logits.float(), torch.from_numpy(b_y).float().to(device))
                total_eval_loss += loss_b.item()
                n_eval_batches += 1

                b_probs = torch.sigmoid(b_logits).cpu().numpy()
                probs_all[b_indices] = b_probs

        val_loss = total_eval_loss / max(1, n_eval_batches)

        # Compute per-label PR-AUC and ROC-AUC
        pr_aucs = []
        roc_aucs = []
        for l_i in range(len(LABELS)):
            y_col = eval_y_np[:, l_i]
            p_col = probs_all[:, l_i]
            if np.sum(y_col) > 0:
                pr_aucs.append(average_precision_score(y_col, p_col))
                try:
                    roc_aucs.append(roc_auc_score(y_col, p_col))
                except ValueError:
                    roc_aucs.append(0.5)

        macro_pr = float(np.mean(pr_aucs)) if pr_aucs else 0.0
        mean_roc = float(np.mean(roc_aucs)) if roc_aucs else 0.5
        return val_loss, macro_pr, mean_roc, probs_all

    for epoch in range(start_epoch, max_epochs + 1):
        ep_t0 = time.perf_counter()
        model.train()

        # Length-bucketed batch generation
        indices = np.random.permutation(n_train)
        chunk_size = 50 * batch_size  # ~6,400 samples
        batches = []

        for ch_start in range(0, n_train, chunk_size):
            chunk = indices[ch_start : ch_start + chunk_size]
            # Sort within chunk by length
            chunk_sorted = chunk[np.argsort(train_lengths_np[chunk])]
            for b_start in range(0, len(chunk_sorted), batch_size):
                b_idx = chunk_sorted[b_start : b_start + batch_size]
                batches.append(b_idx)

        # Shuffle batch execution order
        np.random.shuffle(batches)

        running_loss_accum = torch.zeros(1, device=device)
        total_steps = len(batches)

        pbar = tqdm(batches, desc=f"Epoch {epoch}/{max_epochs}", unit="batch", leave=False)
        for step, b_idx in enumerate(pbar):
            b_lengths = train_lengths_np[b_idx]
            max_b_len = max(8, int(math.ceil(max(b_lengths) / 8.0) * 8))
            max_b_len = min(max_b_len, max_len)

            batch_ids = X_train_gpu[b_idx, :max_b_len]
            batch_y = Y_train_gpu[b_idx]

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                logits = model(batch_ids, b_lengths)
                loss = criterion(logits.float(), batch_y.float())

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Epoch {epoch} Step {step}: NaN/inf loss detected! Falling back to FP32.")
                scaler._enabled = False
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss_accum += loss.detach()

        train_loss = float((running_loss_accum / max(1, total_steps)).item())

        # Validation
        val_loss, val_macro_pr, val_mean_roc, _ = evaluate_validation(X_val_gpu, val_lengths_np, Y_val_np)

        ep_duration = time.perf_counter() - ep_t0
        epoch_times.append(ep_duration)
        samples_per_sec = n_train / ep_duration
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        gpu_util = get_gpu_utilization()

        is_best = False
        if val_macro_pr > best_val_metric:
            best_val_metric = val_macro_pr
            best_epoch = epoch
            patience_counter = 0
            is_best = True
            torch.save(model.state_dict(), best_pt)
        else:
            patience_counter += 1

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
            "patience": patience_counter,
            "is_best": is_best,
        }
        history.append(ep_record)

        summary_line = (
            f"Epoch {epoch}/{max_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Macro PR-AUC: {val_macro_pr:.4f}{' (*BEST*)' if is_best else ''} | "
            f"Val ROC-AUC: {val_mean_roc:.4f} | Time: {ep_duration:.1f}s ({samples_per_sec:.0f} spl/s) | "
            f"VRAM: {peak_vram_mb:.0f}MB | Patience: {patience_counter}/{patience}"
        )
        print(f"  {summary_line}")
        logger.info(summary_line)

        # Save last.pt for resumability
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_metric": best_val_metric,
                "best_epoch": best_epoch,
                "patience_counter": patience_counter,
                "history": history,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(),
            },
            last_pt,
        )

        update_status(
            status_file,
            "RUNNING",
            "5/6_training_epochs",
            {
                "current_epoch": epoch,
                "best_epoch": best_epoch,
                "best_val_macro_pr_auc": round(best_val_metric, 4),
                "elapsed_so_far": round(time.perf_counter() - total_start_time, 2),
            },
        )

        if patience_counter >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch} (patience={patience} reached without improvement).")
            print(f"  Early stopping triggered at epoch {epoch}.")
            break

    stage5_elapsed = time.perf_counter() - stage5_start
    stage_timings["5_training_epochs"] = stage5_elapsed

    # Save history.json
    history_file = model_dir / "history.json"
    with open(history_file, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)

    # ------------------------------------------------------------------ #
    #  [6/6] Load Best Checkpoint, Predict Val/Test, Save Artifacts       #
    # ------------------------------------------------------------------ #
    stage6_start = time.perf_counter()
    print("\n[6/6] Loading best checkpoint, generating predictions and saving artifacts...")
    logger.info(f"[6/6] Loading best checkpoint from epoch {best_epoch} (val PR-AUC: {best_val_metric:.4f})...")
    update_status(status_file, "RUNNING", "6/6_predict_and_save")

    if best_pt.exists():
        best_state = torch.load(best_pt, map_location=device, weights_only=True)
        model.load_state_dict(best_state)

    # Generate full predictions for val and test
    _, _, _, probs_val = evaluate_validation(X_val_gpu, val_lengths_np, Y_val_np)
    _, _, _, probs_test = evaluate_validation(X_test_gpu, test_lengths_np, Y_test_np)

    # Save predictions
    p_val_file = preds_dir / "probs_val.npy"
    p_test_file = preds_dir / "probs_test.npy"
    np.save(p_val_file, probs_val.astype(np.float32))
    np.save(p_test_file, probs_test.astype(np.float32))
    logger.info(f"Saved probs_val.npy shape {probs_val.shape} and probs_test.npy shape {probs_test.shape}")

    # Save final model state dict
    f_model_pt = model_dir / "model.pt"
    torch.save(model.state_dict(), f_model_pt)

    # Delete last.pt after successful run
    if last_pt.exists():
        last_pt.unlink()

    # Calculate model disk size
    model_size_bytes = f_model_pt.stat().st_size
    model_size_mb = round(model_size_bytes / (1024 * 1024), 2)

    total_elapsed = time.perf_counter() - total_start_time
    stage6_elapsed = time.perf_counter() - stage6_start
    stage_timings["6_predict_and_save"] = stage6_elapsed
    mean_epoch_time = float(np.mean(epoch_times)) if epoch_times else 0.0

    meta_payload = {
        "model_id": model_id,
        "model_name": model_name,
        "adapter_type": "bigru_attn",
        "smoke": smoke,
        "framework": f"PyTorch {torch.__version__}",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "labels": LABELS,
        "input_column": input_col,
        "class_weights": class_weights_dict,
        "config": m_cfg,
        "parameters": param_counts,
        "tokenizer_info": vocab_stats,
        "fasttext_info": fasttext_meta,
        "training_history": history,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_val_macro_pr_auc": round(best_val_metric, 4),
        "mean_epoch_time_seconds": round(mean_epoch_time, 2),
        "train_time_seconds": round(total_elapsed, 2),
        "stage_timings_seconds": {k: round(v, 2) for k, v in stage_timings.items()},
        "model_size_bytes": model_size_bytes,
        "model_size_mb": model_size_mb,
        "device_info": {
            "gpu_name": gpu_name,
            "vram_gb": round(total_vram_gb, 2),
            "cuda_version": torch.version.cuda,
            "peak_vram_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1),
        },
        "artifacts": {
            "model_pt": str(f_model_pt.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "vocab_json": str(vocab_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "history_json": str(history_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "probs_val": str(p_val_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "probs_test": str(p_test_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "pandas": pd.__version__,
            "numpy": np.__version__,
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
            "best_epoch": best_epoch,
            "best_val_macro_pr_auc": round(best_val_metric, 4),
            "model_size_mb": model_size_mb,
        },
    )

    # DONE Banner
    banner = f"""
================================================================================
  TRAINING COMPLETED SUCCESSFULLY: {model_id} (smoke={smoke})
================================================================================
  Total Time: {total_elapsed:.2f}s ({total_elapsed / 60:.2f} min)
  Epochs Run: {len(history)} | Best Epoch: {best_epoch} (Val Macro PR-AUC: {best_val_metric:.4f})
  Mean Epoch Time: {mean_epoch_time:.2f}s
  Peak GPU Memory: {torch.cuda.max_memory_allocated() / (1024 * 1024):.1f} MB

  Stage Timings:
    [1/6] Load & Verify:     {stage_timings['1_load_verify']:.2f}s
    [2/6] Tokenize & Vocab:  {stage_timings['2_tokenize_vocab']:.2f}s (Coverage: {coverage:.2f}%)
    [3/6] fastText Matrix:   {stage_timings['3_fasttext_matrix']:.2f}s (Hit Rate: {fasttext_meta['type_hit_rate_pct']:.2f}%)
    [4/6] Build Model & GPU: {stage_timings['4_build_model_gpu']:.2f}s (Trainable: {param_counts['trainable']:,})
    [5/6] Training Epochs:   {stage_timings['5_training_epochs']:.2f}s
    [6/6] Predict & Save:    {stage_timings['6_predict_and_save']:.2f}s

  Model Size on Disk: {model_size_mb} MB ({model_size_bytes:,} bytes)
  Outputs:
    - Models:      {model_dir.relative_to(PROJECT_ROOT)}
    - Predictions: {preds_dir.relative_to(PROJECT_ROOT)}
    - Logs:        {logs_dir.relative_to(PROJECT_ROOT)}
================================================================================
"""
    print(banner)
    logger.info(banner)

    # Beep
    try:
        import winsound
        winsound.Beep(1200, 350)
    except Exception:
        pass


if __name__ == "__main__":
    main()
