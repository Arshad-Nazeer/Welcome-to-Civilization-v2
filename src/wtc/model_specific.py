"""
src/wtc/model_specific.py
-------------------------
Model-specific evaluation routines for Welcome to Civilization.
Plugged into scripts/20_evaluate.py via adapter_type dispatch:
- tfidf_lr: top n-grams analysis (handled in evaluate or here)
- bigru_attn: training curves, attention analysis on true toxic comments,
  and dual-device (GPU + CPU) inference benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .config import LABELS, PROJECT_ROOT, get_model_dir
from .plotting import get_model_color


def run_model_specific_bigru(
    model_id: str,
    smoke: bool,
    reports_dir: Path,
    viz_dir: Path,
    labels: list[str],
    df_test: pd.DataFrame,
    P_test: np.ndarray,
    Y_test: np.ndarray,
) -> dict[str, Any]:
    """
    Run Model 2 specific evaluation:
    1. Training history export and training curves plot.
    2. Attention weight analysis: top 20 tokens by mean attention weight
       among test comments with true toxic=1 (tokens with count >= 30).
    """
    model_dir = get_model_dir(model_id, smoke=smoke)
    tables_dir = reports_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    color = get_model_color(model_id)

    results: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # 1. Training Curves and History                                      #
    # ------------------------------------------------------------------ #
    history_file = model_dir / "history.json"
    if history_file.exists():
        with open(history_file, "r", encoding="utf-8") as fh:
            history = json.load(fh)

        df_hist = pd.DataFrame(history)
        p_hist_csv = tables_dir / "training_history.csv"
        df_hist.to_csv(p_hist_csv, index=False)
        results["training_history_table"] = str(p_hist_csv.relative_to(PROJECT_ROOT)).replace("\\", "/")

        # Plot training curves
        epochs = [h["epoch"] for h in history]
        train_loss = [h["train_loss"] for h in history]
        val_loss = [h["val_loss"] for h in history]
        val_macro_pr = [h["val_macro_pr_auc"] for h in history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=200)

        # Loss curves
        ax1.plot(epochs, train_loss, marker="o", lw=2, color="#1f77b4", label="Train Loss")
        ax1.plot(epochs, val_loss, marker="s", lw=2, color="#d62728", label="Val Loss")
        ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Weighted BCE Loss", fontsize=11, fontweight="bold")
        ax1.set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
        ax1.set_xticks(epochs)
        ax1.legend(loc="upper right")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Validation Macro PR-AUC
        ax2.plot(epochs, val_macro_pr, marker="D", lw=2, color=color, label="Val Macro PR-AUC")
        best_epoch_idx = int(np.argmax(val_macro_pr))
        best_ep = epochs[best_epoch_idx]
        best_val_pr = val_macro_pr[best_epoch_idx]
        ax2.plot(best_ep, best_val_pr, marker="*", markersize=14, color="#d62728", label=f"Best (Ep {best_ep}: {best_val_pr:.4f})")
        ax2.set_xlabel("Epoch", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Validation Macro PR-AUC", fontsize=11, fontweight="bold")
        ax2.set_title("Validation Macro PR-AUC vs Epoch", fontsize=12, fontweight="bold")
        ax2.set_xticks(epochs)
        ax2.legend(loc="lower right")
        ax2.grid(True, linestyle="--", alpha=0.5)

        fig.suptitle(f"BiGRU + Additive Attention Training History ({model_id})", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        p_curves_png = viz_dir / "training_curves.png"
        fig.savefig(p_curves_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        results["training_curves_figure"] = str(p_curves_png.relative_to(PROJECT_ROOT)).replace("\\", "/")

    # ------------------------------------------------------------------ #
    # 2. Attention Analysis on True Toxic Comments                       #
    # ------------------------------------------------------------------ #
    vocab_file = model_dir / "vocab.json"
    model_pt = model_dir / "model.pt"

    if vocab_file.exists() and model_pt.exists() and "toxic" in labels:
        with open(vocab_file, "r", encoding="utf-8") as fh:
            vocab_data = json.load(fh)
        word2idx = vocab_data["word2idx"]
        pad_idx = vocab_data.get("pad_idx", 0)
        unk_idx = vocab_data.get("unk_idx", 1)
        idx2word = {idx: word for word, idx in word2idx.items()}

        # Load trained PyTorch model architecture
        meta_file = model_dir / "meta.json"
        with open(meta_file, "r", encoding="utf-8") as fh:
            meta = json.load(fh)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from .bigru_module import BiGruAttentionModel
        net = BiGruAttentionModel(
            vocab_size=len(word2idx),
            embedding_dim=meta["config"]["embedding"]["dim"],
            hidden_dim=meta["config"]["encoder"]["hidden_dim"],
            attention_dim=meta["config"]["attention"]["dim"],
            num_classes=len(labels),
            pad_idx=pad_idx,
            dropout_gru=meta["config"]["encoder"]["output_dropout"],
            spatial_dropout=meta["config"]["embedding"]["spatial_dropout"],
        )
        state_dict = torch.load(model_pt, map_location=device, weights_only=True)
        net.load_state_dict(state_dict)
        net.to(device)
        net.eval()

        # Find test comments with true toxic == 1
        toxic_idx = labels.index("toxic")
        mask_toxic = (Y_test[:, toxic_idx] == 1)
        toxic_texts = df_test.loc[mask_toxic, "text_clean"].fillna("").tolist()

        tokenizer_regex = re.compile(r"\w+|[^\w\s]")
        token_sum_attn: dict[str, float] = {}
        token_counts: dict[str, int] = {}

        # Process in batches
        batch_size = 256
        with torch.no_grad():
            for i in range(0, len(toxic_texts), batch_size):
                batch_texts = toxic_texts[i : i + batch_size]
                tokenized_batch = []
                lengths_batch = []
                for txt in batch_texts:
                    toks = tokenizer_regex.findall(txt.lower())[:256]
                    ids = [word2idx.get(t, unk_idx) for t in toks]
                    if not ids:
                        ids = [unk_idx]
                        toks = ["<unk>"]
                    tokenized_batch.append((toks, ids))
                    lengths_batch.append(len(ids))

                max_b_len = max(lengths_batch)
                padded_ids = np.full((len(batch_texts), max_b_len), pad_idx, dtype=np.int64)
                for b_idx, (_, ids) in enumerate(tokenized_batch):
                    padded_ids[b_idx, : len(ids)] = ids

                tensor_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
                lengths_cpu = np.array(lengths_batch, dtype=np.int64)

                _, attn_weights = net(tensor_ids, lengths_cpu, return_attn=True)
                attn_weights_np = attn_weights.cpu().numpy()

                for b_idx, (toks, _) in enumerate(tokenized_batch):
                    for t_pos, tok in enumerate(toks):
                        if tok in ("<pad>", "<unk>"):
                            continue
                        w = float(attn_weights_np[b_idx, t_pos])
                        token_sum_attn[tok] = token_sum_attn.get(tok, 0.0) + w
                        token_counts[tok] = token_counts.get(tok, 0) + 1

        # Tokens with count >= 30 (or >= 5 in smoke mode)
        min_tok_count = 5 if smoke else 30
        eligible_tokens = [
            (tok, token_sum_attn[tok] / token_counts[tok], token_counts[tok])
            for tok in token_sum_attn
            if token_counts[tok] >= min_tok_count
        ]

        # Sort by mean attention weight descending
        eligible_tokens.sort(key=lambda x: x[1], reverse=True)
        top20_tokens = eligible_tokens[:20]

        top_tokens_rows = [
            {"rank": rank, "token": tok, "mean_attention": round(mean_w, 6), "count": cnt}
            for rank, (tok, mean_w, cnt) in enumerate(top20_tokens, 1)
        ]
        df_attn = pd.DataFrame(top_tokens_rows)
        p_attn_csv = tables_dir / "top_attention_tokens.csv"
        df_attn.to_csv(p_attn_csv, index=False)
        results["attention_table"] = str(p_attn_csv.relative_to(PROJECT_ROOT)).replace("\\", "/")
        results["top_attention_tokens"] = top_tokens_rows

        # Plot attention bar chart
        if top20_tokens:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
            tokens_plot = [x[0] for x in reversed(top20_tokens)]
            weights_plot = [x[1] for x in reversed(top20_tokens)]
            counts_plot = [x[2] for x in reversed(top20_tokens)]

            y_pos = np.arange(len(tokens_plot))
            bars = ax.barh(y_pos, weights_plot, color=color, alpha=0.85, edgecolor="black", linewidth=0.8)
            ax.set_yticks(y_pos)
            labels_annot = [f"{t} (n={c})" for t, c in zip(tokens_plot, counts_plot)]
            ax.set_yticklabels(labels_annot, fontsize=9, fontweight="bold")
            ax.set_xlabel("Mean Additive Attention Weight", fontsize=11, fontweight="bold")
            ax.set_title(
                f"Top 20 Attended Tokens in True Toxic Comments (count >= {min_tok_count})\n({model_id})",
                fontsize=12,
                fontweight="bold",
                pad=10,
            )
            ax.grid(axis="x", linestyle="--", alpha=0.5)

            for bar, w in zip(bars, weights_plot):
                ax.annotate(
                    f"{w:.4f}",
                    xy=(bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2),
                    va="center",
                    fontsize=8.5,
                )

            plt.tight_layout()
            p_attn_png = viz_dir / "top_attention_tokens.png"
            fig.savefig(p_attn_png, dpi=200)
            plt.close(fig)
            results["attention_figure"] = str(p_attn_png.relative_to(PROJECT_ROOT)).replace("\\", "/")

    return results
