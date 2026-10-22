"""
src/wtc/distilroberta_module.py
-------------------------------
PyTorch model architecture, loss function, optimizer parameter groupings,
and batch collation for Model 3: DistilRoBERTa-base fine-tuning.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler
from transformers import AutoModel, AutoTokenizer


class DistilRobertaClassificationHead(nn.Module):
    """Classification head on top of the <s> (CLS) token."""

    def __init__(self, hidden_size: int = 768, num_classes: int = 6, dropout: float = 0.1) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_classes)

        # Standard initialization
        nn.init.normal_(self.dense.weight, std=0.02)
        nn.init.zeros_(self.dense.bias)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

    def init_bias_log_odds(self, pos_counts: np.ndarray, total_samples: int) -> None:
        """Initialize output bias to log-odds: log(p / (1 - p))."""
        eps = 1e-5
        probs = np.clip(pos_counts / max(1, total_samples), eps, 1.0 - eps)
        log_odds = np.log(probs / (1.0 - probs))
        with torch.no_grad():
            self.out_proj.bias.copy_(torch.from_numpy(log_odds).float())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features is <s> token representation [batch, hidden_size]
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class DistilRobertaForMultilabel(nn.Module):
    """
    DistilRoBERTa encoder + custom classification head for multi-label toxicity.
    Labels are not passed to forward(); loss is calculated externally.
    """

    def __init__(
        self,
        pretrained_name: str = "distilroberta-base",
        num_classes: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.roberta = AutoModel.from_pretrained(pretrained_name)
        hidden_size = self.roberta.config.hidden_size
        self.classifier = DistilRobertaClassificationHead(
            hidden_size=hidden_size,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        cls_token = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_token)
        return logits


class MultilabelLoss(nn.Module):
    """
    Weighted Binary Cross Entropy loss with optional focal factor.
    All calculations are cast to float32 for numerical stability under AMP.
    """

    def __init__(
        self,
        pos_weight: torch.Tensor | None = None,
        focal_gamma: float = 0.0,
    ) -> None:
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None
        self.focal_gamma = float(focal_gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits_f32 = logits.float()
        targets_f32 = targets.float()

        if self.focal_gamma > 0.0:
            bce = F.binary_cross_entropy_with_logits(
                logits_f32,
                targets_f32,
                pos_weight=self.pos_weight,
                reduction="none",
            )
            p = torch.sigmoid(logits_f32)
            p_t = targets_f32 * p + (1.0 - targets_f32) * (1.0 - p)
            focal_factor = (1.0 - p_t) ** self.focal_gamma
            return (focal_factor * bce).mean()
        else:
            return F.binary_cross_entropy_with_logits(
                logits_f32,
                targets_f32,
                pos_weight=self.pos_weight,
                reduction="mean",
            )


def get_layerwise_optimizer_groups(
    model: DistilRobertaForMultilabel,
    base_lr: float,
    weight_decay: float = 0.01,
    layerwise_lr_decay: float = 0.9,
    no_decay_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Construct parameter groups for AdamW with layer-wise learning rate decay (LLRD).
    Classification head receives base_lr; transformer layers receive decayed lr;
    embeddings receive the strongest decay. Bias and LayerNorm weights have 0 weight decay.
    """
    if no_decay_names is None:
        no_decay_names = ["bias", "LayerNorm.weight", "LayerNorm.bias", "layer_norm"]

    def is_no_decay(name: str) -> bool:
        return any(nd.lower() in name.lower() for nd in no_decay_names)

    # 1. Group parameters by layer depth
    # Layers:
    # 7: classifier head (base_lr)
    # 6..1: transformer layers 5 down to 0
    # 0: embeddings
    groups: dict[int, dict[str, list[torch.nn.Parameter]]] = {
        depth: {"decay": [], "no_decay": []} for depth in range(8)
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "classifier" in name:
            depth = 7
        elif "roberta.embeddings" in name:
            depth = 0
        elif "roberta.transformer.layer" in name:
            # Extract layer index e.g. roberta.transformer.layer.4...
            parts = name.split(".")
            idx = int(parts[parts.index("layer") + 1])
            depth = idx + 1  # 0->1, 1->2, ..., 5->6
        else:
            depth = 7

        key = "no_decay" if is_no_decay(name) else "decay"
        groups[depth][key].append(param)

    # 2. Build final list of parameter group dicts
    optimizer_groups = []
    for depth in range(8):
        lr_scale = layerwise_lr_decay ** (7 - depth)
        lr = base_lr * lr_scale

        decay_params = groups[depth]["decay"]
        no_decay_params = groups[depth]["no_decay"]

        if decay_params:
            optimizer_groups.append({
                "params": decay_params,
                "lr": lr,
                "weight_decay": weight_decay,
            })
        if no_decay_params:
            optimizer_groups.append({
                "params": no_decay_params,
                "lr": lr,
                "weight_decay": 0.0,
            })

    return optimizer_groups


class LengthBucketedBatchSampler(Sampler[list[int]]):
    """
    Batches sequences of similar lengths to minimize padding overhead,
    while shuffling bucket order and intra-bucket samples every epoch.
    """

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        shuffle: bool = True,
        bucket_size_multiplier: int = 20,
        seed: int = 42,
    ) -> None:
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bucket_size = batch_size * bucket_size_multiplier
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch if self.shuffle else 42)
        indices = np.argsort(self.lengths)

        batches = []
        # Process in chunks (mega-buckets)
        for i in range(0, len(indices), self.bucket_size):
            bucket = indices[i : i + self.bucket_size]
            if self.shuffle:
                rng.shuffle(bucket)
            for j in range(0, len(bucket), self.batch_size):
                b = bucket[j : j + self.batch_size].tolist()
                batches.append(b)

        if self.shuffle:
            rng.shuffle(batches)

        yield from batches

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.batch_size)
