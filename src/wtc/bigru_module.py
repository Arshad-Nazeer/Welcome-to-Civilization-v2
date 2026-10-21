"""
src/wtc/bigru_module.py
-----------------------
PyTorch neural network architecture for Model 2 / 2b:
Bidirectional GRU + Additive Attention Pooling with fastText embeddings.
Features:
- Configurable frozen or trainable embeddings (freeze_embedding param)
- Spatial Dropout on 300-d embeddings (F.dropout1d across time)
- 1-layer BiGRU with pack_padded_sequence (enforce_sorted=False)
- Additive Attention with float32 masked softmax
- Linear classification head with log-odds bias initialization
- forward(..., return_attn=True) support for interpretability
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiGruAttentionModel(nn.Module):
    """
    BiGRU with Additive Attention Pooling and Frozen Embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 128,
        attention_dim: int = 128,
        num_classes: int = 6,
        pad_idx: int = 0,
        dropout_gru: float = 0.3,
        spatial_dropout: float = 0.25,
        pretrained_embeddings: np.ndarray | torch.Tensor | None = None,
        freeze_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.num_classes = num_classes
        self.pad_idx = pad_idx
        self.spatial_dropout_p = spatial_dropout

        # 1. Embedding Layer (frozen or fine-tuned based on freeze_embedding)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if pretrained_embeddings is not None:
            if isinstance(pretrained_embeddings, np.ndarray):
                weight_tensor = torch.from_numpy(pretrained_embeddings).float()
            else:
                weight_tensor = pretrained_embeddings.float()
            self.embedding.weight.data.copy_(weight_tensor)
        self.embedding.weight.requires_grad = not freeze_embedding

        # 2. Encoder: 1-layer Bidirectional GRU
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.gru_dropout = nn.Dropout(dropout_gru)

        # 3. Additive Attention Pooling
        gru_out_dim = hidden_dim * 2  # 256
        self.attn_proj = nn.Linear(gru_out_dim, attention_dim)
        self.context_vector = nn.Parameter(torch.empty(attention_dim))
        nn.init.normal_(self.context_vector, mean=0.0, std=0.02)

        # 4. Classification Head
        self.fc = nn.Linear(gru_out_dim, num_classes)

    def init_head_bias(self, pos_rates: list[float] | np.ndarray) -> None:
        """Initialize classification head bias to log-odds: log(p / (1 - p))."""
        with torch.no_grad():
            for i, p in enumerate(pos_rates):
                p_clamped = min(max(float(p), 1e-5), 1.0 - 1e-5)
                log_odds = math.log(p_clamped / (1.0 - p_clamped))
                self.fc.bias[i] = log_odds

    def count_parameters(self) -> dict[str, int]:
        """Return counts of trainable, frozen, and total parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {
            "trainable": trainable,
            "frozen": frozen,
            "total": trainable + frozen,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths_cpu: np.ndarray | torch.Tensor,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            input_ids: Tensor of token ids [batch_size, seq_len]
            lengths_cpu: CPU array or tensor of unpadded sequence lengths [batch_size]
            return_attn: If True, returns (logits, attention_weights)
        """
        batch_size, seq_len = input_ids.shape

        # 1. Embeddings
        x = self.embedding(input_ids)  # [B, L, 300]

        # 2. Spatial Dropout (drops whole embedding channels across time)
        if self.training and self.spatial_dropout_p > 0.0:
            x = x.permute(0, 2, 1)  # [B, 300, L]
            x = F.dropout1d(x, p=self.spatial_dropout_p, training=True)
            x = x.permute(0, 2, 1)  # [B, L, 300]

        # 3. Packed sequence through BiGRU
        if isinstance(lengths_cpu, torch.Tensor):
            lengths_seq = lengths_cpu.cpu()
        else:
            lengths_seq = torch.from_numpy(lengths_cpu)
        # Ensure lengths are at least 1
        lengths_seq = torch.clamp(lengths_seq, min=1, max=seq_len)

        packed = pack_padded_sequence(x, lengths_seq, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        gru_out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=seq_len)
        gru_out = self.gru_dropout(gru_out)  # [B, L, 256]

        # 4. Additive Attention Pooling
        # Projection: u = tanh(W * h + b) -> [B, L, 128]
        u = torch.tanh(self.attn_proj(gru_out))
        # Similarity with context vector: [B, L]
        score = torch.matmul(u, self.context_vector)

        # Mask padding positions & Softmax in float32 for AMP numerical stability
        mask = (input_ids != self.pad_idx)
        score_f32 = score.float().masked_fill(~mask, -1e9)
        alpha = F.softmax(score_f32, dim=-1)  # [B, L]
        alpha_casted = alpha.to(gru_out.dtype)

        # Weighted context vector: [B, 256]
        context = torch.sum(gru_out * alpha_casted.unsqueeze(-1), dim=1)

        # 5. Output Logits
        logits = self.fc(context)  # [B, 6]

        if return_attn:
            return logits, alpha
        return logits
