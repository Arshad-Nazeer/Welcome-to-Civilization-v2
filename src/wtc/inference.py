"""
src/wtc/inference.py
--------------------
Inference engine and Moderator wrapper for Welcome to Civilization.
Provides a unified interface:
- Moderator(model_id).predict_proba(texts) -> [n, 6]
- Moderator(model_id).predict(texts) -> [n, 6] (using tuned thresholds)

Registry-based so Model 1 (TF-IDF + LR), Model 2 (BiGRU + Attention),
and Model 3 (DistilRoBERTa) plug in cleanly.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
import re
from typing import Any
import warnings

import joblib
import numpy as np
import pandas as pd
import scipy.sparse

from .config import CFG, LABELS, PROJECT_ROOT, get_model_dir, get_predictions_dir
from .preprocess import clean_text


class BaseModelAdapter(ABC):
    """Abstract base adapter for model inference."""

    @abstractmethod
    def load(self, model_dir: Path, meta: dict[str, Any]) -> None:
        """Load vectorizers/weights/tokenizers into memory."""
        pass

    @abstractmethod
    def predict_proba(self, texts_clean: list[str]) -> np.ndarray:
        """Predict sigmoid probabilities [N, 6] on cleaned texts."""
        pass


class TfidfLrAdapter(BaseModelAdapter):
    """Adapter for Model 1: Word + Char TF-IDF with Logistic Regression."""

    def __init__(self) -> None:
        self.vec_word = None
        self.vec_char = None
        self.classifiers = None
        self.labels: list[str] = LABELS

    def load(self, model_dir: Path, meta: dict[str, Any]) -> None:
        # Check scikit-learn version
        try:
            import sklearn
            installed_skl = sklearn.__version__
            meta_skl = meta.get("versions", {}).get("scikit_learn", "")
            if meta_skl and meta_skl != installed_skl:
                warnings.warn(
                    f"Installed scikit-learn ({installed_skl}) differs from "
                    f"model training version ({meta_skl}).",
                    UserWarning,
                )
        except Exception:
            pass

        self.vec_word = joblib.load(model_dir / "vectorizer_word.joblib")
        self.vec_char = joblib.load(model_dir / "vectorizer_char.joblib")
        self.classifiers = joblib.load(model_dir / "classifiers.joblib")
        self.labels = meta.get("labels", LABELS)

    def predict_proba(self, texts_clean: list[str]) -> np.ndarray:
        X_w = self.vec_word.transform(texts_clean)
        X_c = self.vec_char.transform(texts_clean)
        X = scipy.sparse.hstack([X_w, X_c], format="csr", dtype=np.float32)

        probs = []
        for label in self.labels:
            clf = self.classifiers[label]
            p = clf.predict_proba(X)[:, 1]
            probs.append(p)

        return np.column_stack(probs).astype(np.float32)


class BiGruAttnAdapter(BaseModelAdapter):
    """Adapter for Model 2: BiGRU + Additive Attention with fastText embeddings."""

    def __init__(self) -> None:
        self.model = None
        self.word2idx: dict[str, int] = {}
        self.pad_idx = 0
        self.unk_idx = 1
        self.max_len = 256
        self.labels: list[str] = LABELS
        self.device = None
        self.tokenizer_regex = re.compile(r"\w+|[^\w\s]")

    def load(self, model_dir: Path, meta: dict[str, Any]) -> None:
        import torch
        from .bigru_module import BiGruAttentionModel

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = meta.get("labels", LABELS)

        vocab_file = model_dir / "vocab.json"
        with open(vocab_file, "r", encoding="utf-8") as fh:
            v_data = json.load(fh)
        self.word2idx = v_data["word2idx"]
        self.pad_idx = v_data.get("pad_idx", 0)
        self.unk_idx = v_data.get("unk_idx", 1)
        self.max_len = meta.get("config", {}).get("tokenizer", {}).get("max_len", 256)

        cfg_embed = meta.get("config", {}).get("embedding", {})
        cfg_enc = meta.get("config", {}).get("encoder", {})
        cfg_attn = meta.get("config", {}).get("attention", {})

        self.model = BiGruAttentionModel(
            vocab_size=len(self.word2idx),
            embedding_dim=cfg_embed.get("dim", 300),
            hidden_dim=cfg_enc.get("hidden_dim", 128),
            attention_dim=cfg_attn.get("dim", 128),
            num_classes=len(self.labels),
            pad_idx=self.pad_idx,
            dropout_gru=cfg_enc.get("output_dropout", 0.3),
            spatial_dropout=cfg_embed.get("spatial_dropout", 0.25),
        )

        model_file = model_dir / "model.pt"
        state_dict = torch.load(model_file, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict_proba(self, texts_clean: list[str]) -> np.ndarray:
        import torch

        n_samples = len(texts_clean)
        if n_samples == 0:
            return np.empty((0, len(self.labels)), dtype=np.float32)

        # 1. Tokenize & map to IDs
        tokenized_ids = []
        lengths = []
        for txt in texts_clean:
            toks = self.tokenizer_regex.findall(txt.lower())[: self.max_len]
            ids = [self.word2idx.get(t, self.unk_idx) for t in toks]
            if not ids:
                ids = [self.unk_idx]
            tokenized_ids.append(ids)
            lengths.append(len(ids))

        lengths_arr = np.array(lengths, dtype=np.int64)

        # 2. Length-sorted batching
        sorted_indices = np.argsort(lengths_arr)
        batch_size = 1024 if self.device.type == "cuda" else 128
        probs_ordered = np.zeros((n_samples, len(self.labels)), dtype=np.float32)

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                b_idx = sorted_indices[i : i + batch_size]
                b_lengths = lengths_arr[b_idx]
                max_b_len = int(np.max(b_lengths))

                padded_b = np.full((len(b_idx), max_b_len), self.pad_idx, dtype=np.int64)
                for row_pos, orig_idx in enumerate(b_idx):
                    ids = tokenized_ids[orig_idx]
                    padded_b[row_pos, : len(ids)] = ids

                tensor_b = torch.tensor(padded_b, dtype=torch.long, device=self.device)
                logits = self.model(tensor_b, b_lengths)
                b_probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
                probs_ordered[b_idx] = b_probs

        return probs_ordered


# Registry of model adapters
ADAPTER_REGISTRY: dict[str, type[BaseModelAdapter]] = {
    "1_tfidf_lr": TfidfLrAdapter,
    "tfidf_lr": TfidfLrAdapter,
    "2_bigru_attn": BiGruAttnAdapter,
    "bigru_attn": BiGruAttnAdapter,
}


def register_model_adapter(name: str, adapter_cls: type[BaseModelAdapter]) -> None:
    """Register an adapter class for a model ID or family."""
    ADAPTER_REGISTRY[name] = adapter_cls


class Moderator:
    """
    Production-ready toxicity moderation engine.
    Applies text cleaning pipeline, predicts multi-label probabilities,
    and applies calibrated decision thresholds.
    """

    def __init__(self, model_id: str = "1_tfidf_lr", smoke: bool = False) -> None:
        self.model_id = model_id
        self.smoke = smoke
        self.model_dir = get_model_dir(model_id, smoke=smoke)
        self.preds_dir = get_predictions_dir(model_id, smoke=smoke)

        # 1. Load metadata
        meta_file = self.model_dir / "meta.json"
        if not meta_file.exists():
            known_ids = sorted(set(list(ADAPTER_REGISTRY.keys()) + list(CFG.get("model_ids", {}).values())))
            raise FileNotFoundError(
                f"Model metadata not found at {meta_file}. Run training first.\n"
                f"Available registered model IDs: {known_ids}"
            )
        with open(meta_file, "r", encoding="utf-8") as fh:
            self.meta = json.load(fh)

        self.labels = self.meta.get("labels", LABELS)

        # 2. Load thresholds: models/<id>/ first, fallback to predictions/<id>/
        thresh_file = self.model_dir / "thresholds.json"
        if not thresh_file.exists():
            thresh_file = self.preds_dir / "thresholds.json"

        if thresh_file.exists():
            with open(thresh_file, "r", encoding="utf-8") as fh:
                t_data = json.load(fh)
                self.thresholds = t_data.get("thresholds", {l: 0.5 for l in self.labels})
        else:
            self.thresholds = {l: 0.5 for l in self.labels}

        self.threshold_arr = np.array([self.thresholds[l] for l in self.labels], dtype=np.float32)

        # 3. Resolve and load adapter
        adapter_type = self.meta.get("adapter_type", model_id)
        if adapter_type not in ADAPTER_REGISTRY:
            matched = False
            for reg_key in ADAPTER_REGISTRY:
                if reg_key in model_id or model_id in reg_key:
                    adapter_type = reg_key
                    matched = True
                    break
            if not matched:
                known_ids = sorted(set(list(ADAPTER_REGISTRY.keys()) + list(CFG.get("model_ids", {}).values())))
                raise ValueError(
                    f"Unknown adapter type '{adapter_type}' for model '{model_id}'. "
                    f"Registered IDs: {known_ids}"
                )

        self.adapter = ADAPTER_REGISTRY[adapter_type]()
        self.adapter.load(self.model_dir, self.meta)

    def _normalize_inputs(self, texts: str | list[str] | pd.Series) -> list[str]:
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, pd.Series):
            texts = texts.fillna("").tolist()
        elif not isinstance(texts, (list, tuple)):
            texts = list(texts)
        return [clean_text(t) for t in texts]

    def predict_proba(self, texts: str | list[str] | pd.Series) -> np.ndarray:
        """
        Return predicted probabilities [N, 6] in LABELS order.
        """
        cleaned = self._normalize_inputs(texts)
        if len(cleaned) == 0:
            return np.empty((0, len(self.labels)), dtype=np.float32)
        return self.adapter.predict_proba(cleaned)

    def predict(
        self,
        texts: str | list[str] | pd.Series,
        apply_hierarchical_severe_toxic: bool = False,
    ) -> np.ndarray:
        """
        Return binary predictions [N, 6] using tuned decision thresholds.
        Default apply_hierarchical_severe_toxic=False matches CSV pred_ columns.
        """
        probs = self.predict_proba(texts)
        preds = (probs >= self.threshold_arr).astype(np.int32)

        if apply_hierarchical_severe_toxic and "toxic" in self.labels and "severe_toxic" in self.labels:
            tox_idx = self.labels.index("toxic")
            sev_idx = self.labels.index("severe_toxic")
            preds[:, sev_idx] = preds[:, sev_idx] & preds[:, tox_idx]

        return preds
