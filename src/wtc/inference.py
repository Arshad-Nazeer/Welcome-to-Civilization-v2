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
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy.sparse

from .config import LABELS, PROJECT_ROOT, get_model_dir, get_predictions_dir
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


# Registry of model adapters
ADAPTER_REGISTRY: dict[str, type[BaseModelAdapter]] = {
    "1_tfidf_lr": TfidfLrAdapter,
    "tfidf_lr": TfidfLrAdapter,
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
            raise FileNotFoundError(f"Model metadata not found at {meta_file}. Run training first.")
        with open(meta_file, "r", encoding="utf-8") as fh:
            self.meta = json.load(fh)

        self.labels = self.meta.get("labels", LABELS)

        # 2. Load thresholds
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
            # Try prefix match (e.g. 1_tfidf_lr -> tfidf_lr)
            matched = False
            for reg_key in ADAPTER_REGISTRY:
                if reg_key in model_id or model_id in reg_key:
                    adapter_type = reg_key
                    matched = True
                    break
            if not matched:
                raise ValueError(f"Unknown adapter type '{adapter_type}' for model {model_id}.")

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
        apply_hierarchical_severe_toxic: bool = True,
    ) -> np.ndarray:
        """
        Return binary predictions [N, 6] using tuned decision thresholds.
        Optionally enforces hierarchical rule (severe_toxic requires toxic).
        """
        probs = self.predict_proba(texts)
        preds = (probs >= self.threshold_arr).astype(np.int32)

        if apply_hierarchical_severe_toxic and "toxic" in self.labels and "severe_toxic" in self.labels:
            tox_idx = self.labels.index("toxic")
            sev_idx = self.labels.index("severe_toxic")
            preds[:, sev_idx] = preds[:, sev_idx] & preds[:, tox_idx]

        return preds
