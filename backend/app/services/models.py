"""
backend/app/services/models.py
------------------------------
Model loader and inference service for the Welcome to Civilization moderation API.
Reuses the existing Moderator registry in src/wtc/inference.py.
Loads each model once at startup and auto-detects GPU/CPU.
Provides isolated error handling: if any model fails to load,
it is marked unavailable rather than crashing the server.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any

from ..config import settings

# Ensure project root src is on sys.path to access wtc package
sys.path.insert(0, str(settings.REPO_ROOT_PATH / "src"))

logger = logging.getLogger("wtc.models")

# Exactly the four specified models
SUPPORTED_MODELS: dict[str, dict[str, Any]] = {
    "1_tfidf_lr": {
        "name": "Model 1: TF-IDF + Logistic Regression",
        "role": "Baseline (Linear & N-gram)",
        "description": "Word (50k) and Character (100k) TF-IDF n-grams with independent L2-regularized logistic regressions.",
        "is_external": False,
    },
    "2b_bigru_attn_v2": {
        "name": "Model 2b: BiGRU + Attention",
        "role": "Recurrent Neural Network",
        "description": "Bidirectional GRU with additive attention pooling and fine-tuned fastText embeddings with casing preserved.",
        "is_external": False,
    },
    "3_distilroberta": {
        "name": "Model 3: DistilRoBERTa",
        "role": "Headline Serving Transformer",
        "description": "Fine-tuned DistilRoBERTa-base (~82M parameters) with layer-wise learning rate decay and damped class weights.",
        "is_external": False,
    },
    "detoxify": {
        "name": "Detoxify (external pretrained model, Unitary AI, not trained in this project)",
        "role": "External Pretrained Baseline",
        "description": "Detoxify (external pretrained model, Unitary AI, not trained in this project). BERT-base uncased checkpoint run locally via the detoxify package.",
        "is_external": True,
    },
}

# In-memory model registry cache
_MODERATOR_INSTANCES: dict[str, Any] = {}
_MODEL_STATUSES: dict[str, dict[str, Any]] = {}
_STARTUP_TIME = time.time()


def initialize_models() -> None:
    """Load each model once at startup into memory with isolated try/except."""
    global _MODERATOR_INSTANCES, _MODEL_STATUSES
    logger.info("Initializing moderation models from %s...", settings.REPO_ROOT_PATH / "models")

    from wtc.inference import Moderator
    import torch

    has_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if has_cuda else "CPU"
    logger.info("Hardware acceleration: CUDA=%s (%s)", has_cuda, device_name)

    for model_id, info in SUPPORTED_MODELS.items():
        t0 = time.perf_counter()
        try:
            logger.info("Loading model [%s] (%s)...", model_id, info["name"])
            moderator = Moderator(model_id)

            # Warm-up call with dummy text
            _ = moderator.predict_proba(["warmup test comment"])
            elapsed = time.perf_counter() - t0

            # Determine device used
            model_device = "cpu"
            if hasattr(moderator.adapter, "device"):
                model_device = str(moderator.adapter.device)
            elif model_id in ["2b_bigru_attn_v2", "3_distilroberta", "detoxify"] and has_cuda:
                model_device = "cuda"

            _MODERATOR_INSTANCES[model_id] = moderator
            _MODEL_STATUSES[model_id] = {
                "available": True,
                "device": model_device,
                "load_time_sec": round(elapsed, 2),
                "error_message": None,
                "thresholds": moderator.thresholds,
            }
            logger.info("Model [%s] successfully loaded in %.2fs on %s", model_id, elapsed, model_device)
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.error("Failed to load model [%s]: %s. Marking unavailable.", model_id, exc)
            _MODEL_STATUSES[model_id] = {
                "available": False,
                "device": "unavailable",
                "load_time_sec": round(elapsed, 2),
                "error_message": str(exc),
                "thresholds": {},
            }


def get_model_statuses() -> dict[str, dict[str, Any]]:
    """Return dictionary of statuses and metadata for all models."""
    results = {}
    for model_id, info in SUPPORTED_MODELS.items():
        st = _MODEL_STATUSES.get(model_id, {"available": False, "device": "unknown", "error_message": "Not initialized"})
        results[model_id] = {
            "id": model_id,
            "name": info["name"],
            "role": info["role"],
            "description": info["description"],
            "is_external": info["is_external"],
            "available": st.get("available", False),
            "device": st.get("device", "cpu"),
            "load_time_sec": st.get("load_time_sec", 0.0),
            "error_message": st.get("error_message"),
            "thresholds": st.get("thresholds", {}),
        }
    return results


def predict_single_model(model_id: str, text: str) -> dict[str, Any]:
    """Execute prediction on a single model, returning probabilities and execution time."""
    from wtc.config import LABELS

    status = _MODEL_STATUSES.get(model_id, {})
    if not status.get("available", False) or model_id not in _MODERATOR_INSTANCES:
        return {
            "available": False,
            "probabilities": {l: 0.0 for l in LABELS},
            "execution_time_ms": 0.0,
            "error_message": status.get("error_message") or f"Model '{model_id}' is unavailable.",
        }

    moderator = _MODERATOR_INSTANCES[model_id]
    t0 = time.perf_counter()
    try:
        # Predict continuous probabilities [1, 6]
        probs_matrix = moderator.predict_proba([text])
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        probs_dict = {
            label: float(round(probs_matrix[0, idx], 6))
            for idx, label in enumerate(moderator.labels)
        }
        return {
            "available": True,
            "probabilities": probs_dict,
            "execution_time_ms": round(elapsed_ms, 2),
            "error_message": None,
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.error("Error predicting with model [%s]: %s", model_id, exc)
        return {
            "available": False,
            "probabilities": {l: 0.0 for l in LABELS},
            "execution_time_ms": round(elapsed_ms, 2),
            "error_message": str(exc),
        }


def predict_all_models(text: str) -> dict[str, dict[str, Any]]:
    """Predict probabilities across all four configured models."""
    predictions = {}
    for model_id in SUPPORTED_MODELS:
        predictions[model_id] = predict_single_model(model_id, text)
    return predictions


def get_uptime_seconds() -> float:
    return round(time.time() - _STARTUP_TIME, 1)
