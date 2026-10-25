import sentencepiece
import sys
from pathlib import Path
import numpy as np
import pytest

from app.config import settings
from app.services.models import predict_single_model, initialize_models

# Ensure repo root src is in sys.path
sys.path.insert(0, str(settings.REPO_ROOT_PATH / "src"))
from wtc.config import LABELS, get_predictions_dir
from wtc.data import load_split


@pytest.fixture(scope="module", autouse=True)
def setup_models():
    initialize_models()


def test_api_parity_model3_distilroberta():
    """
    Parity test: Verify that Model 3 inference through the backend service
    reproduces the stored test probabilities in predictions/3_distilroberta/probs_test.npy
    within a float tolerance of 1e-4 for 5 test samples.
    """
    df_test = load_split("test")
    preds_dir = get_predictions_dir("3_distilroberta")
    probs_test = np.load(preds_dir / "probs_test.npy")

    # Evaluate 5 verified test samples
    sample_indices = [0, 1, 2, 4, 6]
    for idx in sample_indices:
        text = df_test.iloc[idx]["text_clean"]
        expected_probs = probs_test[idx]

        res = predict_single_model("3_distilroberta", text)
        assert res["available"] is True
        actual_probs = np.array([res["probabilities"][lbl] for lbl in LABELS])

        diff = float(np.max(np.abs(actual_probs - expected_probs)))
        assert diff < 1e-4, f"Parity mismatch on test row {idx}: max diff = {diff:.6e} >= 1e-4"


def test_api_parity_model1_tfidf():
    """Verify Model 1 TF-IDF parity on 5 test samples."""
    df_test = load_split("test")
    preds_dir = get_predictions_dir("1_tfidf_lr")
    probs_test = np.load(preds_dir / "probs_test.npy")

    sample_indices = [0, 1, 2, 3, 4]
    for idx in sample_indices:
        text = df_test.iloc[idx]["text_clean"]
        expected_probs = probs_test[idx]

        res = predict_single_model("1_tfidf_lr", text)
        assert res["available"] is True
        actual_probs = np.array([res["probabilities"][lbl] for lbl in LABELS])

        diff = float(np.max(np.abs(actual_probs - expected_probs)))
        assert diff < 1e-4, f"Parity mismatch on test row {idx} for Model 1: max diff = {diff:.6e}"
