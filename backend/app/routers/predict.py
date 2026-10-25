"""
backend/app/routers/predict.py
------------------------------
Primary moderation prediction endpoint.
Predicts toxicity across all four models, applies imbalance-aware policy rules,
derives Model 3 headline decision, censors profanity, and persists the record to MongoDB.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
import uuid
from fastapi import APIRouter, HTTPException, status

from ..config import settings
from ..db import save_comment
from ..schemas import PredictRequest, PredictResponse, HeadlineRecommendation, ModelPrediction
from ..services.censor import censor_text
from ..services.models import predict_all_models, get_model_statuses
from ..services.policy import evaluate_all_models_decision, load_policy_config

logger = logging.getLogger("wtc.router.predict")
router = APIRouter(prefix="/api", tags=["Prediction"])


@router.post("/predict", response_model=PredictResponse, status_code=status.HTTP_200_OK)
async def predict_moderation(req: PredictRequest) -> PredictResponse:
    """
    Moderate text across all available models.
    Applies asymmetric decision rules, calculates severity, flags actions,
    and returns Model 3 headline recommendations.
    """
    text = req.text
    if len(text) > settings.MAX_INPUT_CHARS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Input text exceeds maximum allowed length of {settings.MAX_INPUT_CHARS} characters.",
        )

    # 1. Independent profanity censoring
    censor_res = censor_text(text)

    # 2. Predict raw probabilities from all models
    model_predictions_raw = predict_all_models(text)
    model_statuses = get_model_statuses()

    # Collect raw probabilities for available models
    available_statuses = {m: model_statuses[m]["available"] for m in model_predictions_raw}
    raw_probs_by_model = {
        m: model_predictions_raw[m]["probabilities"]
        for m in model_predictions_raw
        if available_statuses.get(m, False)
    }

    # 3. Policy evaluation
    policy_cfg = load_policy_config()
    policy_version = policy_cfg.get("policy_version", "1.0.0")
    model_decisions, headline_data = evaluate_all_models_decision(
        all_models_raw_probs=raw_probs_by_model,
        available_statuses=available_statuses,
    )

    # 4. Assemble per-model response schemas
    assembled_models: dict[str, ModelPrediction] = {}
    for m_id, raw_res in model_predictions_raw.items():
        st = model_statuses.get(m_id, {})
        is_avail = st.get("available", False)

        if is_avail and m_id in model_decisions:
            decision = model_decisions[m_id]
            assembled_models[m_id] = ModelPrediction(
                model_id=m_id,
                model_name=st.get("name", m_id),
                role=st.get("role", "Moderation Model"),
                available=True,
                device=st.get("device", "cpu"),
                probabilities=decision["probabilities"],
                per_label_flags=decision["per_label_flags"],
                normalized_scores=decision["normalized_scores"],
                severity=decision["severity"],
                action=decision["action"],
                action_color=decision["action_color"],
                needs_review=decision["needs_review"],
                reasoning=decision["reasoning"],
                execution_time_ms=raw_res.get("execution_time_ms", 0.0),
                error_message=None,
            )
        else:
            # Model is unavailable
            assembled_models[m_id] = ModelPrediction(
                model_id=m_id,
                model_name=st.get("name", m_id),
                role=st.get("role", "Moderation Model"),
                available=False,
                device="unavailable",
                probabilities={},
                per_label_flags={},
                normalized_scores={},
                severity={"score": 0.0, "band": "Low"},
                action="Unavailable",
                action_color="gray",
                needs_review=False,
                reasoning=["Model artifact unavailable or failed to initialize."],
                execution_time_ms=0.0,
                error_message=raw_res.get("error_message") or st.get("error_message") or "Model unavailable",
            )

    # 5. Assemble Headline Recommendation
    headline = HeadlineRecommendation(
        model_id=headline_data["model_id"],
        model_name=headline_data["model_name"],
        action=headline_data["action"],
        action_color=headline_data["action_color"],
        severity=headline_data["severity"],
        needs_review=headline_data["needs_review"],
        reasoning=headline_data["reasoning"],
    )

    doc_id = uuid.uuid4().hex[:16]
    now_utc = datetime.now(timezone.utc).isoformat()

    response_obj = PredictResponse(
        id=doc_id,
        timestamp=now_utc,
        text_length=len(text),
        original_text=text,  # Stored completely uncensored
        censored_text=censor_res["masked"],
        censored_segments=censor_res["segments"],
        has_profanity=censor_res["has_profanity"],
        policy_version=policy_version,
        headline=headline,
        models=assembled_models,
    )

    # 6. Asynchronously save document to MongoDB
    doc_payload = response_obj.model_dump()
    await save_comment(doc_payload)

    return response_obj
