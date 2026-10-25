"""
backend/app/routers/models.py
-----------------------------
Model registry and metadata endpoint for Welcome to Civilization.
Lists all 4 supported models, their roles, hardware device, and calibrated thresholds.
"""

from __future__ import annotations

from fastapi import APIRouter
from ..schemas import ModelInfo
from ..services.models import get_model_statuses
from ..services.policy import load_policy_config

router = APIRouter(prefix="/api/models", tags=["Models"])


@router.get("", response_model=list[ModelInfo])
async def list_models() -> list[ModelInfo]:
    """Retrieve metadata and runtime availability for all four moderation models."""
    statuses = get_model_statuses()
    policy_cfg = load_policy_config()
    policy_models = policy_cfg.get("models", {})

    items: list[ModelInfo] = []
    for m_id, info in statuses.items():
        pol = policy_models.get(m_id, {})
        thresholds = pol.get("thresholds", info.get("thresholds", {}))
        remove_cuts = pol.get("remove_cuts", {})

        items.append(
            ModelInfo(
                id=m_id,
                name=info["name"],
                role=info["role"],
                description=info["description"],
                available=info["available"],
                device=info["device"],
                thresholds=thresholds,
                remove_cuts=remove_cuts,
                error_message=info.get("error_message"),
            )
        )
    return items
