"""
backend/app/routers/research.py
-------------------------------
Research tab backend endpoints.
Serves benchmark tables, EDA findings, split statistics, model configs,
and paired bootstrap significance tests dynamically from disk without hardcoded numbers.
"""

from __future__ import annotations

from typing import Any
from fastapi import APIRouter

from ..services.research import (
    get_research_overview,
    get_eda_data,
    get_splits_data,
    get_models_config_data,
    get_results_data,
    get_comparisons_data,
    get_imbalance_data,
    get_limitations_data,
)

router = APIRouter(prefix="/api/research", tags=["Research"])


@router.get("/overview")
async def research_overview() -> dict[str, Any]:
    """Overview table and headline benchmark comparisons."""
    return get_research_overview()


@router.get("/eda")
async def research_eda() -> dict[str, Any]:
    """Exploratory data analysis findings, conclusions, and figure links."""
    return get_eda_data()


@router.get("/splits")
async def research_splits() -> dict[str, Any]:
    """Stratified 80/10/10 split statistics and duplicate grouping methodology."""
    return get_splits_data()


@router.get("/models")
async def research_models() -> dict[str, Any]:
    """Model architectures, Optuna hyperparameter trials, and metadata."""
    return get_models_config_data()


@router.get("/results")
async def research_results() -> dict[str, Any]:
    """Test set performance, per-label metrics, and bootstrap confidence intervals."""
    return get_results_data()


@router.get("/comparisons")
async def research_comparisons() -> dict[str, Any]:
    """Paired bootstrap significance testing tables vs Model 2b and Detoxify."""
    return get_comparisons_data()


@router.get("/imbalance")
async def research_imbalance() -> dict[str, Any]:
    """Extreme class imbalance statistics and derivation of asymmetric moderation rules."""
    return get_imbalance_data()


@router.get("/limitations")
async def research_limitations() -> dict[str, Any]:
    """Model constraints, sequence truncation boundaries, and Detoxify comparative footnote."""
    return get_limitations_data()
