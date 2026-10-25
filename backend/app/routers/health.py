"""
backend/app/routers/health.py
-----------------------------
Health check and environment status endpoint.
"""

from __future__ import annotations

from fastapi import APIRouter
import torch

from ..config import settings
from ..db import is_db_connected
from ..schemas import HealthResponse
from ..services.models import get_model_statuses, get_uptime_seconds

router = APIRouter(prefix="/api", tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return API health status, database connectivity, and loaded model count."""
    statuses = get_model_statuses()
    available_count = sum(1 for s in statuses.values() if s.get("available", False))
    total_count = len(statuses)

    has_cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_cuda else None

    # Mask MongoDB credentials in health check response
    uri = settings.effective_mongodb_uri
    if "@" in uri:
        masked_uri = "mongodb+srv://****:****@" + uri.split("@")[-1]
    else:
        masked_uri = uri

    db_ok = is_db_connected()
    api_status = "healthy" if (available_count > 0 and db_ok) else ("degraded" if available_count > 0 else "unhealthy")

    return HealthResponse(
        status=api_status,
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        mongodb_connected=db_ok,
        mongodb_uri_masked=masked_uri,
        models_available=available_count,
        total_models=total_count,
        gpu_available=has_cuda,
        gpu_device_name=gpu_name,
        uptime_seconds=get_uptime_seconds(),
    )
