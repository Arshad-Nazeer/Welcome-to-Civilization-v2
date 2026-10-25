"""
backend/app/routers/censor.py
-----------------------------
Stand-alone profanity detection and masking endpoint.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field
from ..schemas import CensorResponse
from ..services.censor import censor_text

router = APIRouter(prefix="/api/censor", tags=["Censor"])


class CensorInput(BaseModel):
    text: str = Field(..., max_length=5000, description="Text to scan for profanity and mask.")


@router.post("", response_model=CensorResponse)
async def censor_endpoint(req: CensorInput) -> CensorResponse:
    """
    Scan text against the profanity lexicon with leetspeak and inflection normalization.
    Returns masked text and segmented tokens with censored booleans for per-word toggle.
    """
    res = censor_text(req.text)
    return CensorResponse(
        original=res["original"],
        masked=res["masked"],
        has_profanity=res["has_profanity"],
        segments=res["segments"],
    )
