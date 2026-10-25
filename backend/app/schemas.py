"""
backend/app/schemas.py
----------------------
Pydantic schemas for the Welcome to Civilization moderation API.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Payload for submitting comment text to moderation pipeline."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Raw input comment text (maximum 5000 characters).",
        examples=["This is a polite and constructive comment."],
    )

    @field_validator("text")
    @classmethod
    def validate_not_only_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text cannot be empty or only whitespace.")
        return v


class CensorSegment(BaseModel):
    """Text piece with censor flag for granular frontend masking."""
    text: str
    censored: bool


class CensorResponse(BaseModel):
    """Output from profanity lexicon censor engine."""
    original: str
    masked: str
    has_profanity: bool
    segments: list[CensorSegment]


class SeverityInfo(BaseModel):
    """Calibrated 0-10 severity score with severity band."""
    score: float = Field(..., ge=0.0, le=10.0, description="Severity score from 0.0 to 10.0")
    band: str = Field(..., description="Severity category: Low, Medium, High, or Critical")


class ModelPrediction(BaseModel):
    """Full moderation evaluation output from an individual model."""
    model_id: str
    model_name: str
    role: str
    available: bool = True
    device: str = "cpu"
    probabilities: dict[str, float] = Field(default_factory=dict)
    per_label_flags: dict[str, bool] = Field(default_factory=dict)
    normalized_scores: dict[str, float] = Field(default_factory=dict)
    severity: SeverityInfo = Field(default_factory=lambda: SeverityInfo(score=0.0, band="Low"))
    action: str = "Allow"
    action_color: str = "green"  # red, orange, yellow, green
    needs_review: bool = False
    reasoning: list[str] = Field(default_factory=list)
    execution_time_ms: float = 0.0
    error_message: str | None = None


class HeadlineRecommendation(BaseModel):
    """Production headline recommendation derived from Model 3 (DistilRoBERTa)."""
    model_id: str = "3_distilroberta"
    model_name: str = "Model 3: DistilRoBERTa"
    action: str = "Allow"
    action_color: str = "green"
    severity: SeverityInfo = Field(default_factory=lambda: SeverityInfo(score=0.0, band="Low"))
    needs_review: bool = False
    reasoning: list[str] = Field(default_factory=list)


class PredictResponse(BaseModel):
    """Comprehensive multi-model prediction output."""
    id: str
    timestamp: str
    text_length: int
    original_text: str
    censored_text: str
    censored_segments: list[CensorSegment]
    has_profanity: bool
    policy_version: str
    headline: HeadlineRecommendation
    models: dict[str, ModelPrediction]


class CommentItem(BaseModel):
    """Comment record structure stored in database."""
    id: str
    timestamp: str
    text_length: int
    original_text: str
    censored_text: str
    censored_segments: list[CensorSegment]
    has_profanity: bool
    policy_version: str
    headline: HeadlineRecommendation
    models: dict[str, ModelPrediction]


class CommentsListResponse(BaseModel):
    """Paginated collection of past moderation predictions."""
    items: list[dict[str, Any]]
    total: int
    page: int
    limit: int
    total_pages: int


class ModelInfo(BaseModel):
    """Metadata regarding a supported moderation model."""
    id: str
    name: str
    role: str
    description: str
    available: bool
    device: str
    thresholds: dict[str, float]
    remove_cuts: dict[str, float | None]
    error_message: str | None = None


class HealthResponse(BaseModel):
    """API health, runtime environment and model availability."""
    status: str
    version: str
    environment: str
    mongodb_connected: bool
    mongodb_uri_masked: str
    models_available: int
    total_models: int
    gpu_available: bool
    gpu_device_name: str | None
    uptime_seconds: float
