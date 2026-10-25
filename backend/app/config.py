"""
backend/app/config.py
---------------------
Backend configuration loaded via Pydantic Settings.
Reads environment variables from backend/.env or environment.
"""

from __future__ import annotations

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root is two levels up from backend/app
BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    PROJECT_NAME: str = "Welcome to Civilization — Toxicity Moderation API"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"

    # Host & Port
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # MongoDB Atlas / Local
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGO_URI: str | None = None
    MONGODB_DB_NAME: str = "wtc"
    MONGODB_TIMEOUT_MS: int = 5000

    # Content constraints
    MAX_INPUT_CHARS: int = 5000

    # CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*",
    ]

    # Paths to project artifacts
    REPO_ROOT_PATH: Path = REPO_ROOT
    BACKEND_PATH: Path = BACKEND_DIR
    POLICY_PATH: Path = BACKEND_DIR / "policy.yaml"
    CURSE_WORDS_PATH: Path = BACKEND_DIR / "data" / "curse_words.txt"
    EDA_FIGURES_PATH: Path = REPO_ROOT / "reports" / "eda" / "figures"
    VISUALIZATIONS_PATH: Path = REPO_ROOT / "visualizations"
    REPORTS_PATH: Path = REPO_ROOT / "reports"
    DATA_PATH: Path = REPO_ROOT / "data"

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def effective_mongodb_uri(self) -> str:
        """Prefers MONGODB_URI if valid, otherwise falls back to MONGO_URI."""
        if self.MONGODB_URI and self.MONGODB_URI != "mongodb://localhost:27017":
            return self.MONGODB_URI
        if self.MONGO_URI:
            return self.MONGO_URI
        return self.MONGODB_URI


settings = Settings()
