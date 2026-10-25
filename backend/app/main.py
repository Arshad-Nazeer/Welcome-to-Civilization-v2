"""
backend/app/main.py
-------------------
Main FastAPI application entrypoint for Welcome to Civilization.
Configures CORS, lifespan initialization, static files mounts for figures,
and registers all API endpoints.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import settings
from .db import connect_to_mongo, close_mongo_connection
from .routers import predict, comments, models, censor, research, health
from .services.models import initialize_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("wtc.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: connects to DB and pre-loads ML models at startup."""
    logger.info("Starting up Welcome to Civilization Moderation API v%s...", settings.VERSION)

    # 1. Connect to MongoDB Atlas / Local
    await connect_to_mongo()

    # 2. Pre-load all moderation models once
    initialize_models()

    logger.info("Application startup complete. Ready to receive requests.")
    yield

    # Shutdown
    logger.info("Shutting down application...")
    await close_mongo_connection()
    logger.info("Application shutdown finished.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Full-stack AI content moderation system built upon multi-label Jigsaw toxicity models.",
    lifespan=lifespan,
)

# CORS middleware for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static figure files for research tab
if settings.EDA_FIGURES_PATH.exists():
    app.mount("/static/eda", StaticFiles(directory=str(settings.EDA_FIGURES_PATH)), name="eda_figures")

if settings.VISUALIZATIONS_PATH.exists():
    app.mount("/static/vis", StaticFiles(directory=str(settings.VISUALIZATIONS_PATH)), name="model_visualizations")

# Register routers
app.include_router(predict.router)
app.include_router(comments.router)
app.include_router(models.router)
app.include_router(censor.router)
app.include_router(research.router)
app.include_router(health.router)


@app.get("/", tags=["Root"])
async def root_redirect():
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs_url": "/docs",
        "health_url": "/api/health",
    }
