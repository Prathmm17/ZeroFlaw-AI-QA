"""FastAPI application entry point for ZeroFlaw AI-QA.

Provides REST and WebSocket endpoints for industrial real-time defect detection,
persisting inspection results, broadcasting alerts, and rendering analytics.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict

# Configure environment limits for low-memory container environments (e.g. Render 512MB Free Tier)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("ONNX_NUM_THREADS", "1")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.database import init_db
from backend.routers import alerts_router, analytics_router, detection_router, stream_router, video_router

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("visionguard")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown lifecycle events."""
    logger.info("Initializing ZeroFlaw AI-QA Backend...")
    
    # 1. Initialize SQLite database schemas
    init_db()
    logger.info("Database initialized successfully.")

    # Note: ML detector is loaded lazily on the first inference request
    # to guarantee fast startup (<1s) and prevent OOM under 512MB RAM constraints
    logger.info("ZeroFlaw AI-QA system active and ready. Ready to accept connections.")
    yield
    logger.info("ZeroFlaw AI-QA backend shutting down.")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Real-Time AI Quality Assurance and Defect Detection system "
        "for industrial manufacturing lines."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS for frontend access (supporting localhost and Firebase Hosting domains)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_origin_regex=r"^https://.*\.web\.app$|^https://.*\.firebaseapp\.com$|^https://.*\.onrender\.com$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure data directories exist before mounting
settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
settings.SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# Mount static file endpoints for images
app.mount("/data/uploads", StaticFiles(directory=str(settings.UPLOADS_DIR)), name="uploads")
app.mount("/data/results", StaticFiles(directory=str(settings.RESULTS_DIR)), name="results")
app.mount("/data/samples", StaticFiles(directory=str(settings.SAMPLES_DIR)), name="samples")

# Register Routers
app.include_router(detection_router)
app.include_router(alerts_router)
app.include_router(analytics_router)
app.include_router(stream_router)
app.include_router(video_router)


@app.get("/api/health", tags=["System"])
def health_check() -> Dict[str, str]:
    """Verify backend health status without eagerly loading heavy ML weights."""
    from backend.services.detection_service import _detector_instance
    is_ready = _detector_instance.is_ready() if _detector_instance is not None else True
    model_name = _detector_instance.model_name if _detector_instance is not None else settings.MODEL_NAME
    device = _detector_instance.device if _detector_instance is not None else settings.DEVICE
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "model_ready": str(is_ready),
        "model_name": model_name,
        "device": device,
    }


@app.get("/", tags=["System"])
def root() -> Dict[str, str]:
    """Root endpoint welcoming API consumers and linking to documentation."""
    return {
        "message": f"Welcome to {settings.APP_NAME} API v{settings.APP_VERSION}",
        "docs": "/docs",
        "health": "/api/health",
    }

