"""Application configuration for VisionGuard QA.

Centralized settings management using pydantic-settings.
Supports environment variables and .env files with cross-platform Path defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_project_root() -> Path:
    """Resolve the project root directory."""
    current = Path(__file__).resolve().parent
    while current.parent != current:
        if (current / "requirements.txt").is_file():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = get_project_root()


class Settings(BaseSettings):
    """Application settings with environment variable override support."""

    # Application
    APP_NAME: str = "ZeroFlaw AI-QA"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = int(os.environ.get("PORT", "8000"))
    CORS_ORIGINS: list[str] = [
        "*",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
    ]

    # Paths
    PROJECT_ROOT: Path = PROJECT_ROOT
    DATA_DIR: Path = PROJECT_ROOT / "data"
    UPLOADS_DIR: Path = PROJECT_ROOT / "data" / "uploads"
    RESULTS_DIR: Path = PROJECT_ROOT / "data" / "results"
    SAMPLES_DIR: Path = PROJECT_ROOT / "data" / "samples"
    DB_DIR: Path = PROJECT_ROOT / "data" / "db"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    DEFECT_KNOWLEDGE_PATH: Path = PROJECT_ROOT / "backend" / "defect_knowledge.json"

    # Database
    DATABASE_URL: str = f"sqlite:///{DB_DIR / 'visionguard.db'}"

    # ML Inference & Slicing
    MODEL_NAME: str = "best.onnx" if (PROJECT_ROOT / "models" / "best.onnx").is_file() else "best.pt"
    FALLBACK_MODEL_NAME: str = "yolov8n.pt"
    CONFIDENCE_THRESHOLD: float = 0.20
    DEVICE: str = "cpu"
    ENABLE_TILING: bool = True
    SLICE_SIZE: int = 512
    TILING_MIN_DIM: int = 1600  # Default 1600 px on the longest side trigger
    PX_PER_MM: float | None = None  # None = uncalibrated (px); if set, enables physical mm metrology
    STREAM_MAX_SIZE: int = 640  # Max dimension for real-time video stream frames
    STREAM_JPEG_QUALITY: float = 0.6  # WebRTC/WebSocket JPEG compression quality
    ONNX_NUM_THREADS: int = os.cpu_count() or 4

    # Alert Thresholds
    CRITICAL_CONFIDENCE: float = 0.80
    CRITICAL_AREA_RATIO: float = 0.05
    MAJOR_CONFIDENCE: float = 0.55

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Global settings singleton
settings = Settings()

# Ensure directories exist
for directory in [
    settings.DATA_DIR,
    settings.UPLOADS_DIR,
    settings.RESULTS_DIR,
    settings.SAMPLES_DIR,
    settings.DB_DIR,
    settings.MODELS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

