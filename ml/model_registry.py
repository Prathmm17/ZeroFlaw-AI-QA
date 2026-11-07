"""Model registry for VisionGuard QA.

Manages YOLOv8 model loading, path resolution, and industrial defect class mapping.
Supports fine-tuned metallurgical models with automatic fallback to base models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

# PyTorch 2.6+ weights_only compatibility patch for ultralytics checkpoints
import torch

try:
    _orig_torch_load = torch.load

    def _safe_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)

    torch.load = _safe_torch_load
except Exception:
    pass

from ultralytics import YOLO

logger: logging.Logger = logging.getLogger(__name__)

# Standard NEU-DET Industrial Surface Defect Benchmark Classes
DEFECT_CLASSES: dict[int, str] = {
    0: "crazing",           # Network of micro-cracks
    1: "inclusion",         # Non-metallic impurities / slag
    2: "patches",           # Surface oxidation / irregular coating
    3: "pitted_surface",    # Cavities from mechanical / chemical erosion
    4: "rolled_in_scale",   # Mill scale pressed into surface during rolling
    5: "scratches",         # Linear abrasive tool marks
}


class ModelRegistry:
    """Registry that manages YOLOv8 model discovery, loading, and metadata."""

    _BASE_MODEL_NAME: str = "yolov8n.pt"

    def __init__(self, models_dir: Optional[Path] = None) -> None:
        """Initialise the model registry."""
        if models_dir is not None:
            self.models_dir: Path = Path(models_dir)
        else:
            self.models_dir = ModelRegistry.get_project_root() / "models"

        self.models_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ModelRegistry initialised — models_dir: %s", self.models_dir)

    def get_model_path(self, model_name: str = "best.pt") -> Path:
        """Resolve path to weights file, prioritizing industrial fine-tuned models."""
        target_path: Path = self.models_dir / model_name
        if target_path.is_file():
            logger.info("Found model weights: %s", target_path)
            return target_path

        ind_path = self.models_dir / "industrial_best.pt"
        if ind_path.is_file():
            logger.info("Found industrial fine-tuned weights: %s", ind_path)
            return ind_path

        # Fallback to base model
        fallback_path = self.models_dir / self._BASE_MODEL_NAME
        if fallback_path.is_file():
            logger.info("Using cached base model: %s", fallback_path)
            return fallback_path
            return fallback_path

        logger.warning(
            "Model '%s' not found at %s — falling back to base %s",
            model_name,
            fine_tuned_path,
            self._BASE_MODEL_NAME,
        )
        return Path(self._BASE_MODEL_NAME)

    def load_model(self, model_name: str = "best.pt") -> YOLO:
        """Load a YOLO model ready for inference."""
        model_path: Path = self.get_model_path(model_name)
        try:
            logger.info("Loading YOLO model from: %s", model_path)
            model: YOLO = YOLO(str(model_path), task="detect")
            logger.info("Model loaded successfully: %s", model_path)
            return model
        except Exception as exc:
            logger.exception("Failed to load model from '%s'", model_path)
            raise RuntimeError(f"Could not load YOLO model from '{model_path}': {exc}") from exc

    def get_available_models(self) -> list[dict[str, str]]:
        """List every model weight file found in the models directory."""
        models: list[dict[str, str]] = []
        try:
            for weight_file in sorted(self.models_dir.glob("*.pt")):
                if weight_file.is_file():
                    m_type = "industrial" if "industrial" in weight_file.name else "fine-tuned"
                    models.append({
                        "name": weight_file.name,
                        "path": str(weight_file.resolve()),
                        "type": m_type,
                    })
        except OSError as exc:
            logger.error("Error scanning models directory: %s", exc)

        models.append({
            "name": self._BASE_MODEL_NAME,
            "path": self._BASE_MODEL_NAME,
            "type": "base",
        })
        return models

    @staticmethod
    def get_project_root() -> Path:
        """Discover project root containing requirements.txt."""
        current: Path = Path(__file__).resolve().parent
        while True:
            if (current / "requirements.txt").is_file():
                return current
            parent: Path = current.parent
            if parent == current:
                break
            current = parent
        return Path(__file__).resolve().parent.parent
