"""Industrial Defect Model Training & Transfer Pipeline.

Fine-tunes YOLOv8 on the NEU-DET surface benchmark dataset, saving production
weights to models/industrial_best.pt and models/best.pt.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# NumPy 2.x compatibility patch for ultralytics metrics (np.trapz -> np.trapezoid)
import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = getattr(np, "trapezoid", None)

# PyTorch 2.6+ safe_globals / weights_only patch
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

from ml.model_registry import DEFECT_CLASSES, ModelRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("train_industrial")


def train_industrial_model(
    dataset_yaml: Path | None = None,
    epochs: int = 5,
    batch_size: int = 8,
    img_size: int = 640,
    device: str = "cpu",
) -> Path:
    """Train YOLOv8 on the industrial surface defect benchmark.

    Args:
        dataset_yaml: Path to dataset.yaml.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        img_size: Image size for training.
        device: 'cpu' or 'cuda'.

    Returns:
        Path to fine-tuned industrial_best.pt.
    """
    yaml_path = dataset_yaml or (PROJECT_ROOT / "data" / "industrial_benchmark" / "dataset.yaml")
    if not yaml_path.is_file():
        raise FileNotFoundError(f"dataset.yaml not found at {yaml_path}")

    base_model_path = PROJECT_ROOT / "models" / "yolov8n.pt"
    if not base_model_path.is_file():
        base_model_path = Path("yolov8n.pt")

    logger.info("Initializing YOLO model from %s...", base_model_path)
    model = YOLO(str(base_model_path))

    project_dir = PROJECT_ROOT / "runs" / "industrial_train"
    logger.info(
        "Starting Industrial Fine-Tuning (%d epochs, batch=%d, device=%s)...",
        epochs,
        batch_size,
        device,
    )

    results = model.train(
        data=str(yaml_path).replace("\\", "/"),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=str(project_dir),
        name="neu_det",
        exist_ok=True,
        verbose=True,
        plots=False,
        freeze=10,
        warmup_epochs=0.0,
        lr0=0.01,
    )

    # Resolve resulting best.pt from run output
    run_best = project_dir / "neu_det" / "weights" / "best.pt"
    run_last = project_dir / "neu_det" / "weights" / "last.pt"

    trained_weights = run_best if run_best.is_file() else run_last
    if not trained_weights.is_file():
        raise RuntimeError(f"Training completed but weights not found at {run_best} or {run_last}")

    # Copy to models directory
    dest_industrial = PROJECT_ROOT / "models" / "industrial_best.pt"
    dest_best = PROJECT_ROOT / "models" / "best.pt"

    shutil.copyfile(trained_weights, dest_industrial)
    shutil.copyfile(trained_weights, dest_best)

    logger.info("Industrial weights successfully saved to: %s", dest_industrial)
    logger.info("Activated as primary model at: %s", dest_best)
    return dest_industrial


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VisionGuard Industrial Model")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    args = parser.parse_args()

    trained = train_industrial_model(epochs=args.epochs, batch_size=args.batch, device=args.device)
    print("\nIndustrial training complete! Weights:", trained)
