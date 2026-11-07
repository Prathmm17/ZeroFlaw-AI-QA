"""Deep Fine-Tuning Pipeline on Multi-Domain Industrial Dataset.

Fine-tunes YOLOv8 on 600 real industrial specimens across SEM micrographs,
polymer assemblies, and brushed steel surfaces with Cosine Annealing, saving
production weights to models/industrial_v2_best.pt and models/best.pt.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# NumPy 2.x patch
import numpy as np
if not hasattr(np, "trapz"):
    np.trapz = getattr(np, "trapezoid", None)

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("train_real_industrial")


def train_v2_model(
    epochs: int = 8,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = "cpu",
) -> Path:
    yaml_path = PROJECT_ROOT / "data" / "real_industrial_dataset" / "dataset.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"dataset.yaml not found at {yaml_path}")

    base_weights = PROJECT_ROOT / "models" / "yolov8n.pt"
    if not base_weights.is_file():
        base_weights = Path("yolov8n.pt")

    logger.info("Initializing YOLO from %s...", base_weights)
    model = YOLO(str(base_weights))

    project_dir = PROJECT_ROOT / "runs" / "industrial_v2_train"

    logger.info("Starting multi-domain fine-tuning (%d epochs, batch=%d, device=%s)...", epochs, batch_size, device)
    results = model.train(
        data=str(yaml_path).replace("\\", "/"),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=str(project_dir),
        name="multi_domain",
        exist_ok=True,
        verbose=True,
        plots=False,
        freeze=10,
        warmup_epochs=0.0,
        cos_lr=True,
        lr0=0.01,
    )

    run_best = project_dir / "multi_domain" / "weights" / "best.pt"
    run_last = project_dir / "multi_domain" / "weights" / "last.pt"
    trained_weights = run_best if run_best.is_file() else run_last

    dest_v2 = PROJECT_ROOT / "models" / "industrial_v2_best.pt"
    dest_best = PROJECT_ROOT / "models" / "best.pt"

    shutil.copyfile(trained_weights, dest_v2)
    shutil.copyfile(trained_weights, dest_best)

    logger.info("Trained weights saved to: %s", dest_v2)
    logger.info("Activated as primary model at: %s", dest_best)
    return dest_v2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    trained = train_v2_model(epochs=args.epochs, batch_size=args.batch, device=args.device)
    print("\n[SUCCESS] Multi-domain model training complete! Weights:", trained)

