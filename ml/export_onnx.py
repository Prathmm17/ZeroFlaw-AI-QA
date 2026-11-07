"""ONNX Model Exporter for Edge & CPU Optimization.

Exports PyTorch YOLO weights (models/industrial_best.pt) to ONNX format
with dynamic input dimensions and metadata for low-latency edge deployment.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Project root in sys.path
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
logger = logging.getLogger("export_onnx")


def export_model_to_onnx(
    weights_path: Path | None = None,
    img_size: int = 640,
    opset: int = 12,
) -> Path:
    """Export fine-tuned YOLO model to ONNX format.

    Args:
        weights_path: Path to .pt weights file.
        img_size: Input tensor dimension.
        opset: ONNX operator set version.

    Returns:
        Path to exported .onnx model file.
    """
    model_file = weights_path or (PROJECT_ROOT / "models" / "industrial_best.pt")
    if not model_file.is_file():
        model_file = PROJECT_ROOT / "models" / "best.pt"

    if not model_file.is_file():
        raise FileNotFoundError(f"Model weights not found at {model_file}")

    logger.info("Loading PyTorch model from %s...", model_file)
    model = YOLO(str(model_file))

    logger.info("Exporting model to ONNX (imgsz=%d, opset=%d, dynamic=True)...", img_size, opset)
    exported_path = model.export(
        format="onnx",
        imgsz=img_size,
        dynamic=True,
        opset=opset,
        simplify=False,
    )

    onnx_file = Path(exported_path)
    logger.info("Successfully exported ONNX model to: %s (Size: %.2f MB)", onnx_file, onnx_file.stat().st_size / (1024 * 1024))
    return onnx_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export VisionGuard YOLO to ONNX")
    parser.add_argument("--weights", type=str, default="", help="Path to .pt weights")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    args = parser.parse_args()

    pt_path = Path(args.weights) if args.weights else None
    exported = export_model_to_onnx(weights_path=pt_path, img_size=args.imgsz)
    print("\nONNX Export complete:", exported)

