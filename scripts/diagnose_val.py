"""Diagnostic Script: Ultralytics val on models/best.pt."""

from __future__ import annotations

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
# PyTorch 2.6 weights_only patch
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

from ultralytics import YOLO

def main():
    model_path = PROJECT_ROOT / "models" / "best.pt"
    data_yaml = PROJECT_ROOT / "data" / "real_industrial_dataset" / "dataset.yaml"

    print(f"Loading model: {model_path}")
    print(f"Dataset YAML: {data_yaml}")

    model = YOLO(str(model_path))

    # Run validation
    # Ultralytics val supports split='val' or split='test' (if defined)
    # Check if 'test' split is defined in yaml, otherwise use val
    print("\nRunning Ultralytics val (split='val', plots=True)...")
    metrics = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=640,
        batch=16,
        device="cpu",
        plots=True,
        save_json=False,
        project=str(PROJECT_ROOT / "runs" / "diagnostics"),
        name="val_best",
        exist_ok=True,
    )

    print("\n" + "=" * 70)
    print("ULTRALYTICS VALIDATION RESULTS ON models/best.pt")
    print("=" * 70)
    print(f"Overall Precision (P)    : {metrics.results_dict.get('metrics/precision(B)', 0.0):.4f}")
    print(f"Overall Recall (R)       : {metrics.results_dict.get('metrics/recall(B)', 0.0):.4f}")
    print(f"Overall mAP@50           : {metrics.results_dict.get('metrics/mAP50(B)', 0.0):.4f}")
    print(f"Overall mAP@50-95        : {metrics.results_dict.get('metrics/mAP50-95(B)', 0.0):.4f}")
    print("-" * 70)
    print("Per-Class AP:")
    for idx, name in enumerate(metrics.names.values()):
        ap50 = metrics.box.ap50[idx] if idx < len(metrics.box.ap50) else 0.0
        ap = metrics.box.ap[idx] if idx < len(metrics.box.ap) else 0.0
        p = metrics.box.p[idx] if idx < len(metrics.box.p) else 0.0
        r = metrics.box.r[idx] if idx < len(metrics.box.r) else 0.0
        print(f"  Class {idx} ({name:<16}) : AP50={ap50:.4f} | AP50-95={ap:.4f} | P={p:.4f} | R={r:.4f}")

    print("=" * 70)
    save_dir = Path(metrics.save_dir)
    print(f"Results & Plots saved to: {save_dir}")
    cm_path = save_dir / "confusion_matrix.png"
    cm_norm_path = save_dir / "confusion_matrix_normalized.png"
    print(f"Confusion Matrix saved   : {cm_path.is_file()} ({cm_path})")
    print(f"Normalized Matrix saved  : {cm_norm_path.is_file()} ({cm_norm_path})")

if __name__ == "__main__":
    main()

