"""Evaluate models/best.pt on the validation split and sweep confidence thresholds per class.

Finds the optimal F1 threshold for each defect class, saves the result to
backend/class_thresholds.json, and prints mAP50, Precision, Recall, and the per-class table.
"""

from __future__ import annotations

import json
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


def main() -> None:
    model_path = PROJECT_ROOT / "models" / "best.pt"
    dataset_yaml = PROJECT_ROOT / "data" / "real_industrial_dataset" / "dataset.yaml"
    out_json = PROJECT_ROOT / "backend" / "class_thresholds.json"

    print("================================================================================")
    print("       VISIONGUARD QA - MODEL VALIDATION & PER-CLASS F1 THRESHOLD SWEEP         ")
    print("================================================================================")
    print(f"Model: {model_path}")
    print(f"Dataset YAML: {dataset_yaml}")
    print(f"Output Path: {out_json}\n")

    model = YOLO(str(model_path))

    # Run validation
    metrics = model.val(data=str(dataset_yaml), split="val", verbose=False)

    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    mp = float(metrics.box.mp)
    mr = float(metrics.box.mr)

    px = metrics.box.px  # 1000 threshold values [0.0 to 1.0]
    f1_curve = metrics.box.f1_curve
    p_curve = metrics.box.p_curve
    r_curve = metrics.box.r_curve
    ap50_per_class = metrics.box.ap50
    class_indices = metrics.box.ap_class_index

    per_class_results = {}
    thresholds_flat = {}

    print("\n--------------------------------------------------------------------------------")
    print(f"{'Class Name':<18} | {'Opt Thresh':<10} | {'F1 Score':<10} | {'Precision':<10} | {'Recall':<10} | {'AP50':<8}")
    print("--------------------------------------------------------------------------------")

    for idx, cls_id in enumerate(class_indices):
        cls_name = model.names[int(cls_id)]
        f1_vals = f1_curve[idx]
        best_idx = int(np.argmax(f1_vals))
        best_thresh = round(float(px[best_idx]), 3)
        best_f1 = round(float(f1_vals[best_idx]), 3)
        prec = round(float(p_curve[idx][best_idx]), 3)
        rec = round(float(r_curve[idx][best_idx]), 3)
        ap50 = round(float(ap50_per_class[idx]), 3)

        per_class_results[cls_name] = {
            "class_id": int(cls_id),
            "threshold": best_thresh,
            "f1": best_f1,
            "precision": prec,
            "recall": rec,
            "ap50": ap50,
        }
        thresholds_flat[cls_name] = best_thresh

        print(
            f"{cls_name:<18} | {best_thresh:<10.3f} | {best_f1:<10.3f} | {prec:<10.3f} | {rec:<10.3f} | {ap50:<8.3f}"
        )

    print("--------------------------------------------------------------------------------")
    print(f"Overall mAP50    : {map50:.4f} ({map50*100:.2f}%)")
    print(f"Overall mAP50-95 : {map50_95:.4f} ({map50_95*100:.2f}%)")
    print(f"Overall Precision: {mp:.4f} ({mp*100:.2f}%)")
    print(f"Overall Recall   : {mr:.4f} ({mr*100:.2f}%)")
    print("================================================================================\n")

    payload = {
        "model": str(model_path.name),
        "dataset": "real_industrial_dataset",
        "split": "val",
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "precision": round(mp, 4),
        "recall": round(mr, 4),
        "thresholds": thresholds_flat,
        "per_class": per_class_results,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved optimized per-class thresholds to: {out_json}")


if __name__ == "__main__":
    main()

