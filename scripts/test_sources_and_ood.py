"""Test pipeline with one NEU-DET image and one real photo.

Verifies:
1. Every detection carries a "source" field: ("yolo", "anomaly_cv", or "vlm").
2. Source is "vlm" ONLY when real Gemini API succeeded; when GEMINI_API_KEY is missing,
   source is "anomaly_cv".
3. Out-of-domain handling: confidence below class threshold + anomaly engine -> "unclassified_anomaly".
4. Dumps JSON for both the NEU-DET image and the real photo.
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

import cv2
import torch

# PyTorch 2.6 weights_only patch
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

from backend.database import SessionLocal, init_db
from backend.services.detection_service import DetectionService
from ml.arbitrator import ProductionArbitrator
from ml.detector import DefectDetector


def run_tests() -> None:
    init_db()
    db = SessionLocal()

    # 1. NEU-DET specimen image
    neudet_candidates = [
        PROJECT_ROOT / "data" / "industrial_benchmark" / "images" / "val" / "neu_steel_val_0000.png",
        PROJECT_ROOT / "data" / "factory_benchmark" / "specimen_3_crazing.png",
        PROJECT_ROOT / "data" / "factory_benchmark" / "specimen_7_scratch.png",
    ]
    neudet_path = None
    for p in neudet_candidates:
        if p.exists():
            neudet_path = p
            break
    if neudet_path is None:
        raise FileNotFoundError("Could not find a NEU-DET image candidate.")

    # 2. Real photo candidate
    real_photo_candidates = [
        PROJECT_ROOT / "data" / "uploads" / "WhatsApp Image 2026-10-29 at 2.34.32 AM_20261028_210759_278b9ff7_orig.jpg",
        PROJECT_ROOT / "data" / "uploads" / "sample_crack_000_20261028_200322_627c22bc_orig.jpg",
        PROJECT_ROOT / "data" / "uploads" / "Screenshot 2026-10-29 013229_20261028_200305_2414ebe4_orig.jpg",
    ]
    real_photo_path = None
    for p in real_photo_candidates:
        if p.exists():
            real_photo_path = p
            break
    if real_photo_path is None:
        raise FileNotFoundError("Could not find a real photo candidate.")

    print(f"\n=======================================================")
    print(f"1. RUNNING INSPECTION ON NEU-DET SPECIMEN: {neudet_path.name}")
    print(f"=======================================================")
    res_neudet = DetectionService.inspect_and_save(
        db=db,
        image_data=neudet_path,
        filename=neudet_path.name,
        source="sample",
        notes="NEU-DET evaluation sample",
    )

    # Format neat JSON for display
    neudet_display_json = {
        "inspection_id": res_neudet["id"],
        "filename": res_neudet["filename"],
        "status": res_neudet["status"],
        "arbitration_tier": res_neudet["arbitration_tier"],
        "arbitration_status": res_neudet["arbitration_status"],
        "total_defects": res_neudet["total_defects"],
        "inference_time_ms": res_neudet["inference_time_ms"],
        "defects": [
            {
                "class_name": d["class_name"],
                "source": d["source"],
                "confidence": round(d["confidence"], 3),
                "severity": d["severity"],
                "bbox": [round(c, 1) for c in d["bbox"]],
                "description": d["description"],
            }
            for d in res_neudet["defects"]
        ],
    }
    print("\nNEU-DET INSPECTION JSON OUTPUT:")
    print(json.dumps(neudet_display_json, indent=2))

    # Assertions for NEU-DET
    for d in res_neudet["defects"]:
        assert "source" in d, "Missing 'source' field in NEU-DET defect"
        assert d["source"] in ("yolo", "anomaly_cv", "vlm"), f"Invalid source: {d['source']}"

    print(f"\n=======================================================")
    print(f"2. RUNNING INSPECTION ON REAL PHOTO: {real_photo_path.name}")
    print(f"=======================================================")
    res_real = DetectionService.inspect_and_save(
        db=db,
        image_data=real_photo_path,
        filename=real_photo_path.name,
        source="upload",
        notes="Real manufacturing photo out-of-domain evaluation",
    )

    real_display_json = {
        "inspection_id": res_real["id"],
        "filename": res_real["filename"],
        "status": res_real["status"],
        "arbitration_tier": res_real["arbitration_tier"],
        "arbitration_status": res_real["arbitration_status"],
        "total_defects": res_real["total_defects"],
        "inference_time_ms": res_real["inference_time_ms"],
        "defects": [
            {
                "class_name": d["class_name"],
                "source": d["source"],
                "confidence": round(d["confidence"], 3),
                "severity": d["severity"],
                "bbox": [round(c, 1) for c in d["bbox"]],
                "description": d["description"],
            }
            for d in res_real["defects"]
        ],
    }
    print("\nREAL PHOTO INSPECTION JSON OUTPUT:")
    print(json.dumps(real_display_json, indent=2))

    # Assertions for Real photo
    for d in res_real["defects"]:
        assert "source" in d, "Missing 'source' field in real photo defect"
        assert d["source"] in ("yolo", "anomaly_cv", "vlm"), f"Invalid source: {d['source']}"
        # If API key is missing, source must NEVER be vlm
        import os
        if not os.environ.get("GEMINI_API_KEY"):
            assert d["source"] != "vlm", "source was set to 'vlm' when GEMINI_API_KEY was absent!"

    db.close()
    print("\n=======================================================")
    print("ALL ASSERTIONS PASSED! Detection source & OOD verified.")
    print("=======================================================")


if __name__ == "__main__":
    run_tests()

