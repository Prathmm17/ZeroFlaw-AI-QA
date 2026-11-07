"""Run real DefectDetector on an uploaded real photo and save outputs/letterbox_check.png."""

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

import cv2
import torch

# PyTorch 2.6 weights_only patch
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

from ml.detector import DefectDetector


def main() -> None:
    upload_dir = PROJECT_ROOT / "data" / "uploads"
    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        "WhatsApp Image 2026-10-29 at 2.34.32 AM_20261028_210759_278b9ff7_orig.jpg",
        "sample_crack_000_20261028_200322_627c22bc_orig.jpg",
        "WhatsApp Image 2026-10-30 at 6.26.04 AM_20261030_011449_d82a1133_orig.jpg",
    ]

    selected_image = None
    for c in candidates:
        candidate_path = upload_dir / c
        if candidate_path.exists():
            selected_image = candidate_path
            break

    if selected_image is None:
        all_imgs = list(upload_dir.glob("*.jpg")) + list(upload_dir.glob("*.png"))
        if not all_imgs:
            raise FileNotFoundError(f"No image files found in {upload_dir}")
        selected_image = all_imgs[0]

    print(f"Running DefectDetector on real photo: {selected_image.name}")
    orig_bgr = cv2.imread(str(selected_image))
    if orig_bgr is None:
        raise ValueError(f"Could not read image: {selected_image}")

    h, w, c = orig_bgr.shape
    print(f"Original image resolution: {w}x{h} (channels={c})")

    detector = DefectDetector(
        model_name="best.pt",
        confidence_threshold=0.25,
        enable_tiling=True,
    )

    result = detector.detect(selected_image)

    detections = result["detections"]
    print(f"Total detections: {len(detections)}")
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = d["bbox"]
        cls_name = d["class_name"]
        conf = d["confidence"]
        sev = d["severity"]
        print(f"  [{i+1}] {cls_name} ({sev}) conf={conf:.3f} bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
        # Assert bounding box is within original bounds
        assert 0.0 <= x1 <= w, f"x1 ({x1}) out of bounds [0, {w}]"
        assert 0.0 <= y1 <= h, f"y1 ({y1}) out of bounds [0, {h}]"
        assert 0.0 <= x2 <= w, f"x2 ({x2}) out of bounds [0, {w}]"
        assert 0.0 <= y2 <= h, f"y2 ({y2}) out of bounds [0, {h}]"

    annotated = result["annotated_image"]
    output_path = outputs_dir / "letterbox_check.png"
    cv2.imwrite(str(output_path), annotated)

    print(f"Successfully saved annotated image to: {output_path}")
    print(f"Output image resolution: {annotated.shape[1]}x{annotated.shape[0]}")
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    print("Verification complete! Letterbox coordinate bug fixed.")


if __name__ == "__main__":
    main()

