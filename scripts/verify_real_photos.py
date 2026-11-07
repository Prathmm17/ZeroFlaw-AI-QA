"""Test user-uploaded real 3D photos against domain guard, box area ceiling, and source labeling rules."""

from __future__ import annotations

import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# NumPy 2.x patch
if not hasattr(np, "trapz"):
    np.trapz = getattr(np, "trapezoid", None)

import torch
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

from ml.arbitrator import ProductionArbitrator
from ml.detector import DefectDetector


def test_real_photo(photo_path: Path, label: str):
    print("=" * 80)
    print(f"TESTING REAL PHOTO: {label} ({photo_path.name})")
    print("=" * 80)
    assert photo_path.is_file(), f"Photo not found at {photo_path}"

    img = cv2.imread(str(photo_path))
    assert img is not None, f"Failed reading image {photo_path}"
    h, w = img.shape[:2]
    img_area = w * h
    print(f"Image Resolution: {w}x{h} px (Total Area: {img_area:,} px^2)")

    detector = DefectDetector(confidence_threshold=0.20)
    arbitrator = ProductionArbitrator(edge_detector=detector)

    # Run full production arbitration
    res = arbitrator.arbitrate(img, filename=photo_path.name, deep_verify=False)

    print(f"Domain Trust Assessment: {res.domain_trust}")
    print(f"Domain Warning: {res.domain_warning}")
    print(f"Arbitration Verdict: {res.verdict}")
    print(f"Total Detections: {len(res.detections)}")

    # Verification Checks
    assert res.domain_trust.upper() == "LOW", f"Expected Domain Trust LOW, got: {res.domain_trust}"
    assert res.domain_warning is not None and "differs from the training domain" in res.domain_warning
    assert res.verdict != "REJECTED", f"Expected verdict NOT to be REJECTED on out-of-domain photo, got: {res.verdict}"

    for i, d in enumerate(res.detections):
        x1, y1, x2, y2 = d.bbox
        bw = x2 - x1
        bh = y2 - y1
        box_area = bw * bh
        area_pct = (box_area / img_area) * 100.0

        print(f"\n  [Box #{i+1}] {d.class_name} ({d.confidence*100:.1f}% conf)")
        print(f"    • Coordinates: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
        print(f"    • Dimensions: {bw:.1f}x{bh:.1f} px")
        print(f"    • Area % of Image: {area_pct:.2f}% (d.area_ratio={d.area_ratio*100:.2f}%)")
        print(f"    • Source Field: '{d.source}'")
        print(f"    • Severity: {d.severity.upper()}")

        # Constraint 1: Box sits within image bounds
        assert 0 <= x1 < w and 0 <= x2 <= w
        assert 0 <= y1 < h and 0 <= y2 <= h

        # Constraint 2: No box above 35% area
        assert area_pct <= 35.0, f"Box #{i+1} area {area_pct:.2f}% exceeds 35% limit!"
        assert d.area_ratio <= 0.35, f"Defect area_ratio {d.area_ratio:.4f} exceeds 0.35 limit!"

        # Constraint 3: No "VLM" badge when it is anomaly_cv / offline
        if not d.vlm_verified:
            assert d.source in ("yolo", "anomaly_cv"), f"Invalid source '{d.source}' without real VLM call"
            assert d.source != "vlm", "Claimed 'vlm' source without real VLM API execution!"

    print("-" * 80)
    print(f"ALL CONSTRAINTS PASSED FOR: {label}!")
    print("=" * 80 + "\n")


def main():
    uploaded_dir = Path(r"C:\Users\hp\.gemini\antigravity\brain\97c07de2-6855-40d5-9e92-c9a70691aa04\.user_uploaded")
    cast_part = uploaded_dir / "media_1793221728328_aa688074.png"
    pipe_part = uploaded_dir / "media_1793221698910_a449b49f.png"

    test_real_photo(cast_part, "User Real Photo 1 (Cast Part)")
    test_real_photo(pipe_part, "User Real Photo 2 (Pipe Component)")


if __name__ == "__main__":
    main()

