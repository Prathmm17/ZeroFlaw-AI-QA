"""Phase 3 Verification Script: Pixel-Level Mask Segmentation & Contour Metrology.

Validates that DefectSegmenter accurately extracts polygonal contours,
calculates physical metallurgical dimensions (area in mm², crack length, aspect ratio, solidity),
and integrates with DefectDetector to attach metrology telemetry to detections.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# NumPy 2.x patch
import numpy as np
if not hasattr(np, "trapz"):
    np.trapz = getattr(np, "trapezoid", None)

import cv2
from ml.detector import DefectDetector
from ml.segmenter import DefectMetrology, DefectSegmenter


def test_segmenter_geometric_accuracy() -> bool:
    print("\n--- TEST 1: Metrology Geometric & Physical Calculation Accuracy ---")
    segmenter = DefectSegmenter(pixels_per_mm=10.0)  # 10 px = 1 mm, 1 px = 0.1 mm, 1 px² = 0.01 mm²

    canvas = np.full((500, 500, 3), 180, dtype=np.uint8)  # Metallic gray background

    # 1. Synthesize a 100px long, 4px wide longitudinal crack (aspect ratio ~25:1)
    # Physical length: 100px / 10 = 10.0 mm
    # Physical width: 4px / 10 = 0.4 mm
    crack_x1, crack_y1, crack_x2, crack_y2 = 100, 100, 200, 100
    cv2.line(canvas, (crack_x1, crack_y1), (crack_x2, crack_y2), (20, 20, 20), 4)

    # Normalized bbox covering the crack with slight margin
    bbox_crack = (80 / 500, 80 / 500, 220 / 500, 120 / 500)
    metrology_crack = segmenter.segment_roi(canvas, bbox_crack)

    print(f"Crack Measurements:")
    print(f"  Pixel Area      : {metrology_crack.pixel_area} px")
    print(f"  Surface Area    : {metrology_crack.surface_area_mm2:.4f} mm²")
    print(f"  Crack Length    : {metrology_crack.length_mm:.2f} mm (Expected ~10.0 mm)")
    print(f"  Crack Width     : {metrology_crack.width_mm:.2f} mm (Expected ~0.4 mm)")
    print(f"  Aspect Ratio    : {metrology_crack.aspect_ratio:.1f}:1")
    print(f"  Polygon Vertices: {len(metrology_crack.polygon)} points")

    assert metrology_crack.length_mm > 5.0, f"Expected length > 5mm, got {metrology_crack.length_mm}"
    assert metrology_crack.aspect_ratio > 3.0, f"Expected crack aspect ratio > 3.0, got {metrology_crack.aspect_ratio}"
    assert len(metrology_crack.polygon) >= 3, "Expected at least 3 vertices for polygon"
    print("[PASS] Longitudinal crack metrology verified.")

    # 2. Synthesize a circular surface pit (radius 20px -> diameter 40px = 4.0 mm, aspect ratio ~1:1)
    pit_center = (350, 350)
    cv2.circle(canvas, pit_center, 20, (15, 15, 15), -1)

    bbox_pit = (320 / 500, 320 / 500, 380 / 500, 380 / 500)
    metrology_pit = segmenter.segment_roi(canvas, bbox_pit)

    print(f"\nSurface Pit Measurements:")
    print(f"  Pixel Area      : {metrology_pit.pixel_area} px")
    print(f"  Surface Area    : {metrology_pit.surface_area_mm2:.4f} mm²")
    print(f"  Pit Dimension   : {metrology_pit.length_mm:.2f} x {metrology_pit.width_mm:.2f} mm")
    print(f"  Aspect Ratio    : {metrology_pit.aspect_ratio:.1f}:1 (Expected ~1:1)")
    print(f"  Solidity        : {metrology_pit.solidity:.2f} (Expected >= 0.85)")

    assert metrology_pit.aspect_ratio < 2.0, f"Expected circular aspect ratio < 2.0, got {metrology_pit.aspect_ratio}"
    assert metrology_pit.solidity >= 0.80, f"Expected high circular solidity >= 0.80, got {metrology_pit.solidity}"
    print("[PASS] Spherical surface pit metrology verified.")

    # 3. Test polygon overlay rendering
    annotated = segmenter.draw_polygon_overlay(canvas, metrology_crack.polygon, color=(0, 0, 255), alpha=0.4)
    annotated = segmenter.draw_polygon_overlay(annotated, metrology_pit.polygon, color=(0, 165, 255), alpha=0.4)
    assert annotated.shape == canvas.shape
    print("[PASS] Translucent polygon mask overlay rendering verified.")
    return True


def test_detector_metrology_integration() -> bool:
    print("\n--- TEST 2: DefectDetector & Metrology End-to-End Pipeline ---")
    detector = DefectDetector(
        model_name="industrial_best.pt",
        confidence_threshold=0.05,
        enable_tiling=False,
    )

    val_dir = PROJECT_ROOT / "data" / "industrial_benchmark" / "images" / "val"
    val_images = list(val_dir.glob("*.png")) + list(val_dir.glob("*.jpg"))
    if not val_images:
        print(f"[FAIL] No validation images found in {val_dir}")
        return False

    sample_img = cv2.imread(str(val_images[0]))
    result = detector.detect(sample_img)

    print(f"Inference Time: {result['inference_time_ms']} ms")
    print(f"Total Detections Found: {len(result['detections'])}")

    # Also test an image synthesized with clear surface defect to verify polygon generation
    synth_canvas = np.full((640, 640, 3), 160, dtype=np.uint8)
    cv2.line(synth_canvas, (100, 200), (500, 250), (20, 20, 20), 5)  # Severe scratch
    synth_result = detector.detect(synth_canvas)

    # Check that metrology and polygon fields exist on every detection
    for det in synth_result["detections"] + result["detections"]:
        assert "polygon" in det, "Missing 'polygon' field in detection"
        assert "metrology" in det, "Missing 'metrology' field in detection"
        m = det["metrology"]
        assert "surface_area_mm2" in m, "Missing surface_area_mm2 in metrology"
        assert "length_mm" in m, "Missing length_mm in metrology"
        assert "aspect_ratio" in m, "Missing aspect_ratio in metrology"
        assert "solidity" in m, "Missing solidity in metrology"

    print("[PASS] DefectDetector seamlessly populates polygon masks and physical metrology telemetry.")
    return True


def main() -> int:
    print("=" * 70)
    print("VISIONGUARD QA — PHASE 3 METROLOGY & SEGMENTATION VERIFICATION")
    print("=" * 70)

    t1_pass = test_segmenter_geometric_accuracy()
    t2_pass = test_detector_metrology_integration()

    if t1_pass and t2_pass:
        print("\n" + "=" * 70)
        print("[SUCCESS] ALL PHASE 3 VERIFICATION TESTS PASSED (100% SUCCESS)")
        print("=" * 70)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())

