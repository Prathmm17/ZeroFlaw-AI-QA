"""Verification script for Honest Measurements in VisionGuard QA.

Tests both modes:
1. Mode 1: px_per_mm is None (Uncalibrated Mode)
   - Area, length, and width displayed in pixels (px).
   - Label: 'uncalibrated (px)'.
   - NEVER prints mm / mm² values (surface_area_mm2, length_mm, width_mm are None).
2. Mode 2: px_per_mm is Set (Calibrated Mode, e.g. via UI calibration helper)
   - Area in mm², length in mm, width in mm.
   - Label: 'calibrated'.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Patch PyTorch weights_only
import torch
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

import cv2
import numpy as np

from backend.config import settings
from ml.detector import DefectDetector
from ml.segmenter import DefectSegmenter


def run_honest_measurement_test():
    print("=" * 80)
    print("VISIONGUARD QA — HONEST MEASUREMENTS & CALIBRATION VERIFICATION")
    print("=" * 80)

    # Build a realistic steel specimen with two distinct defect flaws:
    # 1. A long scratch / crack across the upper region
    # 2. A circular pit / inclusion in the lower region
    np.random.seed(42)
    test_img = np.full((600, 800, 3), 160, dtype=np.uint8)
    noise = np.random.randint(-8, 8, (600, 800, 3), dtype=np.int16)
    test_img = np.clip(test_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Flaw 1: Longitudinal scratch defect
    cv2.line(test_img, (140, 180), (560, 210), (30, 30, 30), 6)
    cv2.line(test_img, (280, 195), (460, 225), (45, 45, 45), 3)

    # Flaw 2: Pitted cavity defect
    cv2.circle(test_img, (380, 420), 32, (25, 25, 25), -1)

    h, w = test_img.shape[:2]
    print(f"Synthesized steel specimen: {w}x{h} px with crack and pit flaw regions.\n")

    # Initialize detector
    detector = DefectDetector(model_name="best.pt", confidence_threshold=0.20)

    # -------------------------------------------------------------------------
    # TEST MODE 1: px_per_mm is None (UNCALIBRATED)
    # -------------------------------------------------------------------------
    print("-" * 80)
    print("TEST MODE 1: px_per_mm = None (Uncalibrated Mode)")
    print("-" * 80)
    detector.set_calibration(None)

    res_uncal = detector.detect(test_img)
    dets_uncal = res_uncal["detections"]

    print(f"Calibration Scale: {res_uncal['px_per_mm']}")
    print(f"Is Calibrated:     {res_uncal['is_calibrated']}")
    print(f"Calibration Label: '{res_uncal['calibration_label']}'")
    print(f"Detections Count:  {len(dets_uncal)}")
    print("\nDefect Metrology Table (Mode 1 - Uncalibrated):")
    print(f"{'#':<3} | {'Class':<22} | {'Area':<18} | {'Length':<16} | {'Width':<16} | {'Label':<18}")
    print("-" * 100)

    for i, d in enumerate(dets_uncal, 1):
        m = d.get("metrology", {})
        c_label = m.get("calibration_label", "N/A")
        area_str = f"{m.get('area_px', m.get('pixel_area'))} px"
        len_str = f"{m.get('length_px')} px"
        wid_str = f"{m.get('width_px')} px"

        # STRICT VERIFICATION: mm values must NOT exist
        assert m.get("surface_area_mm2") is None, f"FAIL: surface_area_mm2 found in uncalibrated mode: {m.get('surface_area_mm2')}"
        assert m.get("length_mm") is None, f"FAIL: length_mm found in uncalibrated mode: {m.get('length_mm')}"
        assert m.get("width_mm") is None, f"FAIL: width_mm found in uncalibrated mode: {m.get('width_mm')}"
        assert m.get("is_calibrated") is False, "FAIL: is_calibrated should be False"
        assert c_label == "uncalibrated (px)", f"FAIL: wrong label: {c_label}"

        print(f"{i:<3} | {d['class_name']:<22} | {area_str:<18} | {len_str:<16} | {wid_str:<16} | {c_label:<18}")

    print("\n>> VERIFICATION PASSED: No mm values leaked. Honest 'uncalibrated (px)' displayed.")

    # -------------------------------------------------------------------------
    # TEST MODE 2: px_per_mm is Set (CALIBRATED)
    # -------------------------------------------------------------------------
    calib_scale = 16.5  # 16.5 pixels per millimeter (e.g. from UI calibration line helper)
    print("\n" + "-" * 80)
    print(f"TEST MODE 2: px_per_mm = {calib_scale} px/mm (Calibrated Mode)")
    print("-" * 80)
    detector.set_calibration(calib_scale)

    res_cal = detector.detect(test_img)
    dets_cal = res_cal["detections"]

    print(f"Calibration Scale: {res_cal['px_per_mm']} px/mm")
    print(f"Is Calibrated:     {res_cal['is_calibrated']}")
    print(f"Calibration Label: '{res_cal['calibration_label']}'")
    print(f"Detections Count:  {len(dets_cal)}")
    print("\nDefect Metrology Table (Mode 2 - Calibrated):")
    print(f"{'#':<3} | {'Class':<22} | {'Area':<18} | {'Length':<16} | {'Width':<16} | {'Label':<18}")
    print("-" * 100)

    for i, d in enumerate(dets_cal, 1):
        m = d.get("metrology", {})
        c_label = m.get("calibration_label", "N/A")
        area_str = f"{m.get('surface_area_mm2')} mm²"
        len_str = f"{m.get('length_mm')} mm"
        wid_str = f"{m.get('width_mm')} mm"

        # VERIFICATION: mm values MUST be present and calibrated
        assert m.get("surface_area_mm2") is not None, "FAIL: surface_area_mm2 is None in calibrated mode"
        assert m.get("length_mm") is not None, "FAIL: length_mm is None in calibrated mode"
        assert m.get("width_mm") is not None, "FAIL: width_mm is None in calibrated mode"
        assert m.get("is_calibrated") is True, "FAIL: is_calibrated should be True"
        assert c_label == "calibrated", f"FAIL: wrong label: {c_label}"

        print(f"{i:<3} | {d['class_name']:<22} | {area_str:<18} | {len_str:<16} | {wid_str:<16} | {c_label:<18}")

    print("\n>> VERIFICATION PASSED: Calibrated mm and mm² successfully computed and labeled.")
    print("=" * 80)
    print("ALL HONEST MEASUREMENT CRITERIA SUCCESSFULLY VERIFIED.")
    print("=" * 80)


if __name__ == "__main__":
    run_honest_measurement_test()
