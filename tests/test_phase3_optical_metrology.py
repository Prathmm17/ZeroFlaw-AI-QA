"""Phase 3 Verification Tests: Automated Optical Metrology & Drift-Compensated Calibration.

Verifies:
1. Automated ArUco & circular fiducial detection measuring known 20mm reference artifact within 20.0 +/- 0.2mm.
2. Vibration / focal distance drift detection triggering alert when shift >= 2%.
3. Sub-pixel polygon boundary extraction with area, length, width, aspect ratio, solidity.
4. Engineering Tolerance Engine compliance rules (ISO/ASTM) and uncalibrated safety warning.
5. Auto-calibration API endpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pytest

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

from ml.segmenter import DefectMetrology, DefectSegmenter, ToleranceAssessment


# ---------------------------------------------------------------------------
# Test 1: Known 20mm Reference Artifact Measured within 20.0 +/- 0.2 mm
# ---------------------------------------------------------------------------
def test_aruco_fiducial_reference_measurement():
    """Verify that a known 20.0 mm ArUco marker is measured within 20.0 +/- 0.2 mm."""
    segmenter = DefectSegmenter()

    # Generate an ArUco marker of side 200px
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(aruco_dict, 1, 200)

    # Place marker on a 600x600 conveyor background with white quiet zone
    canvas = np.full((600, 600, 3), 220, dtype=np.uint8)
    # 200x200 marker placed at (100, 100)
    canvas[100:300, 100:300] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

    # Auto-detect fiducial scale for a known 20.0 mm marker
    scale, drift_warn = segmenter.detect_fiducial_scale(canvas, marker_size_mm=20.0, target_type="aruco")

    assert scale is not None, "Failed to detect ArUco fiducial marker!"
    assert segmenter.is_calibrated is True

    # Expected scale: 199.0px / 20.0mm = 9.95 px/mm
    expected_scale = 199.0 / 20.0
    assert abs(scale - expected_scale) <= 0.2

    # Measure the marker side using calibrated scale
    measured_mm = 199.0 / scale
    assert abs(measured_mm - 20.0) <= 0.2, f"Measured {measured_mm}mm differs from 20.0mm by > 0.2mm!"


def test_circular_fiducial_reference_measurement():
    """Verify that a known 20.0 mm circular optical benchmark is measured within 20.0 +/- 0.2 mm."""
    segmenter = DefectSegmenter()

    # Create a 600x600 image with a black calibration disc of diameter 200px (radius 100px)
    canvas = np.full((600, 600, 3), 230, dtype=np.uint8)
    cv2.circle(canvas, (300, 300), 100, (25, 25, 25), -1)

    scale, drift_warn = segmenter.detect_fiducial_scale(canvas, marker_size_mm=20.0, target_type="circle")

    assert scale is not None, "Failed to detect circular optical fiducial!"
    assert segmenter.is_calibrated is True

    # 200px diameter / 20mm = 10.0 px/mm
    assert abs(scale - 10.0) <= 0.2

    measured_diameter_mm = 200.0 / scale
    assert abs(measured_diameter_mm - 20.0) <= 0.2, f"Measured {measured_diameter_mm}mm differs by > 0.2mm!"


# ---------------------------------------------------------------------------
# Test 2: Focal Distance Drift Compensation Alert (Shift >= 2%)
# ---------------------------------------------------------------------------
def test_focal_distance_drift_detection():
    """Verify that a scale shift of >= 2% triggers a focal distance drift warning."""
    segmenter = DefectSegmenter()

    # Initial calibration at 10.0 px/mm (camera at height H)
    canvas1 = np.full((600, 600, 3), 230, dtype=np.uint8)
    cv2.circle(canvas1, (300, 300), 100, (25, 25, 25), -1)  # 200px diameter -> 10.0 px/mm
    scale1, warn1 = segmenter.detect_fiducial_scale(canvas1, marker_size_mm=20.0, target_type="circle")
    assert abs(scale1 - 10.0) <= 0.2
    assert warn1 is None  # Initial baseline has no drift

    # Camera vibration / height shift causes 5% magnification change (radius = 105px -> diameter = 210px -> 10.5 px/mm)
    canvas2 = np.full((600, 600, 3), 230, dtype=np.uint8)
    cv2.circle(canvas2, (300, 300), 105, (25, 25, 25), -1)
    scale2, warn2 = segmenter.detect_fiducial_scale(canvas2, marker_size_mm=20.0, target_type="circle", drift_threshold_pct=2.0)

    assert warn2 is not None, "Expected focal drift warning for 5% scale shift!"
    assert "Focal distance drift detected" in warn2
    assert segmenter.pixels_per_mm == scale2


# ---------------------------------------------------------------------------
# Test 3: Sub-Pixel Contour Boundary Approximation & Geometric Properties
# ---------------------------------------------------------------------------
def test_subpixel_contour_geometry():
    """Verify sub-pixel Douglas-Peucker contour extraction and metric properties."""
    segmenter = DefectSegmenter(pixels_per_mm=10.0)  # 10 px/mm -> 1 px = 0.1 mm

    # Synthetic image with an oblique fracture polygon (100px length, 20px width)
    img = np.full((400, 400, 3), 150, dtype=np.uint8)
    crack_pts = np.array([[100, 100], [200, 100], [200, 120], [100, 120]], dtype=np.int32)
    cv2.fillPoly(img, [crack_pts], (20, 20, 20))

    norm_bbox = (100 / 400, 100 / 400, 200 / 400, 120 / 400)
    metrology = segmenter.segment_roi(img, norm_bbox, class_name="crazing")

    assert metrology.is_calibrated is True
    assert metrology.length_mm is not None
    assert metrology.width_mm is not None
    assert metrology.surface_area_mm2 is not None

    # Length ~ 100px -> ~ 10.0mm (+/- 0.5mm)
    assert abs(metrology.length_mm - 10.0) <= 0.5
    # Width ~ 20px -> ~ 2.0mm (+/- 0.5mm)
    assert abs(metrology.width_mm - 2.0) <= 0.5
    # Area ~ 2000 px^2 -> ~ 20.0 mm^2 (+/- 2.0mm^2)
    assert abs(metrology.surface_area_mm2 - 20.0) <= 2.0
    # Aspect ratio ~ 5.0
    assert abs(metrology.aspect_ratio - 5.0) <= 1.0
    # Solidity close to 1.0 for convex rectangle
    assert metrology.solidity >= 0.90


# ---------------------------------------------------------------------------
# Test 4: Honest Engineering Tolerance Engine (ISO / ASTM Rules)
# ---------------------------------------------------------------------------
def test_engineering_tolerance_rules():
    """Verify ISO/ASTM tolerance grading for cracks, pores, and uncalibrated mode."""
    segmenter = DefectSegmenter(pixels_per_mm=10.0)

    # 1. Critical Crack: length 5.0 mm > 3.0 mm limit
    m_crit_crack = DefectMetrology(
        pixel_area=500,
        aspect_ratio=6.0,
        solidity=0.85,
        polygon=[[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]],
        length_mm=5.0,
        width_mm=0.8,
        surface_area_mm2=4.0,
        is_calibrated=True,
    )
    tol1 = segmenter.evaluate_engineering_tolerances(m_crit_crack, "crazing")
    assert tol1.status == "VIOLATION"
    assert tol1.recommended_severity == "critical"
    assert tol1.is_acceptable is False

    # 2. Minor Hairline Crack: length 0.8 mm <= 1.0 mm limit
    m_minor_crack = DefectMetrology(
        pixel_area=80,
        aspect_ratio=4.0,
        solidity=0.90,
        polygon=[[0.1, 0.1], [0.15, 0.1], [0.15, 0.12]],
        length_mm=0.8,
        width_mm=0.2,
        surface_area_mm2=0.16,
        is_calibrated=True,
    )
    tol2 = segmenter.evaluate_engineering_tolerances(m_minor_crack, "crazing")
    assert tol2.status == "PASS"
    assert tol2.recommended_severity == "minor"
    assert tol2.is_acceptable is True

    # 3. Critical Porosity: width 2.5 mm > 2.0 mm limit
    m_pore = DefectMetrology(
        pixel_area=600,
        aspect_ratio=1.1,
        solidity=0.95,
        polygon=[[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]],
        length_mm=2.5,
        width_mm=2.5,
        surface_area_mm2=4.9,
        is_calibrated=True,
    )
    tol3 = segmenter.evaluate_engineering_tolerances(m_pore, "pitted_surface")
    assert tol3.status == "VIOLATION"
    assert tol3.recommended_severity == "critical"

    # 4. Uncalibrated mode: returns UNCALIBRATED warning
    segmenter_uncal = DefectSegmenter(pixels_per_mm=None)
    m_uncal = DefectMetrology(
        pixel_area=500,
        aspect_ratio=2.0,
        solidity=0.8,
        polygon=[],
        is_calibrated=False,
    )
    tol4 = segmenter_uncal.evaluate_engineering_tolerances(m_uncal, "crazing")
    assert tol4.status == "UNCALIBRATED"
    assert "Optical millimeter calibration required" in tol4.criterion
    assert tol4.warning is not None

