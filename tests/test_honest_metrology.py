"""Unit tests for Honest Metrology measurements and optical scale calibration."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# NumPy 2.x patch
if not hasattr(np, "trapz"):
    np.trapz = getattr(np, "trapezoid", None)

import torch
# PyTorch 2.6 weights_only patch
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

from backend.config import settings
from backend.models import CalibrationRequest, CalibrationResponse
from backend.routers.detection import get_camera_calibration, set_camera_calibration
from backend.services.detection_service import get_detector
from ml.detector import DefectDetector
from ml.segmenter import DefectMetrology, DefectSegmenter


def test_segmenter_uncalibrated_mode():
    """Verify that when pixels_per_mm is None, no mm values are produced."""
    segmenter = DefectSegmenter(pixels_per_mm=None)
    assert not segmenter.is_calibrated
    assert segmenter.calibration_label == "uncalibrated (px)"

    # Synthetic image with a bright rectangle defect
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    img[100:150, 100:200] = 255  # 50x100 rectangle

    # Normalized bbox: [100/400, 100/400, 200/400, 150/400] = [0.25, 0.25, 0.50, 0.375]
    bbox = (0.25, 0.25, 0.50, 0.375)
    metrology = segmenter.segment_roi(img, bbox)

    assert isinstance(metrology, DefectMetrology)
    assert metrology.is_calibrated is False
    assert metrology.calibration_label == "uncalibrated (px)"
    # STRICT RULE: mm values MUST be None
    assert metrology.surface_area_mm2 is None
    assert metrology.length_mm is None
    assert metrology.width_mm is None

    # Pixel values must be present and accurate
    assert metrology.pixel_area > 0
    assert metrology.area_px == metrology.pixel_area
    assert metrology.length_px is not None and metrology.length_px > 0
    assert metrology.width_px is not None and metrology.width_px > 0


def test_segmenter_calibrated_mode():
    """Verify that when pixels_per_mm is set, mm and mm2 values are computed accurately."""
    px_per_mm = 20.0  # 1 mm = 20 px -> 1 px = 0.05 mm
    segmenter = DefectSegmenter(pixels_per_mm=px_per_mm)
    assert segmenter.is_calibrated
    assert segmenter.calibration_label == "calibrated"

    img = np.zeros((400, 400, 3), dtype=np.uint8)
    img[100:150, 100:200] = 255  # 50x100 rectangle

    bbox = (0.25, 0.25, 0.50, 0.375)
    metrology = segmenter.segment_roi(img, bbox)

    assert metrology.is_calibrated is True
    assert metrology.calibration_label == "calibrated"
    assert metrology.surface_area_mm2 is not None
    assert metrology.length_mm is not None
    assert metrology.width_mm is not None

    expected_mm2 = round(metrology.pixel_area * (1.0 / px_per_mm) ** 2, 4)
    assert metrology.surface_area_mm2 == expected_mm2
    assert metrology.px_per_mm == px_per_mm


def test_segmenter_dynamic_toggle():
    """Verify transitioning between uncalibrated and calibrated via set_calibration."""
    segmenter = DefectSegmenter(pixels_per_mm=None)
    assert not segmenter.is_calibrated

    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[50:100, 50:100] = 200
    bbox = (0.25, 0.25, 0.50, 0.50)

    # 1. First run uncalibrated
    m_uncal = segmenter.segment_roi(img, bbox)
    assert m_uncal.surface_area_mm2 is None
    assert m_uncal.calibration_label == "uncalibrated (px)"

    # 2. Switch to calibrated
    segmenter.set_calibration(10.0)
    m_cal = segmenter.segment_roi(img, bbox)
    assert m_cal.is_calibrated is True
    assert m_cal.surface_area_mm2 is not None
    assert m_cal.calibration_label == "calibrated"

    # 3. Switch back to uncalibrated
    segmenter.set_calibration(None)
    m_reset = segmenter.segment_roi(img, bbox)
    assert m_reset.is_calibrated is False
    assert m_reset.surface_area_mm2 is None
    assert m_reset.calibration_label == "uncalibrated (px)"


def test_calibration_api_endpoints():
    """Verify GET and POST calibration functions."""
    # 1. Reset to uncalibrated
    res = set_camera_calibration(CalibrationRequest(px_per_mm=None))
    assert res.calibrated is False
    assert res.calibration_label == "uncalibrated (px)"
    assert res.px_per_mm is None

    # 2. Query calibration status
    status = get_camera_calibration()
    assert status.calibrated is False
    assert status.calibration_label == "uncalibrated (px)"

    # 3. Calibrate by specifying line length (px) and known physical length (mm)
    # e.g., 200 pixels line represents 10.0 mm -> scale = 20.0 px/mm
    res = set_camera_calibration(
        CalibrationRequest(line_length_px=200.0, known_length_mm=10.0)
    )
    assert res.calibrated is True
    assert res.calibration_label == "calibrated"
    assert res.px_per_mm == 20.0

    # 4. Check status reflects updated calibration
    status = get_camera_calibration()
    assert status.px_per_mm == 20.0
    assert status.calibrated is True

    # 5. Clean up - reset back to uncalibrated
    reset = set_camera_calibration(CalibrationRequest(px_per_mm=None))
    assert reset.calibrated is False
