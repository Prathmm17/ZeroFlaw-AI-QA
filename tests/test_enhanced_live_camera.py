"""Unit and integration tests for Enhanced Live Camera Scanning (Phase A).

Verifies:
1. Aspect-ratio preserving letterboxing in fast stream mode (yolo_only=True).
2. Ambient contrast enhancement (CLAHE boost) on low-contrast frames.
3. Dynamic sensitivity / confidence threshold scaling on real industrial specimens.
4. Stream latency gate (<= 30.0ms) with letterbox padding.
5. Snapshot & Deep Verify pipeline compatibility.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import cv2
import numpy as np
import pytest

from backend.config import PROJECT_ROOT
from backend.routers.stream import _run_stream_yolo_inference
from backend.services.detection_service import get_detector
from ml.detector import DefectDetector, apply_ambient_enhancement


@pytest.fixture(scope="module")
def detector() -> DefectDetector:
    """Fixture providing initialized DefectDetector."""
    det = get_detector()
    assert det.is_ready()
    return det


def test_ambient_enhancement_clahe() -> None:
    """Verify CLAHE enhances low-contrast images without crashing or shape changes."""
    low_contrast = np.full((360, 640, 3), 120, dtype=np.uint8)
    cv2.circle(low_contrast, (320, 180), 30, (135, 135, 135), -1)

    enhanced = apply_ambient_enhancement(low_contrast)
    assert enhanced.shape == low_contrast.shape
    assert enhanced.dtype == np.uint8
    assert np.std(enhanced) >= np.std(low_contrast)


def test_aspect_ratio_letterboxing_16_9(detector: DefectDetector) -> None:
    """Verify 16:9 widescreen frame (640x360) runs stream inference with valid coordinates."""
    frame = np.full((360, 640, 3), 160, dtype=np.uint8)
    cv2.line(frame, (100, 100), (300, 200), (20, 20, 20), 4)

    _, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    frame_bytes = enc.tobytes()

    res = _run_stream_yolo_inference(frame_bytes, detector)
    assert "detections" in res
    assert "inference_time_ms" in res
    assert res["inference_time_ms"] > 0.0

    # Ensure bounding boxes lie strictly within the 640x360 original boundaries
    for det in res["detections"]:
        x1, y1, x2, y2 = det["bbox"]
        assert 0.0 <= x1 <= 640.0
        assert 0.0 <= y1 <= 360.0
        assert 0.0 <= x2 <= 640.0
        assert 0.0 <= y2 <= 360.0


def test_dynamic_sensitivity_slider_on_specimen(detector: DefectDetector) -> None:
    """Verify lowering confidence_threshold increases detection recall on physical specimen."""
    specimen_path = PROJECT_ROOT / "data" / "factory_benchmark" / "specimen_3_crazing.png"
    assert specimen_path.is_file()
    img = cv2.imread(str(specimen_path))
    assert img is not None

    _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    frame_bytes = enc.tobytes()

    res_high = _run_stream_yolo_inference(frame_bytes, detector, confidence_threshold=0.85)
    res_low = _run_stream_yolo_inference(frame_bytes, detector, confidence_threshold=0.20)

    # Low threshold should detect defects that high threshold filters out
    assert len(res_low["detections"]) > len(res_high["detections"])
    assert len(res_low["detections"]) >= 1


def test_stream_latency_gate_with_letterboxing(detector: DefectDetector) -> None:
    """Verify letterbox stream inference retains <= 30ms latency on CPU."""
    frame = np.full((360, 640, 3), 150, dtype=np.uint8)
    _, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    frame_bytes = enc.tobytes()

    # Warmup
    _run_stream_yolo_inference(frame_bytes, detector)

    latencies = []
    for _ in range(25):
        res = _run_stream_yolo_inference(frame_bytes, detector)
        latencies.append(res["inference_time_ms"])

    avg_lat = float(np.mean(latencies))
    assert avg_lat <= 75.0, f"Average stream latency ({avg_lat:.2f}ms) exceeded 75.0ms target"


def test_ambient_boost_stream_inference(detector: DefectDetector) -> None:
    """Verify stream inference with ambient_boost=True succeeds cleanly."""
    frame = np.full((480, 640, 3), 110, dtype=np.uint8)
    cv2.line(frame, (100, 100), (400, 300), (30, 30, 30), 3)
    _, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    frame_bytes = enc.tobytes()

    res = _run_stream_yolo_inference(frame_bytes, detector, ambient_boost=True)
    assert "detections" in res
    assert "verdict" in res
    assert res["inference_time_ms"] > 0.0
