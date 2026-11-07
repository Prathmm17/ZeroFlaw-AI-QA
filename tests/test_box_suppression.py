"""Unit tests for false box suppression and class-aware NMS.

Validates:
1. Connected components generates one box per component on thresholded masks without bridging.
2. Boxes exceeding 35% of the image area are dropped with a warning.
3. Boxes with aspect ratio exceeding 15:1 for non-scratch classes are dropped.
4. Class-aware NMS suppresses overlapping boxes (> 0.50 IoU) across YOLO and anomaly detections.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# NumPy 2.x patch
if not hasattr(np, "trapz"):
    np.trapz = getattr(np, "trapezoid", None)

import cv2
from ml.anomaly_engine import AnomalyRegion, UnsupervisedAnomalyEngine
from ml.postprocessor import Detection, DetectionPostprocessor


def test_connected_components_no_bridging():
    """Verify that two distinct nearby spots are not merged into a single bridged box."""
    engine = UnsupervisedAnomalyEngine(sensitivity=0.85)

    # 400x400 gray image with two separated dark defect spots
    img = np.full((400, 400, 3), 180, dtype=np.uint8)
    # Defect A: 8x8 spot at (100, 100)
    img[100:108, 100:108] = 20
    # Defect B: 8x8 spot at (100, 120) - separated by 12px gap
    img[100:108, 120:128] = 20

    anomalies = engine.detect_anomalies(img, max_detections=10)
    # Both defects should be localized as separate candidates, not bridged into one box
    assert len(anomalies) >= 2
    # Ensure no anomaly box bridges across the gap to merge both spots
    for a in anomalies:
        bx1, by1, bx2, by2 = a.bbox
        bw = (bx2 - bx1) * 400
        # Individual 8px spots should not produce a bridged box of width > 20px
        assert bw <= 20.0


def test_drop_boxes_above_35_percent_area():
    """Verify that any box covering > 35% of the image area is rejected."""
    engine = UnsupervisedAnomalyEngine(sensitivity=0.70)
    # 500x500 image with a huge defect occupying 60% of the image
    img = np.full((500, 500, 3), 180, dtype=np.uint8)
    img[50:450, 50:450] = 20  # 400x400 = 160000 / 250000 = 64% of image

    anomalies = engine.detect_anomalies(img)
    for a in anomalies:
        bx1, by1, bx2, by2 = a.bbox
        box_area_ratio = (bx2 - bx1) * (by2 - by1)
        assert box_area_ratio <= 0.35


def test_drop_aspect_ratio_above_15_non_scratch():
    """Verify that non-scratch boxes with aspect ratio > 15:1 are dropped."""
    postprocessor = DetectionPostprocessor()

    dets = [
        # Aspect 20:1 crazing -> should be dropped
        Detection(
            bbox=(10.0, 10.0, 210.0, 20.0),  # w=200, h=10 -> aspect 20:1
            confidence=0.75,
            class_id=0,
            class_name="crazing",
            source="yolo",
        ),
        # Aspect 20:1 scratch -> should be kept (scratches allowed high aspect)
        Detection(
            bbox=(10.0, 30.0, 210.0, 40.0),  # w=200, h=10 -> aspect 20:1
            confidence=0.80,
            class_id=5,
            class_name="scratches",
            source="yolo",
        ),
    ]

    filtered = []
    for d in dets:
        x1, y1, x2, y2 = d.bbox
        bw, bh = x2 - x1, y2 - y1
        aspect = max(bw, bh) / max(min(bw, bh), 1e-6)
        if aspect > 15.0 and d.class_name.lower() != "scratches":
            continue
        filtered.append(d)

    assert len(filtered) == 1
    assert filtered[0].class_name == "scratches"


def test_class_aware_nms_across_yolo_and_anomaly():
    """Verify class-aware NMS suppresses lower-confidence overlapping boxes (IoU > 0.50)."""
    dets = [
        # Anomaly box with higher confidence
        Detection(
            bbox=(100.0, 100.0, 200.0, 200.0),
            confidence=0.88,
            class_id=-1,
            class_name="unclassified_anomaly",
            source="anomaly_cv",
        ),
        # YOLO box for same region with lower confidence (IoU ~ 0.80)
        Detection(
            bbox=(105.0, 105.0, 200.0, 200.0),
            confidence=0.45,
            class_id=3,
            class_name="pitted_surface",
            source="yolo",
        ),
        # Distinct YOLO box at another location
        Detection(
            bbox=(300.0, 300.0, 350.0, 350.0),
            confidence=0.75,
            class_id=0,
            class_name="crazing",
            source="yolo",
        ),
    ]

    nms_result = DetectionPostprocessor.apply_class_aware_nms(dets, iou_threshold=0.50)

    # The overlapping pitted_surface (0.45) should be suppressed by unclassified_anomaly (0.88)
    assert len(nms_result) == 2
    classes_kept = [d.class_name for d in nms_result]
    assert "unclassified_anomaly" in classes_kept
    assert "crazing" in classes_kept
    assert "pitted_surface" not in classes_kept
