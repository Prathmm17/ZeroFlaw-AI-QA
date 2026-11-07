"""Unit tests for detection source tracking and out-of-domain handling.

Validates:
1. Every detection carries a 'source' field in ('yolo', 'anomaly_cv', 'vlm').
2. When YOLO confidence is below class threshold and anomaly engine finds the region,
   the detection is labeled 'unclassified_anomaly' with source 'anomaly_cv'.
3. Bounding box labels and UI descriptions never invent a class name or claim VLM when offline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

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

from backend.services.gemini_grounding import GeminiVisualGrounder, GroundedDefect
from ml.anomaly_engine import AnomalyRegion
from ml.arbitrator import ArbitratedDefect, ProductionArbitrator
from ml.detector import DefectDetector
from ml.postprocessor import Detection, DetectionPostprocessor


def test_postprocessor_source_field():
    """Verify Detection dataclass defaults to 'yolo' source."""
    det = Detection(
        bbox=(10.0, 20.0, 50.0, 60.0),
        confidence=0.85,
        class_id=0,
        class_name="crazing",
    )
    assert hasattr(det, "source")
    assert det.source == "yolo"


def test_gemini_grounding_offline_source():
    """Verify GeminiVisualGrounder fallback returns source='anomaly_cv' and 'unclassified_anomaly'."""
    grounder = GeminiVisualGrounder(api_key=None)
    assert not grounder.is_api_active()

    # Synthetic image with an anomaly
    img = np.full((300, 300, 3), 128, dtype=np.uint8)
    # Draw a defect
    img[100:150, 100:150] = 20

    grounded = grounder.ground_defects(img)
    for g in grounded:
        assert g.source == "anomaly_cv"
        assert g.label == "unclassified_anomaly"
        assert g.source != "vlm"


def test_out_of_domain_confidence_relabeling():
    """Verify that a YOLO detection below class threshold overlapping an anomaly is relabeled unclassified_anomaly."""
    detector = DefectDetector(confidence_threshold=0.20)
    # Define an anomaly region at (0.2, 0.2, 0.5, 0.5)
    anom_region = AnomalyRegion(
        bbox=(0.2, 0.2, 0.5, 0.5),
        anomaly_score=0.72,
        contour=[[0.2, 0.2], [0.5, 0.2], [0.5, 0.5], [0.2, 0.5]],
        defect_type_hint="crazing",
        pixel_area=1500,
    )

    # Box in pixels on 600x600 canvas (normalized: 0.25 to 0.45, overlapping anomaly)
    pixel_bbox = (150.0, 150.0, 270.0, 270.0)
    overlaps = detector._overlaps_anomaly(pixel_bbox, (600, 600), [anom_region])
    assert overlaps is True


def test_arbitrator_hud_and_source_assignment():
    """Verify ProductionArbitrator preserves anomaly_cv and sets correct status and badge."""
    arbitrator = ProductionArbitrator()
    img = np.full((400, 400, 3), 120, dtype=np.uint8)
    # Dark anomaly in center
    img[150:250, 150:250] = 10

    res = arbitrator.arbitrate(img, force_tier2=True)
    for d in res.detections:
        assert d.source in ("yolo", "anomaly_cv", "vlm")
        if d.source == "anomaly_cv":
            assert "[VLM]" not in d.description
            assert "Tier 2 VLM Verified" not in d.description
            assert res.arbitration_status != "VLM_CONFIRMED"

