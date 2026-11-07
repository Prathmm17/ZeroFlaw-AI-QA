"""Unit tests for Domain Guard, per-class thresholding, and routing logic.

Validates:
1. Domain Guard returns trust HIGH on in-domain flat steel images and trust LOW on out-of-domain photos.
2. When trust is LOW, warning message matches:
   "Image differs from the training domain (flat steel surface). Result may be unreliable."
3. Detections with trust LOW and source anomaly_cv are at most MAJOR.
4. Item verdict under trust LOW becomes "FLAGGED for human review", never auto-REJECT.
5. CRITICAL severity is assigned ONLY when confidence >= class threshold AND area/class rules apply.
6. Calibrated thresholds from backend/class_thresholds.json are loaded at inference.
"""

from __future__ import annotations

import json
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
import torch
# PyTorch 2.6 weights_only patch
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

from ml.arbitrator import ProductionArbitrator
from ml.detector import DefectDetector
from ml.domain_guard import DOMAIN_WARNING_MESSAGE, DomainGuard
from ml.postprocessor import Detection, DetectionPostprocessor


def test_class_thresholds_json_exists_and_loaded():
    """Verify backend/class_thresholds.json exists and detector loads them."""
    json_path = PROJECT_ROOT / "backend" / "class_thresholds.json"
    assert json_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "thresholds" in data
    assert "crazing" in data["thresholds"]
    assert "scratches" in data["thresholds"]

    detector = DefectDetector()
    for cls_name, thresh in data["thresholds"].items():
        assert cls_name in detector.class_thresholds
        assert detector.class_thresholds[cls_name] == thresh


def test_domain_guard_in_domain_high_trust():
    """Verify DomainGuard returns trust HIGH for synthetic/real flat steel images with high conf."""
    guard = DomainGuard()
    # 400x400 neutral flat grayscale steel (saturation = 0)
    img = np.full((400, 400, 3), 128, dtype=np.uint8)
    assessment = guard.assess(img, max_yolo_confidence=0.85)

    assert assessment.trust == "HIGH"
    assert assessment.warning is None


def test_domain_guard_out_of_domain_low_trust():
    """Verify DomainGuard returns trust LOW and exact warning for high saturation or low conf."""
    guard = DomainGuard()
    # High color saturation (e.g. bronze or colored background) with low model confidence
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    img[:, :] = (30, 100, 200)  # BGR colored image

    assessment = guard.assess(img, max_yolo_confidence=0.10)
    assert assessment.trust == "LOW"
    assert assessment.warning == DOMAIN_WARNING_MESSAGE


def test_routing_critical_requires_confidence_above_threshold():
    """Verify CRITICAL is assigned ONLY when confidence >= class threshold."""
    post = DetectionPostprocessor(
        class_thresholds={"crazing": 0.50, "patches": 0.45}
    )

    # Crazing with confidence 0.40 (< class threshold 0.50): CANNOT be critical
    sev_below = post.classify_severity(
        confidence=0.40,
        area_ratio=0.06,
        class_name="crazing",
        domain_trust="HIGH",
        source="yolo",
    )
    assert sev_below != "critical"

    # Crazing with confidence 0.65 (>= class threshold 0.50) and area 0.04: CAN be critical
    sev_above = post.classify_severity(
        confidence=0.65,
        area_ratio=0.04,
        class_name="crazing",
        domain_trust="HIGH",
        source="yolo",
    )
    assert sev_above == "critical"


def test_routing_low_trust_anomaly_cv_capped_at_major():
    """Verify detections with trust LOW and source anomaly_cv are at most MAJOR."""
    post = DetectionPostprocessor()

    # High energy anomaly with large area, but under trust LOW:
    sev = post.classify_severity(
        confidence=0.92,
        area_ratio=0.20,
        class_name="unclassified_anomaly",
        domain_trust="LOW",
        source="anomaly_cv",
    )
    assert sev == "major"
    assert sev != "critical"


def test_arbitrator_low_trust_verdict_flagged_never_reject():
    """Verify that under trust LOW, the item verdict becomes 'FLAGGED for human review', never auto-REJECT."""
    arbitrator = ProductionArbitrator()

    # Real photo known to have trust LOW (Cast Part)
    cast_photo = PROJECT_ROOT / "data" / "uploads" / "WhatsApp Image 2026-10-30 at 6.26.03 AM_20261030_011330_6eded82a_orig.jpg"
    if cast_photo.exists():
        result = arbitrator.arbitrate(cast_photo)
        assert result.domain_trust == "LOW"
        assert result.domain_warning == DOMAIN_WARNING_MESSAGE
        assert result.verdict == "FLAGGED for human review"
        assert result.verdict != "REJECTED"
        assert result.critical_count == 0  # No auto-reject criticals under LOW trust

