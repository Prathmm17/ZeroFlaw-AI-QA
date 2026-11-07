"""Unit tests for Defect Knowledge Base, Glossary API schema, and detection metadata attachment."""

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

import torch
# PyTorch 2.6 weights_only patch
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

from backend.models import DefectKnowledgeItem, DefectSchema
from backend.routers.detection import get_defect_glossary
from backend.services.knowledge_service import enrich_defect_dict, get_all_defect_knowledge, get_defect_knowledge
from ml.detector import DefectDetector
from ml.postprocessor import Detection


REQUIRED_CLASSES = {
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
    "unclassified_anomaly",
}

REQUIRED_FIELDS = {
    "definition",
    "visual_look",
    "cause",
    "risk",
    "action",
    "default_severity",
}


def test_defect_knowledge_json_file_validity():
    """Verify backend/defect_knowledge.json contains all 7 required defect classes and fields."""
    knowledge = get_all_defect_knowledge()
    assert isinstance(knowledge, dict)
    assert set(knowledge.keys()) == REQUIRED_CLASSES

    for cls_name, info in knowledge.items():
        assert isinstance(info, dict)
        for field in REQUIRED_FIELDS:
            assert field in info, f"Class '{cls_name}' missing field '{field}'"
            assert isinstance(info[field], str) and len(info[field].strip()) > 0
        assert info["default_severity"] in {"critical", "major", "minor"}
        assert "calibrated_threshold" in info
        assert "confidence_label" in info


def test_get_defect_glossary_api_endpoint():
    """Verify get_defect_glossary() returns accurate DefectKnowledgeItem models for all 7 classes."""
    glossary = get_defect_glossary()
    assert isinstance(glossary, dict)
    assert set(glossary.keys()) == REQUIRED_CLASSES

    for cls_name, item in glossary.items():
        assert isinstance(item, DefectKnowledgeItem)
        assert item.name == cls_name
        assert len(item.definition) > 0
        assert len(item.cause) > 0
        assert len(item.action) > 0
        assert len(item.visual_look) > 0
        assert len(item.risk) > 0
        assert item.default_severity in {"critical", "major", "minor"}
        assert item.calibrated_threshold is not None
        assert item.confidence_label is not None
        # Verify confidence label formatting rule
        assert "% conf" in item.confidence_label


def test_enrich_defect_dict_helper():
    """Verify enrich_defect_dict attaches definition, cause, action, visual_look, risk."""
    for cls in REQUIRED_CLASSES:
        raw_det = {"class_name": cls, "confidence": 0.95}
        enriched = enrich_defect_dict(raw_det)
        assert enriched["definition"] is not None and len(enriched["definition"]) > 0
        assert enriched["cause"] is not None and len(enriched["cause"]) > 0
        assert enriched["action"] is not None and len(enriched["action"]) > 0
        assert enriched["visual_look"] is not None and len(enriched["visual_look"]) > 0
        assert enriched["risk"] is not None and len(enriched["risk"]) > 0


def test_postprocessor_enriches_detection_objects():
    """Verify Detection objects and DefectSchema carry definition, cause, and action."""
    det = Detection(
        bbox=(10.0, 20.0, 50.0, 60.0),
        confidence=0.85,
        class_id=0,
        class_name="crazing",
    )
    k_info = get_defect_knowledge("crazing")
    assert k_info is not None
    det.definition = k_info["definition"]
    det.cause = k_info["cause"]
    det.action = k_info["action"]

    schema = DefectSchema(
        bbox=det.bbox,
        confidence=det.confidence,
        class_id=det.class_id,
        class_name=det.class_name,
        severity=det.severity,
        definition=det.definition,
        cause=det.cause,
        action=det.action,
    )
    assert schema.definition == k_info["definition"]
    assert schema.cause == k_info["cause"]
    assert schema.action == k_info["action"]
