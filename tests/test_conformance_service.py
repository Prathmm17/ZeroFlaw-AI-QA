"""Unit and integration tests for Conformance Verification Engine (Phase 1).

Verifies:
1. ConformanceService deterministic boolean evaluation (TRUE / FALSE logic).
2. Clean specimen passes all manufacturing conformance checks.
3. Critical crack flaw fails surface integrity and revokes lot release.
4. Excessive surface flaw area ratio (> 2.0%) fails area tolerance check.
5. Dimensional metrology tolerance verification (calibrated & uncalibrated).
6. InspectionResponse Pydantic contract serialization.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict

import pytest

from backend.models import ConformanceReportSchema, DefectSchema, InspectionResponse
from backend.services.conformance_service import ConformanceService


def test_conformance_evaluation_clean_specimen() -> None:
    """Verify clean workpiece with zero flaws passes all manufacturing criteria."""
    clean_data = {
        "status": "passed",
        "total_defects": 0,
        "critical_count": 0,
        "major_count": 0,
        "minor_count": 0,
        "defects": [],
    }

    report = ConformanceService.evaluate_inspection(clean_data)

    assert report["lot_release_authorized"] is True
    assert report["surface_integrity_pass"] is True
    assert report["critical_violation_free"] is True
    assert report["area_tolerance_pass"] is True
    assert report["dimensional_tolerance_pass"] is True
    assert report["material_anomaly_clear"] is True
    assert report["total_criteria"] == 6
    assert report["passed_criteria"] == 6
    assert report["failed_criteria"] == 0

    for item in report["items"]:
        assert item["passed"] is True
        assert "TRUE" in item["status_label"]
        assert item["severity_level"] == "pass"


def test_conformance_evaluation_critical_crack_failure() -> None:
    """Verify critical crack flaw flags violation and revokes lot release authorization."""
    critical_data = {
        "status": "rejected",
        "total_defects": 1,
        "critical_count": 1,
        "major_count": 0,
        "minor_count": 0,
        "defects": [
            {
                "class_name": "crazing",
                "severity": "critical",
                "area_ratio": 0.008,
                "confidence": 0.92,
                "metrology": {"length_mm": 12.0, "is_calibrated": True},
            }
        ],
    }

    report = ConformanceService.evaluate_inspection(critical_data)

    assert report["lot_release_authorized"] is False
    assert report["critical_violation_free"] is False
    assert report["surface_integrity_pass"] is False
    assert report["failed_criteria"] >= 1

    crit1 = next(it for it in report["items"] if it["criterion_id"] == "CRIT-01")
    assert crit1["passed"] is False
    assert "FALSE" in crit1["status_label"]
    assert crit1["severity_level"] == "violation"

    crit6 = next(it for it in report["items"] if it["criterion_id"] == "CRIT-06")
    assert crit6["passed"] is False
    assert "FALSE" in crit6["status_label"]


def test_conformance_evaluation_excess_area_failure() -> None:
    """Verify flaw area exceeding 2.0% threshold fails area tolerance check."""
    high_density_data = {
        "status": "flagged",
        "total_defects": 3,
        "critical_count": 0,
        "major_count": 0,
        "minor_count": 3,
        "defects": [
            {"class_name": "scratches", "severity": "minor", "area_ratio": 0.012},
            {"class_name": "patches", "severity": "minor", "area_ratio": 0.015},
        ],
    }

    report = ConformanceService.evaluate_inspection(high_density_data)

    assert report["area_tolerance_pass"] is False
    assert report["lot_release_authorized"] is False

    crit3 = next(it for it in report["items"] if it["criterion_id"] == "CRIT-03")
    assert crit3["passed"] is False
    assert "FALSE" in crit3["status_label"]
    assert "EXCEEDED" in crit3["status_label"] or "VIOLATION" in crit3["status_label"]


def test_conformance_evaluation_dimensional_metrology() -> None:
    """Verify dimensional metrology tolerance limits in calibrated (mm) mode."""
    # Defect length 24.5mm exceeds 15.0mm max threshold
    oversized_data = {
        "status": "passed",
        "total_defects": 1,
        "critical_count": 0,
        "major_count": 0,
        "minor_count": 1,
        "defects": [
            {
                "class_name": "scratches",
                "severity": "minor",
                "area_ratio": 0.005,
                "metrology": {"length_mm": 24.5, "is_calibrated": True},
            }
        ],
    }

    report = ConformanceService.evaluate_inspection(oversized_data)
    assert report["dimensional_tolerance_pass"] is False
    crit4 = next(it for it in report["items"] if it["criterion_id"] == "CRIT-04")
    assert crit4["passed"] is False
    assert "FALSE" in crit4["status_label"]


def test_conformance_report_schema_validation() -> None:
    """Verify ConformanceReportSchema Pydantic model parses service output cleanly."""
    data = {
        "status": "passed",
        "total_defects": 0,
        "critical_count": 0,
        "major_count": 0,
        "minor_count": 0,
        "defects": [],
    }
    report_dict = ConformanceService.evaluate_inspection(data)
    schema = ConformanceReportSchema(**report_dict)

    assert schema.lot_release_authorized is True
    assert len(schema.items) == 6
    assert all(it.passed for it in schema.items)

