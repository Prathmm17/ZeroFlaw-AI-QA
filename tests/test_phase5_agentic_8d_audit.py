"""Unit and integration tests for Phase 5: Agentic 8D Quality Root-Cause Auditor & Enterprise UI Polish.

Verification Gates:
1. Complete, valid 8D diagnostic report generation adhering strictly to IATF 16949 / ISO 9001 (D1-D8).
2. Pure Python vector PDF report generation with valid PDF 1.4 byte structure.
3. API PDF export endpoint (GET /api/detect/{id}/8d-report/pdf) returning 200 and application/pdf.
4. Statistical Process Control (SPC) Cp and Cpk capability calculation with Six-Sigma classification.
5. Defect Type Pareto 80/20 cumulative distribution sorting and calculation.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

os.environ["OFFLINE_MODE"] = "1"

import numpy as np
import pytest

from backend.services.pdf_service import generate_8d_report_pdf
from backend.services.vlm_service import EightDDiagnosticReport, QualityEngineeringAuditor


def test_8d_report_generation_all_disciplines_d1_to_d8() -> None:
    """Verify 8D audit report populates all automotive D1 through D8 disciplines."""
    auditor = QualityEngineeringAuditor(api_key=None)
    mock_image = np.full((640, 640, 3), 128, dtype=np.uint8)

    detections = [
        {
            "class_name": "crazing",
            "confidence": 0.942,
            "severity": "critical",
            "bbox": (120.0, 80.0, 240.0, 210.0),
            "area_ratio": 0.038,
            "metrology": {
                "pixel_area": 15600,
                "surface_area_mm2": 156.0,
                "length_mm": 24.0,
                "width_mm": 13.0,
                "aspect_ratio": 1.85,
                "solidity": 0.88,
                "is_calibrated": True,
            },
        }
    ]

    report = auditor.audit_inspection(
        inspection_id=501,
        image=mock_image,
        detections=detections,
        material_type="Cold-Rolled Automotive Strip (SPCC)",
        line_speed_mpm=120.0,
    )

    # Assert standard 8D metadata
    assert report.inspection_id == 501
    assert report.primary_defect_type == "crazing"
    assert report.engineering_severity.upper() == "CRITICAL"
    assert "ISO" in report.standard_reference or "ASTM" in report.standard_reference

    # Assert all D1 through D8 disciplines exist and are non-empty
    assert report.d1_team and len(report.d1_team) > 10
    assert report.d2_problem_description and len(report.d2_problem_description) > 10
    assert report.d3_containment_action and len(report.d3_containment_action) > 10
    assert report.d4_root_cause_analysis and len(report.d4_root_cause_analysis) > 10
    assert report.d5_corrective_actions and len(report.d5_corrective_actions) > 10
    assert report.d6_implementation_verification and len(report.d6_implementation_verification) > 10
    assert report.d7_preventive_action and len(report.d7_preventive_action) > 10
    assert report.d8_team_recognition and len(report.d8_team_recognition) > 10

    # Assert markdown formatted report contains all D1-D8 headings
    md = report.markdown_report
    assert "8D Quality Discrepancy" in md
    assert "## D1: Cross-Functional Team Establishment" in md
    assert "## D2: Problem Description" in md
    assert "## D3: Interim Containment Actions" in md
    assert "## D4: Root Cause Analysis" in md
    assert "## D5: Chosen Permanent Corrective Action" in md
    assert "## D6: Implemented Corrective Actions & Verification" in md
    assert "## D7: Actions to Prevent Recurrence" in md
    assert "## D8: Team Recognition" in md


def test_pure_python_pdf_generator_structure() -> None:
    """Verify pure Python PDF service builds a valid binary PDF with D1-D8 content."""
    report_data = {
        "inspection_id": 501,
        "primary_defect_type": "crazing",
        "engineering_severity": "CRITICAL",
        "standard_reference": "IATF 16949 / ISO 9001:2015 §8.7",
        "tooling_station": "Secondary Quench Chamber & Spray Header 3",
        "defect_mechanism": "Thermal fatigue cracking along austenite grain boundaries",
        "probable_root_cause": "Spray nozzle #4 clogged creating localized thermal shock delta T > 220C",
        "containment_action": "Quarantine rolling batch and isolate outfeed cooling bed",
        "corrective_action": "Purge descaling nozzles and recalibrate infrared pyrometer array",
        "d1_team": "Lead Metallurgist, Rolling Superintendent, Automation QA Engineer",
        "d2_problem_description": "Network of surface crazing micro-fractures detected on specimen",
        "d3_containment_action": "Hold current coil lot in containment bay; 100% optical inspection",
        "d4_root_cause_analysis": "Root cause confirmed as clogged secondary spray header #4",
        "d5_corrective_actions": "Replace spray nozzle tips, inspect descaler pump pressure",
        "d6_implementation_verification": "Conduct trial rolling run with Cp > 1.67 over 50 specimens",
        "d7_preventive_action": "Implement weekly automated pressure differential check on manifold",
        "d8_team_recognition": "Commend Line 2 maintenance shift and update QMS log",
        "metrology_summary": {
            "defects_evaluated": 1,
            "max_surface_area_mm2": 156.0,
            "mean_aspect_ratio": 1.85,
        },
    }

    pdf_bytes = generate_8d_report_pdf(report_data)

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    # Must adhere to PDF specification
    assert pdf_bytes.startswith(b"%PDF-1.4")
    assert b"%%EOF" in pdf_bytes

    # Must contain essential report tokens in PDF streams
    assert (b"ZeroFlaw" in pdf_bytes) or (b"VISIONGUARD" in pdf_bytes)
    assert b"8D DISCREPANCY" in pdf_bytes
    assert b"D1" in pdf_bytes
    assert b"D4" in pdf_bytes
    assert b"D5" in pdf_bytes
    assert b"ROOT CAUSE ANALYSIS" in pdf_bytes
    assert b"PERMANENT CORRECTIVE ACTION" in pdf_bytes


def test_api_8d_report_pdf_export_endpoint() -> None:
    """Verify export_8d_report_pdf router endpoint generates valid downloadable PDF."""
    from backend.database import SessionLocal, Inspection as InspectionModel
    from backend.routers.detection import export_8d_report_pdf
    import datetime

    db = SessionLocal()
    try:
        insp = InspectionModel(
            timestamp=datetime.datetime.utcnow(),
            source="test_phase5",
            filename="phase5_specimen.jpg",
            original_path="",
            result_path="",
            total_defects=1,
            critical_count=1,
            major_count=0,
            minor_count=0,
            max_severity="critical",
            status="rejected",
            inference_time_ms=22.5,
            total_time_ms=28.0,
            audit_report_json=json.dumps({
                "inspection_id": 9999,
                "primary_defect_type": "crazing",
                "engineering_severity": "CRITICAL",
                "standard_reference": "IATF 16949 / ISO 9001:2015",
                "defect_mechanism": "Thermal fatigue micro-fissuring",
                "probable_root_cause": "Coolant quench delta T > 45C",
                "tooling_station": "Finishing Stand F2",
                "containment_action": "Quarantine affected lot immediately",
                "corrective_action": "Replace spray nozzle manifold",
                "d1_team": "QA Director, Lead Metallurgist",
                "d2_problem_description": "Surface crazing network",
                "d3_containment_action": "Quarantine affected coil lot",
                "d4_root_cause_analysis": "Coolant quench delta T exceeded limits",
                "d5_corrective_actions": "Re-engineer coolant header spray geometry",
                "d6_implementation_verification": "Validated thermal profile over 5 heats",
                "d7_preventive_action": "Update Control Plan and PFMEA",
                "d8_team_recognition": "Team recognized for rapid resolution",
                "metrology_summary": {"defects_evaluated": 1},
            }),
        )
        db.add(insp)
        db.commit()
        db.refresh(insp)
        insp_id = insp.id

        # Request the PDF export through the router endpoint
        res = export_8d_report_pdf(inspection_id=insp_id, db=db)
        assert res.status_code == 200
        assert res.media_type == "application/pdf"
        assert "attachment" in res.headers["Content-Disposition"]
        assert f"8D_Report_VG_INSP_{insp_id:06d}.pdf" in res.headers["Content-Disposition"]
        assert res.body.startswith(b"%PDF-1.4")
        assert b"%%EOF" in res.body
    finally:
        db.close()


def test_analytics_spc_capability_and_six_sigma_calculation() -> None:
    """Verify statistical process control calculations (Cp, Cpk) and Six-Sigma status."""
    from backend.database import SessionLocal
    from backend.routers.analytics import get_analytics_overview

    db = SessionLocal()
    try:
        overview = get_analytics_overview(db=db)
        assert overview.total_inspections >= 0
        assert overview.cp is not None
        assert overview.cpk is not None
        assert isinstance(overview.cp, float)
        assert isinstance(overview.cpk, float)
        assert overview.cp > 0.0
        assert overview.cpk >= 0.0

        # Verify Six-Sigma process status mapping
        valid_statuses = {
            "WORLD_CLASS_SIX_SIGMA",
            "CAPABLE_PROCESS",
            "MARGINALLY_CAPABLE",
            "REQUIRES_INVESTIGATION",
        }
        assert overview.process_capability_status in valid_statuses
    finally:
        db.close()


def test_analytics_pareto_cumulative_80_20_distribution() -> None:
    """Verify defect Pareto cumulative percentages sort descending and reach 100%."""
    from backend.database import SessionLocal
    from backend.routers.analytics import get_analytics_overview

    db = SessionLocal()
    try:
        overview = get_analytics_overview(db=db)
        pareto = overview.pareto_cumulative
        assert isinstance(pareto, dict)

        if pareto:
            percentages = list(pareto.values())
            # Percentages must be monotonically non-decreasing
            for i in range(len(percentages) - 1):
                assert percentages[i] <= percentages[i + 1]

            # Final cumulative percentage must be 100.0%
            assert pytest.approx(percentages[-1], 0.1) == 100.0

            # All values must be bounded between 0 and 100
            for pct in percentages:
                assert 0.0 <= pct <= 100.0
    finally:
        db.close()
