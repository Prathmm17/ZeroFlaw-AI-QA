"""Integration tests for Browser Inspection Certificate & Conformance Report (Phase 2).

Verifies:
1. GET /api/detect/{id}/report/view returns valid HTML5 certificate directly viewable in browser.
2. Certificate HTML contains Conformance Verification Truth Matrix (TRUE / FALSE indicators).
3. GET /api/detect/{id}/8d-report/pdf supports ?inline=true for direct in-browser rendering.
4. Error handling for non-existent inspections (404 Not Found).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.config import PROJECT_ROOT, settings
from backend.database import Inspection, SessionLocal
from backend.main import app

client = TestClient(app)


@pytest.fixture(scope="module")
def sample_inspection_id() -> int:
    """Ensure at least one inspection exists in the database for reporting tests."""
    import datetime
    import json
    db = SessionLocal()
    try:
        insp = db.query(Inspection).first()
        if insp:
            return insp.id

        new_insp = Inspection(
            timestamp=datetime.datetime.utcnow(),
            source="test_report",
            filename="test_specimen.png",
            original_path="data/uploads/test.png",
            result_path="data/results/test_res.png",
            total_defects=1,
            critical_count=1,
            major_count=0,
            minor_count=0,
            max_severity="critical",
            status="rejected",
            inference_time_ms=18.5,
            total_time_ms=22.0,
            audit_report_json=json.dumps({
                "inspection_id": 1,
                "d3_containment_action": "Quarantine lot immediately",
                "d4_root_cause_analysis": "Thermal fatigue micro-fissuring",
                "d5_corrective_actions": "Redress work rolls",
                "standard_reference": "ISO 9001:2015 / IATF 16949",
            }),
        )
        db.add(new_insp)
        db.commit()
        db.refresh(new_insp)
        return new_insp.id
    finally:
        db.close()


def test_view_inspection_certificate_html(sample_inspection_id: int) -> None:
    """Verify view_inspection_certificate_html renders standalone HTML5 certificate with TRUE/FALSE matrix."""
    from backend.routers.detection import view_inspection_certificate_html
    db = SessionLocal()
    try:
        resp = view_inspection_certificate_html(sample_inspection_id, db=db)
        assert resp.status_code == 200
        assert "text/html" in resp.media_type

        html_content = resp.body.decode("utf-8")
        # Document structure assertions
        assert "<!DOCTYPE html>" in html_content
        assert ("ZeroFlaw AI-QA — Quality Assurance Certificate" in html_content) or ("VisionGuard QA — Quality Assurance Certificate" in html_content)
        assert f"VG-INSP-{sample_inspection_id:06d}" in html_content

        # Conformance Truth Matrix assertions (must explicitly present TRUE / FALSE)
        assert "Quality Conformance Verification Matrix" in html_content
        assert ("TRUE (PASS)" in html_content) or ("FALSE (VIOLATION)" in html_content)
        assert "CRIT-01" in html_content
        assert "CRIT-03" in html_content
        assert "CRIT-06" in html_content

        # Evidence & Metrology sections
        assert "Optical Inspection Evidence" in html_content
        assert "Detailed Defect Telemetry" in html_content
        assert "8D Root-Cause" in html_content

        # Browser print trigger
        assert "window.print()" in html_content
    finally:
        db.close()


def test_export_pdf_attachment_vs_inline(sample_inspection_id: int) -> None:
    """Verify inline=True opens PDF directly in browser while default downloads as attachment."""
    from backend.routers.detection import export_8d_report_pdf
    db = SessionLocal()
    try:
        # 1. Default (download attachment)
        resp_attach = export_8d_report_pdf(sample_inspection_id, inline=False, db=db)
        assert resp_attach.status_code == 200
        assert resp_attach.media_type == "application/pdf"
        assert "attachment;" in resp_attach.headers["Content-Disposition"]

        # 2. Inline (opens in browser tab)
        resp_inline = export_8d_report_pdf(sample_inspection_id, inline=True, db=db)
        assert resp_inline.status_code == 200
        assert resp_inline.media_type == "application/pdf"
        assert "inline;" in resp_inline.headers["Content-Disposition"]
    finally:
        db.close()


def test_view_report_nonexistent_id_404() -> None:
    """Verify 404 is returned gracefully for invalid inspection ID."""
    from fastapi import HTTPException
    from backend.routers.detection import view_inspection_certificate_html
    db = SessionLocal()
    try:
        with pytest.raises(HTTPException) as exc_info:
            view_inspection_certificate_html(9999999, db=db)
        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail.lower()
    finally:
        db.close()

