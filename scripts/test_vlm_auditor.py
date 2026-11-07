"""Phase 4 Verification Script: Gemini Flash / Metallurgical Root-Cause QA Auditor.

Tests the QualityEngineeringAuditor service and the 8D Discrepancy Quality Report
generation pipeline over direct unit tests and live HTTP API endpoints.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# NumPy 2.x patch
import numpy as np
if not hasattr(np, "trapz"):
    np.trapz = getattr(np, "trapezoid", None)

import cv2
from backend.services.vlm_service import EightDDiagnosticReport, QualityEngineeringAuditor


def test_unit_auditor() -> bool:
    print("\n--- TEST 1: Unit Test of QualityEngineeringAuditor & 8D Synthesis ---")
    auditor = QualityEngineeringAuditor()

    canvas = np.full((640, 640, 3), 160, dtype=np.uint8)

    # Test cases across metallurgical classes
    test_cases = [
        ("crazing", "ASTM E381", "Critical"),
        ("scratches", "ISO 1302", "Major"),
        ("inclusion", "ASTM E45", "Critical"),
        ("pitted_surface", "ASTM A899", "Major"),
        ("rolled_in_scale", "ISO 8501-1", "Major"),
    ]

    for defect_type, expected_std, expected_sev in test_cases:
        detections = [{
            "class_name": defect_type,
            "confidence": 0.88,
            "severity": expected_sev.lower(),
            "bbox": (0.2, 0.2, 0.4, 0.4),
            "polygon": [[0.2, 0.2], [0.4, 0.2], [0.4, 0.4]],
            "metrology": {
                "surface_area_mm2": 3.45,
                "length_mm": 8.2,
                "width_mm": 0.42,
                "aspect_ratio": 19.5,
                "solidity": 0.89,
            },
        }]

        report: EightDDiagnosticReport = auditor.audit_inspection(
            inspection_id=999,
            image=canvas,
            detections=detections,
        )

        print(f"Audited: {defect_type.upper():<16} | Standard: {report.standard_reference:<16} | Severity: {report.engineering_severity}")
        assert defect_type in report.primary_defect_type.lower()
        assert expected_std in report.standard_reference
        assert len(report.probable_root_cause) > 20
        assert len(report.corrective_action) > 20
        assert len(report.containment_action) > 10
        assert "# 8D Quality Discrepancy" in report.markdown_report

    print("[PASS] Unit tests for QualityEngineeringAuditor verified across all metallurgical defect types.")
    return True


def test_api_8d_report_endpoints() -> bool:
    print("\n--- TEST 2: Live HTTP Endpoints for 8D Audit & Report Retrieval ---")
    base_url = "http://127.0.0.1:8000"

    # 1. Trigger detection on a sample to create an inspection record
    samples_url = f"{base_url}/api/detect/samples"
    with urllib.request.urlopen(samples_url, timeout=5) as resp:
        samples_data = json.loads(resp.read().decode())
        if isinstance(samples_data, list):
            first_item = samples_data[0]
            sample_name = first_item if isinstance(first_item, str) else first_item["filename"]
        else:
            sample_name = samples_data["samples"][0]["filename"]

    print(f"Using sample: {sample_name}")
    req = urllib.request.Request(f"{base_url}/api/detect/sample/{sample_name}", data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        insp = json.loads(resp.read().decode())
        insp_id = insp["id"]

    print(f"Created inspection #{insp_id}")

    # 2. Trigger 8D Audit via POST /api/detect/{insp_id}/audit
    audit_url = f"{base_url}/api/detect/{insp_id}/audit"
    audit_req = urllib.request.Request(audit_url, data=b"", method="POST")
    with urllib.request.urlopen(audit_req, timeout=10) as resp:
        assert resp.status == 200
        audit_res = json.loads(resp.read().decode())

    print(f"[OK] Audit completed for #{insp_id}:")
    print(f"     Defect Mechanism   : {audit_res.get('defect_mechanism')}")
    print(f"     Tooling Station    : {audit_res.get('tooling_station')}")
    print(f"     Standard Reference : {audit_res.get('standard_reference')}")
    print(f"     Engineering Risk   : {audit_res.get('engineering_severity')}")

    # 3. Retrieve formatted report via GET /api/detect/{insp_id}/8d-report
    get_url = f"{base_url}/api/detect/{insp_id}/8d-report"
    with urllib.request.urlopen(get_url, timeout=5) as resp:
        assert resp.status == 200
        retrieved_report = json.loads(resp.read().decode())

    assert retrieved_report["inspection_id"] == insp_id
    assert "markdown_report" in retrieved_report
    print("[PASS] Live HTTP endpoints for 8D Audit and report retrieval verified.")
    return True


def main() -> int:
    print("=" * 70)
    print("VISIONGUARD QA — PHASE 4 ROOT-CAUSE QA AUDITOR VERIFICATION")
    print("=" * 70)

    t1_pass = test_unit_auditor()
    t2_pass = test_api_8d_report_endpoints()

    if t1_pass and t2_pass:
        print("\n" + "=" * 70)
        print("[SUCCESS] ALL PHASE 4 VERIFICATION TESTS PASSED (100% SUCCESS)")
        print("=" * 70)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
