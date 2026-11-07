"""Deterministic Quality Conformance Verification Engine.

Computes explicit boolean conformance checks (TRUE / FALSE) for industrial QA
inspections according to ISO 9001 / IATF 16949 and ASTM quality standards.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Standard manufacturing tolerance limits
MAX_DEFECT_AREA_RATIO_TOLERANCE = 0.020  # Max 2.0% surface flaw density
MAX_ALLOWABLE_CRITICAL_DEFECTS = 0
MAX_ALLOWABLE_MAJOR_DEFECTS = 0
MAX_LINEAR_LENGTH_MM_TOLERANCE = 15.0   # Max 15mm single flaw length
MAX_LINEAR_LENGTH_PX_TOLERANCE = 140.0  # Max 140px uncalibrated flaw length


class ConformanceService:
    """Evaluates industrial inspection records against deterministic quality tolerance criteria."""

    @classmethod
    def evaluate_inspection(
        cls,
        inspection_data: Any,
        defects: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate deterministic boolean conformance checklist for an inspection.

        Args:
            inspection_data: SQLAlchemy Inspection object or dictionary containing inspection metrics.
            defects: Optional list of defect records or DefectSchema objects.

        Returns:
            Dictionary matching ConformanceReportSchema with individual checklist items.
        """
        # 1. Extract high-level inspection counters
        if isinstance(inspection_data, dict):
            status = str(inspection_data.get("status", "passed")).lower()
            critical_count = int(inspection_data.get("critical_count", 0))
            major_count = int(inspection_data.get("major_count", 0))
            minor_count = int(inspection_data.get("minor_count", 0))
            total_defects = int(inspection_data.get("total_defects", 0))
            raw_defects = defects or inspection_data.get("defects", [])
        else:
            status = str(getattr(inspection_data, "status", "passed")).lower()
            critical_count = int(getattr(inspection_data, "critical_count", 0))
            major_count = int(getattr(inspection_data, "major_count", 0))
            minor_count = int(getattr(inspection_data, "minor_count", 0))
            total_defects = int(getattr(inspection_data, "total_defects", 0))
            raw_defects = defects if defects is not None else getattr(inspection_data, "defects", [])

        # 2. Compute aggregate defect area ratio & max dimensional length
        total_area_ratio = 0.0
        max_flaw_length = 0.0
        is_calibrated = False
        has_unclassified_anomaly = False

        for d in raw_defects:
            if isinstance(d, dict):
                area_ratio = float(d.get("area_ratio", 0.0) or 0.0)
                cname = str(d.get("class_name", "")).lower()
                metro = d.get("metrology")
            else:
                area_ratio = float(getattr(d, "area_ratio", 0.0) or 0.0)
                cname = str(getattr(d, "class_name", "")).lower()
                metro_raw = getattr(d, "metrology", None)
                if metro_raw is None:
                    metro_json = getattr(d, "metrology_json", None)
                    metro = json.loads(metro_json) if metro_json else None
                elif isinstance(metro_raw, str):
                    metro = json.loads(metro_raw)
                else:
                    metro = metro_raw

            total_area_ratio += area_ratio
            if "anomaly" in cname:
                has_unclassified_anomaly = True

            if metro and isinstance(metro, dict):
                if metro.get("is_calibrated") and metro.get("length_mm") is not None:
                    is_calibrated = True
                    flaw_len = float(metro.get("length_mm", 0.0) or 0.0)
                else:
                    flaw_len = float(metro.get("length_px", 0.0) or metro.get("pixel_area", 0.0) ** 0.5 or 0.0)
                if flaw_len > max_flaw_length:
                    max_flaw_length = flaw_len

        # 3. Check Criteria (True / False)

        # Criterion 1: Zero Critical Defects
        c1_pass = (critical_count <= MAX_ALLOWABLE_CRITICAL_DEFECTS)
        c1_item = {
            "criterion_id": "CRIT-01",
            "name": "Zero Critical Structural Flaws",
            "description": "Zero catastrophic structural cracks, deep crazing, or severe substrate ruptures allowed.",
            "passed": c1_pass,
            "status_label": "TRUE (PASS)" if c1_pass else "FALSE (VIOLATION)",
            "severity_level": "pass" if c1_pass else "violation",
            "recommendation": (
                "Continuous automated conveyor feed authorized."
                if c1_pass
                else "Immediate line-halt and batch quarantine required. Initiate 8D containment."
            ),
        }

        # Criterion 2: Major Flaw Containment
        c2_pass = (major_count <= MAX_ALLOWABLE_MAJOR_DEFECTS)
        c2_item = {
            "criterion_id": "CRIT-02",
            "name": "Major Surface Flaw Containment",
            "description": "Zero major repairable inclusions or roll-in scale exceeding tolerance boundaries.",
            "passed": c2_pass,
            "status_label": "TRUE (PASS)" if c2_pass else "FALSE (VIOLATION)",
            "severity_level": "pass" if c2_pass else "warning",
            "recommendation": (
                "Surface integrity conforms to Class-A manufacturing standards."
                if c2_pass
                else "Divert workpiece via pneumatic arm to Station B for offline mechanical redress."
            ),
        }

        # Criterion 3: Defect Area Density (< 2.0%)
        c3_pass = (total_area_ratio <= MAX_DEFECT_AREA_RATIO_TOLERANCE)
        c3_item = {
            "criterion_id": "CRIT-03",
            "name": "Defect Area Density (< 2.0% Threshold)",
            "description": (
                f"Total cumulative flaw area must not exceed 2.0% of total workpiece surface area "
                f"(Measured: {total_area_ratio * 100:.2f}%)."
            ),
            "passed": c3_pass,
            "status_label": "TRUE (PASS)" if c3_pass else "FALSE (VIOLATION)",
            "severity_level": "pass" if c3_pass else "violation",
            "recommendation": (
                "Defect density well within permissible ASTM / ISO limits."
                if c3_pass
                else "Excessive surface flaw coverage detected; inspect mill lubricant headers and roll pressure."
            ),
        }

        # Criterion 4: Dimensional Metrology Limits
        if is_calibrated:
            c4_pass = (max_flaw_length <= MAX_LINEAR_LENGTH_MM_TOLERANCE)
            c4_desc = (
                f"Maximum continuous linear defect length must not exceed {MAX_LINEAR_LENGTH_MM_TOLERANCE} mm "
                f"(Peak detected: {max_flaw_length:.1f} mm)."
            )
        else:
            c4_pass = (max_flaw_length <= MAX_LINEAR_LENGTH_PX_TOLERANCE)
            c4_desc = (
                f"Maximum defect pixel extent must not exceed {MAX_LINEAR_LENGTH_PX_TOLERANCE} px "
                f"(Peak detected: {max_flaw_length:.1f} px)."
            )

        c4_item = {
            "criterion_id": "CRIT-04",
            "name": "Dimensional Metrology Specification",
            "description": c4_desc,
            "passed": c4_pass,
            "status_label": "TRUE (PASS)" if c4_pass else "FALSE (VIOLATION)",
            "severity_level": "pass" if c4_pass else "warning",
            "recommendation": (
                "All defect dimensions comply with geometric product specifications."
                if c4_pass
                else "Flaw dimensions exceed tolerance limits; redress rolls and adjust slitting shears."
            ),
        }

        # Criterion 5: Material Matrix Structural Saliency
        c5_pass = (not has_unclassified_anomaly)
        c5_item = {
            "criterion_id": "CRIT-05",
            "name": "Material Matrix Structural Saliency",
            "description": "Zero unexpected morphological anomalies, thermal burns, or out-of-domain matrix disruptions.",
            "passed": c5_pass,
            "status_label": "TRUE (PASS)" if c5_pass else "FALSE (VIOLATION)",
            "severity_level": "pass" if c5_pass else "warning",
            "recommendation": (
                "Substrate microstructure conforms to nominal baseline texture."
                if c5_pass
                else "Unclassified texture variation detected; execute Tier 2 VLM deep grounding."
            ),
        }

        # Criterion 6: Final Manufacturing Lot Release Authorization
        overall_release = (
            c1_pass
            and c3_pass
            and (status == "passed" or (critical_count == 0 and major_count == 0))
        )
        c6_item = {
            "criterion_id": "CRIT-06",
            "name": "Manufacturing Lot Release Authorization",
            "description": "Final quality disposition authorizing release to shipping or continuous line assembly.",
            "passed": overall_release,
            "status_label": "TRUE (RELEASE APPROVED)" if overall_release else "FALSE (RELEASE QUARANTINED)",
            "severity_level": "pass" if overall_release else "violation",
            "recommendation": (
                "Workpiece conforming to engineering quality standards. Certificate validated."
                if overall_release
                else "Workpiece non-conforming. Release revoked; quarantine batch for QA disposition."
            ),
        }

        items = [c1_item, c2_item, c3_item, c4_item, c5_item, c6_item]
        passed_count = sum(1 for it in items if it["passed"])
        failed_count = len(items) - passed_count

        return {
            "lot_release_authorized": overall_release,
            "surface_integrity_pass": c1_pass and c2_pass,
            "critical_violation_free": c1_pass,
            "area_tolerance_pass": c3_pass,
            "dimensional_tolerance_pass": c4_pass,
            "material_anomaly_clear": c5_pass,
            "total_criteria": len(items),
            "passed_criteria": passed_count,
            "failed_criteria": failed_count,
            "items": items,
        }

