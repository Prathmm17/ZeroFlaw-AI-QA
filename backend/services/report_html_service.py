"""HTML5 Industrial Inspection Certificate & Conformance Report Generator.

Produces standalone, print-optimized, responsive HTML5 quality certificates
conforming to ISO 9001 / IATF 16949 standards with zero external dependencies.
"""

from __future__ import annotations

import datetime
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.config import settings
from backend.services.conformance_service import ConformanceService


def generate_inspection_certificate_html(
    inspection: Any,
    report_dict: Optional[Dict[str, Any]] = None,
    conformance_data: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate self-contained HTML5 quality certificate for direct browser rendering.

    Args:
        inspection: SQLAlchemy Inspection model or dictionary of inspection metrics.
        report_dict: Optional 8D audit report dictionary (D1-D8 data).
        conformance_data: Optional evaluated conformance truth matrix.

    Returns:
        Full standalone HTML5 string with inline CSS styling and print support.
    """
    if isinstance(inspection, dict):
        insp_id = int(inspection.get("id", 1))
        filename = str(inspection.get("filename", "specimen_image.png"))
        orig_url = str(inspection.get("original_url", ""))
        res_url = str(inspection.get("result_url", ""))
        ts_val = inspection.get("timestamp", datetime.datetime.now())
        status = str(inspection.get("status", "passed")).lower()
        max_sev = str(inspection.get("max_severity", "none")).lower()
        inf_ms = float(inspection.get("inference_time_ms", 0.0))
        tot_ms = float(inspection.get("total_time_ms", 0.0))
        defects_list = inspection.get("defects", [])
        px_scale = inspection.get("px_per_mm")
        is_cal = bool(inspection.get("is_calibrated", False))
    else:
        insp_id = int(getattr(inspection, "id", 1))
        filename = str(getattr(inspection, "filename", "specimen_image.png"))
        orig_name = Path(getattr(inspection, "original_path", "")).name
        res_name = Path(getattr(inspection, "result_path", "")).name
        orig_url = f"/data/uploads/{orig_name}"
        res_url = f"/data/results/{res_name}"
        ts_val = getattr(inspection, "timestamp", datetime.datetime.now())
        status = str(getattr(inspection, "status", "passed")).lower()
        max_sev = str(getattr(inspection, "max_severity", "none")).lower()
        inf_ms = float(getattr(inspection, "inference_time_ms", 0.0))
        tot_ms = float(getattr(inspection, "total_time_ms", 0.0))
        defects_list = getattr(inspection, "defects", [])
        px_scale = getattr(inspection, "px_per_mm", None)
        is_cal = bool(px_scale is not None and px_scale > 0)

    # Format timestamp
    if isinstance(ts_val, datetime.datetime):
        ts_str = ts_val.strftime("%Y-%m-%d %H:%M:%S UTC")
    else:
        ts_str = str(ts_val)

    # 1. Evaluate Conformance Matrix (True / False)
    if not conformance_data:
        conformance_data = ConformanceService.evaluate_inspection(inspection, defects=defects_list)

    lot_authorized = bool(conformance_data.get("lot_release_authorized", False))
    conf_items = conformance_data.get("items", [])
    passed_count = conformance_data.get("passed_criteria", 0)
    total_criteria = conformance_data.get("total_criteria", len(conf_items))

    # 2. Extract 8D audit details if present
    audit = report_dict or {}
    if not audit and not isinstance(inspection, dict) and getattr(inspection, "audit_report_json", None):
        try:
            audit = json.loads(inspection.audit_report_json)
        except Exception:
            audit = {}

    d3_containment = audit.get("d3_containment_action") or audit.get("containment_action") or "Standard automated continuous inspection routing."
    d4_root_cause = audit.get("d4_root_cause_analysis") or audit.get("probable_root_cause") or "Surface integrity conforms to baseline nominal topography."
    d5_corrective = audit.get("d5_corrective_actions") or audit.get("corrective_action") or "Maintain nominal mill roll lubrication and clean high-pressure spray headers."
    standard_ref = audit.get("standard_reference") or "ISO 9001:2015 / IATF 16949 Cl. 8.5.1"

    # Status Pill Colors
    if lot_authorized:
        badge_bg = "#064e3b"
        badge_border = "#059669"
        badge_text = "#34d399"
        verdict_title = "RELEASE AUTHORIZED [TRUE]"
        verdict_sub = "All manufacturing quality gates and tolerance thresholds satisfied."
    else:
        badge_bg = "#4c0519"
        badge_border = "#e11d48"
        badge_text = "#fda4af"
        verdict_title = "QUARANTINE / REJECT [FALSE]"
        verdict_sub = "Critical flaw or tolerance violation detected. Material lot quarantined."

    # Build Conformance Rows HTML
    conformance_rows_html = []
    for item in conf_items:
        passed = bool(item.get("passed", False))
        crit_id = html.escape(str(item.get("criterion_id", "")))
        name = html.escape(str(item.get("name", "")))
        desc = html.escape(str(item.get("description", "")))
        rec = html.escape(str(item.get("recommendation", "")))

        if passed:
            pill_html = '<span class="status-pill pill-pass">TRUE (PASS) ✓</span>'
            row_class = "row-pass"
        else:
            pill_html = '<span class="status-pill pill-fail">FALSE (VIOLATION) ✗</span>'
            row_class = "row-fail"

        conformance_rows_html.append(f"""
        <tr class="{row_class}">
            <td class="font-mono font-bold">{crit_id}</td>
            <td>
                <div class="font-bold text-white">{name}</div>
                <div class="text-xs text-muted">{desc}</div>
            </td>
            <td class="text-center">{pill_html}</td>
            <td class="text-xs">{rec}</td>
        </tr>
        """)

    # Build Defect Table Rows HTML
    defect_rows_html = []
    for idx, d in enumerate(defects_list, 1):
        if isinstance(d, dict):
            cname = html.escape(str(d.get("class_name", "defect")).upper())
            sev = str(d.get("severity", "minor")).lower()
            conf = float(d.get("confidence", 0.0))
            area_ratio = float(d.get("area_ratio", 0.0) or 0.0)
            bbox = d.get("bbox", (0, 0, 0, 0))
            metro = d.get("metrology") or {}
            mech = html.escape(str(d.get("failure_mechanism") or "Surface texture disruption"))
        else:
            cname = html.escape(str(getattr(d, "class_name", "defect")).upper())
            sev = str(getattr(d, "severity", "minor")).lower()
            conf = float(getattr(d, "confidence", 0.0))
            area_ratio = float(getattr(d, "area_ratio", 0.0) or 0.0)
            bbox = (
                getattr(d, "bbox_x1", 0),
                getattr(d, "bbox_y1", 0),
                getattr(d, "bbox_x2", 0),
                getattr(d, "bbox_y2", 0),
            )
            metro_raw = getattr(d, "metrology", None)
            if metro_raw is None:
                metro_json = getattr(d, "metrology_json", None)
                metro = json.loads(metro_json) if metro_json else {}
            elif isinstance(metro_raw, str):
                metro = json.loads(metro_raw)
            else:
                metro = metro_raw or {}
            mech = html.escape(str(getattr(d, "failure_mechanism", "Surface texture disruption") or "Surface texture disruption"))

        sev_class = f"sev-{sev}"
        if metro.get("is_calibrated") and metro.get("surface_area_mm2") is not None:
            dim_str = f"Area: {metro.get('surface_area_mm2')} mm² | L: {metro.get('length_mm')} mm | W: {metro.get('width_mm')} mm"
        else:
            area_px = metro.get("area_px") or metro.get("pixel_area") or round((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 1)
            dim_str = f"Area: {area_px} px | Ratio: {metro.get('aspect_ratio', 1.0)}:1"

        defect_rows_html.append(f"""
        <tr>
            <td class="font-mono text-center">#{idx}</td>
            <td class="font-bold">{cname}</td>
            <td class="text-center"><span class="sev-badge {sev_class}">{sev.upper()}</span></td>
            <td class="font-mono text-center">{(conf * 100):.1f}%</td>
            <td class="font-mono text-center">{(area_ratio * 100):.3f}%</td>
            <td class="text-xs font-mono">{dim_str}</td>
            <td class="text-xs text-muted">{mech}</td>
        </tr>
        """)

    if not defect_rows_html:
        defect_rows_html.append("""
        <tr>
            <td colspan="7" class="text-center text-muted" style="padding: 24px;">
                ✓ ZERO DEFECTS DETECTED — Substrate conforms 100% to surface quality specifications.
            </td>
        </tr>
        """)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quality Certificate — VG-INSP-{insp_id:06d}</title>
    <style>
        :root {{
            --bg-dark: #0a0e17;
            --card-bg: #111827;
            --border: #1f2937;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --primary: #38bdf8;
            --emerald: #10b981;
            --rose: #f43f5e;
            --amber: #f59e0b;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-main);
            line-height: 1.5;
            padding: 24px;
        }}
        .container {{
            max-width: 1040px;
            margin: 0 auto;
        }}
        /* Action Bar (Top Floating on Web, Hidden on Print) */
        .action-bar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: #1e293b;
            padding: 12px 20px;
            border-radius: 12px;
            border: 1px solid #334155;
            margin-bottom: 24px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        }}
        .action-btns {{
            display: flex;
            gap: 12px;
        }}
        .btn {{
            cursor: pointer;
            border: none;
            border-radius: 8px;
            padding: 8px 16px;
            font-size: 13px;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            text-decoration: none;
            transition: all 0.15s ease;
        }}
        .btn-primary {{
            background: #0284c7;
            color: white;
        }}
        .btn-primary:hover {{
            background: #0369a1;
        }}
        .btn-secondary {{
            background: #334155;
            color: #e2e8f0;
        }}
        .btn-secondary:hover {{
            background: #475569;
        }}
        /* Certificate Document Sheet */
        .certificate-sheet {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
        }}
        .cert-header {{
            background: linear-gradient(135deg, #0b1528 0%, #172554 100%);
            border-bottom: 3px solid var(--primary);
            padding: 24px 32px;
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }}
        .cert-title h1 {{
            font-size: 20px;
            font-weight: 800;
            letter-spacing: -0.5px;
            color: white;
            text-transform: uppercase;
        }}
        .cert-title p {{
            font-size: 12px;
            color: #93c5fd;
            margin-top: 4px;
        }}
        .cert-meta {{
            text-align: right;
            font-family: monospace;
            font-size: 12px;
        }}
        .cert-meta .ref-id {{
            color: var(--primary);
            font-size: 14px;
            font-weight: bold;
        }}
        .cert-body {{
            padding: 32px;
        }}
        /* Verdict Box */
        .verdict-box {{
            background: {badge_bg};
            border: 1px solid {badge_border};
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 28px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .verdict-title {{
            font-size: 16px;
            font-weight: 800;
            color: {badge_text};
            font-family: monospace;
        }}
        .verdict-sub {{
            font-size: 12px;
            color: {badge_text};
            opacity: 0.9;
            margin-top: 2px;
        }}
        .section-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 14px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--primary);
            margin: 28px 0 14px 0;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border);
        }}
        /* Tables */
        .table-wrap {{
            overflow-x: auto;
            border: 1px solid var(--border);
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        th {{
            background: #0f172a;
            color: #cbd5e1;
            text-align: left;
            padding: 10px 14px;
            font-weight: 600;
            border-bottom: 1px solid var(--border);
        }}
        td {{
            padding: 12px 14px;
            border-bottom: 1px solid #1e293b;
        }}
        tr:last-child td {{
            border-bottom: none;
        }}
        /* Conformance Pills */
        .status-pill {{
            display: inline-block;
            font-family: monospace;
            font-weight: bold;
            font-size: 11px;
            padding: 4px 10px;
            border-radius: 6px;
            white-space: nowrap;
        }}
        .pill-pass {{
            background: #064e3b;
            color: #34d399;
            border: 1px solid #059669;
        }}
        .pill-fail {{
            background: #4c0519;
            color: #fda4af;
            border: 1px solid #e11d48;
        }}
        .row-pass {{
            background: rgba(16, 185, 129, 0.03);
        }}
        .row-fail {{
            background: rgba(244, 63, 94, 0.06);
        }}
        /* Severity Badges */
        .sev-badge {{
            display: inline-block;
            font-family: monospace;
            font-size: 10px;
            font-weight: bold;
            padding: 2px 8px;
            border-radius: 4px;
        }}
        .sev-critical {{ background: #881337; color: #fecdd3; border: 1px solid #e11d48; }}
        .sev-major {{ background: #78350f; color: #fef3c7; border: 1px solid #f59e0b; }}
        .sev-minor {{ background: #0c4a6e; color: #e0f2fe; border: 1px solid #0284c7; }}
        .sev-none {{ background: #064e3b; color: #d1fae5; border: 1px solid #10b981; }}

        /* Image Evidence Grid */
        .evidence-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 24px;
        }}
        .evidence-card {{
            background: #090d16;
            border: 1px solid var(--border);
            border-radius: 10px;
            overflow: hidden;
        }}
        .evidence-label {{
            padding: 8px 12px;
            font-size: 11px;
            font-weight: 700;
            font-family: monospace;
            background: #111827;
            border-bottom: 1px solid var(--border);
            color: #cbd5e1;
        }}
        .evidence-img {{
            width: 100%;
            height: 240px;
            object-fit: contain;
            display: block;
            background: #000;
        }}

        /* 8D Section Grid */
        .eight-d-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
        }}
        .eight-d-card {{
            background: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 12px 14px;
        }}
        .eight-d-step {{
            font-family: monospace;
            font-size: 10px;
            font-weight: bold;
            color: #38bdf8;
            margin-bottom: 4px;
        }}
        .eight-d-content {{
            font-size: 12px;
            color: #e2e8f0;
        }}

        /* Metadata Details */
        .meta-strip {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            background: #0f172a;
            padding: 12px 16px;
            border-radius: 8px;
            border: 1px solid var(--border);
            font-size: 12px;
            margin-bottom: 20px;
        }}
        .meta-item .lbl {{
            color: var(--text-muted);
            font-size: 11px;
        }}
        .meta-item .val {{
            font-weight: 600;
            color: white;
            font-family: monospace;
        }}

        /* Footer */
        .cert-footer {{
            border-top: 1px solid var(--border);
            padding: 20px 32px;
            font-size: 11px;
            color: var(--text-muted);
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #0b1120;
        }}

        /* Print Media Styles */
        @media print {{
            body {{
                background: white !important;
                color: black !important;
                padding: 0 !important;
            }}
            .action-bar {{
                display: none !important;
            }}
            .container {{
                max-width: 100% !important;
            }}
            .certificate-sheet {{
                border: none !important;
                box-shadow: none !important;
                border-radius: 0 !important;
                background: white !important;
                color: black !important;
            }}
            .cert-header {{
                background: #f1f5f9 !important;
                color: black !important;
                border-bottom: 2px solid #0284c7 !important;
            }}
            .cert-title h1 {{ color: #0f172a !important; }}
            .cert-title p {{ color: #475569 !important; }}
            .cert-meta {{ color: #334155 !important; }}
            .verdict-box {{
                background: #f8fafc !important;
                border: 2px solid {badge_border} !important;
            }}
            th {{
                background: #e2e8f0 !important;
                color: #0f172a !important;
            }}
            td {{
                border-bottom: 1px solid #cbd5e1 !important;
                color: #0f172a !important;
            }}
            .text-white {{ color: #0f172a !important; }}
            .text-muted {{ color: #475569 !important; }}
            .evidence-card {{
                border: 1px solid #cbd5e1 !important;
            }}
            .evidence-img {{
                height: 190px !important;
            }}
            .eight-d-card {{
                background: #f8fafc !important;
                border: 1px solid #e2e8f0 !important;
            }}
            .eight-d-content {{ color: #0f172a !important; }}
            .meta-strip {{
                background: #f8fafc !important;
                border: 1px solid #e2e8f0 !important;
            }}
            .meta-item .val {{ color: #0f172a !important; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Action Bar (Browser View only) -->
        <div class="action-bar">
            <div>
                <span style="font-weight: bold; color: white; font-size: 14px;">ZeroFlaw AI-QA Certificate Viewer</span>
                <span style="color: #94a3b8; font-size: 12px; margin-left: 8px;">Inspection #{insp_id}</span>
            </div>
            <div class="action-btns">
                <button onclick="window.print()" class="btn btn-primary" title="Print this certificate or save as clean PDF">
                    🖨️ Print / Save as PDF
                </button>
                <a href="/api/detect/{insp_id}/8d-report/pdf?inline=true" target="_blank" class="btn btn-secondary" title="View official vector PDF in browser tab">
                    📄 Official Vector PDF
                </a>
                <button onclick="window.close()" class="btn btn-secondary">
                    ✕ Close
                </button>
            </div>
        </div>

        <!-- Official Quality Certificate Sheet -->
        <div class="certificate-sheet">
            <!-- Header -->
            <div class="cert-header">
                <div class="cert-title">
                    <h1>ZeroFlaw AI-QA — Quality Assurance Certificate</h1>
                    <p>Automated Optical Surface Inspection & Root-Cause Conformance Report</p>
                </div>
                <div class="cert-meta">
                    <div class="ref-id">VG-INSP-{insp_id:06d}</div>
                    <div style="color: #94a3b8; margin-top: 2px;">{ts_str}</div>
                    <div style="color: #38bdf8; font-size: 11px; margin-top: 4px;">{html.escape(standard_ref)}</div>
                </div>
            </div>

            <div class="cert-body">
                <!-- Lot Disposition Verdict Banner -->
                <div class="verdict-box">
                    <div>
                        <div class="verdict-title">{verdict_title}</div>
                        <div class="verdict-sub">{verdict_sub}</div>
                    </div>
                    <div style="text-align: right; font-family: monospace;">
                        <span style="font-size: 18px; font-weight: 800; color: {badge_text};">{passed_count} / {total_criteria}</span>
                        <div style="font-size: 11px; color: {badge_text}; opacity: 0.85;">CRITERIA CONFORMING</div>
                    </div>
                </div>

                <!-- Specimen Metadata Strip -->
                <div class="meta-strip">
                    <div class="meta-item">
                        <div class="lbl">Specimen File:</div>
                        <div class="val">{html.escape(filename)}</div>
                    </div>
                    <div class="meta-item">
                        <div class="lbl">Optical Scale:</div>
                        <div class="val">{f"{px_scale:.2f} px/mm" if is_cal else "Uncalibrated (px)"}</div>
                    </div>
                    <div class="meta-item">
                        <div class="lbl">Inference Latency:</div>
                        <div class="val">{inf_ms:.1f} ms</div>
                    </div>
                    <div class="meta-item">
                        <div class="lbl">Defects Flagged:</div>
                        <div class="val">{len(defects_list)} items ({max_sev.upper()})</div>
                    </div>
                </div>

                <!-- Section 1: Conformance Verification Matrix (TRUE / FALSE) -->
                <div class="section-header">
                    <span>1. Quality Conformance Verification Matrix (Truth Evaluation)</span>
                </div>
                <div class="table-wrap">
                    <table>
                        <thead>
                            <tr>
                                <th style="width: 10%;">Code</th>
                                <th style="width: 45%;">Quality Standard Criterion & Requirement</th>
                                <th style="width: 20%; text-align: center;">Conformance State</th>
                                <th style="width: 25%;">Quality Action Recommendation</th>
                            </tr>
                        </thead>
                        <tbody>
                            {"".join(conformance_rows_html)}
                        </tbody>
                    </table>
                </div>

                <!-- Section 2: Visual Inspection Evidence Side-by-Side -->
                <div class="section-header">
                    <span>2. Optical Inspection Evidence (Raw vs AI Bounding Overlay)</span>
                </div>
                <div class="evidence-grid">
                    <div class="evidence-card">
                        <div class="evidence-label">A. RAW SPECIMEN CAPTURE</div>
                        <img src="{orig_url}" alt="Raw Specimen Capture" class="evidence-img">
                    </div>
                    <div class="evidence-card">
                        <div class="evidence-label">B. AI DEFECT ANNOTATION & METROLOGY BOUNDS</div>
                        <img src="{res_url}" alt="AI Defect Overlay" class="evidence-img">
                    </div>
                </div>

                <!-- Section 3: Flaw Metrology & Telemetry Breakdown -->
                <div class="section-header">
                    <span>3. Detailed Defect Telemetry & Dimensional Metrology</span>
                </div>
                <div class="table-wrap">
                    <table>
                        <thead>
                            <tr>
                                <th style="width: 5%; text-align: center;">#</th>
                                <th style="width: 18%;">Defect Type</th>
                                <th style="width: 12%; text-align: center;">Severity</th>
                                <th style="width: 10%; text-align: center;">Confidence</th>
                                <th style="width: 12%; text-align: center;">Area Density</th>
                                <th style="width: 25%;">Dimensions (Metrology)</th>
                                <th style="width: 18%;">Failure Mechanism</th>
                            </tr>
                        </thead>
                        <tbody>
                            {"".join(defect_rows_html)}
                        </tbody>
                    </table>
                </div>

                <!-- Section 4: 8D Root-Cause Corrective Action Summary -->
                <div class="section-header">
                    <span>4. 8D Root-Cause & Corrective Action Summary</span>
                </div>
                <div class="eight-d-grid">
                    <div class="eight-d-card">
                        <div class="eight-d-step">D3 — IMMEDIATE CONTAINMENT ACTION</div>
                        <div class="eight-d-content">{html.escape(str(d3_containment))}</div>
                    </div>
                    <div class="eight-d-card">
                        <div class="eight-d-step">D4 — ROOT-CAUSE MECHANISM (5-WHY / VLM)</div>
                        <div class="eight-d-content">{html.escape(str(d4_root_cause))}</div>
                    </div>
                    <div class="eight-d-card">
                        <div class="eight-d-step">D5 — PERMANENT CORRECTIVE ACTION (PCA)</div>
                        <div class="eight-d-content">{html.escape(str(d5_corrective))}</div>
                    </div>
                    <div class="eight-d-card">
                        <div class="eight-d-step">D7 — SYSTEMIC PREVENTION & STANDARDIZATION</div>
                        <div class="eight-d-content">Update Process FMEA (PFMEA-VG-084); recalibrate optical line sensors at 120-hour intervals.</div>
                    </div>
                </div>
            </div>

            <!-- Footer -->
            <div class="cert-footer">
                <div>
                    <strong>CONFIDENTIAL QUALITY DOCUMENT</strong> — Conforms to ISO 9001:2015 & IATF 16949 Section 8.5.1
                </div>
                <div>
                    Generated by ZeroFlaw AI-QA Autonomous System • {ts_str}
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

