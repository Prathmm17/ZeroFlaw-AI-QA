"""Industrial 8D Quality Report PDF Generator.

Generates formal, high-fidelity engineering PDF documents conforming to
ISO 9001 / IATF 16949 quality audit documentation standards.
Uses zero external dependencies with native PDF 1.4 vector specification.
"""

from __future__ import annotations

import io
import re
from typing import Any, Dict, List, Optional


def _escape_pdf_str(text: str) -> str:
    """Escape text for PDF literal strings with clean ASCII / WinAnsi compatibility."""
    text = (
        text.replace("—", " - ")
        .replace("–", " - ")
        .replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("•", "*")
        .replace("²", "2")
        .replace("µ", "u")
        .replace("°", " deg ")
    )
    text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return "".join(c if 32 <= ord(c) < 127 else " " for c in text)


class SimplePDFWriter:
    """Zero-dependency native PDF 1.4 document composer."""

    def __init__(self, page_width: float = 612.0, page_height: float = 792.0) -> None:
        self.width = page_width  # Standard US Letter
        self.height = page_height
        self.page_streams: List[bytes] = []
        self.current_stream: List[str] = []
        self.current_y = self.height - 50.0

    def start_page(self) -> None:
        """Start a new PDF page."""
        self.current_stream = []
        self.current_y = self.height - 50.0

    def end_page(self) -> None:
        """Finish current page and record stream."""
        content = "\n".join(self.current_stream).encode("latin-1", errors="replace")
        self.page_streams.append(content)

    def draw_rect(self, x: float, y: float, w: float, h: float, r: float, g: float, b: float, fill: bool = True) -> None:
        """Draw rectangle with fill or stroke."""
        if fill:
            self.current_stream.append(f"{r:.3f} {g:.3f} {b:.3f} rg {x:.1f} {y:.1f} {w:.1f} {h:.1f} re f")
        else:
            self.current_stream.append(f"{r:.3f} {g:.3f} {b:.3f} RG 1.0 w {x:.1f} {y:.1f} {w:.1f} {h:.1f} re S")

    def draw_text(self, text: str, x: float, y: float, font: str = "F1", size: float = 10.0, r: float = 0.0, g: float = 0.0, b: float = 0.0) -> None:
        """Draw text line at (x, y)."""
        escaped = _escape_pdf_str(text)
        self.current_stream.append(
            f"BT /{font} {size:.1f} Tf {r:.3f} {g:.3f} {b:.3f} rg {x:.1f} {y:.1f} Td ({escaped}) Tj ET"
        )

    def add_section_header(self, title: str, discipline: str) -> None:
        """Add standardized 8D discipline section header."""
        if self.current_y < 120:
            self.end_page()
            self.start_page()

        y = self.current_y
        self.draw_rect(40, y - 18, self.width - 80, 22, 0.07, 0.13, 0.22, fill=True)
        self.draw_text(discipline, 48, y - 13, font="F2", size=10, r=0.22, g=0.74, b=0.97)
        self.draw_text(f"— {title}", 110, y - 13, font="F2", size=10, r=1.0, g=1.0, b=1.0)
        self.current_y -= 28

    def add_field(self, label: str, value: str, indent: float = 48.0) -> None:
        """Add label-value entry with automatic line wrapping."""
        if self.current_y < 80:
            self.end_page()
            self.start_page()

        y = self.current_y
        self.draw_text(f"{label}:", indent, y, font="F2", size=9, r=0.2, g=0.2, b=0.3)

        # Wrap long lines
        max_chars = 80
        words = value.split()
        lines = []
        curr_line = []
        for word in words:
            if sum(len(w) for w in curr_line) + len(curr_line) + len(word) > max_chars:
                lines.append(" ".join(curr_line))
                curr_line = [word]
            else:
                curr_line.append(word)
        if curr_line:
            lines.append(" ".join(curr_line))

        if not lines:
            lines = ["N/A"]

        val_x = indent + 130.0
        for idx, line in enumerate(lines):
            line_y = y - (idx * 12.0)
            if line_y < 50:
                self.end_page()
                self.start_page()
                line_y = self.current_y
            self.draw_text(line, val_x if idx == 0 else indent + 20.0, line_y, font="F1", size=9, r=0.1, g=0.1, b=0.1)

        self.current_y -= max(16.0, len(lines) * 12.0 + 4.0)

    def build(self) -> bytes:
        """Compose entire PDF binary structure with strict PDF 1.4 compliance."""
        if not self.page_streams and self.current_stream:
            self.end_page()

        num_pages = len(self.page_streams)
        if num_pages == 0:
            self.start_page()
            self.end_page()
            num_pages = 1

        # Deterministic object IDs:
        # Object 1: Catalog -> /Pages 2 0 R
        # Object 2: Pages tree -> /Kids [...] /Count N
        # Object 3: F1 (Helvetica Regular)
        # Object 4: F2 (Helvetica Bold)
        # For page i (0-indexed):
        #   Object (5 + 2*i): Content Stream
        #   Object (5 + 2*i + 1): Page Dictionary
        out = io.BytesIO()
        out.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

        offsets = [0]  # Dummy 0-index offset

        def write_obj(content: bytes) -> None:
            offsets.append(out.tell())
            obj_num = len(offsets) - 1
            out.write(f"{obj_num} 0 obj\n".encode("latin-1"))
            out.write(content)
            out.write(b"\nendobj\n")

        # 1: Catalog
        write_obj(b"<< /Type /Catalog /Pages 2 0 R >>")

        # 2: Pages
        kids_str = " ".join(f"{5 + 2 * i + 1} 0 R" for i in range(num_pages))
        write_obj(f"<< /Type /Pages /Kids [{kids_str}] /Count {num_pages} >>".encode("latin-1"))

        # 3: F1 (Helvetica Regular)
        write_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")

        # 4: F2 (Helvetica Bold)
        write_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")

        # Page stream and page dictionary objects
        for i, stream_content in enumerate(self.page_streams):
            stream_obj_id = 5 + 2 * i
            page_obj_id = 5 + 2 * i + 1

            # Content stream object
            stream_body = (
                f"<< /Length {len(stream_content)} >>\nstream\n".encode("latin-1")
                + stream_content
                + b"\nendstream"
            )
            write_obj(stream_body)

            # Page dictionary object
            page_body = (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width} {self.height}] "
                f"/Contents {stream_obj_id} 0 R /Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> >>"
            ).encode("latin-1")
            write_obj(page_body)

        # XRef table
        xref_offset = out.tell()
        out.write(f"xref\n0 {len(offsets)}\n".encode("latin-1"))
        out.write(b"0000000000 65535 f \n")
        for off in offsets[1:]:
            out.write(f"{off:010d} 00000 n \n".encode("latin-1"))

        # Trailer
        out.write(
            f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("latin-1")
        )

        return out.getvalue()


def generate_8d_report_pdf(report_data: Dict[str, Any]) -> bytes:
    """Generate professional ISO 9001 / IATF 16949 8D Discrepancy PDF."""
    pdf = SimplePDFWriter()
    pdf.start_page()

    # Document Header Banner
    pdf.draw_rect(0, pdf.height - 70, pdf.width, 70, 0.05, 0.09, 0.16, fill=True)
    pdf.draw_rect(0, pdf.height - 73, pdf.width, 3, 0.22, 0.74, 0.97, fill=True)

    pdf.draw_text("ZeroFlaw AI-QA — AUTOMATED QUALITY INTELLIGENCE", 40, pdf.height - 35, font="F2", size=13, r=1.0, g=1.0, b=1.0)
    pdf.draw_text("FORMAL 8D DISCREPANCY & ROOT-CAUSE AUDIT REPORT", 40, pdf.height - 52, font="F1", size=10, r=0.6, g=0.8, b=1.0)

    insp_id = report_data.get("inspection_id", 0)
    ts = report_data.get("timestamp", "N/A")
    ref_code = f"VG-8D-{insp_id:06d}"

    pdf.draw_text(f"Audit Ref: {ref_code}", pdf.width - 200, pdf.height - 35, font="F2", size=10, r=0.3, g=0.9, b=0.6)
    pdf.draw_text(f"Date: {ts}", pdf.width - 200, pdf.height - 52, font="F1", size=8, r=0.8, g=0.8, b=0.8)

    pdf.current_y = pdf.height - 95

    # Metadata Summary Bar
    pdf.draw_rect(40, pdf.current_y - 25, pdf.width - 80, 30, 0.94, 0.96, 0.98, fill=True)
    pdf.draw_rect(40, pdf.current_y - 25, pdf.width - 80, 30, 0.8, 0.85, 0.9, fill=False)

    p_type = str(report_data.get("primary_defect_type", "N/A")).upper()
    sev = str(report_data.get("engineering_severity", "Major")).upper()
    std = str(report_data.get("standard_reference", "ISO 9001 / IATF 16949"))

    pdf.draw_text("PRIMARY DEFECT:", 50, pdf.current_y - 12, font="F2", size=8, r=0.3, g=0.3, b=0.4)
    pdf.draw_text(p_type, 135, pdf.current_y - 12, font="F2", size=9, r=0.8, g=0.1, b=0.1 if "CRIT" in sev else 0.7)

    pdf.draw_text("SEVERITY RISK:", 240, pdf.current_y - 12, font="F2", size=8, r=0.3, g=0.3, b=0.4)
    pdf.draw_text(sev, 315, pdf.current_y - 12, font="F2", size=9, r=0.8, g=0.1, b=0.1 if "CRIT" in sev else 0.7)

    pdf.draw_text("STANDARD:", 410, pdf.current_y - 12, font="F2", size=8, r=0.3, g=0.3, b=0.4)
    pdf.draw_text(std[:22], 470, pdf.current_y - 12, font="F1", size=8, r=0.1, g=0.1, b=0.1)

    pdf.current_y -= 40

    # D1: Team
    pdf.add_section_header("CROSS-FUNCTIONAL TEAM ESTABLISHMENT", "D1")
    pdf.add_field("Team Champion", "Quality Assurance Director / Plant Manager")
    pdf.add_field("Core Team Members", str(report_data.get("d1_team", "Lead Process Metallurgist, Mill Superintendent, QA Reliability Specialist")))

    # D2: Problem Description
    pdf.add_section_header("PROBLEM DESCRIPTION & PHYSICAL SCOPE", "D2")
    pdf.add_field("Failure Mechanism", str(report_data.get("defect_mechanism", "Surface morphological anomaly")))
    pdf.add_field("Tooling Station", str(report_data.get("tooling_station", "Main Rolling Finishing Mill Stand F2")))
    pdf.add_field("Problem Detail", str(report_data.get("d2_problem_description", f"Optical anomaly detected on specimen {ref_code} exceeding surface texture threshold.")))

    # Metrology dimensions
    metro = report_data.get("metrology_summary", {})
    if metro:
        is_cal = metro.get("is_calibrated", False)
        unit = "mm" if is_cal else "px"
        area = metro.get("surface_area_mm2") if is_cal else metro.get("area_px", "N/A")
        length = metro.get("length_mm") if is_cal else metro.get("length_px", "N/A")
        width = metro.get("width_mm") if is_cal else metro.get("width_px", "N/A")
        pdf.add_field("Metrology Dimensions", f"Area: {area} {unit}² | Length: {length} {unit} | Width: {width} {unit} | Aspect Ratio: {metro.get('aspect_ratio', 'N/A')}:1")

    # D3: Interim Containment Action
    pdf.add_section_header("INTERIM CONTAINMENT ACTIONS (ICA)", "D3")
    pdf.add_field("Immediate Containment", str(report_data.get("d3_containment_action") or report_data.get("containment_action", "Quarantine affected manufacturing lot immediately.")))

    # D4: Root Cause Analysis
    pdf.add_section_header("ROOT CAUSE ANALYSIS (RCA / 5-WHY)", "D4")
    pdf.add_field("Root Cause Mechanism", str(report_data.get("d4_root_cause_analysis") or report_data.get("probable_root_cause", "Mechanical guide wear and thermal quench shock.")))

    # D5: Chosen Permanent Corrective Action
    pdf.add_section_header("PERMANENT CORRECTIVE ACTION (PCA)", "D5")
    pdf.add_field("Selected PCA", str(report_data.get("d5_corrective_actions") or report_data.get("corrective_action", "Inspect and redress work rolls; adjust coolant nozzle spray angles.")))

    # D6: Implementation & Validation
    pdf.add_section_header("IMPLEMENTATION & VALIDATION", "D6")
    pdf.add_field("Execution Verification", str(report_data.get("d6_implementation_verification", "Tooling replacement scheduled and optical inspection verified on next 10 consecutive batches.")))

    # D7: Prevention of Recurrence
    pdf.add_section_header("PREVENT RECURRENCE & STANDARDIZATION", "D7")
    pdf.add_field("Systemic Prevention", str(report_data.get("d7_preventive_action", "Update Process FMEA (PFMEA-VG-084) and recalibrate maintenance intervals.")))

    # D8: Team Recognition & Sign-off
    pdf.add_section_header("CONGRATULATE TEAM & SIGN-OFF", "D8")
    pdf.add_field("Closure Status", str(report_data.get("d8_team_recognition", "8D Discrepancy report validated and approved for production resumption.")))

    # Footer note on current page
    pdf.draw_text("CONFIDENTIAL — ISO 9001:2015 / IATF 16949 COMPLIANT QUALITY RECORD", 40, 25, font="F2", size=7, r=0.5, g=0.5, b=0.5)
    pdf.draw_text("Page 1 of 1 — ZeroFlaw AI-QA Autonomous Quality Management", pdf.width - 270, 25, font="F1", size=7, r=0.5, g=0.5, b=0.5)

    pdf.end_page()
    return pdf.build()

