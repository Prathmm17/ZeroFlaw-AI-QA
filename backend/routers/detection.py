"""Detection API routes for image uploads, sample runs, and inspection history."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database import DefectRecord, Inspection, get_db
from backend.models import (
    CalibrationRequest,
    CalibrationResponse,
    DefectKnowledgeItem,
    DefectSchema,
    InspectionListResponse,
    InspectionResponse,
)
from backend.services.detection_service import DetectionService, get_detector
from backend.services.knowledge_service import get_all_defect_knowledge, get_defect_knowledge

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/detect", tags=["Defect Detection"])


def _to_inspection_response(inspection: Inspection) -> InspectionResponse:
    """Format an SQLAlchemy Inspection model into an InspectionResponse Pydantic schema."""
    orig_name = Path(inspection.original_path).name
    res_name = Path(inspection.result_path).name

    defects_schema: List[DefectSchema] = []
    for d in inspection.defects:
        k_info = get_defect_knowledge(d.class_name) or {}
        defects_schema.append(
            DefectSchema(
                bbox=(d.bbox_x1, d.bbox_y1, d.bbox_x2, d.bbox_y2),
                confidence=d.confidence,
                class_id=d.class_id,
                class_name=d.class_name,
                severity=d.severity,
                area_ratio=d.area_ratio,
                description=d.description,
                polygon=json.loads(d.polygon_json) if getattr(d, "polygon_json", None) else None,
                metrology=json.loads(d.metrology_json) if getattr(d, "metrology_json", None) else None,
                tier=getattr(d, "tier", "tier1_edge") or "tier1_edge",
                vlm_verified=bool(getattr(d, "vlm_verified", False)),
                failure_mechanism=getattr(d, "failure_mechanism", None),
                source=getattr(d, "source", "yolo") or "yolo",
                definition=k_info.get("definition"),
                cause=k_info.get("cause"),
                action=k_info.get("action"),
                visual_look=k_info.get("visual_look"),
                risk=k_info.get("risk"),
            )
        )

    detector = get_detector()
    is_cal = bool(detector.px_per_mm is not None and detector.px_per_mm > 0)
    px_scale = detector.px_per_mm
    for d_s in defects_schema:
        if d_s.metrology and "is_calibrated" in d_s.metrology:
            is_cal = bool(d_s.metrology["is_calibrated"])
            px_scale = d_s.metrology.get("px_per_mm", px_scale)
            break
    cal_label = "calibrated" if is_cal else "uncalibrated (px)"

    from backend.services.conformance_service import ConformanceService
    conformance_data = ConformanceService.evaluate_inspection(inspection, defects=defects_schema)

    return InspectionResponse(
        id=inspection.id,
        timestamp=inspection.timestamp,
        source=inspection.source,
        filename=inspection.filename,
        original_url=f"/data/uploads/{orig_name}",
        result_url=f"/data/results/{res_name}",
        total_defects=inspection.total_defects,
        critical_count=inspection.critical_count,
        major_count=inspection.major_count,
        minor_count=inspection.minor_count,
        max_severity=inspection.max_severity,
        status=inspection.status,
        inference_time_ms=inspection.inference_time_ms,
        total_time_ms=inspection.total_time_ms,
        notes=inspection.notes,
        audit_report=json.loads(inspection.audit_report_json) if getattr(inspection, "audit_report_json", None) else None,
        conformance_report=conformance_data,
        arbitration_tier=getattr(inspection, "arbitration_tier", "tier1_edge") or "tier1_edge",
        arbitration_status=getattr(inspection, "arbitration_status", "EDGE_CONFIRMED") or "EDGE_CONFIRMED",
        domain_trust=getattr(inspection, "domain_trust", "HIGH") or "HIGH",
        domain_warning=getattr(inspection, "domain_warning", None),
        px_per_mm=px_scale,
        is_calibrated=is_cal,
        calibration_label=cal_label,
        defects=defects_schema,
    )


@router.get("/glossary", response_model=Dict[str, DefectKnowledgeItem])
def get_defect_glossary() -> Dict[str, DefectKnowledgeItem]:
    """Retrieve full engineering defect knowledge glossary across all classes."""
    data = get_all_defect_knowledge()
    return {k: DefectKnowledgeItem(**v) for k, v in data.items()}


@router.post("/upload", response_model=InspectionResponse, status_code=status.HTTP_201_CREATED)
async def upload_and_detect(
    file: UploadFile = File(...),
    notes: Optional[str] = None,
    deep_verify: bool = Query(False, description="Enable Tier 2 VLM / Anomaly deep verification"),
    db: Session = Depends(get_db),
) -> InspectionResponse:
    """Upload an industrial camera image, run defect detection, save results and route alerts."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename in upload.")

    allowed_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format '{file_ext}'. Allowed: {', '.join(allowed_exts)}",
        )

    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        res_dict = DetectionService.inspect_and_save(
            db=db,
            image_data=contents,
            filename=file.filename,
            source="upload",
            notes=notes,
            deep_verify=deep_verify,
        )

        inspection = db.query(Inspection).filter(Inspection.id == res_dict["id"]).first()
        if not inspection:
            raise HTTPException(status_code=500, detail="Failed to retrieve inspection record.")

        return _to_inspection_response(inspection)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error processing uploaded image")
        raise HTTPException(
            status_code=500, detail=f"Inference pipeline failed: {str(exc)}"
        ) from exc


@router.get("/samples", response_model=List[str])
def list_sample_images() -> List[str]:
    """List sample defect images available in data/samples/ for fast demo testing."""
    sample_files = []
    if settings.SAMPLES_DIR.is_dir():
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for f in settings.SAMPLES_DIR.glob(ext):
                sample_files.append(f.name)
    return sorted(sample_files)


@router.post("/sample/{sample_name}", response_model=InspectionResponse)
def detect_sample(
    sample_name: str,
    deep_verify: bool = Query(False, description="Enable Tier 2 VLM / Anomaly deep verification"),
    db: Session = Depends(get_db),
) -> InspectionResponse:
    """Run defect detection on a pre-generated sample image."""
    sample_path = settings.SAMPLES_DIR / sample_name
    if not sample_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Sample '{sample_name}' not found in {settings.SAMPLES_DIR}",
        )

    try:
        res_dict = DetectionService.inspect_and_save(
            db=db,
            image_data=sample_path,
            filename=sample_name,
            source="sample",
            notes="Processed from demo sample catalog",
            deep_verify=deep_verify,
        )

        inspection = db.query(Inspection).filter(Inspection.id == res_dict["id"]).first()
        if not inspection:
            raise HTTPException(status_code=500, detail="Failed to retrieve inspection.")

        return _to_inspection_response(inspection)
    except Exception as exc:
        logger.exception("Failed to inspect sample %s", sample_name)
        raise HTTPException(
            status_code=500, detail=f"Inspection failed on sample: {exc}"
        ) from exc


@router.get("/history", response_model=InspectionListResponse)
def get_inspection_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status_filter: Optional[str] = Query(None, alias="status"),
    severity_filter: Optional[str] = Query(None, alias="severity"),
    db: Session = Depends(get_db),
) -> InspectionListResponse:
    """Retrieve paginated inspection history with optional status and severity filters."""
    query = db.query(Inspection)

    if status_filter and isinstance(status_filter, str):
        query = query.filter(Inspection.status == status_filter.lower())
    if severity_filter and isinstance(severity_filter, str):
        query = query.filter(Inspection.max_severity == severity_filter.lower())

    total = query.count()
    offset = (page - 1) * page_size
    inspections = (
        query.order_by(Inspection.timestamp.desc())
        .offset(offset)
        .limit(page_size)
        .all()
    )

    items = [_to_inspection_response(insp) for insp in inspections]
    pages = (total + page_size - 1) // page_size if total > 0 else 1

    return InspectionListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        pages=pages,
    )


@router.get("/{inspection_id}", response_model=InspectionResponse)
def get_inspection_detail(
    inspection_id: int,
    db: Session = Depends(get_db),
) -> InspectionResponse:
    """Retrieve full details of a specific inspection by ID."""
    inspection = db.query(Inspection).filter(Inspection.id == inspection_id).first()
    if not inspection:
        raise HTTPException(status_code=404, detail=f"Inspection #{inspection_id} not found.")
    return _to_inspection_response(inspection)


@router.post("/{inspection_id}/audit")
def audit_inspection_root_cause(
    inspection_id: int,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Execute Root-Cause Quality Engineering Audit (8D Report) for an inspection."""
    inspection = db.query(Inspection).filter(Inspection.id == inspection_id).first()
    if not inspection:
        raise HTTPException(status_code=404, detail=f"Inspection #{inspection_id} not found.")

    try:
        from backend.services.vlm_service import QualityEngineeringAuditor
        auditor = QualityEngineeringAuditor()

        # Load original image safely
        image = None
        if inspection.original_path and os.path.exists(inspection.original_path):
            try:
                image = cv2.imread(inspection.original_path)
            except Exception as read_err:
                logger.warning("Could not read original path %s: %s", inspection.original_path, read_err)

        if image is None:
            image = np.full((640, 640, 3), 128, dtype=np.uint8)

        # Format detections list safely
        detections = []
        for d in inspection.defects:
            poly = []
            if getattr(d, "polygon_json", None):
                try:
                    poly = json.loads(d.polygon_json) if isinstance(d.polygon_json, str) else d.polygon_json
                except Exception:
                    poly = []

            metro = {}
            if getattr(d, "metrology_json", None):
                try:
                    metro = json.loads(d.metrology_json) if isinstance(d.metrology_json, str) else d.metrology_json
                except Exception:
                    metro = {}

            detections.append({
                "class_name": d.class_name,
                "confidence": d.confidence,
                "severity": d.severity,
                "bbox": (d.bbox_x1, d.bbox_y1, d.bbox_x2, d.bbox_y2),
                "polygon": poly,
                "metrology": metro,
                "source": getattr(d, "source", "yolo") or "yolo",
            })

        report = auditor.audit_inspection(
            inspection_id=inspection.id,
            image=image,
            detections=detections,
        )
        report_dict = asdict(report)
        inspection.audit_report_json = json.dumps(report_dict)
        db.commit()
        db.refresh(inspection)
        return report_dict
    except Exception as exc:
        logger.exception("Failed to execute 8D audit for inspection #%s: %s", inspection_id, exc)
        raise HTTPException(
            status_code=500,
            detail=f"8D Audit computation failed: {exc}",
        ) from exc


@router.get("/{inspection_id}/8d-report")
def get_8d_report(
    inspection_id: int,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Retrieve or generate 8D Discrepancy Quality Report for an inspection."""
    inspection = db.query(Inspection).filter(Inspection.id == inspection_id).first()
    if not inspection:
        raise HTTPException(status_code=404, detail=f"Inspection #{inspection_id} not found.")

    if inspection.audit_report_json:
        return json.loads(inspection.audit_report_json)

    # If not yet audited, trigger audit now
    return audit_inspection_root_cause(inspection_id=inspection_id, db=db)


@router.get("/{inspection_id}/report/view")
def view_inspection_certificate_html(
    inspection_id: int,
    db: Session = Depends(get_db),
):
    """Render full standalone HTML5 quality certificate directly in the browser."""
    from fastapi.responses import HTMLResponse
    from backend.services.conformance_service import ConformanceService
    from backend.services.report_html_service import generate_inspection_certificate_html

    inspection = db.query(Inspection).filter(Inspection.id == inspection_id).first()
    if not inspection:
        raise HTTPException(status_code=404, detail=f"Inspection #{inspection_id} not found.")

    report_dict = None
    if inspection.audit_report_json:
        try:
            report_dict = json.loads(inspection.audit_report_json)
        except Exception:
            report_dict = None

    conformance_data = ConformanceService.evaluate_inspection(inspection)
    html_content = generate_inspection_certificate_html(
        inspection=inspection,
        report_dict=report_dict,
        conformance_data=conformance_data,
    )

    return HTMLResponse(content=html_content, status_code=status.HTTP_200_OK)


@router.get("/{inspection_id}/8d-report/pdf")
def export_8d_report_pdf(
    inspection_id: int,
    inline: bool = Query(False, description="Open PDF inline in browser instead of downloading"),
    db: Session = Depends(get_db),
):
    """Generate formal ISO 9001 / IATF 16949 8D Discrepancy PDF.

    If inline=True, opens directly in the browser tab; otherwise triggers file download.
    """
    from fastapi.responses import Response
    from backend.services.pdf_service import generate_8d_report_pdf

    report_dict = get_8d_report(inspection_id=inspection_id, db=db)
    pdf_bytes = generate_8d_report_pdf(report_dict)

    disp_type = "inline" if (inline is True) else "attachment"
    filename = f"8D_Report_VG_INSP_{inspection_id:06d}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"{disp_type}; filename={filename}",
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )


@router.post("/{inspection_id}/vlm-grounding")
def ground_defects_vlm(
    inspection_id: int,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Execute Gemini Flash zero-shot visual grounding on an inspection image."""
    inspection = db.query(Inspection).filter(Inspection.id == inspection_id).first()
    if not inspection:
        raise HTTPException(status_code=404, detail=f"Inspection #{inspection_id} not found.")

    image = cv2.imread(inspection.original_path)
    if image is None:
        raise HTTPException(status_code=400, detail="Original image not found on disk.")

    from backend.services.gemini_grounding import GeminiVisualGrounder
    grounder = GeminiVisualGrounder()
    grounded = grounder.ground_defects(image, material_hint=inspection.filename)

    h, w = image.shape[:2]
    formatted = []
    for g in grounded:
        ymin, xmin, ymax, xmax = g.box_2d
        x1, y1, x2, y2 = xmin * w, ymin * h, xmax * w, ymax * h
        formatted.append({
            "box_2d": [ymin, xmin, ymax, xmax],
            "pixel_box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "label": g.label,
            "confidence": g.confidence,
            "severity": g.severity,
            "failure_mechanism": g.failure_mechanism,
            "source": getattr(g, "source", "anomaly_cv"),
        })

    return {
        "inspection_id": inspection.id,
        "is_vlm_active": grounder.is_api_active(),
        "grounded_count": len(formatted),
        "defects": formatted,
    }


@router.get("/calibration", response_model=CalibrationResponse)
def get_camera_calibration() -> CalibrationResponse:
    """Get current camera optical calibration scale."""
    detector = get_detector()
    scale = detector.px_per_mm
    is_cal = bool(scale is not None and scale > 0)
    label = "calibrated" if is_cal else "uncalibrated (px)"
    return CalibrationResponse(
        px_per_mm=scale,
        calibrated=is_cal,
        calibration_label=label,
        message=f"Current scale: {scale} px/mm ({label})" if is_cal else "Scale is uncalibrated (px). Measurements displayed in pixels only.",
    )


@router.post("/calibration", response_model=CalibrationResponse)
def set_camera_calibration(req: CalibrationRequest) -> CalibrationResponse:
    """Set optical calibration scale in px/mm, or compute from a reference measurement."""
    detector = get_detector()
    computed_scale: Optional[float] = None

    if req.px_per_mm is not None and req.px_per_mm > 0:
        computed_scale = round(float(req.px_per_mm), 4)
    elif req.line_length_px is not None and req.known_length_mm is not None:
        if req.known_length_mm <= 0:
            raise HTTPException(status_code=400, detail="known_length_mm must be greater than zero.")
        computed_scale = round(float(req.line_length_px) / float(req.known_length_mm), 4)
    elif req.px_per_mm is None and req.line_length_px is None:
        # Reset to uncalibrated
        computed_scale = None

    detector.set_calibration(computed_scale)
    settings.PX_PER_MM = computed_scale
    is_cal = bool(computed_scale is not None and computed_scale > 0)
    label = "calibrated" if is_cal else "uncalibrated (px)"

    return CalibrationResponse(
        px_per_mm=computed_scale,
        calibrated=is_cal,
        calibration_label=label,
        message=(
            f"Successfully calibrated optical scale to {computed_scale:.2f} px/mm"
            if is_cal
            else "Camera scale reset to uncalibrated (px mode)."
        ),
    )


@router.post("/auto-calibrate", response_model=CalibrationResponse)
async def auto_calibrate_from_fiducial(
    file: UploadFile = File(...),
    reference_size_mm: float = Query(20.0, description="Known physical dimension of reference target in mm"),
    target_type: str = Query("auto", description="Target type: 'aruco', 'circle', or 'auto'"),
) -> CalibrationResponse:
    """Automatically detect ArUco or circular optical benchmark to calibrate optical scale."""
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Failed to decode image.")

    detector = get_detector()
    scale, drift_warn = detector._segmenter.detect_fiducial_scale(
        image,
        marker_size_mm=reference_size_mm,
        target_type=target_type,
    )

    if scale is None:
        raise HTTPException(
            status_code=422,
            detail="No ArUco marker or circular optical benchmark detected in image.",
        )

    detector.set_calibration(scale)
    settings.PX_PER_MM = scale

    msg = f"Auto-calibration successful: {scale:.2f} px/mm (Target: {reference_size_mm}mm)"
    if drift_warn:
        msg += f" | {drift_warn}"

    return CalibrationResponse(
        px_per_mm=scale,
        calibrated=True,
        calibration_label="calibrated",
        message=msg,
    )


@router.post("/{inspection_id}/recalculate-metrology", response_model=InspectionResponse)
def recalculate_metrology(
    inspection_id: int,
    db: Session = Depends(get_db),
) -> InspectionResponse:
    """Recalculate defect metrology for an inspection using current calibration scale."""
    inspection = db.query(Inspection).filter(Inspection.id == inspection_id).first()
    if not inspection:
        raise HTTPException(status_code=404, detail=f"Inspection #{inspection_id} not found.")

    image = cv2.imread(inspection.original_path)
    if image is None:
        raise HTTPException(status_code=400, detail="Original image not found on disk.")

    detector = get_detector()
    img_h, img_w = image.shape[:2]

    for d in inspection.defects:
        norm_bbox = (
            max(0.0, float(d.bbox_x1 / img_w)),
            max(0.0, float(d.bbox_y1 / img_h)),
            min(1.0, float(d.bbox_x2 / img_w)),
            min(1.0, float(d.bbox_y2 / img_h)),
        )
        metrology = detector._segmenter.segment_roi(image, norm_bbox)
        d.metrology_json = json.dumps(asdict(metrology))
        if metrology.polygon and len(metrology.polygon) >= 3:
            d.polygon_json = json.dumps(metrology.polygon)

    db.commit()
    db.refresh(inspection)
    return _to_inspection_response(inspection)


