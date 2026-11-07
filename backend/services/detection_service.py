"""Inference orchestration and database persistence service for VisionGuard QA.

Ties ML DefectDetector with SQLAlchemy persistence and AlertService.
"""

from __future__ import annotations

import datetime
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database import DefectRecord, Inspection
from backend.services.alert_service import AlertService
from ml.arbitrator import ProductionArbitrator
from ml.detector import DefectDetector

logger = logging.getLogger(__name__)

# Global singleton arbitrator & detector instances
_detector_instance: Optional[DefectDetector] = None
_arbitrator_instance: Optional[ProductionArbitrator] = None


def get_detector() -> DefectDetector:
    """Retrieve or initialize the DefectDetector singleton."""
    global _detector_instance
    if _detector_instance is None:
        logger.info(
            "Initializing DefectDetector (model=%s, conf=%.2f, device=%s, tiling=%s, slice_size=%d)",
            settings.MODEL_NAME,
            settings.CONFIDENCE_THRESHOLD,
            settings.DEVICE,
            settings.ENABLE_TILING,
            settings.SLICE_SIZE,
        )
        _detector_instance = DefectDetector(
            model_name=settings.MODEL_NAME,
            confidence_threshold=settings.CONFIDENCE_THRESHOLD,
            device=settings.DEVICE,
            enable_tiling=settings.ENABLE_TILING,
            slice_size=settings.SLICE_SIZE,
            tiling_min_dim=settings.TILING_MIN_DIM,
        )
    return _detector_instance


def get_arbitrator() -> ProductionArbitrator:
    """Retrieve or initialize the ProductionArbitrator singleton."""
    global _arbitrator_instance
    if _arbitrator_instance is None:
        detector = get_detector()
        _arbitrator_instance = ProductionArbitrator(edge_detector=detector)
    return _arbitrator_instance


class DetectionService:
    """Orchestrates image inspection, DB persistence, and alert trigger dispatch."""

    @staticmethod
    def inspect_and_save(
        db: Session,
        image_data: Union[Path, np.ndarray, bytes],
        filename: str,
        source: str = "upload",
        notes: Optional[str] = None,
        deep_verify: bool = False,
    ) -> Dict[str, Any]:
        """Perform end-to-end defect detection, record results to DB, and trigger alerts.

        Args:
            db: Active database session.
            image_data: Path, numpy image array, or raw bytes.
            filename: Original or synthetic filename.
            source: Source of image ('upload', 'sample', 'camera').
            notes: Optional human inspection notes.
            deep_verify: Whether to run Tier 2 VLM / Anomaly escalation (default False).

        Returns:
            Dictionary containing inspection metrics, defect list, URLs, and status.
        """
        arbitrator = get_arbitrator()
        
        # 1. Run Two-Tier Production Arbitration pipeline
        arb_res = arbitrator.arbitrate(image_data, filename=filename, deep_verify=deep_verify)
        
        detections: List[Dict[str, Any]] = [
            {
                "bbox": d.bbox,
                "confidence": d.confidence,
                "class_id": d.class_id,
                "class_name": d.class_name,
                "severity": d.severity,
                "area_ratio": d.area_ratio,
                "description": d.description,
                "tier": d.tier,
                "vlm_verified": d.vlm_verified,
                "failure_mechanism": d.failure_mechanism,
                "polygon": d.polygon,
                "metrology": d.metrology,
                "source": getattr(d, "source", "yolo"),
                "definition": getattr(d, "definition", None),
                "cause": getattr(d, "cause", None),
                "action": getattr(d, "action", None),
                "visual_look": getattr(d, "visual_look", None),
                "risk": getattr(d, "risk", None),
            }
            for d in arb_res.detections
        ]
        summary: Dict[str, Any] = arb_res.summary
        annotated_image: np.ndarray = arb_res.annotated_image
        inference_time_ms: float = arb_res.tier1_latency_ms
        total_time_ms: float = arb_res.total_latency_ms

        # 2. Generate unique filenames for persistence
        timestamp_slug = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        safe_name = Path(filename).stem
        
        orig_filename = f"{safe_name}_{timestamp_slug}_{unique_id}_orig.jpg"
        result_filename = f"{safe_name}_{timestamp_slug}_{unique_id}_result.jpg"
        
        orig_path = settings.UPLOADS_DIR / orig_filename
        result_path = settings.RESULTS_DIR / result_filename

        # Save annotated result image
        cv2.imwrite(str(result_path), annotated_image)

        # Save or copy original image
        if isinstance(image_data, np.ndarray):
            cv2.imwrite(str(orig_path), image_data)
        elif isinstance(image_data, bytes):
            with open(orig_path, "wb") as f:
                f.write(image_data)
        elif isinstance(image_data, Path):
            import shutil
            shutil.copyfile(image_data, orig_path)

        # 3. Determine inspection overall severity and status
        critical_count = arb_res.critical_count
        major_count = arb_res.major_count
        minor_count = arb_res.minor_count
        max_severity = arb_res.max_severity
        status = arb_res.verdict.lower()

        # 4. Save Inspection record to Database
        inspection = Inspection(
            timestamp=datetime.datetime.utcnow(),
            source=source,
            filename=filename,
            original_path=str(orig_path),
            result_path=str(result_path),
            total_defects=len(detections),
            critical_count=critical_count,
            major_count=major_count,
            minor_count=minor_count,
            max_severity=max_severity,
            status=status,
            inference_time_ms=inference_time_ms,
            total_time_ms=total_time_ms,
            notes=notes,
            arbitration_tier=arb_res.arbitration_tier,
            arbitration_status=arb_res.arbitration_status,
            domain_trust=getattr(arb_res, "domain_trust", "HIGH"),
            domain_warning=getattr(arb_res, "domain_warning", None),
        )
        db.add(inspection)
        db.commit()
        db.refresh(inspection)

        # 5. Save individual DefectRecords
        defect_entities: List[DefectRecord] = []
        for det in detections:
            bbox = det["bbox"]
            defect_rec = DefectRecord(
                inspection_id=inspection.id,
                class_id=det["class_id"],
                class_name=det["class_name"],
                confidence=det["confidence"],
                severity=det["severity"],
                bbox_x1=bbox[0],
                bbox_y1=bbox[1],
                bbox_x2=bbox[2],
                bbox_y2=bbox[3],
                area_ratio=det.get("area_ratio", 0.0),
                description=det.get("description", ""),
                polygon_json=json.dumps(det.get("polygon", [])) if det.get("polygon") else None,
                metrology_json=json.dumps(det.get("metrology", {})) if det.get("metrology") else None,
                tier=det.get("tier", "tier1_edge"),
                vlm_verified=det.get("vlm_verified", False),
                failure_mechanism=det.get("failure_mechanism"),
                source=det.get("source", "yolo"),
            )
            db.add(defect_rec)
            defect_entities.append(defect_rec)

        db.commit()

        # 6. Evaluate and route alerts
        created_alerts = AlertService.process_inspection_alerts(db, inspection, detections)

        # 7. Construct URLs for static image access
        original_url = f"/data/uploads/{orig_filename}"
        result_url = f"/data/results/{result_filename}"

        return {
            "id": inspection.id,
            "timestamp": inspection.timestamp,
            "source": inspection.source,
            "filename": inspection.filename,
            "original_url": original_url,
            "result_url": result_url,
            "total_defects": inspection.total_defects,
            "critical_count": inspection.critical_count,
            "major_count": inspection.major_count,
            "minor_count": inspection.minor_count,
            "max_severity": inspection.max_severity,
            "status": inspection.status,
            "inference_time_ms": inspection.inference_time_ms,
            "total_time_ms": inspection.total_time_ms,
            "notes": inspection.notes,
            "arbitration_tier": inspection.arbitration_tier,
            "arbitration_status": inspection.arbitration_status,
            "domain_trust": getattr(inspection, "domain_trust", "HIGH"),
            "domain_warning": getattr(inspection, "domain_warning", None),
            "defects": detections,
            "alerts_created": len(created_alerts),
        }

