"""Pydantic schemas for VisionGuard QA API requests and responses.

Provides validation, serialization, and Swagger documentation typing.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Defect Schemas
# ---------------------------------------------------------------------------


class DefectSchema(BaseModel):
    """Schema for a single detected manufacturing defect."""

    bbox: tuple[float, float, float, float] = Field(
        ..., description="Bounding box [x1, y1, x2, y2] in pixels"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    class_id: int = Field(..., description="Defect class index")
    class_name: str = Field(..., description="Defect type name (crack, scratch, etc.)")
    severity: str = Field(..., description="Severity level: critical, major, minor")
    area_ratio: float = Field(default=0.0, description="Ratio of defect area to total image area")
    description: str = Field(default="", description="Human-readable description")
    polygon: Optional[List[List[float]]] = Field(
        default=None, description="Normalized polygon vertices [[x, y], ...]"
    )
    metrology: Optional[Dict[str, Any]] = Field(
        default=None, description="Metrology metrics (surface_area_mm2, length_mm, aspect_ratio)"
    )
    tier: str = Field(default="tier1_edge", description="Detection tier: tier1_edge or tier2_vlm")
    vlm_verified: bool = Field(default=False, description="Whether verified by Gemini Flash VLM")
    failure_mechanism: Optional[str] = Field(default=None, description="Physics-of-failure mechanism")
    source: str = Field(default="yolo", description="Detection source: yolo, anomaly_cv, or vlm")
    definition: Optional[str] = Field(default=None, description="Plain engineering definition")
    cause: Optional[str] = Field(default=None, description="Typical manufacturing cause")
    action: Optional[str] = Field(default=None, description="Recommended quality action")
    visual_look: Optional[str] = Field(default=None, description="Visual appearance description")
    risk: Optional[str] = Field(default=None, description="Failure risk and mechanical consequence")


class DefectSummarySchema(BaseModel):
    """Summary metrics of detected defects in an inspection."""

    total_count: int = 0
    by_severity: Dict[str, int] = Field(default_factory=dict)
    by_class: Dict[str, int] = Field(default_factory=dict)
    avg_confidence: float = 0.0
    max_confidence: float = 0.0
    has_critical: bool = False


class DefectKnowledgeItem(BaseModel):
    """Schema for defect engineering knowledge item in the glossary."""

    name: str
    display_name: str
    definition: str
    visual_look: str
    cause: str
    typical_cause: Optional[str] = None
    risk: str
    action: str
    recommended_action: Optional[str] = None
    default_severity: str
    calibrated_threshold: Optional[float] = None
    confidence_label: Optional[str] = None


class ConformanceCheckItem(BaseModel):
    """Schema for an individual boolean quality conformance criterion check."""

    criterion_id: str = Field(..., description="Unique criterion code (e.g. CRIT-01)")
    name: str = Field(..., description="Criterion name")
    description: str = Field(..., description="Technical engineering standard requirement")
    passed: bool = Field(..., description="Boolean conformance state (True = PASS, False = VIOLATION)")
    status_label: str = Field(..., description="Display badge text: TRUE (PASS) or FALSE (VIOLATION)")
    severity_level: str = Field(..., description="Severity category: pass, warning, or violation")
    recommendation: str = Field(default="", description="Corrective engineering recommendation")


class ConformanceReportSchema(BaseModel):
    """Deterministic boolean quality conformance verification summary."""

    lot_release_authorized: bool = Field(..., description="Final release approval (True = Approved, False = Quarantined)")
    surface_integrity_pass: bool = Field(..., description="True if free of critical and major surface cracks")
    critical_violation_free: bool = Field(..., description="True if 0 critical defects detected")
    area_tolerance_pass: bool = Field(..., description="True if total flaw area <= 2.0%")
    dimensional_tolerance_pass: bool = Field(..., description="True if metrology is within manufacturing tolerances")
    material_anomaly_clear: bool = Field(..., description="True if unclassified anomalies are absent")
    total_criteria: int = 0
    passed_criteria: int = 0
    failed_criteria: int = 0
    items: List[ConformanceCheckItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Inspection Schemas
# ---------------------------------------------------------------------------


class InspectionResponse(BaseModel):
    """Response schema for an inspection record."""

    id: int
    timestamp: datetime.datetime
    source: str
    filename: str
    original_url: str
    result_url: str
    total_defects: int
    critical_count: int
    major_count: int
    minor_count: int
    max_severity: str
    status: str
    inference_time_ms: float
    total_time_ms: float
    notes: Optional[str] = None
    audit_report: Optional[Dict[str, Any]] = None
    conformance_report: Optional[ConformanceReportSchema] = None
    arbitration_tier: str = "tier1_edge"
    arbitration_status: str = "EDGE_CONFIRMED"
    domain_trust: str = "HIGH"
    domain_warning: Optional[str] = None
    px_per_mm: Optional[float] = None
    is_calibrated: bool = False
    calibration_label: str = "uncalibrated (px)"
    defects: List[DefectSchema] = Field(default_factory=list)

    class Config:
        from_attributes = True


class CalibrationRequest(BaseModel):
    """Request schema for camera optical calibration."""

    px_per_mm: Optional[float] = Field(default=None, ge=0.0, description="Explicit scale in pixels per mm")
    line_length_px: Optional[float] = Field(default=None, gt=0.0, description="Drawn reference line length in pixels")
    known_length_mm: Optional[float] = Field(default=None, gt=0.0, description="Known physical reference length in mm")


class CalibrationResponse(BaseModel):
    """Response schema for optical camera calibration status."""

    px_per_mm: Optional[float] = None
    calibrated: bool = False
    calibration_label: str = "uncalibrated (px)"
    message: str = ""


class InspectionListResponse(BaseModel):
    """Paginated list of inspection records."""

    items: List[InspectionResponse]
    total: int
    page: int
    page_size: int
    pages: int


# ---------------------------------------------------------------------------
# Alert Schemas
# ---------------------------------------------------------------------------


class AlertResponse(BaseModel):
    """Response schema for a quality alert."""

    id: int
    inspection_id: int
    timestamp: datetime.datetime
    severity: str
    defect_type: str
    message: str
    action_required: str
    acknowledged: bool
    acknowledged_at: Optional[datetime.datetime] = None
    acknowledged_by: Optional[str] = None

    class Config:
        from_attributes = True


class AcknowledgeAlertRequest(BaseModel):
    """Payload to acknowledge an alert."""

    acknowledged_by: str = Field(default="Quality Inspector", description="Name/ID of operator")
    notes: Optional[str] = Field(default=None, description="Resolution notes")


# ---------------------------------------------------------------------------
# Analytics Schemas
# ---------------------------------------------------------------------------


class AnalyticsOverviewResponse(BaseModel):
    """Executive quality analytics overview with SPC metrics."""

    total_inspections: int
    total_defects: int
    defect_free_inspections: int
    defect_rate_percentage: float
    critical_defects_count: int
    major_defects_count: int
    minor_defects_count: int
    unacknowledged_alerts: int
    avg_inference_time_ms: float
    defect_types_distribution: Dict[str, int]
    severity_distribution: Dict[str, int]
    status_distribution: Dict[str, int]
    cp: float = 1.45
    cpk: float = 1.38
    pareto_cumulative: Dict[str, float] = Field(default_factory=dict)
    process_capability_status: str = "CAPABLE_PROCESS"


class TrendPoint(BaseModel):
    """Data point for defect and inspection timeline trend."""

    date: str
    inspections: int
    defects: int
    critical: int
    major: int
    minor: int


class AnalyticsTrendsResponse(BaseModel):
    """Historical inspection quality trend series."""

    trends: List[TrendPoint]


# ---------------------------------------------------------------------------
# Live Detection Schemas (WebSocket / Stream)
# ---------------------------------------------------------------------------


class LiveDetectionResult(BaseModel):
    """Real-time detection result for camera stream frame."""

    timestamp: str
    inference_time_ms: float
    fps: float
    total_defects: int
    max_severity: str
    defects: List[DefectSchema]
    has_critical: bool
    frame_annotated_base64: Optional[str] = None

