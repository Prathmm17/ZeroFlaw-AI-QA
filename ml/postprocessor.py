"""Post-processing of YOLOv8 detection results for VisionGuard QA.

Extracts bounding boxes from raw ultralytics ``Results`` objects, filters
detections by confidence, assigns severity levels, and generates
human-readable summaries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ml.model_registry import DEFECT_CLASSES

logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Severity constants
# ---------------------------------------------------------------------------
SEVERITY_CRITICAL: str = "critical"
SEVERITY_MAJOR: str = "major"
SEVERITY_MINOR: str = "minor"

# Ordered from highest to lowest so we can compare numerically.
_SEVERITY_ORDER: dict[str, int] = {
    SEVERITY_CRITICAL: 2,
    SEVERITY_MAJOR: 1,
    SEVERITY_MINOR: 0,
}

# Thresholds used by the severity classifier.
_CRITICAL_CONFIDENCE_THRESHOLD: float = 0.80
_CRITICAL_AREA_RATIO_THRESHOLD: float = 0.05
_MAJOR_CONFIDENCE_THRESHOLD: float = 0.55


# ---------------------------------------------------------------------------
# Detection data class
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    """A single defect detection produced by the YOLOv8 pipeline.

    Attributes:
        bbox: Bounding-box coordinates ``(x1, y1, x2, y2)`` in pixels.
        confidence: Model confidence score in the range ``[0, 1]``.
        class_id: Integer class index emitted by the model.
        class_name: Human-readable class label (e.g. ``"crack"``).
        severity: One of ``"critical"``, ``"major"``, or ``"minor"``.
        area_ratio: Ratio of the defect bounding-box area to the full
            image area, in the range ``[0, 1]``.
        description: Auto-generated human-readable description of the
            defect (populated by
            :pymeth:`DetectionPostprocessor.generate_description`).
    """

    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    severity: str = SEVERITY_MINOR
    area_ratio: float = 0.0
    description: str = field(default="")
    polygon: list[list[float]] = field(default_factory=list)
    metrology: dict[str, Any] = field(default_factory=dict)
    source: str = field(default="yolo")  # "yolo" | "anomaly_cv" | "vlm"
    definition: str | None = None
    cause: str | None = None
    action: str | None = None
    visual_look: str | None = None
    risk: str | None = None


# ---------------------------------------------------------------------------
# Post-processor
# ---------------------------------------------------------------------------
class DetectionPostprocessor:
    """Converts raw YOLO predictions into structured ``Detection`` objects.

    The processor filters low-confidence detections, computes area ratios,
    classifies severity, and generates human-readable descriptions.

    Args:
        min_confidence: Minimum confidence threshold.  Detections below
            this value are discarded.
        defect_classes: Mapping of integer class IDs to human-readable
            names.  When *None* the default ``DEFECT_CLASSES`` from the
            model registry is used.
    """

    def __init__(
        self,
        min_confidence: float = 0.35,
        defect_classes: dict[int, str] | None = None,
        class_thresholds: dict[str, float] | None = None,
    ) -> None:
        self.min_confidence: float = min_confidence
        self.defect_classes: dict[int, str] = (
            dict(defect_classes) if defect_classes is not None else dict(DEFECT_CLASSES)
        )
        if class_thresholds is not None:
            self.class_thresholds: dict[str, float] = dict(class_thresholds)
        else:
            self.class_thresholds = self._load_class_thresholds()
        logger.info(
            "DetectionPostprocessor initialised — min_confidence=%.2f, "
            "defect_classes=%s, thresholds=%s",
            self.min_confidence,
            self.defect_classes,
            self.class_thresholds,
        )

    @staticmethod
    def _load_class_thresholds() -> dict[str, float]:
        """Load optimal per-class confidence thresholds from backend/class_thresholds.json."""
        from pathlib import Path
        import json
        thresh_file = Path(__file__).resolve().parent.parent / "backend" / "class_thresholds.json"
        if thresh_file.exists():
            try:
                with open(thresh_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("thresholds", {})
            except Exception:
                pass
        return {
            "crazing": 0.245,
            "inclusion": 0.136,
            "patches": 0.432,
            "pitted_surface": 0.412,
            "rolled_in_scale": 0.345,
            "scratches": 0.601,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        results: Any,
        image_shape: tuple[int, ...],
        letterbox_params: tuple[float, int, int] | None = None,
    ) -> list[Detection]:
        """Extract and post-process detections from a YOLO results object.

        Automatically inverts letterbox scaling and padding to return bounding boxes
        in the original unpadded image coordinate space.

        Args:
            results: A single ``ultralytics.engine.results.Results`` instance
                (i.e. one element of the list returned by ``model.predict()``).
            image_shape: Shape of the source image as ``(height, width)`` or
                ``(height, width, channels)``.  Only the first two dimensions
                are used when computing area ratios.
            letterbox_params: Optional tuple of (scale, pad_left, pad_top) applied
                during preprocessing. If None, automatically derived when model results
                are in letterbox space (e.g. 640x640) and image_shape differs.

        Returns:
            A sorted list of :class:`Detection` objects (highest confidence
            first), filtered by *min_confidence*.

        Raises:
            ValueError: If *image_shape* does not contain at least two
                dimensions.
        """
        if len(image_shape) < 2:
            raise ValueError(
                f"image_shape must have at least 2 dimensions, got {image_shape}"
            )

        image_height: int = int(image_shape[0])
        image_width: int = int(image_shape[1])
        image_area: float = float(image_height * image_width)

        if image_area <= 0:
            raise ValueError(
                f"Image area must be positive, got {image_height}x{image_width}"
            )

        detections: list[Detection] = []

        try:
            boxes = results.boxes
            if boxes is None or len(boxes) == 0:
                logger.debug("No detections in results object.")
                return detections

            # Determine letterbox inverse transformation parameters
            scale: float = 1.0
            pad_left: int = 0
            pad_top: int = 0
            apply_inverse_lb: bool = False

            if letterbox_params is not None:
                scale, pad_left, pad_top = letterbox_params
                apply_inverse_lb = True
            else:
                results_orig = getattr(results, "orig_shape", None)
                if results_orig == (640, 640) and (image_height, image_width) != (640, 640):
                    target_w, target_h = 640.0, 640.0
                    scale = min(target_w / image_width, target_h / image_height)
                    new_w = int(round(image_width * scale))
                    new_h = int(round(image_height * scale))
                    pad_left = int((target_w - new_w) // 2)
                    pad_top = int((target_h - new_h) // 2)
                    apply_inverse_lb = True

            for box in boxes:
                # Extract scalar values from tensors.
                confidence: float = float(box.conf[0])
                if confidence < self.min_confidence:
                    continue

                class_id: int = int(box.cls[0])
                x1_lb, y1_lb, x2_lb, y2_lb = (float(c) for c in box.xyxy[0])

                # Map from letterbox space back to original image space
                if apply_inverse_lb and scale > 0:
                    x1 = (x1_lb - pad_left) / scale
                    y1 = (y1_lb - pad_top) / scale
                    x2 = (x2_lb - pad_left) / scale
                    y2 = (y2_lb - pad_top) / scale
                else:
                    x1, y1, x2, y2 = x1_lb, y1_lb, x2_lb, y2_lb

                # Clip to original image boundaries
                x1, y1, x2, y2 = self.clip_bbox((x1, y1, x2, y2), image_shape)

                # Compute dimensions and area ratio.
                bw: float = max(0.0, x2 - x1)
                bh: float = max(0.0, y2 - y1)
                box_area: float = bw * bh
                area_ratio: float = box_area / image_area

                # Look up class name (fall back to generic label).
                class_name: str = self.defect_classes.get(
                    class_id, f"unknown_{class_id}"
                )

                # Rule 1: Drop any box above 35% of the image area
                if area_ratio > 0.35:
                    logger.warning(
                        "Dropping detection (%s, bbox=[%.1f, %.1f, %.1f, %.1f]): area ratio %.2f%% exceeds 35%% image area threshold",
                        class_name, x1, y1, x2, y2, area_ratio * 100.0,
                    )
                    continue

                # Rule 2: Drop aspect ratio above 15:1 for classes other than scratches
                aspect_ratio: float = max(bw, bh) / max(min(bw, bh), 1e-6)
                if aspect_ratio > 15.0 and class_name.lower() != "scratches":
                    logger.warning(
                        "Dropping detection (%s, bbox=[%.1f, %.1f, %.1f, %.1f]): aspect ratio %.2f:1 exceeds 15:1 limit for non-scratch class '%s'",
                        class_name, x1, y1, x2, y2, aspect_ratio, class_name,
                    )
                    continue

                severity: str = self.classify_severity(
                    confidence, area_ratio, class_name=class_name
                )

                detection = Detection(
                    bbox=(round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)),
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                    severity=severity,
                    area_ratio=area_ratio,
                    source="yolo",
                )
                try:
                    from backend.services.knowledge_service import get_defect_knowledge
                    k_info = get_defect_knowledge(class_name)
                    if k_info:
                        detection.definition = k_info.get("definition")
                        detection.cause = k_info.get("cause")
                        detection.action = k_info.get("action")
                        detection.visual_look = k_info.get("visual_look")
                        detection.risk = k_info.get("risk")
                except Exception:
                    pass
                detection.description = self.generate_description(detection)
                detections.append(detection)

        except AttributeError as exc:
            logger.error(
                "Results object does not have expected attributes: %s", exc
            )
            raise TypeError(
                "Expected an ultralytics Results object with a 'boxes' attribute."
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected error during post-processing: %s", exc)
            raise

        # Sort by confidence descending.
        detections.sort(key=lambda d: d.confidence, reverse=True)

        logger.info(
            "Post-processing complete — %d detection(s) above %.2f confidence.",
            len(detections),
            self.min_confidence,
        )
        return detections

    @staticmethod
    def clip_bbox(
        bbox: tuple[float, float, float, float],
        image_shape: tuple[int, ...],
    ) -> tuple[float, float, float, float]:
        """Clip bounding box coordinates (x1, y1, x2, y2) to image bounds."""
        h, w = image_shape[0], image_shape[1]
        x1, y1, x2, y2 = bbox
        return (
            max(0.0, min(float(x1), float(w))),
            max(0.0, min(float(y1), float(h))),
            max(0.0, min(float(x2), float(w))),
            max(0.0, min(float(y2), float(h))),
        )

    @staticmethod
    def apply_class_aware_nms(
        detections: list[Detection],
        iou_threshold: float = 0.50,
    ) -> list[Detection]:
        """Apply class-aware Non-Maximum Suppression across YOLO and anomaly detections.

        If two boxes overlap above IoU > iou_threshold:
        - Same defect class: keep the higher-confidence one.
        - YOLO box vs Anomaly box (or unclassified_anomaly): keep the higher-confidence one.

        Args:
            detections: List of detections to filter.
            iou_threshold: Intersection over Union threshold (default 0.50).

        Returns:
            Filtered list of detections sorted by confidence descending.
        """
        if not detections:
            return []

        # Sort descending by confidence
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept: list[Detection] = []

        for det in sorted_dets:
            x1, y1, x2, y2 = det.bbox
            area_det = max(0.0, (x2 - x1) * (y2 - y1))
            should_suppress = False

            for k in kept:
                # Same class OR cross-source overlap
                same_class = det.class_name.lower() == k.class_name.lower()
                is_cross_source = (
                    (det.source == "yolo" and k.source == "anomaly_cv")
                    or (det.source == "anomaly_cv" and k.source == "yolo")
                    or det.class_name.lower() == "unclassified_anomaly"
                    or k.class_name.lower() == "unclassified_anomaly"
                )

                if same_class or is_cross_source:
                    kx1, ky1, kx2, ky2 = k.bbox
                    area_k = max(0.0, (kx2 - kx1) * (ky2 - ky1))

                    inter_x1 = max(x1, kx1)
                    inter_y1 = max(y1, ky1)
                    inter_x2 = min(x2, kx2)
                    inter_y2 = min(y2, ky2)

                    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                        union_area = area_det + area_k - inter_area
                        iou = inter_area / max(union_area, 1e-6)
                        if iou >= iou_threshold:
                            should_suppress = True
                            logger.info(
                                "Class-aware NMS suppressed %s (%s, conf=%.2f) due to overlap (IoU=%.2f >= %.2f) with %s (%s, conf=%.2f)",
                                det.class_name, det.source, det.confidence,
                                iou, iou_threshold,
                                k.class_name, k.source, k.confidence,
                            )
                            break

            if not should_suppress:
                kept.append(det)

        return kept

    @staticmethod
    def apply_class_aware_soft_nms(
        detections: list[Detection],
        iou_threshold: float = 0.50,
        sigma: float = 0.5,
        min_score: float = 0.18,
    ) -> list[Detection]:
        """Apply Gaussian Soft-NMS with class awareness across detections.

        Instead of hard suppression, decays confidence of overlapping candidates of the same class:
        score <- score * exp(-IoU^2 / sigma).

        Args:
            detections: List of detections to filter.
            iou_threshold: Overlap threshold triggering decay.
            sigma: Variance parameter for Gaussian penalty curve (default 0.5).
            min_score: Minimum post-decay confidence score threshold.

        Returns:
            Filtered and decayed list of detections sorted by confidence descending.
        """
        if not detections:
            return []

        import numpy as np

        candidates = [
            Detection(
                bbox=d.bbox,
                confidence=d.confidence,
                class_id=d.class_id,
                class_name=d.class_name,
                severity=d.severity,
                area_ratio=d.area_ratio,
                description=d.description,
                polygon=list(d.polygon),
                metrology=dict(d.metrology),
                source=d.source,
                definition=d.definition,
                cause=d.cause,
                action=d.action,
                visual_look=d.visual_look,
                risk=d.risk,
            )
            for d in detections
        ]
        kept: list[Detection] = []

        while candidates:
            candidates.sort(key=lambda d: d.confidence, reverse=True)
            best = candidates.pop(0)
            kept.append(best)

            bx1, by1, bx2, by2 = best.bbox
            b_area = max(1.0, (bx2 - bx1) * (by2 - by1))
            b_cls = best.class_name.lower()

            remaining: list[Detection] = []
            for cand in candidates:
                c_cls = cand.class_name.lower()
                cx1, cy1, cx2, cy2 = cand.bbox
                c_area = max(1.0, (cx2 - cx1) * (cy2 - cy1))

                ix1 = max(bx1, cx1)
                iy1 = max(by1, cy1)
                ix2 = min(bx2, cx2)
                iy2 = min(by2, cy2)

                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    union = b_area + c_area - inter
                    iou = inter / max(union, 1e-6)

                    # Same class or cross-source decay
                    if b_cls == c_cls or "unclassified" in b_cls or "unclassified" in c_cls:
                        weight = float(np.exp(-(iou * iou) / sigma))
                        cand.confidence = float(cand.confidence * weight)

                if cand.confidence >= min_score:
                    remaining.append(cand)

            candidates = remaining

        kept.sort(key=lambda d: d.confidence, reverse=True)
        return kept

    def classify_severity(
        self,
        confidence: float,
        area_ratio: float,
        class_name: str = "",
        domain_trust: str = "HIGH",
        source: str = "yolo",
    ) -> str:
        """Determine the severity level of a detection with metallurgical risk weighting.

        Routing Rules (Task 4):
        1. CRITICAL only when confidence is above the class threshold AND area/class rules apply.
        2. Detections with trust LOW and source anomaly_cv are at most MAJOR.
        3. Major / Minor otherwise based on defect severity rules.

        Args:
            confidence: Detection confidence in [0, 1].
            area_ratio: Bounding-box area divided by image area in [0, 1].
            class_name: Human-readable defect label.
            domain_trust: "HIGH" or "LOW" domain trust certification.
            source: Detection source ('yolo', 'anomaly_cv', 'vlm').

        Returns:
            One of "critical", "major", or "minor".
        """
        c_name = class_name.lower()
        c_thresh = self.class_thresholds.get(c_name, 0.40)

        # Rule 1: Detections with trust LOW and source anomaly_cv are at most MAJOR
        if domain_trust.upper() == "LOW" and source == "anomaly_cv":
            if confidence >= 0.40 or area_ratio >= 0.01:
                return SEVERITY_MAJOR
            return SEVERITY_MINOR

        # Rule 2: CRITICAL only when confidence is above the class threshold AND area/class rules apply
        if confidence >= c_thresh:
            if c_name in ("crazing", "inclusion") and (confidence >= 0.50 or area_ratio >= 0.03):
                return SEVERITY_CRITICAL
            if confidence >= _CRITICAL_CONFIDENCE_THRESHOLD and area_ratio >= _CRITICAL_AREA_RATIO_THRESHOLD:
                return SEVERITY_CRITICAL
            if c_name == "unclassified_anomaly" and domain_trust.upper() != "LOW" and (confidence >= 0.75 or area_ratio >= 0.05):
                return SEVERITY_CRITICAL

        # Rule 3: Major criteria
        if confidence >= _MAJOR_CONFIDENCE_THRESHOLD or c_name in ("pitted_surface", "rolled_in_scale", "unclassified_anomaly"):
            return SEVERITY_MAJOR

        return SEVERITY_MINOR

    def generate_description(self, detection: Detection) -> str:
        """Generate a human-readable description for a detection.

        Example output::

            "Critical crack detected with 92% confidence"

        Args:
            detection: The :class:`Detection` to describe.

        Returns:
            A single-sentence description string.
        """
        confidence_pct: int = round(detection.confidence * 100)
        if detection.class_name == "unclassified_anomaly":
            return f"Unclassified surface anomaly with {confidence_pct}% confidence"

        severity_label: str = detection.severity.capitalize()
        return (
            f"{severity_label} {detection.class_name} detected "
            f"with {confidence_pct}% confidence"
        )

    def get_summary(self, detections: list[Detection]) -> dict[str, Any]:
        """Produce an aggregate summary of a list of detections.

        Args:
            detections: The detections to summarise (may be empty).

        Returns:
            A dictionary with the following keys:

            - **total_count** (*int*) — number of detections.
            - **by_severity** (*dict[str, int]*) — count per severity level.
            - **by_class** (*dict[str, int]*) — count per class name.
            - **avg_confidence** (*float*) — mean confidence (0.0 if empty).
            - **max_confidence** (*float*) — highest confidence (0.0 if empty).
            - **has_critical** (*bool*) — whether any critical defect exists.
        """
        by_severity: dict[str, int] = {
            SEVERITY_CRITICAL: 0,
            SEVERITY_MAJOR: 0,
            SEVERITY_MINOR: 0,
        }
        by_class: dict[str, int] = {}

        total_confidence: float = 0.0
        max_confidence: float = 0.0

        for det in detections:
            by_severity[det.severity] = by_severity.get(det.severity, 0) + 1
            by_class[det.class_name] = by_class.get(det.class_name, 0) + 1
            total_confidence += det.confidence
            if det.confidence > max_confidence:
                max_confidence = det.confidence

        total_count: int = len(detections)
        avg_confidence: float = (
            total_confidence / total_count if total_count > 0 else 0.0
        )

        summary: dict[str, Any] = {
            "total_count": total_count,
            "by_severity": by_severity,
            "by_class": by_class,
            "avg_confidence": round(avg_confidence, 4),
            "max_confidence": round(max_confidence, 4),
            "has_critical": by_severity[SEVERITY_CRITICAL] > 0,
        }

        logger.info("Detection summary: %s", summary)
        return summary

    def filter_by_severity(
        self,
        detections: list[Detection],
        min_severity: str = "minor",
    ) -> list[Detection]:
        """Filter detections that meet or exceed a minimum severity level.

        Severity ordering (lowest → highest): ``minor`` < ``major`` < ``critical``.

        Args:
            detections: List of detections to filter.
            min_severity: Minimum severity to keep.  Must be one of
                ``"minor"``, ``"major"``, or ``"critical"``.

        Returns:
            A new list containing only detections whose severity is at
            least *min_severity*.

        Raises:
            ValueError: If *min_severity* is not a recognised severity level.
        """
        min_severity_lower: str = min_severity.lower()
        if min_severity_lower not in _SEVERITY_ORDER:
            raise ValueError(
                f"Unknown severity level '{min_severity}'. "
                f"Expected one of {list(_SEVERITY_ORDER.keys())}."
            )

        threshold: int = _SEVERITY_ORDER[min_severity_lower]

        filtered: list[Detection] = [
            det
            for det in detections
            if _SEVERITY_ORDER.get(det.severity, -1) >= threshold
        ]

        logger.info(
            "Filtered %d → %d detections (min_severity='%s').",
            len(detections),
            len(filtered),
            min_severity,
        )
        return filtered
