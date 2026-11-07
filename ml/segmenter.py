"""Defect Contour Segmentation and Physical Metrology Module.

Extracts pixel-accurate polygonal masks, boundary contours, and physical
dimensions (area in mm², crack length, aspect ratio, solidity) from detected
defect regions of interest.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DefectMetrology:
    """Physical and geometric metrology measurements for a segmented defect.

    Attributes:
        pixel_area: Total number of defect pixels in the mask.
        aspect_ratio: Ratio of length to width (>= 1.0). High values indicate linear cracks/scratches.
        solidity: Ratio of contour area to its convex hull area (roughness indicator).
        polygon: List of normalized (x, y) coordinates relative to image [0.0, 1.0].
        area_px: Number of defect pixels (uncalibrated area).
        surface_area_mm2: Calibrated physical surface area in mm² (None when uncalibrated).
        length_mm: Maximum physical dimension (calibrated mm) along principal axis (None when uncalibrated).
        width_mm: Minimum physical dimension (calibrated mm) perpendicular to length (None when uncalibrated).
        length_px: Maximum dimension in pixels along principal axis.
        width_px: Minimum dimension in pixels perpendicular to length.
        is_calibrated: Boolean indicating whether optical mm calibration was applied.
        calibration_label: "calibrated" when calibrated; "uncalibrated (px)" when scale is None.
        px_per_mm: The optical scale in px/mm used, or None if uncalibrated.
    """

    pixel_area: int
    aspect_ratio: float
    solidity: float
    polygon: List[List[float]]
    area_px: Optional[int] = None
    surface_area_mm2: Optional[float] = None
    length_mm: Optional[float] = None
    width_mm: Optional[float] = None
    length_px: Optional[float] = None
    width_px: Optional[float] = None
    is_calibrated: bool = False
    calibration_label: str = "uncalibrated (px)"
    px_per_mm: Optional[float] = None
    tolerance_status: Optional[str] = None
    tolerance_note: Optional[str] = None


@dataclass
class ToleranceAssessment:
    """Engineering tolerance evaluation according to ISO/ASTM quality standards.

    Attributes:
        status: "PASS" | "WARNING" | "VIOLATION" | "UNCALIBRATED".
        defect_class: Label of the inspected defect class.
        recommended_severity: Calibrated severity level ("critical" | "major" | "minor").
        tolerance_limit_mm: Maximum allowable threshold for this class in millimeters.
        measured_dimension_mm: Actual physical measurement in millimeters.
        criterion: Human-readable engineering tolerance rule description.
        is_acceptable: Whether component passes tolerance for this defect.
        warning: Safety or calibration advisory if uncalibrated or out-of-tolerance.
    """

    status: str
    defect_class: str
    recommended_severity: str
    tolerance_limit_mm: Optional[float]
    measured_dimension_mm: Optional[float]
    criterion: str
    is_acceptable: bool
    warning: Optional[str] = None


class DefectSegmenter:
    """Pixel-level defect segmenter and industrial metrology analyzer.

    Extracts high-fidelity polygonal masks from defect bounding boxes
    and calculates industrial metallurgical metrology.

    Args:
        pixels_per_mm: Calibrated optical camera magnification (pixels per millimetre).
            Default is None (honest uncalibrated mode in pixels).
    """

    def __init__(self, pixels_per_mm: Optional[float] = None) -> None:
        self.pixels_per_mm: Optional[float] = None
        self.mm_per_pixel: Optional[float] = None
        self.mm2_per_pixel2: Optional[float] = None
        self.is_calibrated: bool = False
        self.calibration_label: str = "uncalibrated (px)"
        self.last_calibrated_scale: Optional[float] = None
        self.set_calibration(pixels_per_mm)

    def set_calibration(self, pixels_per_mm: Optional[float]) -> None:
        """Set or update optical calibration scale (pixels per millimeter).

        If None or <= 0: operates in uncalibrated mode (px only).
        If > 0: operates in calibrated mode (mm and mm²).
        """
        if pixels_per_mm is not None and float(pixels_per_mm) > 0:
            self.last_calibrated_scale = self.pixels_per_mm
            self.pixels_per_mm = float(pixels_per_mm)
            self.mm_per_pixel = 1.0 / self.pixels_per_mm
            self.mm2_per_pixel2 = self.mm_per_pixel ** 2
            self.is_calibrated = True
            self.calibration_label = "calibrated"
        else:
            self.pixels_per_mm = None
            self.mm_per_pixel = None
            self.mm2_per_pixel2 = None
            self.is_calibrated = False
            self.calibration_label = "uncalibrated (px)"

    def detect_fiducial_scale(
        self,
        image: np.ndarray,
        marker_size_mm: float = 20.0,
        target_type: str = "auto",
        drift_threshold_pct: float = 2.0,
    ) -> Tuple[Optional[float], Optional[str]]:
        """Automatically detect ArUco or circular optical benchmark to calibrate px_per_mm.

        Also performs drift compensation: if detected scale shifts >= 2% from previous calibration,
        recalibrates automatically and issues a vibration/drift alert.

        Args:
            image: Image containing conveyor fixture or reference artifact.
            marker_size_mm: Physical reference dimension in mm (side length for square, diameter for circle).
            target_type: "aruco", "circle", or "auto".
            drift_threshold_pct: Threshold percentage to trigger focal drift alert (default 2.0%).

        Returns:
            Tuple of (detected_px_per_mm, drift_warning_message_or_None).
        """
        if image is None or image.size == 0:
            return None, None

        img_h, img_w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        computed_scale: Optional[float] = None

        # 1. Try ArUco Marker Detection (Square Fiducial: perimeter = 4 * side)
        if target_type in ("aruco", "auto") and hasattr(cv2, "aruco"):
            for dict_id in (cv2.aruco.DICT_4X4_50, cv2.aruco.DICT_5X5_50, cv2.aruco.DICT_6X6_50):
                try:
                    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
                    if hasattr(cv2.aruco, "ArucoDetector"):
                        det = cv2.aruco.ArucoDetector(aruco_dict)
                        corners, ids, _ = det.detectMarkers(image)
                    else:
                        corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict)

                    if corners and len(corners) > 0:
                        pts = corners[0][0]
                        perim_px = float(cv2.arcLength(pts, True))
                        # Square marker perimeter = 4 * side
                        computed_scale = round(perim_px / (4.0 * marker_size_mm), 4)
                        logger.info(
                            "ArUco fiducial detected (id=%s): perim=%.1f px, scale=%.2f px/mm",
                            ids[0][0] if ids is not None else "?", perim_px, computed_scale,
                        )
                        break
                except Exception as exc:
                    logger.debug("ArUco search failed on dict %s: %s", dict_id, exc)

        # 2. Try Circular Optical Fiducial Benchmark (Circle: diameter = 2 * r)
        if computed_scale is None and target_type in ("circle", "auto"):
            # Threshold to find high-contrast circular calibration disc
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best_circ = 0.0
            best_circle_d = 0.0

            for c in cnts:
                area = cv2.contourArea(c)
                perim = cv2.arcLength(c, True)
                if perim > 0:
                    circ = 4.0 * np.pi * area / (perim * perim)
                    if circ > 0.85 and area > 400:
                        (cx, cy), r = cv2.minEnclosingCircle(c)
                        d_px = 2.0 * float(r)
                        if circ > best_circ:
                            best_circ = circ
                            best_circle_d = d_px

            if best_circle_d > 0:
                computed_scale = round(best_circle_d / float(marker_size_mm), 4)
                logger.info(
                    "Circular optical fiducial detected: diameter=%.1f px, scale=%.2f px/mm (circularity=%.3f)",
                    best_circle_d, computed_scale, best_circ,
                )

        if computed_scale is None:
            return None, None

        # Check for focal distance / vibration drift against previous scale
        drift_warning: Optional[str] = None
        if self.pixels_per_mm is not None and self.pixels_per_mm > 0:
            drift_pct = abs(computed_scale - self.pixels_per_mm) / self.pixels_per_mm * 100.0
            if drift_pct >= drift_threshold_pct:
                drift_warning = (
                    f"Focal distance drift detected ({drift_pct:.2f}% >= {drift_threshold_pct:.1f}%). "
                    f"Scale auto-compensated from {self.pixels_per_mm:.2f} to {computed_scale:.2f} px/mm."
                )
                logger.warning(drift_warning)

        self.set_calibration(computed_scale)
        return computed_scale, drift_warning

    def evaluate_engineering_tolerances(
        self,
        metrology: DefectMetrology,
        class_name: str,
    ) -> ToleranceAssessment:
        """Evaluate defect metrology against ISO/ASTM engineering quality standards.

        Args:
            metrology: DefectMetrology object containing calibrated mm measurements.
            class_name: Defect class label (e.g. 'crazing', 'pitted_surface', 'scratches').

        Returns:
            ToleranceAssessment dataclass detailing compliance and recommended severity.
        """
        c_name = class_name.lower()

        # Uncalibrated mode safety guard
        if not self.is_calibrated or metrology.length_mm is None:
            return ToleranceAssessment(
                status="UNCALIBRATED",
                defect_class=class_name,
                recommended_severity="major",
                tolerance_limit_mm=None,
                measured_dimension_mm=None,
                criterion="Optical millimeter calibration required before engineering tolerance gating.",
                is_acceptable=False,
                warning="Physical millimeter dimensions unavailable. Optical calibration required.",
            )

        l_mm = metrology.length_mm
        w_mm = metrology.width_mm or 0.0
        area_mm2 = metrology.surface_area_mm2 or 0.0

        # Fracture / Crack (crazing) standard: Max permissible crack length <= 3.0mm
        if c_name in ("crazing", "crack", "unclassified_anomaly"):
            limit = 3.0
            hairline_limit = 1.0
            if l_mm > limit:
                return ToleranceAssessment(
                    status="VIOLATION",
                    defect_class=class_name,
                    recommended_severity="critical",
                    tolerance_limit_mm=limit,
                    measured_dimension_mm=l_mm,
                    criterion=f"Critical fracture length ({l_mm:.2f}mm) exceeds ISO safety limit of {limit:.1f}mm",
                    is_acceptable=False,
                )
            elif l_mm > hairline_limit:
                return ToleranceAssessment(
                    status="WARNING",
                    defect_class=class_name,
                    recommended_severity="major",
                    tolerance_limit_mm=hairline_limit,
                    measured_dimension_mm=l_mm,
                    criterion=f"Crack length ({l_mm:.2f}mm) exceeds hairline tolerance ({hairline_limit:.1f}mm); repairable flaw",
                    is_acceptable=False,
                )
            else:
                return ToleranceAssessment(
                    status="PASS",
                    defect_class=class_name,
                    recommended_severity="minor",
                    tolerance_limit_mm=hairline_limit,
                    measured_dimension_mm=l_mm,
                    criterion=f"Micro-fissure length ({l_mm:.2f}mm) within acceptable surface tolerance (<= {hairline_limit:.1f}mm)",
                    is_acceptable=True,
                )

        # Porosity / Cavitation (pitted_surface): Max pore diameter <= 2.0mm, cluster area <= 4.0mm²
        elif c_name in ("pitted_surface", "porosity"):
            limit_d = 2.0
            limit_area = 4.0
            if w_mm > limit_d or area_mm2 > limit_area:
                return ToleranceAssessment(
                    status="VIOLATION",
                    defect_class=class_name,
                    recommended_severity="critical",
                    tolerance_limit_mm=limit_d,
                    measured_dimension_mm=w_mm,
                    criterion=f"Pore diameter ({w_mm:.2f}mm) or area ({area_mm2:.2f}mm²) exceeds structural threshold",
                    is_acceptable=False,
                )
            elif w_mm > 0.5 or area_mm2 > 0.8:
                return ToleranceAssessment(
                    status="WARNING",
                    defect_class=class_name,
                    recommended_severity="major",
                    tolerance_limit_mm=0.5,
                    measured_dimension_mm=w_mm,
                    criterion=f"Pore diameter ({w_mm:.2f}mm) exceeds precision machining tolerance (0.5mm)",
                    is_acceptable=False,
                )
            else:
                return ToleranceAssessment(
                    status="PASS",
                    defect_class=class_name,
                    recommended_severity="minor",
                    tolerance_limit_mm=0.5,
                    measured_dimension_mm=w_mm,
                    criterion=f"Micro-pit diameter ({w_mm:.2f}mm) within acceptable casting roughness allowance",
                    is_acceptable=True,
                )

        # Scratches: Max scratch length <= 15.0mm
        elif c_name in ("scratches", "scratch"):
            limit = 15.0
            if l_mm > limit:
                return ToleranceAssessment(
                    status="WARNING",
                    defect_class=class_name,
                    recommended_severity="major",
                    tolerance_limit_mm=limit,
                    measured_dimension_mm=l_mm,
                    criterion=f"Continuous scratch length ({l_mm:.2f}mm) exceeds cosmetic surface limit of {limit:.1f}mm",
                    is_acceptable=False,
                )
            else:
                return ToleranceAssessment(
                    status="PASS",
                    defect_class=class_name,
                    recommended_severity="minor",
                    tolerance_limit_mm=limit,
                    measured_dimension_mm=l_mm,
                    criterion=f"Scratch length ({l_mm:.2f}mm) within permissible light surface allowance",
                    is_acceptable=True,
                )

        # Inclusions & Patches: Max allowable patch area <= 5.0mm²
        else:
            limit_area = 5.0
            if area_mm2 > limit_area:
                return ToleranceAssessment(
                    status="VIOLATION",
                    defect_class=class_name,
                    recommended_severity="critical",
                    tolerance_limit_mm=limit_area,
                    measured_dimension_mm=area_mm2,
                    criterion=f"Impurity inclusion area ({area_mm2:.2f}mm²) exceeds metallurgical limit of {limit_area:.1f}mm²",
                    is_acceptable=False,
                )
            elif area_mm2 > 1.5:
                return ToleranceAssessment(
                    status="WARNING",
                    defect_class=class_name,
                    recommended_severity="major",
                    tolerance_limit_mm=1.5,
                    measured_dimension_mm=area_mm2,
                    criterion=f"Inclusion area ({area_mm2:.2f}mm²) requires surface polishing",
                    is_acceptable=False,
                )
            else:
                return ToleranceAssessment(
                    status="PASS",
                    defect_class=class_name,
                    recommended_severity="minor",
                    tolerance_limit_mm=1.5,
                    measured_dimension_mm=area_mm2,
                    criterion="Within standard metallurgical tolerance",
                    is_acceptable=True,
                )

    def segment_roi(
        self,
        full_image: np.ndarray,
        bbox: Tuple[float, float, float, float],
        class_name: str = "defect",
    ) -> DefectMetrology:
        """Extract polygon mask and physical metrology for a defect bounding box.

        Args:
            full_image: Full image array (H, W, C) in BGR or grayscale.
            bbox: Normalized bounding box (x1, y1, x2, y2) in range [0, 1].

        Returns:
            DefectMetrology dataclass with geometric and physical properties.
        """
        img_h, img_w = full_image.shape[:2]

        # Convert normalized bbox to absolute pixel coordinates
        x1 = max(0, int(round(bbox[0] * img_w)))
        y1 = max(0, int(round(bbox[1] * img_h)))
        x2 = min(img_w, int(round(bbox[2] * img_w)))
        y2 = min(img_h, int(round(bbox[3] * img_h)))

        roi_w = max(1, x2 - x1)
        roi_h = max(1, y2 - y1)

        # Fallback if box is too small for segmentation
        if roi_w < 3 or roi_h < 3:
            norm_box_polygon = [
                [bbox[0], bbox[1]],
                [bbox[2], bbox[1]],
                [bbox[2], bbox[3]],
                [bbox[0], bbox[3]],
            ]
            px_area = roi_w * roi_h
            len_px = float(max(roi_w, roi_h))
            wid_px = float(min(roi_w, roi_h))

            if self.is_calibrated and self.mm2_per_pixel2 and self.mm_per_pixel:
                surf_mm2 = round(px_area * self.mm2_per_pixel2, 4)
                l_mm = round(len_px * self.mm_per_pixel, 3)
                w_mm = round(wid_px * self.mm_per_pixel, 3)
            else:
                surf_mm2 = None
                l_mm = None
                w_mm = None

            return DefectMetrology(
                pixel_area=px_area,
                area_px=px_area,
                surface_area_mm2=surf_mm2,
                length_mm=l_mm,
                width_mm=w_mm,
                length_px=round(len_px, 1),
                width_px=round(wid_px, 1),
                aspect_ratio=round(len_px / max(wid_px, 1), 2),
                solidity=1.0,
                polygon=norm_box_polygon,
                is_calibrated=self.is_calibrated,
                calibration_label=self.calibration_label,
                px_per_mm=self.pixels_per_mm,
            )

        roi = full_image[y1:y2, x1:x2]
        if len(roi.shape) == 3:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_roi = roi.copy()

        # Apply CLAHE to accentuate micro-fracture boundaries against metallic grain
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_roi)

        # Adaptive Gaussian thresholding to isolate defect pixels
        thresh = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15,
            4,
        )

        # Morphological opening and closing to suppress granular camera sensor noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

        # Find external contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_contour = None
        max_area = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                best_contour = cnt

        # If no significant contour found inside ROI, fallback to oriented ellipse/box
        if best_contour is None or len(best_contour) < 3 or max_area < 2.0:
            norm_box_polygon = [
                [bbox[0], bbox[1]],
                [bbox[2], bbox[1]],
                [bbox[2], bbox[3]],
                [bbox[0], bbox[3]],
            ]
            px_area = roi_w * roi_h
            len_px = float(max(roi_w, roi_h))
            wid_px = float(min(roi_w, roi_h))

            if self.is_calibrated and self.mm2_per_pixel2 and self.mm_per_pixel:
                surf_mm2 = round(px_area * self.mm2_per_pixel2, 4)
                l_mm = round(len_px * self.mm_per_pixel, 3)
                w_mm = round(wid_px * self.mm_per_pixel, 3)
            else:
                surf_mm2 = None
                l_mm = None
                w_mm = None

            return DefectMetrology(
                pixel_area=px_area,
                area_px=px_area,
                surface_area_mm2=surf_mm2,
                length_mm=l_mm,
                width_mm=w_mm,
                length_px=round(len_px, 1),
                width_px=round(wid_px, 1),
                aspect_ratio=round(len_px / max(wid_px, 1), 2),
                solidity=1.0,
                polygon=norm_box_polygon,
                is_calibrated=self.is_calibrated,
                calibration_label=self.calibration_label,
                px_per_mm=self.pixels_per_mm,
            )

        # Simplify polygon using Douglas-Peucker algorithm
        epsilon = 0.015 * cv2.arcLength(best_contour, True)
        approx = cv2.approxPolyDP(best_contour, epsilon, True)

        # Minimum oriented bounding rectangle for true crack orientation & length
        rect = cv2.minAreaRect(best_contour)
        rect_w, rect_h = rect[1]
        phys_length_px = float(max(rect_w, rect_h))
        phys_width_px = float(min(rect_w, rect_h))

        aspect_ratio = phys_length_px / max(phys_width_px, 0.1)

        # Convex hull & solidity
        hull = cv2.convexHull(best_contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(max_area / max(hull_area, 1e-4))
        solidity = min(1.0, max(0.0, solidity))

        # Project ROI-local contour coordinates back to normalized full-image coordinates
        norm_polygon: List[List[float]] = []
        for pt in approx:
            px_x = x1 + pt[0][0]
            px_y = y1 + pt[0][1]
            norm_x = min(1.0, max(0.0, float(px_x / img_w)))
            norm_y = min(1.0, max(0.0, float(px_y / img_h)))
            norm_polygon.append([round(norm_x, 5), round(norm_y, 5)])

        # Ensure polygon has at least 3 points
        if len(norm_polygon) < 3:
            norm_polygon = [
                [bbox[0], bbox[1]],
                [bbox[2], bbox[1]],
                [bbox[2], bbox[3]],
                [bbox[0], bbox[3]],
            ]

        pixel_area = int(max_area)
        if self.is_calibrated and self.mm2_per_pixel2 and self.mm_per_pixel:
            surface_area_mm2 = round(pixel_area * self.mm2_per_pixel2, 4)
            length_mm = round(phys_length_px * self.mm_per_pixel, 3)
            width_mm = round(phys_width_px * self.mm_per_pixel, 3)
        else:
            surface_area_mm2 = None
            length_mm = None
            width_mm = None

        raw_metrology = DefectMetrology(
            pixel_area=pixel_area,
            area_px=pixel_area,
            surface_area_mm2=surface_area_mm2,
            length_mm=length_mm,
            width_mm=width_mm,
            length_px=round(phys_length_px, 1),
            width_px=round(phys_width_px, 1),
            aspect_ratio=round(aspect_ratio, 2),
            solidity=round(solidity, 3),
            polygon=norm_polygon,
            is_calibrated=self.is_calibrated,
            calibration_label=self.calibration_label,
            px_per_mm=self.pixels_per_mm,
        )

        tol = self.evaluate_engineering_tolerances(raw_metrology, class_name=class_name)
        raw_metrology.tolerance_status = tol.status
        raw_metrology.tolerance_note = tol.criterion
        return raw_metrology

    def draw_polygon_overlay(
        self,
        image: np.ndarray,
        polygon: List[List[float]],
        color: Tuple[int, int, int] = (0, 0, 255),
        alpha: float = 0.35,
    ) -> np.ndarray:
        """Render a translucent filled polygon with anti-aliased border onto image.

        Args:
            image: Image array (H, W, 3) BGR.
            polygon: Normalized polygon points [[x1, y1], [x2, y2], ...].
            color: BGR tuple for fill and contour.
            alpha: Transparency weight for overlay (0.0 to 1.0).

        Returns:
            Annotated image array with polygon mask.
        """
        if len(polygon) < 3:
            return image

        h, w = image.shape[:2]
        pts = np.array(
            [[int(round(p[0] * w)), int(round(p[1] * h))] for p in polygon],
            dtype=np.int32,
        )

        overlay = image.copy()
        cv2.fillPoly(overlay, [pts], color)
        annotated = cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)
        cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
        return annotated

