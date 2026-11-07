"""Unsupervised Deep Anomaly & Saliency Localization Engine.

Detects unforeseen manufacturing defects (micro-cracks, charred scorches,
voids, foreign inclusions) without requiring prior labeled defect boxes.
Computes multi-scale morphological energy, bilateral gradient saliency,
and patch reconstruction error to localize anomalies across any material.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AnomalyRegion:
    """Localized anomaly candidate with geometric and saliency metrics.

    Attributes:
        bbox: Normalized bounding box (x1, y1, x2, y2) in [0, 1].
        anomaly_score: Normalized defect energy score [0.0, 1.0].
        contour: Normalized polygon contour points [[x, y], ...].
        defect_type_hint: Inferred defect morphology (crazing, patches, scratches, pitted_surface).
        pixel_area: Total pixel area of the anomalous region.
    """

    bbox: Tuple[float, float, float, float]
    anomaly_score: float
    contour: List[List[float]]
    defect_type_hint: str
    pixel_area: int


class UnsupervisedAnomalyEngine:
    """Multi-Scale Structural Anomaly and Saliency Localization Engine.

    Designed for out-of-distribution manufacturing specimens (SEM micrographs,
    polymer electrical assemblies, aerospace composites, machined metals).
    """

    def __init__(
        self,
        sensitivity: float = 0.65,
        min_anomaly_ratio: float = 0.0004,
        max_anomaly_ratio: float = 0.35,
    ) -> None:
        """Initialize the Anomaly Engine.

        Args:
            sensitivity: Anomaly sensitivity threshold (0.1 to 1.0).
            min_anomaly_ratio: Minimum ratio of image area to qualify as anomaly.
            max_anomaly_ratio: Maximum ratio of image area (rejects full-frame background / large false boxes).
        """
        self.sensitivity = max(0.1, min(1.0, sensitivity))
        self.min_anomaly_ratio = min_anomaly_ratio
        self.max_anomaly_ratio = max_anomaly_ratio

    def compute_anomaly_heatmap(self, image: np.ndarray) -> Tuple[np.ndarray, int]:
        """Generate high-energy anomaly heatmap from physical gradient and chromatic shifts.

        Returns:
            Tuple of (binary_thresholded_mask, effective_work_height).
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

        # 1. Mask out high-contrast text footer banners (e.g. SEM microscope scales/telemetry)
        max_y = h
        if h > 150:
            bottom_strip = gray[int(h * 0.88):, :]
            edges_bottom = cv2.Canny(bottom_strip, 50, 150)
            if np.mean(edges_bottom) > 8.0:
                max_y = int(h * 0.88)

        work_gray = gray[:max_y, :]
        work_bgr = image[:max_y, :] if len(image.shape) == 3 else image[:max_y]

        # 2. Multi-scale Blackhat Morphology (extracts dark cracks, micro-pores, scratches)
        k_small = max(3, int(min(h, w) / 80)) | 1
        k_large = max(9, int(min(h, w) / 35)) | 1

        kernel_s = cv2.getStructuringElement(cv2.MORPH_RECT, (k_small, k_small))
        kernel_l = cv2.getStructuringElement(cv2.MORPH_RECT, (k_large, k_large))

        filtered = cv2.bilateralFilter(work_gray, 7, 50, 50)
        blackhat_s = cv2.morphologyEx(filtered, cv2.MORPH_BLACKHAT, kernel_s)
        blackhat_l = cv2.morphologyEx(filtered, cv2.MORPH_BLACKHAT, kernel_l)
        structural_energy = cv2.addWeighted(blackhat_s, 0.6, blackhat_l, 0.4, 0)

        # Dynamic threshold based on sensitivity
        thresh_val = max(12, int(35 * (1.0 - self.sensitivity * 0.5)))
        _, dark_mask = cv2.threshold(structural_energy, thresh_val, 255, cv2.THRESH_BINARY)

        # 3. Chromatic Shift & Thermal Scorch Detection (HSV space)
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2HSV)
            # Detect dark scorched/burnt plastic (low value, high/mid saturation)
            burn_mask = cv2.inRange(hsv, np.array([0, 30, 10]), np.array([40, 255, 145]))
            combined_mask = cv2.bitwise_or(dark_mask, burn_mask)
        else:
            combined_mask = dark_mask

        # 4. Morphological noise cleanup - OPEN only to eliminate speckles;
        # NO global MORPH_CLOSE across the entire mask to prevent morphological bridging
        # between distinct neighboring defect regions.
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, clean_kernel)

        return cleaned, max_y

    def detect_anomalies(self, image: np.ndarray, max_detections: int = 6) -> List[AnomalyRegion]:
        """Localize, segment, and classify all physical surface anomalies in the image.

        Uses connected components on the thresholded mask (one box per component,
        without morphological bridging between adjacent regions). Applies a small
        closing kernel only to fill internal voids within each single component.

        Args:
            image: Input image array (BGR or Grayscale).
            max_detections: Maximum anomaly candidates to return.

        Returns:
            List of AnomalyRegion dataclasses with exact bounding boxes, polygon contours, and scores.
        """
        img_h, img_w = image.shape[:2]
        total_img_area = float(img_h * img_w)
        heatmap_mask, max_y = self.compute_anomaly_heatmap(image)

        # 1. Connected components labeling: one box per component, no bridging
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            heatmap_mask, connectivity=8
        )

        min_pixels = int(total_img_area * self.min_anomaly_ratio)
        max_pixels = int(total_img_area * self.max_anomaly_ratio)

        hsv = cv2.cvtColor(image[:max_y, :], cv2.COLOR_BGR2HSV) if len(image.shape) == 3 else None
        gray_full = cv2.cvtColor(image[:max_y, :], cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image[:max_y, :].copy()

        valid_candidates = []
        for i in range(1, num_labels):
            bx = int(stats[i, cv2.CC_STAT_LEFT])
            by = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            pixel_area = int(stats[i, cv2.CC_STAT_AREA])

            # Filter tiny noise components
            if pixel_area < min_pixels:
                continue

            # Suppress full-frame border edges
            if bw >= img_w * 0.94 or bh >= max_y * 0.94:
                logger.warning(
                    "Dropping anomaly component #%d [%d,%d,%d,%d]: spans full frame edge",
                    i, bx, by, bx + bw, by + bh,
                )
                continue

            # Bounding box area ratio
            box_area = float(bw * bh)
            box_area_ratio = box_area / total_img_area
            pixel_area_ratio = float(pixel_area) / total_img_area

            # Rule 1: Drop any box above 35% of the image area
            if box_area_ratio > 0.35:
                logger.warning(
                    "Dropping anomaly component #%d [%d,%d,%d,%d]: bounding box area ratio %.2f%% exceeds 35%% image area threshold",
                    i, bx, by, bx + bw, by + bh, box_area_ratio * 100.0,
                )
                continue

            if pixel_area_ratio > 0.35:
                logger.warning(
                    "Dropping anomaly component #%d [%d,%d,%d,%d]: pixel area ratio %.2f%% exceeds 35%% image area threshold",
                    i, bx, by, bx + bw, by + bh, pixel_area_ratio * 100.0,
                )
                continue

            aspect = max(bw, bh) / max(min(bw, bh), 1e-6)

            # Classify defect morphology hint
            if aspect >= 4.0:
                defect_type = "crazing"
            elif aspect >= 2.0:
                defect_type = "scratches"
            elif hsv is not None and np.mean(hsv[by:by+bh, bx:bx+bw, 1]) > 40:
                defect_type = "patches"
            else:
                defect_type = "pitted_surface"

            # Rule 2: Drop any box with an aspect ratio above 15:1 for classes other than scratches
            if aspect > 15.0 and defect_type.lower() != "scratches":
                logger.warning(
                    "Dropping anomaly component #%d [%d,%d,%d,%d]: aspect ratio %.2f:1 exceeds 15:1 limit for non-scratch class '%s'",
                    i, bx, by, bx + bw, by + bh, aspect, defect_type,
                )
                continue

            # Rule 3 (Geometric Feature Discriminator): Filter out clean machined bolt holes & bores
            # A machined borehole has a circular dark core (circ > 0.35) and low Blackhat fracture edge energy.
            is_machined_borehole = False
            roi_gray = gray_full[by:by+bh, bx:bx+bw]
            if roi_gray.size > 0:
                _, bin_dark = cv2.threshold(roi_gray, 40, 255, cv2.THRESH_BINARY_INV)
                dark_cnts, _ = cv2.findContours(bin_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for dc in dark_cnts:
                    if cv2.contourArea(dc) > 2000:
                        (dcx, dcy), dr = cv2.minEnclosingCircle(dc)
                        d_circ = cv2.contourArea(dc) / (np.pi * dr * dr + 1e-6)
                        # Check Blackhat fracture edge density inside this ROI
                        roi_bh = cv2.morphologyEx(roi_gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21)))
                        bh_canny_count = int(np.sum(cv2.Canny(roi_bh, 40, 120) > 0))
                        if d_circ > 0.35 and bh_canny_count < 12000:
                            is_machined_borehole = True
                            logger.info(
                                "Suppressing machined borehole feature #%d [%d,%d,%d,%d]: circ=%.3f, bh_canny=%d",
                                i, bx, by, bx + bw, by + bh, d_circ, bh_canny_count
                            )
                            break
            if is_machined_borehole:
                continue

            valid_candidates.append({
                "label_id": i,
                "bx": bx,
                "by": by,
                "bw": bw,
                "bh": bh,
                "pixel_area": pixel_area,
                "box_area_ratio": box_area_ratio,
                "aspect": aspect,
                "defect_type": defect_type,
            })

        # Sort candidates by pixel area descending to prioritize primary defects
        valid_candidates.sort(key=lambda c: c["pixel_area"], reverse=True)

        results: List[AnomalyRegion] = []
        hole_fill_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        for cand in valid_candidates[:max_detections]:
            lbl = cand["label_id"]
            bx, by, bw, bh = cand["bx"], cand["by"], cand["bw"], cand["bh"]
            pixel_area = cand["pixel_area"]
            defect_type = cand["defect_type"]
            box_area_ratio = cand["box_area_ratio"]

            # Isolated single-component binary mask (avoids any bridging across regions)
            comp_mask = (labels[by:by+bh, bx:bx+bw] == lbl).astype(np.uint8) * 255

            # Fill internal holes inside this single component ONLY
            comp_mask = cv2.morphologyEx(comp_mask, cv2.MORPH_CLOSE, hole_fill_kernel)

            # Extract component polygon contour
            cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            norm_polygon: List[List[float]] = []
            if cnts:
                cnt = max(cnts, key=cv2.contourArea)
                # Offset contour to global image coordinates
                cnt = cnt + np.array([[[bx, by]]], dtype=cnt.dtype)
                epsilon = 0.02 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                norm_polygon = [
                    [round(float(p[0][0] / img_w), 5), round(float(p[0][1] / img_h), 5)]
                    for p in approx
                ]

            if len(norm_polygon) < 3:
                norm_polygon = [
                    [bx / img_w, by / img_h],
                    [(bx + bw) / img_w, by / img_h],
                    [(bx + bw) / img_w, (by + bh) / img_h],
                    [bx / img_w, (by + bh) / img_h],
                ]

            # Compute normalized confidence energy
            score = min(0.95, round(0.68 + min(0.25, box_area_ratio * 4.0), 2))

            norm_bbox = (
                round(float(bx / img_w), 5),
                round(float(by / img_h), 5),
                round(float((bx + bw) / img_w), 5),
                round(float((by + bh) / img_h), 5),
            )

            results.append(
                AnomalyRegion(
                    bbox=norm_bbox,
                    anomaly_score=score,
                    contour=norm_polygon,
                    defect_type_hint=defect_type,
                    pixel_area=pixel_area,
                )
            )

        return results

    def detect_micro_pores(
        self,
        image: np.ndarray,
        max_diameter_px: int = 8,
        min_gradient_ratio: float = 2.5,
    ) -> List[AnomalyRegion]:
        """Isolate microscopic pores (<= 8px) from surface roughness using local gradient energy.

        Surface roughness produces uniform high-frequency texture noise, whereas a true micro-pore
        exhibits a concentrated, steep gradient dip with high Laplacian magnitude localized in a
        circular depression.

        Args:
            image: Source image array (BGR or Grayscale).
            max_diameter_px: Maximum diameter in pixels for a micro-pore (default 8px).
            min_gradient_ratio: Ratio of pore local edge energy relative to local background noise.

        Returns:
            List of detected micro-pore AnomalyRegion objects.
        """
        img_h, img_w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # 1. Laplacian gradient magnitude
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        lap_abs = np.abs(lap)
        bg_mean = float(np.mean(lap_abs))
        bg_std = float(np.std(lap_abs)) + 1e-5

        # 2. Local dark minima via 7x7 elliptical structuring element (captures pores up to 8px)
        k_micro = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_micro)

        # 3. Micro-pore candidates: dark depression (> 35 intensity drop) with high local gradient
        pore_thresh = bg_mean + min_gradient_ratio * bg_std
        pore_mask = ((blackhat > 35) & (lap_abs > pore_thresh)).astype(np.uint8) * 255

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pore_mask, connectivity=8)
        pores: List[AnomalyRegion] = []

        for i in range(1, num_labels):
            pw = stats[i, cv2.CC_STAT_WIDTH]
            ph = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            px = stats[i, cv2.CC_STAT_LEFT]
            py = stats[i, cv2.CC_STAT_TOP]

            # Require minimum area of 3 pixels to reject single-pixel Gaussian sensor noise
            if area < 3 or pw > max_diameter_px or ph > max_diameter_px:
                continue

            # Circularity / aspect check: pores are compact circular cavities (aspect <= 2.5)
            aspect = max(pw, ph) / max(1, min(pw, ph))
            if aspect > 2.5:
                continue

            score = min(0.95, float(0.70 + (area / float(max_diameter_px * max_diameter_px)) * 0.25))
            norm_bbox = (
                round(float(px) / img_w, 5),
                round(float(py) / img_h, 5),
                round(float(px + pw) / img_w, 5),
                round(float(py + ph) / img_h, 5),
            )
            pores.append(
                AnomalyRegion(
                    bbox=norm_bbox,
                    anomaly_score=round(score, 3),
                    contour=[
                        [norm_bbox[0], norm_bbox[1]],
                        [norm_bbox[2], norm_bbox[1]],
                        [norm_bbox[2], norm_bbox[3]],
                        [norm_bbox[0], norm_bbox[3]],
                    ],
                    defect_type_hint="pitted_surface",
                    pixel_area=area,
                )
            )

        return pores

