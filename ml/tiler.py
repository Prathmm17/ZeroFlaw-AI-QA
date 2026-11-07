"""High-Resolution Slicing and Tiling Engine for VisionGuard QA.

Implements Slicing Aided Hyper Inference (SAHI) style overlapping window slicing
to preserve sub-millimeter micro-defects and hairline fractures on high-resolution
industrial camera images without destructive downsampling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ImageSlice:
    """A sub-window slice cut from a high-resolution source image.

    Attributes:
        image: Numpy BGR image array corresponding to the window patch.
        x1: Left coordinate of slice relative to the original source image.
        y1: Top coordinate of slice relative to the original source image.
        x2: Right coordinate of slice relative to the original source image.
        y2: Bottom coordinate of slice relative to the original source image.
    """

    image: np.ndarray
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


class ImageTiler:
    """Splits high-resolution images into overlapping patches and fuses slice predictions."""

    def __init__(
        self,
        slice_size: int = 512,
        overlap_ratio: float = 0.20,
        iou_threshold: float = 0.45,
    ) -> None:
        """Initialise the image tiler.

        Args:
            slice_size: Square dimension (width and height in px) for each slice.
            overlap_ratio: Overlap percentage (0.0 to 0.5) between neighboring slices.
            iou_threshold: Intersection-over-Union threshold for global box fusion NMS.
        """
        if slice_size <= 0:
            raise ValueError(f"slice_size must be positive, got {slice_size}")
        if not 0.0 <= overlap_ratio < 1.0:
            raise ValueError(f"overlap_ratio must be in [0.0, 1.0), got {overlap_ratio}")

        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        self.iou_threshold = iou_threshold
        logger.info(
            "ImageTiler initialized (slice_size=%d, overlap=%.2f, iou_thresh=%.2f)",
            slice_size,
            overlap_ratio,
            iou_threshold,
        )

    def slice_image(self, image: np.ndarray) -> List[ImageSlice]:
        """Slice an image into an overlapping regular grid covering the entire canvas.

        Args:
            image: High-resolution BGR numpy image array.

        Returns:
            List of ImageSlice instances covering every pixel of the source image.
        """
        img_h, img_w = image.shape[:2]
        slices: List[ImageSlice] = []

        # If image is smaller than or equal to slice_size, return as single full slice
        if img_w <= self.slice_size and img_h <= self.slice_size:
            slices.append(ImageSlice(image=image.copy(), x1=0, y1=0, x2=img_w, y2=img_h))
            return slices

        step = int(self.slice_size * (1.0 - self.overlap_ratio))
        step = max(1, step)

        # Generate X coordinates
        x_starts: List[int] = []
        x = 0
        while x + self.slice_size < img_w:
            x_starts.append(x)
            x += step
        x_starts.append(max(0, img_w - self.slice_size))
        x_starts = sorted(list(set(x_starts)))

        # Generate Y coordinates
        y_starts: List[int] = []
        y = 0
        while y + self.slice_size < img_h:
            y_starts.append(y)
            y += step
        y_starts.append(max(0, img_h - self.slice_size))
        y_starts = sorted(list(set(y_starts)))

        for y1 in y_starts:
            y2 = min(img_h, y1 + self.slice_size)
            for x1 in x_starts:
                x2 = min(img_w, x1 + self.slice_size)
                patch = image[y1:y2, x1:x2].copy()
                slices.append(ImageSlice(image=patch, x1=x1, y1=y1, x2=x2, y2=y2))

        logger.debug(
            "Sliced image %dx%d into %d tiles of size %dx%d",
            img_w,
            img_h,
            len(slices),
            self.slice_size,
            self.slice_size,
        )
        return slices

    def detect_component_envelope(self, image: np.ndarray, margin: int = 32) -> Tuple[int, int, int, int]:
        """Detect active component bounding envelope (x1, y1, x2, y2) to skip dead conveyor margins.

        Args:
            image: High-resolution source image (BGR or Grayscale).
            margin: Pixel safety margin added around the component boundary.

        Returns:
            Tuple of (x1, y1, x2, y2) bounding box enclosing the industrial part.
        """
        img_h, img_w = image.shape[:2]
        small_h = 256
        scale = float(small_h) / float(img_h)
        small_w = max(1, int(round(img_w * scale)))

        small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if len(small.shape) == 3 else small

        # Otsu thresholding with bilateral filtering to isolate component foreground from conveyor background
        filtered = cv2.bilateralFilter(gray, 7, 50, 50)
        _, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert if background is bright
        if np.mean(thresh) > 127:
            thresh = cv2.bitwise_not(thresh)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            valid_cnts = [c for c in contours if cv2.contourArea(c) > (small_h * small_w * 0.04)]
            if valid_cnts:
                all_pts = np.vstack(valid_cnts)
                bx, by, bw, bh = cv2.boundingRect(all_pts)
                gx1 = max(0, int(round(bx / scale)) - margin)
                gy1 = max(0, int(round(by / scale)) - margin)
                gx2 = min(img_w, int(round((bx + bw) / scale)) + margin)
                gy2 = min(img_h, int(round((by + bh) / scale)) + margin)

                # Ensure envelope is at least slice_size in both dimensions
                if (gx2 - gx1) >= self.slice_size and (gy2 - gy1) >= self.slice_size:
                    logger.info(
                        "Detected component envelope: [%d, %d, %d, %d] on %dx%d image (saving background tiles)",
                        gx1, gy1, gx2, gy2, img_w, img_h,
                    )
                    return (gx1, gy1, gx2, gy2)

        return (0, 0, img_w, img_h)

    def slice_image_adaptive(
        self,
        image: np.ndarray,
        focus_envelope: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[ImageSlice]:
        """Adaptive resolution-aware tiling at native 1:1 optical scale without downsampling loss.

        Slices the active component bounding envelope into 512x512 patches with 20% overlap.

        Args:
            image: High-resolution source image (e.g. 2048x2048, 4K).
            focus_envelope: Optional (x1, y1, x2, y2) region. If None, auto-detected.

        Returns:
            List of ImageSlice instances covering the component at 1:1 optical scale.
        """
        img_h, img_w = image.shape[:2]
        if focus_envelope is None:
            focus_envelope = self.detect_component_envelope(image)

        ex1, ey1, ex2, ey2 = focus_envelope
        ew = ex2 - ex1
        eh = ey2 - ey1

        # If envelope is smaller than slice_size, crop and pad or return directly
        if ew <= self.slice_size and eh <= self.slice_size:
            patch = image[ey1:ey2, ex1:ex2].copy()
            return [ImageSlice(image=patch, x1=ex1, y1=ey1, x2=ex2, y2=ey2)]

        step = max(1, int(round(self.slice_size * (1.0 - self.overlap_ratio))))

        x_starts: List[int] = []
        cx = ex1
        while cx + self.slice_size < ex2:
            x_starts.append(cx)
            cx += step
        x_starts.append(max(0, ex2 - self.slice_size))
        x_starts = sorted(list(set(x_starts)))

        y_starts: List[int] = []
        cy = ey1
        while cy + self.slice_size < ey2:
            y_starts.append(cy)
            cy += step
        y_starts.append(max(0, ey2 - self.slice_size))
        y_starts = sorted(list(set(y_starts)))

        slices: List[ImageSlice] = []
        for y1 in y_starts:
            y2 = min(img_h, y1 + self.slice_size)
            for x1 in x_starts:
                x2 = min(img_w, x1 + self.slice_size)
                patch = image[y1:y2, x1:x2].copy()
                slices.append(ImageSlice(image=patch, x1=x1, y1=y1, x2=x2, y2=y2))

        logger.info(
            "Adaptive Slicing (1:1 Native Scale): envelope=[%d,%d,%d,%d] -> %d patches of size %dx%d (stride=%d)",
            ex1, ey1, ex2, ey2, len(slices), self.slice_size, self.slice_size, step,
        )
        return slices

    def stitch_seam_fractures(
        self,
        detections: List[Dict[str, Any]],
        image_shape: Tuple[int, ...],
        proximity_px: float = 24.0,
    ) -> List[Dict[str, Any]]:
        """Merge split crack segments crossing patch seams into unified continuous detections.

        Args:
            detections: List of detection dictionaries with global coordinates.
            image_shape: (height, width) of the full image.
            proximity_px: Maximum distance in pixels between adjacent seam endpoints to trigger merge.

        Returns:
            List of detections with split seam fractures stitched together.
        """
        if len(detections) <= 1:
            return detections

        img_h, img_w = image_shape[:2]
        total_area = float(img_h * img_w)
        linear_classes = {"crazing", "scratches", "unclassified_anomaly"}

        merged: List[Dict[str, Any]] = []
        used = [False] * len(detections)

        for i in range(len(detections)):
            if used[i]:
                continue

            curr = dict(detections[i])
            c_cls = curr.get("class_name", "").lower()

            if c_cls not in linear_classes:
                merged.append(curr)
                used[i] = True
                continue

            # Iteratively search for adjacent seam fragments to stitch
            stitched = True
            while stitched:
                stitched = False
                bx1, by1, bx2, by2 = curr["bbox"]

                for j in range(len(detections)):
                    if i == j or used[j]:
                        continue

                    cand = detections[j]
                    cand_cls = cand.get("class_name", "").lower()
                    if cand_cls not in linear_classes and cand_cls != c_cls:
                        continue

                    cx1, cy1, cx2, cy2 = cand["bbox"]

                    # Check horizontal seam proximity (box1 right touches box2 left or vice versa)
                    h_touch = (
                        abs(bx2 - cx1) <= proximity_px
                        and max(0.0, min(by2, cy2) - max(by1, cy1)) > 0.20 * min(by2 - by1, cy2 - cy1)
                    ) or (
                        abs(cx2 - bx1) <= proximity_px
                        and max(0.0, min(by2, cy2) - max(by1, cy1)) > 0.20 * min(by2 - by1, cy2 - cy1)
                    )

                    # Check vertical seam proximity (box1 bottom touches box2 top or vice versa)
                    v_touch = (
                        abs(by2 - cy1) <= proximity_px
                        and max(0.0, min(bx2, cx2) - max(bx1, cx1)) > 0.20 * min(bx2 - bx1, cx2 - cx1)
                    ) or (
                        abs(cy2 - by1) <= proximity_px
                        and max(0.0, min(bx2, cx2) - max(bx1, cx1)) > 0.20 * min(bx2 - bx1, cx2 - cx1)
                    )

                    # Substantial overlap (IoU > 0.25 between collinear cracks)
                    ix1 = max(bx1, cx1)
                    iy1 = max(by1, cy1)
                    ix2 = min(bx2, cx2)
                    iy2 = min(by2, cy2)
                    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                    overlap = inter > 0 and (inter / max(1.0, min((bx2 - bx1) * (by2 - by1), (cx2 - cx1) * (cy2 - cy1)))) > 0.30

                    if h_touch or v_touch or overlap:
                        # Merge cand into curr
                        nx1 = round(min(bx1, cx1), 1)
                        ny1 = round(min(by1, cy1), 1)
                        nx2 = round(max(bx2, cx2), 1)
                        ny2 = round(max(by2, cy2), 1)
                        nw = nx2 - nx1
                        nh = ny2 - ny1

                        curr["bbox"] = (nx1, ny1, nx2, ny2)
                        curr["confidence"] = max(curr["confidence"], cand["confidence"])
                        curr["area_ratio"] = round((nw * nh) / total_area, 6)

                        # Merge polygons if present
                        p1 = curr.get("polygon") or []
                        p2 = cand.get("polygon") or []
                        if p1 and p2:
                            curr["polygon"] = p1 + p2

                        used[j] = True
                        stitched = True
                        logger.info(
                            "Stitched split seam fracture: [%.1f,%.1f,%.1f,%.1f] + [%.1f,%.1f,%.1f,%.1f] -> [%.1f,%.1f,%.1f,%.1f]",
                            bx1, by1, bx2, by2, cx1, cy1, cx2, cy2, nx1, ny1, nx2, ny2,
                        )
                        break

            used[i] = True
            merged.append(curr)

        return merged

    def merge_detections(
        self,
        slice_detections: List[Tuple[ImageSlice, List[Dict[str, Any]]]],
        image_shape: Tuple[int, ...],
        use_soft_nms: bool = True,
    ) -> List[Dict[str, Any]]:
        """Map slice detections back to global coordinates and apply seam stitching and Soft-NMS.

        Args:
            slice_detections: Pairs of (ImageSlice, list_of_detections_in_slice).
            image_shape: (height, width) of the original high-resolution image.
            use_soft_nms: Whether to apply Gaussian Soft-NMS (default True) to prevent fracture loss.

        Returns:
            Filtered list of detections in global image coordinates.
        """
        all_boxes: List[List[int]] = []
        all_confidences: List[float] = []
        all_class_ids: List[int] = []
        all_raw_data: List[Dict[str, Any]] = []

        img_h, img_w = image_shape[:2]
        total_image_area = float(img_h * img_w)

        for sl, detections in slice_detections:
            for det in detections:
                bx1, by1, bx2, by2 = det["bbox"]
                
                # Global coordinate mapping
                gx1 = max(0.0, min(float(img_w), float(bx1 + sl.x1)))
                gy1 = max(0.0, min(float(img_h), float(by1 + sl.y1)))
                gx2 = max(0.0, min(float(img_w), float(bx2 + sl.x1)))
                gy2 = max(0.0, min(float(img_h), float(by2 + sl.y1)))

                gw = gx2 - gx1
                gh = gy2 - gy1

                if gw <= 0 or gh <= 0:
                    continue

                # Recalculate true area ratio relative to full image
                true_area_ratio = (gw * gh) / total_image_area

                det_copy = dict(det)
                det_copy["bbox"] = (round(gx1, 1), round(gy1, 1), round(gx2, 1), round(gy2, 1))
                det_copy["area_ratio"] = round(true_area_ratio, 6)

                all_boxes.append([int(gx1), int(gy1), int(gw), int(gh)])
                all_confidences.append(float(det["confidence"]))
                all_class_ids.append(int(det["class_id"]))
                all_raw_data.append(det_copy)

        if not all_raw_data:
            return []

        # 1. First stitch linear split crack segments crossing patch seams
        stitched_candidates = self.stitch_seam_fractures(all_raw_data, image_shape=image_shape)

        if not use_soft_nms:
            # Standard hard NMS fallback
            boxes = [
                [int(d["bbox"][0]), int(d["bbox"][1]), int(d["bbox"][2] - d["bbox"][0]), int(d["bbox"][3] - d["bbox"][1])]
                for d in stitched_candidates
            ]
            scores = [float(d["confidence"]) for d in stitched_candidates]
            indices = cv2.dnn.NMSBoxes(
                bboxes=boxes,
                scores=scores,
                score_threshold=0.0,
                nms_threshold=self.iou_threshold,
            )
            merged = [stitched_candidates[i] for i in indices.flatten()] if len(indices) > 0 else []
        else:
            # Gaussian Soft-NMS: smoothly attenuates overlapping duplicate boxes without sudden fracture drops
            sigma = 0.5
            candidates = [dict(d) for d in stitched_candidates]
            merged: List[Dict[str, Any]] = []

            while candidates:
                # Pick detection with maximum confidence
                candidates.sort(key=lambda d: d["confidence"], reverse=True)
                best = candidates.pop(0)
                merged.append(best)

                bx1, by1, bx2, by2 = best["bbox"]
                b_area = max(1.0, (bx2 - bx1) * (by2 - by1))
                b_cls = best.get("class_name", "").lower()

                remaining: List[Dict[str, Any]] = []
                for cand in candidates:
                    c_cls = cand.get("class_name", "").lower()
                    cx1, cy1, cx2, cy2 = cand["bbox"]
                    c_area = max(1.0, (cx2 - cx1) * (cy2 - cy1))

                    # Calculate IoU
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
                            cand["confidence"] = float(cand["confidence"] * weight)
                            cand["description"] = f"{cand['class_name'].capitalize()} detected with {round(cand['confidence']*100)}% confidence"

                    # Only keep candidate if score >= min_threshold (0.18)
                    if cand["confidence"] >= 0.18:
                        remaining.append(cand)

                candidates = remaining

        # Sort by confidence descending
        merged.sort(key=lambda d: d["confidence"], reverse=True)
        logger.info(
            "Merged %d slice detections into %d global detections (stitched & Soft-NMS iou=%.2f)",
            len(all_raw_data),
            len(merged),
            self.iou_threshold,
        )
        return merged

