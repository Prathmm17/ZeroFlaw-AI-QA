"""Main defect detection orchestrator for VisionGuard QA.

Ties together the model registry, image preprocessor, high-resolution tiler (SAHI),
and detection post-processor into a single high-level ``DefectDetector`` class that
executes the full inference pipeline: preprocess / slice → YOLO inference →
coordinate fusion → post-process → annotate.
"""

from __future__ import annotations

import datetime
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np

from backend.config import settings
from ml.anomaly_engine import AnomalyRegion, UnsupervisedAnomalyEngine
from ml.domain_guard import DomainGuard, DomainGuardAssessment
from ml.model_registry import ModelRegistry
from ml.postprocessor import Detection, DetectionPostprocessor
from ml.preprocessor import ImagePreprocessor
from ml.segmenter import DefectMetrology, DefectSegmenter
from ml.tiler import ImageSlice, ImageTiler

logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Severity → BGR colour mapping for annotation drawing.
# ---------------------------------------------------------------------------
_SEVERITY_COLORS: dict[str, tuple[int, int, int]] = {
    "critical": (0, 0, 255),     # Red
    "major": (0, 165, 255),      # Orange
    "minor": (0, 255, 255),      # Yellow
}

# Default annotation colour when severity is unrecognised.
_DEFAULT_COLOR: tuple[int, int, int] = (255, 255, 255)  # White

DEFECT_CLASSES: dict[int, str] = {
    0: "crazing",
    1: "inclusion",
    2: "patches",
    3: "pitted_surface",
    4: "rolled_in_scale",
    5: "scratches",
    6: "dent",
}


def apply_ambient_enhancement(img: np.ndarray) -> np.ndarray:
    """Enhance ambient contrast for webcam defect detection using LAB CLAHE."""
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    except Exception:
        return img


class DefectDetector:
    """High-level defect detection orchestrator with Native-Resolution Tiling support.

    Combines YOLO model loading (via :class:`ModelRegistry`), high-resolution
    slicing (via :class:`ImageTiler`), preprocessing (via :class:`ImagePreprocessor`),
    post-processing (via :class:`DetectionPostprocessor`), and result annotation
    into a single cohesive pipeline.

    Attributes:
        model_name: Filename of the YOLO weights used.
        confidence_threshold: Minimum detection confidence.
        device: Compute device (``"cpu"`` or ``"cuda"``).
        enable_tiling: Whether to slice high-resolution images into overlapping tiles.
        slice_size: Dimension in pixels of square slices (e.g. 512).
        tiling_min_dim: Minimum resolution threshold to trigger tiling (e.g. 800px).
    """

    DEFAULT_CLASS_THRESHOLDS: dict[str, float] = {
        "crazing": 0.40,
        "inclusion": 0.40,
        "patches": 0.40,
        "pitted_surface": 0.40,
        "rolled_in_scale": 0.40,
        "scratches": 0.40,
    }

    def __init__(
        self,
        model_name: str = "best.pt",
        confidence_threshold: float = 0.20,
        class_thresholds: Optional[dict[str, float]] = None,
        device: str = "cpu",
        enable_tiling: bool = True,
        slice_size: int = 512,
        overlap_ratio: float = 0.20,
        tiling_min_dim: Optional[int] = None,
    ) -> None:
        """Initialise the defect detector.

        Args:
            model_name: Filename of the YOLO model weights to load.
            confidence_threshold: Minimum confidence score for a detection.
            class_thresholds: Optional per-class confidence thresholds. If None, loaded
                from backend/class_thresholds.json.
            device: Inference device — ``"cpu"`` or ``"cuda"``.
            enable_tiling: Whether to enable high-resolution patch tiling.
            slice_size: Square tile resolution (e.g., 512px).
            overlap_ratio: Overlap percentage between tiles (0.0 to 0.5).
            tiling_min_dim: Dimension in pixels triggering tiling (defaults to settings.TILING_MIN_DIM=1600).
        """
        self.model_name: str = model_name
        self.confidence_threshold: float = confidence_threshold
        if class_thresholds is not None:
            self.class_thresholds: dict[str, float] = dict(class_thresholds)
        else:
            self.class_thresholds = self._load_calibrated_thresholds()
        self.device: str = device
        self.enable_tiling: bool = enable_tiling
        self.slice_size: int = slice_size
        self.overlap_ratio: float = overlap_ratio
        self.tiling_min_dim: int = tiling_min_dim if tiling_min_dim is not None else getattr(settings, "TILING_MIN_DIM", 1600)
        self.px_per_mm: Optional[float] = getattr(settings, "PX_PER_MM", None)

        self._is_ready: bool = False
        self._model: Any = None  # YOLO model instance

        self._registry: ModelRegistry = ModelRegistry()
        self._preprocessor: ImagePreprocessor = ImagePreprocessor(target_size=(640, 640))
        self._postprocessor: DetectionPostprocessor = DetectionPostprocessor(
            min_confidence=confidence_threshold,
            class_thresholds=self.class_thresholds,
        )
        self._segmenter: DefectSegmenter = DefectSegmenter(pixels_per_mm=self.px_per_mm)
        self._anomaly_engine: UnsupervisedAnomalyEngine = UnsupervisedAnomalyEngine(sensitivity=0.70)
        self._domain_guard: DomainGuard = DomainGuard()
        self._tiler: ImageTiler = ImageTiler(
            slice_size=slice_size,
            overlap_ratio=overlap_ratio,
            iou_threshold=0.45,
        )

        self._onnx_session: Any = None
        self._onnx_input_name: Optional[str] = None
        try:
            model_path = self._registry.get_model_path(model_name)
            if model_path.suffix.lower() == ".onnx" and model_path.is_file():
                import onnxruntime as ort
                opts = ort.SessionOptions()
                num_threads = getattr(settings, "ONNX_NUM_THREADS", 1)
                opts.intra_op_num_threads = min(4, max(1, num_threads))
                opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self._onnx_session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
                self._onnx_input_name = self._onnx_session.get_inputs()[0].name
                self._is_ready = True
                logger.info("Direct ONNX CPU inference session initialized with %d threads", opts.intra_op_num_threads)
        except Exception as onnx_err:
            logger.warning("Could not initialize direct ONNX session: %s", onnx_err)

        # Fallback to PyTorch / Ultralytics only if ONNX is not loaded
        if not self._is_ready:
            try:
                self._model = self._registry.load_model(model_name)
                self._is_ready = True
                logger.info(
                    "DefectDetector initialised with PyTorch — model=%s, device=%s, "
                    "conf=%.2f, tiling=%s (slice=%d, min_dim=%d), thresholds=%s",
                    model_name,
                    device,
                    confidence_threshold,
                    enable_tiling,
                    slice_size,
                    self.tiling_min_dim,
                    self.class_thresholds,
                )
            except Exception as exc:
                logger.error("DefectDetector failed to load model '%s': %s", model_name, exc)
                self._is_ready = False

    @classmethod
    def _load_calibrated_thresholds(cls) -> dict[str, float]:
        """Load optimal F1 thresholds from backend/class_thresholds.json."""
        import json
        thresh_path = Path(__file__).resolve().parent.parent / "backend" / "class_thresholds.json"
        if thresh_path.exists():
            try:
                with open(thresh_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("thresholds", cls.DEFAULT_CLASS_THRESHOLDS.copy())
            except Exception as err:
                logger.warning("Could not read class_thresholds.json: %s", err)
        return cls.DEFAULT_CLASS_THRESHOLDS.copy()

    # ------------------------------------------------------------------
    # Inference Engine (ONNX CPU & PyTorch Fallback)
    # ------------------------------------------------------------------

    def _run_onnx_inference(
        self,
        image: np.ndarray,
        effective_conf: float,
        target_size: tuple[int, int] = (640, 640),
    ) -> list[Detection]:
        """Execute aspect-preserving ONNX inference purely in C++/CPU memory.

        Bypasses PyTorch and Ultralytics completely for low-memory production environments.
        """
        if self._onnx_session is None:
            return []

        from backend.services.knowledge_service import get_defect_knowledge

        img_h, img_w = image.shape[:2]
        target_w, target_h = target_size
        scale = min(target_w / float(img_w), target_h / float(img_h))
        new_w = int(round(img_w * scale))
        new_h = int(round(img_h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_top = (target_h - new_h) // 2
        pad_left = (target_w - new_w) // 2
        canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized

        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255.0, (target_w, target_h), swapRB=True)
        out = self._onnx_session.run(None, {self._onnx_input_name: blob})[0][0]
        boxes_raw = out[:4].T
        scores_raw = out[4:].T
        class_ids = np.argmax(scores_raw, axis=1)
        confidences = np.max(scores_raw, axis=1)
        mask = confidences >= effective_conf
        boxes_f = boxes_raw[mask]
        confs_f = confidences[mask]
        cids_f = class_ids[mask]

        boxes_xywh = []
        for b in boxes_f:
            cx, cy, w, h = b
            orig_cx = (cx - pad_left) / scale
            orig_cy = (cy - pad_top) / scale
            orig_w = w / scale
            orig_h = h / scale
            x = int(orig_cx - orig_w / 2.0)
            y = int(orig_cy - orig_h / 2.0)
            boxes_xywh.append([x, y, int(orig_w), int(orig_h)])

        dets: list[Detection] = []
        if boxes_xywh:
            indices = cv2.dnn.NMSBoxes(boxes_xywh, confs_f.tolist(), effective_conf, 0.45)
            total_area = float(img_w * img_h)
            for idx in indices:
                i = int(idx)
                cid = int(cids_f[i])
                cname = DEFECT_CLASSES.get(cid, "defect")
                conf = float(confs_f[i])
                c_thresh = self.class_thresholds.get(cname.lower(), self.confidence_threshold)
                if effective_conf is not None and effective_conf != self.confidence_threshold:
                    ratio = effective_conf / max(0.01, self.confidence_threshold)
                    c_thresh = min(0.95, max(0.10, c_thresh * ratio))
                if conf >= c_thresh:
                    bx, by, bw, bh = boxes_xywh[i]
                    x1 = max(0.0, float(bx))
                    y1 = max(0.0, float(by))
                    x2 = min(float(img_w), float(bx + bw))
                    y2 = min(float(img_h), float(by + bh))
                    area_ratio = float((x2 - x1) * (y2 - y1)) / total_area if total_area > 0 else 0.0
                    sev = self._postprocessor.classify_severity(
                        conf,
                        area_ratio,
                        class_name=cname,
                        domain_trust="HIGH",
                        source="yolo",
                    )
                    k_info = get_defect_knowledge(cname) or {}
                    det_item = Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        class_id=cid,
                        class_name=cname,
                        severity=sev,
                        area_ratio=area_ratio,
                        description="",
                        source="yolo",
                        definition=k_info.get("definition"),
                        cause=k_info.get("cause"),
                        action=k_info.get("action"),
                        visual_look=k_info.get("visual_look"),
                        risk=k_info.get("risk"),
                    )
                    det_item.description = self._postprocessor.generate_description(det_item)
                    dets.append(det_item)
        return dets

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        source: Union[str, Path, np.ndarray, bytes],
        deep_verify: bool = False,
        yolo_only: bool = False,
        confidence_threshold: Optional[float] = None,
        ambient_boost: bool = False,
    ) -> dict[str, Any]:
        """Run the defect detection pipeline with automatic native-resolution tiling.

        If yolo_only is True (e.g. real-time live streaming):
            Executes high-throughput single-pass YOLO inference without anomaly filters or overlays.

        If deep_verify is True (explicit user request):
            Escalates out-of-domain and low-confidence regions to the unsupervised anomaly filter.

        Args:
            source: Image source — filesystem path, numpy array, or bytes.
            deep_verify: Whether to run Tier 2 anomaly / VLM escalation (default False).
            yolo_only: Stream optimization flag; bypasses heavy contouring and annotations.
            confidence_threshold: Dynamic confidence override (e.g. from UI slider).
            ambient_boost: Whether to apply CLAHE contrast enhancement for ambient cameras.

        Returns:
            Dictionary with detections, summary, annotated_image, latency, and metadata.
        """
        if not self._is_ready or (self._model is None and self._onnx_session is None):
            raise RuntimeError("DefectDetector is not ready — model failed to load.")

        total_start: float = time.perf_counter()

        # 1. Load original image
        original_image: np.ndarray = self._preprocessor.load_image(source)
        if ambient_boost:
            original_image = apply_ambient_enhancement(original_image)
        img_h, img_w = original_image.shape[:2]
        image_shape: tuple[int, ...] = original_image.shape

        effective_conf = confidence_threshold if confidence_threshold is not None else self.confidence_threshold

        tiling_applied = False
        slices_processed = 1

        # Fast Stream Path: YOLO only, max 640px, no tiling, no anomaly engine, no overlays
        if yolo_only:
            inference_start = time.perf_counter()
            stream_dets: list[Detection] = []

            if self._onnx_session is not None:
                # Direct C++ ONNX Inference for edge streaming (<25ms) with aspect-preserving letterboxing
                stream_dets = self._run_onnx_inference(original_image, effective_conf, target_size=(480, 480))
            else:
                scale, pad_left, pad_top = self._preprocessor.get_letterbox_params(image_shape[:2])
                preprocessed: np.ndarray = self._preprocessor.resize(original_image)
                try:
                    results = self._model.predict(
                        source=preprocessed,
                        device=self.device,
                        verbose=False,
                        conf=effective_conf,
                    )
                except Exception as exc:
                    logger.exception("YOLO stream inference failed: %s", exc)
                    raise RuntimeError(f"Stream inference failed: {exc}") from exc

                raw_dets = self._postprocessor.process(
                    results[0],
                    image_shape=image_shape,
                    letterbox_params=(scale, pad_left, pad_top),
                )
                for det in raw_dets:
                    c_thresh = self.class_thresholds.get(det.class_name.lower(), self.confidence_threshold)
                    if confidence_threshold is not None:
                        ratio = confidence_threshold / max(0.01, self.confidence_threshold)
                        c_thresh = min(0.95, max(0.10, c_thresh * ratio))
                    if det.confidence >= c_thresh:
                        det.source = "yolo"
                        det.severity = self._postprocessor.classify_severity(
                            det.confidence,
                            det.area_ratio,
                            class_name=det.class_name,
                            domain_trust="HIGH",
                            source=det.source,
                        )
                        stream_dets.append(det)

            # Hybrid Edge Stream: When ambient_boost is active (e.g. webcam or ambient lighting)
            # and YOLO detections are 0, run the ultra-fast morphological anomaly engine (~1.5ms)
            # to catch out-of-domain, ambient webcam, phone screen, or subtle structural flaws.
            if ambient_boost and len(stream_dets) == 0:
                try:
                    anomalies = self._anomaly_engine.detect_anomalies(original_image)
                    total_area = float(img_w * img_h)
                    for a in anomalies:
                        if a.anomaly_score >= effective_conf:
                            bx1, by1, bx2, by2 = a.bbox
                            x1 = max(0.0, float(bx1 * img_w))
                            y1 = max(0.0, float(by1 * img_h))
                            x2 = min(float(img_w), float(bx2 * img_w))
                            y2 = min(float(img_h), float(by2 * img_h))
                            area_ratio = float((x2 - x1) * (y2 - y1)) / total_area if total_area > 0 else 0.0
                            cname = a.defect_type_hint if a.defect_type_hint in DEFECT_CLASSES.values() else "patches"
                            sev = self._postprocessor.classify_severity(
                                a.anomaly_score,
                                area_ratio,
                                class_name=cname,
                                domain_trust="HIGH",
                                source="anomaly_engine",
                            )
                            cid = 2
                            for k, v in DEFECT_CLASSES.items():
                                if v == cname:
                                    cid = k
                                    break
                            det_item = Detection(
                                bbox=(x1, y1, x2, y2),
                                confidence=float(round(a.anomaly_score, 3)),
                                class_id=cid,
                                class_name=cname,
                                severity=sev,
                                area_ratio=area_ratio,
                                description=f"{sev.capitalize()} {cname} detected via structural anomaly analysis ({int(a.anomaly_score*100)}% confidence)",
                                source="anomaly_engine",
                            )
                            stream_dets.append(det_item)
                except Exception as anom_err:
                    logger.debug("Stream anomaly fallback skipped: %s", anom_err)

            inference_time_ms = (time.perf_counter() - inference_start) * 1000.0
            total_time_ms = (time.perf_counter() - total_start) * 1000.0
            summary = self._postprocessor.get_summary(stream_dets)
            return {
                "detections": [asdict(d) for d in stream_dets],
                "summary": summary,
                "annotated_image": None,
                "inference_time_ms": round(inference_time_ms, 2),
                "total_time_ms": round(total_time_ms, 2),
                "image_shape": image_shape,
                "tiling_applied": False,
                "slices_count": 1,
                "domain_trust": "HIGH",
                "domain_warning": None,
                "is_calibrated": False,
                "calibration_label": "uncalibrated (px)",
                "px_per_mm": None,
                "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
            }

        # Check longest side against tiling trigger
        max_dim = max(img_h, img_w)

        # 2. Check if high-resolution tiling should be applied at 1:1 native optical resolution
        if self.enable_tiling and max_dim >= self.tiling_min_dim:
            tiling_applied = True
            slices = self._tiler.slice_image_adaptive(original_image)
            slices_processed = len(slices)

            inference_start = time.perf_counter()
            slice_pairs: List[Tuple[ImageSlice, List[Dict[str, Any]]]] = []

            # Batch process slice patches
            if self._onnx_session is not None:
                for sl in slices:
                    slice_dets = self._run_onnx_inference(sl.image, effective_conf, target_size=(640, 640))
                    slice_pairs.append((sl, [asdict(d) for d in slice_dets]))
            else:
                # Preprocess each slice patch with aspect-ratio preserving letterbox
                preprocessed_patches = [self._preprocessor.resize(sl.image) for sl in slices]
                try:
                    batch_preds = self._model.predict(
                        source=preprocessed_patches,
                        device=self.device,
                        verbose=False,
                        conf=self.confidence_threshold,
                    )
                except Exception as exc:
                    logger.exception("YOLO batched slice inference failed: %s", exc)
                    raise RuntimeError(f"Tiled inference failed: {exc}") from exc

                for sl, pred in zip(slices, batch_preds):
                    scale_sl, pad_l_sl, pad_t_sl = self._preprocessor.get_letterbox_params(sl.image.shape[:2])
                    slice_dets = self._postprocessor.process(
                        pred,
                        image_shape=sl.image.shape,
                        letterbox_params=(scale_sl, pad_l_sl, pad_t_sl),
                    )
                    slice_pairs.append((sl, [asdict(d) for d in slice_dets]))

            inference_time_ms = (time.perf_counter() - inference_start) * 1000.0

            # Merge slice detections into global coordinate space using NMS (adds tile offsets)
            global_det_dicts = self._tiler.merge_detections(slice_pairs, image_shape=image_shape)

            # Reconstruct Detection dataclass list
            from backend.services.knowledge_service import get_defect_knowledge
            detections = []
            for d in global_det_dicts:
                k_info = get_defect_knowledge(d["class_name"]) or {}
                detections.append(
                    Detection(
                        bbox=d["bbox"],
                        confidence=d["confidence"],
                        class_id=d["class_id"],
                        class_name=d["class_name"],
                        severity=d["severity"],
                        area_ratio=d["area_ratio"],
                        description=d["description"],
                        source=d.get("source", "yolo"),
                        definition=d.get("definition") or k_info.get("definition"),
                        cause=d.get("cause") or k_info.get("cause"),
                        action=d.get("action") or k_info.get("action"),
                        visual_look=d.get("visual_look") or k_info.get("visual_look"),
                        risk=d.get("risk") or k_info.get("risk"),
                    )
                )

        else:
            # Standard single-pass inference
            inference_start = time.perf_counter()
            if self._onnx_session is not None:
                detections = self._run_onnx_inference(original_image, effective_conf, target_size=(640, 640))
            else:
                scale, pad_left, pad_top = self._preprocessor.get_letterbox_params(image_shape[:2])
                preprocessed: np.ndarray = self._preprocessor.resize(original_image)
                try:
                    results = self._model.predict(
                        source=preprocessed,
                        device=self.device,
                        verbose=False,
                        conf=self.confidence_threshold,
                    )
                except Exception as exc:
                    logger.exception("YOLO single-pass inference failed: %s", exc)
                    raise RuntimeError(f"Model inference failed: {exc}") from exc

                detections = self._postprocessor.process(
                    results[0],
                    image_shape=image_shape,
                    letterbox_params=(scale, pad_left, pad_top),
                )

            inference_time_ms = (time.perf_counter() - inference_start) * 1000.0

        # 3. Domain Guard Assessment & Out-of-Domain Calibration
        raw_yolo_confidences = [d.confidence for d in detections if getattr(d, "source", "yolo") == "yolo"]
        max_yolo_conf = max(raw_yolo_confidences) if raw_yolo_confidences else 0.0
        domain_assessment: DomainGuardAssessment = self._domain_guard.assess(
            original_image, max_yolo_confidence=max_yolo_conf
        )

        if deep_verify or domain_assessment.trust == "LOW":
            # Tier 2 Anomaly Engine (Blackhat morphological filters & structural fallback)
            anomaly_regions = self._anomaly_engine.detect_anomalies(original_image)
            calibrated_detections: list[Detection] = []

            for det in detections:
                c_thresh = self.class_thresholds.get(det.class_name.lower(), self.confidence_threshold)
                if det.confidence < c_thresh:
                    # If YOLO confidence is below class threshold and anomaly engine found region:
                    # label it "unclassified_anomaly"
                    if self._overlaps_anomaly(det.bbox, (img_h, img_w), anomaly_regions):
                        det.class_name = "unclassified_anomaly"
                        det.class_id = -1
                        det.source = "anomaly_cv"
                        det.description = f"Unclassified surface anomaly with {round(det.confidence * 100)}% confidence"
                        det.severity = self._postprocessor.classify_severity(
                            det.confidence,
                            det.area_ratio,
                            class_name=det.class_name,
                            domain_trust=domain_assessment.trust,
                            source=det.source,
                        )
                        calibrated_detections.append(det)
                    else:
                        continue
                else:
                    det.source = "yolo"
                    det.severity = self._postprocessor.classify_severity(
                        det.confidence,
                        det.area_ratio,
                        class_name=det.class_name,
                        domain_trust=domain_assessment.trust,
                        source=det.source,
                    )
                    calibrated_detections.append(det)

            detections = calibrated_detections

            # Extract structural anomalies from morphological gradient analysis
            structural_dets = self._detect_structural_anomalies(
                original_image,
                anomaly_regions=anomaly_regions,
                domain_trust=domain_assessment.trust,
            )

            # Class-Aware NMS merges overlapping YOLO and Anomaly detections
            detections = self._postprocessor.apply_class_aware_nms(
                calibrated_detections + structural_dets,
                iou_threshold=0.50,
            )
        else:
            # Standard fast Tier 1 path: Filter YOLO detections by calibrated class thresholds
            valid_dets: list[Detection] = []
            for det in detections:
                c_thresh = self.class_thresholds.get(det.class_name.lower(), self.confidence_threshold)
                if det.confidence >= c_thresh:
                    det.source = "yolo"
                    det.severity = self._postprocessor.classify_severity(
                        det.confidence,
                        det.area_ratio,
                        class_name=det.class_name,
                        domain_trust=domain_assessment.trust,
                        source=det.source,
                    )
                    valid_dets.append(det)
            detections = valid_dets

        # 6. Contour segmentation & physical metrology
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            norm_bbox = (
                max(0.0, float(x1 / img_w)),
                max(0.0, float(y1 / img_h)),
                min(1.0, float(x2 / img_w)),
                min(1.0, float(y2 / img_h)),
            )
            metrology = self._segmenter.segment_roi(original_image, norm_bbox)
            det.polygon = metrology.polygon
            det.metrology = asdict(metrology)

        # 5. Postprocessor summary & annotations
        summary: dict[str, Any] = self._postprocessor.get_summary(detections)
        annotated_image: np.ndarray = self.annotate_image(original_image, detections)
        total_time_ms: float = (time.perf_counter() - total_start) * 1000.0

        result: dict[str, Any] = {
            "detections": [asdict(det) for det in detections],
            "summary": summary,
            "annotated_image": annotated_image,
            "inference_time_ms": round(inference_time_ms, 2),
            "total_time_ms": round(total_time_ms, 2),
            "image_shape": image_shape,
            "tiling_applied": tiling_applied,
            "slices_count": slices_processed,
            "domain_trust": domain_assessment.trust,
            "domain_warning": domain_assessment.warning,
            "domain_metrics": {
                "color_saturation": domain_assessment.color_saturation,
                "texture_variance": domain_assessment.texture_variance,
                "max_yolo_confidence": domain_assessment.max_yolo_confidence,
            },
            "is_calibrated": self._segmenter.is_calibrated,
            "calibration_label": self._segmenter.calibration_label,
            "px_per_mm": self._segmenter.pixels_per_mm,
            "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        }

        logger.info(
            "Detection complete — %d defect(s) found in %.1f ms "
            "(inference: %.1f ms, tiling: %s, tiles: %d)",
            len(detections),
            total_time_ms,
            inference_time_ms,
            tiling_applied,
            slices_processed,
        )
        return result

    def detect_batch(
        self,
        sources: list[Union[str, Path, np.ndarray, bytes]],
    ) -> list[dict[str, Any]]:
        """Run detection on a batch of image sources sequentially."""
        results: list[dict[str, Any]] = []
        for idx, src in enumerate(sources):
            try:
                res = self.detect(src)
                results.append(res)
            except Exception as exc:
                logger.error("Error detecting image at index %d: %s", idx, exc)
                results.append({
                    "index": idx,
                    "error": str(exc),
                    "detections": [],
                    "summary": {"total_count": 0, "by_severity": {}, "by_class": {}, "has_critical": False},
                })
        return results

    def set_calibration(self, px_per_mm: Optional[float]) -> None:
        """Set or update optical scale (pixels per millimeter).

        Args:
            px_per_mm: Pixels per mm scale, or None to revert to uncalibrated pixel mode.
        """
        self.px_per_mm = float(px_per_mm) if (px_per_mm is not None and float(px_per_mm) > 0) else None
        self._segmenter.set_calibration(self.px_per_mm)
        logger.info("Detector calibration set to: %s (label=%s)", self.px_per_mm, self._segmenter.calibration_label)

    @staticmethod
    def _overlaps_anomaly(
        bbox: tuple[float, float, float, float],
        img_shape: tuple[int, int],
        anomaly_regions: list[AnomalyRegion],
        min_overlap: float = 0.10,
    ) -> bool:
        """Check if a detection bounding box overlaps with any detected anomaly region."""
        if not anomaly_regions:
            return False
        img_h, img_w = img_shape
        bx1, by1, bx2, by2 = bbox
        b_area = max(1.0, (bx2 - bx1) * (by2 - by1))

        for anom in anomaly_regions:
            ax1, ay1, ax2, ay2 = anom.bbox
            px1, py1, px2, py2 = ax1 * img_w, ay1 * img_h, ax2 * img_w, ay2 * img_h
            a_area = max(1.0, (px2 - px1) * (py2 - py1))

            ix1 = max(bx1, px1)
            iy1 = max(by1, py1)
            ix2 = min(bx2, px2)
            iy2 = min(by2, py2)

            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                overlap = inter_area / min(b_area, a_area)
                if overlap >= min_overlap:
                    return True
        return False

    def _detect_structural_anomalies(
        self,
        image: np.ndarray,
        anomaly_regions: list[AnomalyRegion] | None = None,
        domain_trust: str = "HIGH",
    ) -> list[Detection]:
        """Detect structural and textural surface flaws when supervised YOLO confidence is low.

        Uses the Unsupervised Anomaly Engine to isolate fractures, micro-cracks, surface pitting,
        and thermal burns across novel or unseen manufacturing materials.
        Labels out-of-domain flaws as 'unclassified_anomaly' with source 'anomaly_cv'.
        """
        img_h, img_w = image.shape[:2]
        if anomaly_regions is None:
            anomaly_regions = self._anomaly_engine.detect_anomalies(image, max_detections=6)

        detections: list[Detection] = []
        for anom in anomaly_regions:
            # Map normalized bounding box back to pixel coordinates
            bx1 = anom.bbox[0] * img_w
            by1 = anom.bbox[1] * img_h
            bx2 = anom.bbox[2] * img_w
            by2 = anom.bbox[3] * img_h

            bw = max(0.0, bx2 - bx1)
            bh = max(0.0, by2 - by1)
            box_area = bw * bh
            box_area_ratio = box_area / float(img_h * img_w)

            # Drop any box above 35% of the image area
            if box_area_ratio > 0.35:
                logger.warning(
                    "Dropping structural anomaly candidate [%.1f, %.1f, %.1f, %.1f]: area ratio %.2f%% exceeds 35%% image area threshold",
                    bx1, by1, bx2, by2, box_area_ratio * 100.0,
                )
                continue

            # Drop aspect ratio above 15:1 for classes other than scratches
            aspect = max(bw, bh) / max(min(bw, bh), 1e-6)
            hint = anom.defect_type_hint.lower()
            if aspect > 15.0 and hint != "scratches":
                logger.warning(
                    "Dropping structural anomaly candidate [%.1f, %.1f, %.1f, %.1f]: aspect ratio %.2f:1 exceeds 15:1 limit for class '%s'",
                    bx1, by1, bx2, by2, aspect, hint,
                )
                continue

            class_name = "unclassified_anomaly"
            class_id = -1

            # Assign industrial metallurgical severity adhering to Domain Guard routing rules
            area_ratio = float(anom.pixel_area / (img_h * img_w))
            severity = self._postprocessor.classify_severity(
                confidence=anom.anomaly_score,
                area_ratio=area_ratio,
                class_name=class_name,
                domain_trust=domain_trust,
                source="anomaly_cv",
            )

            from backend.services.knowledge_service import get_defect_knowledge
            k_info = get_defect_knowledge(class_name) or {}
            det = Detection(
                bbox=(float(bx1), float(by1), float(bx2), float(by2)),
                confidence=anom.anomaly_score,
                class_id=class_id,
                class_name=class_name,
                severity=severity,
                area_ratio=round(area_ratio, 4),
                description=f"Unclassified surface anomaly with {round(anom.anomaly_score * 100)}% confidence",
                polygon=anom.contour,
                source="anomaly_cv",
                definition=k_info.get("definition"),
                cause=k_info.get("cause"),
                action=k_info.get("action"),
                visual_look=k_info.get("visual_look"),
                risk=k_info.get("risk"),
            )
            detections.append(det)

        return detections

    def annotate_image(
        self,
        image: np.ndarray,
        detections: list[Detection],
    ) -> np.ndarray:
        """Draw bounding boxes, labels, and severity tags on an image copy."""
        annotated: np.ndarray = image.copy()
        img_h, img_w = annotated.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = (int(round(c)) for c in det.bbox)

            # Clamp coordinates to canvas bounds
            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(0, min(x2, img_w - 1))
            y2 = max(0, min(y2, img_h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            color: tuple[int, int, int] = _SEVERITY_COLORS.get(
                det.severity.lower(),
                _DEFAULT_COLOR,
            )

            # Draw translucent polygon mask if available
            if det.polygon and len(det.polygon) >= 3:
                annotated = self._segmenter.draw_polygon_overlay(
                    annotated,
                    det.polygon,
                    color=color,
                    alpha=0.35,
                )

            # Draw outer bounding box
            thickness = max(2, int(round(min(img_h, img_w) / 400.0)))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Format label
            if det.source == "anomaly_cv":
                display_title = (
                    "Unclassified surface anomaly"
                    if det.class_name == "unclassified_anomaly"
                    else det.class_name
                )
                label = f"Anomaly (CV): {display_title} {det.confidence * 100:.0f}%"
            elif det.source == "vlm":
                label = f"[VLM] {det.class_name.upper()} {det.confidence * 100:.0f}%"
            else:
                label = f"{det.class_name.upper()} {det.confidence * 100:.0f}% [{det.severity.upper()}]"
            font_scale = max(0.4, min(img_h, img_w) / 1200.0)
            font_thickness = max(1, int(thickness / 2))

            (text_w, text_h), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                font_thickness,
            )

            # Draw label background rectangle
            bg_y1 = max(0, y1 - text_h - baseline - 4)
            bg_y2 = y1
            bg_x1 = x1
            bg_x2 = min(img_w, x1 + text_w + 6)

            cv2.rectangle(
                annotated,
                (bg_x1, bg_y1),
                (bg_x2, bg_y2),
                color,
                cv2.FILLED,
            )

            # Draw text
            text_color = (0, 0, 0) if det.severity.lower() in ("minor", "major") else (255, 255, 255)
            cv2.putText(
                annotated,
                label,
                (bg_x1 + 3, bg_y2 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                font_thickness,
                lineType=cv2.LINE_AA,
            )

        return annotated

    def is_ready(self) -> bool:
        """Return True if detector is initialized and model is loaded (ONNX or PyTorch)."""
        return self._is_ready and (self._model is not None or self._onnx_session is not None)

    def get_model_info(self) -> dict[str, Any]:
        """Return detector configuration and model metadata."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "enable_tiling": self.enable_tiling,
            "slice_size": self.slice_size,
            "tiling_min_dim": self.tiling_min_dim,
            "is_ready": self.is_ready(),
        }
