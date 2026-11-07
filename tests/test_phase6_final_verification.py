"""Phase 6: End-to-End Stress Testing, Multi-Domain Benchmark Verification & Final Sign-Off.

Verification Gates:
1. Multi-Domain Industrial Validation Split (120 images):
   - mAP50 >= 96.0% (Automotive benchmark).
   - Precision >= 95.0% and Recall >= 93.0%.
2. Clean Master & Defect Separation:
   - Clean master baseline (specimen_9_clean_master.png): 0 false positives (100.0% precision).
   - Defect specimens: 0 false escapes on true flaw specimens.
3. High-Resolution Microscopic Defect Localization:
   - 2048x2048 specimen with micro-fissure: successfully localized via SAHI tiling.
4. Real-Time Stream Performance:
   - Edge latency <= 30.0ms.
   - Sustained throughput >= 20 FPS (exceeding 15+ FPS gate).
5. Full System Health & Zero Uncaught Exceptions:
   - System health status healthy, database initialized, all API schemas verified.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import pytest

# NumPy 2.x patch
if not hasattr(np, "trapz"):
    np.trapz = getattr(np, "trapezoid", None)

import torch

try:
    _orig_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    torch.load = _patched_load
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from backend.database import SessionLocal
from backend.routers.analytics import get_analytics_overview
from backend.routers.detection import get_inspection_history, list_sample_images
from backend.routers.stream import _run_stream_yolo_inference
from backend.services.analytics_service import AnalyticsService
from ml.arbitrator import ProductionArbitrator
from ml.detector import DefectDetector


@pytest.fixture(scope="module")
def edge_detector() -> DefectDetector:
    """Fixture providing initialized DefectDetector with ONNX engine."""
    return DefectDetector(model_name="best.onnx", enable_tiling=True)


@pytest.fixture(scope="module")
def arbitrator(edge_detector: DefectDetector) -> ProductionArbitrator:
    """Fixture providing ProductionArbitrator."""
    return ProductionArbitrator(edge_detector=edge_detector)


# ---------------------------------------------------------------------------
# Test 1: Multi-Domain Industrial Benchmark Split (120 Images)
# ---------------------------------------------------------------------------
def test_multidomain_industrial_benchmark_map50() -> None:
    """Verify YOLO model achieves mAP50 >= 96% and Recall >= 93% on 120-image val split."""
    from ultralytics import YOLO

    dataset_yaml = PROJECT_ROOT / "data" / "real_industrial_dataset" / "dataset.yaml"
    weights_path = PROJECT_ROOT / "models" / "best.pt"

    assert dataset_yaml.is_file(), f"Dataset config missing at {dataset_yaml}"
    assert weights_path.is_file(), f"Model weights missing at {weights_path}"

    model = YOLO(str(weights_path))
    val_results = model.val(
        data=str(dataset_yaml).replace("\\", "/"),
        plots=False,
        verbose=False,
        device="cpu",
        project=str(PROJECT_ROOT / "runs" / "val_eval"),
        name="industrial_val",
        exist_ok=True,
    )

    map50 = float(val_results.box.map50)
    precision = float(val_results.box.mp)
    recall = float(val_results.box.mr)

    # Verification criteria from master plan
    assert map50 >= 0.96, f"mAP50 ({map50*100:.2f}%) fell below 96.0% target"
    assert recall >= 0.93, f"Recall ({recall*100:.2f}%) fell below 93.0% target"
    assert precision >= 0.95, f"Precision ({precision*100:.2f}%) fell below 95.0% target"


# ---------------------------------------------------------------------------
# Test 2: Clean Master Specimen Zero False-Alarm Baseline
# ---------------------------------------------------------------------------
def test_clean_master_specimen_zero_false_positives(edge_detector: DefectDetector) -> None:
    """Verify clean master part (specimen_9_clean_master.png) has 0 defects (100% precision)."""
    clean_path = PROJECT_ROOT / "data" / "factory_benchmark" / "specimen_9_clean_master.png"
    assert clean_path.is_file()

    img = cv2.imread(str(clean_path))
    assert img is not None

    result = edge_detector.detect(img)
    detections = result.get("detections", [])

    # Must be completely defect free
    assert len(detections) == 0, f"False positive detected on clean master specimen: {detections}"


# ---------------------------------------------------------------------------
# Test 3: Defect Specimen Recall & Zero False Escapes
# ---------------------------------------------------------------------------
def test_defect_specimen_zero_false_escapes(edge_detector: DefectDetector) -> None:
    """Verify confirmed defect specimens trigger detection without false escapes."""
    fdir = PROJECT_ROOT / "data" / "factory_benchmark"
    test_cases = [
        ("specimen_10_multi_weld.png", "multi-weld flaw"),
        ("specimen_3_crazing.png", "crazing micro-fractures"),
        ("specimen_4_inclusion.png", "slag/oxide inclusion"),
    ]

    for fname, desc in test_cases:
        p = fdir / fname
        assert p.is_file()
        img = cv2.imread(str(p))
        assert img is not None

        res = edge_detector.detect(img)
        dets = res.get("detections", [])
        assert len(dets) >= 1, f"False escape! Failed to detect {desc} in {fname}"


# ---------------------------------------------------------------------------
# Test 4: High-Resolution 2048 Microscopic Defect Localization
# ---------------------------------------------------------------------------
def test_high_res_microscopic_defect_localization(edge_detector: DefectDetector) -> None:
    """Verify SAHI tiling detects micro-defects on high-resolution (2048x2048) specimen."""
    hairline_path = PROJECT_ROOT / "data" / "samples" / "benchmark_2048_hairline.png"
    assert hairline_path.is_file()

    img = cv2.imread(str(hairline_path))
    assert img is not None
    assert img.shape[:2] == (2048, 2048)

    res = edge_detector.detect(img)
    dets = res.get("detections", [])
    assert len(dets) >= 1, "Microscopic hairline crack in 2048 specimen was missed!"


# ---------------------------------------------------------------------------
# Test 5: Real-Time Stream Edge Latency (<= 30ms) & Throughput (>= 20 FPS)
# ---------------------------------------------------------------------------
def test_stream_edge_latency_and_throughput(edge_detector: DefectDetector) -> None:
    """Verify stream inference maintains average latency <= 30.0ms and throughput >= 20 FPS."""
    # Synthesize standard 640x480 binary frame
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    cv2.circle(frame, (320, 240), 15, (40, 40, 40), -1)
    _, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
    frame_bytes = encoded.tobytes()

    # Warmup
    _run_stream_yolo_inference(frame_bytes, edge_detector)

    latencies: List[float] = []
    n_frames = 50
    t_start = time.perf_counter()

    for _ in range(n_frames):
        res = _run_stream_yolo_inference(frame_bytes, edge_detector)
        latencies.append(res["inference_time_ms"])

    total_time = time.perf_counter() - t_start
    mean_lat_ms = float(np.mean(latencies))
    fps = n_frames / total_time

    assert mean_lat_ms <= 30.0, f"Average stream latency ({mean_lat_ms:.2f}ms) exceeded 30.0ms gate"
    assert fps >= 20.0, f"Streaming throughput ({fps:.1f} FPS) fell below 20.0 FPS gate"


# ---------------------------------------------------------------------------
# Test 6: Full System Integration & Database Integrity
# ---------------------------------------------------------------------------
def test_system_health_and_analytics_integrity() -> None:
    """Verify database and analytics aggregation endpoints respond with zero errors."""
    db = SessionLocal()
    try:
        overview = AnalyticsService.get_overview(db=db)
        assert overview.total_inspections >= 0
        assert overview.defect_rate_percentage >= 0.0
        assert overview.cp is not None
        assert overview.cpk is not None
        assert overview.process_capability_status is not None

        samples = list_sample_images()
        assert isinstance(samples, list)
        assert len(samples) > 0

        history = get_inspection_history(page=1, page_size=10, db=db)
        assert history.page == 1
        assert history.total >= 0
    finally:
        db.close()
