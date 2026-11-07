"""Benchmark SAHI native-resolution tiling ON vs OFF on a real industrial photograph.

Evaluates:
- Detections count
- Processing latency (Inference & Total)
- Slices processed
- Largest box area %
- Phone-style photo downscaling
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# NumPy 2.x patch
import numpy as np
if not hasattr(np, "trapz"):
    np.trapz = getattr(np, "trapezoid", None)

import torch
# PyTorch 2.6 weights_only patch
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

import cv2
from ml.detector import DefectDetector


def run_benchmark() -> None:
    test_photo = PROJECT_ROOT / "data" / "uploads" / "WhatsApp Image 2026-10-30 at 6.26.03 AM_20261030_011330_6eded82a_orig.jpg"
    if not test_photo.exists():
        print(f"Error: Specimen photo not found at {test_photo}")
        return

    img = cv2.imread(str(test_photo))
    h, w = img.shape[:2]
    img_area = float(h * w)

    print("================================================================================")
    print("           VISIONGUARD QA - SAHI TILING BENCHMARK EVALUATION                    ")
    print("================================================================================")
    print(f"Specimen: Cast Part with Cavitation Pitting ({test_photo.name[:35]}...)")
    print(f"Native Resolution: {w} x {h} px ({int(img_area):,} pixels)\n")

    # 1. Tiling OFF: Single-pass letterbox inference (640x640)
    detector_no_tiling = DefectDetector(
        enable_tiling=False,
    )
    # Warmup
    _ = detector_no_tiling.detect(img)
    t0 = time.perf_counter()
    res_no_tiling = detector_no_tiling.detect(img)
    lat_no_tiling = (time.perf_counter() - t0) * 1000.0

    dets_no_tiling = res_no_tiling["detections"]
    areas_no_tiling = [
        ((d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1])) / img_area * 100.0
        for d in dets_no_tiling
    ]
    max_area_no_tiling = max(areas_no_tiling) if areas_no_tiling else 0.0

    # 2. Tiling ON: SAHI multi-tile native resolution slicing (slice=512, trigger=1000)
    detector_tiled = DefectDetector(
        enable_tiling=True,
        tiling_min_dim=1000,
        slice_size=512,
        overlap_ratio=0.20,
    )
    # Warmup
    _ = detector_tiled.detect(img)
    t1 = time.perf_counter()
    res_tiled = detector_tiled.detect(img)
    lat_tiled = (time.perf_counter() - t1) * 1000.0

    dets_tiled = res_tiled["detections"]
    areas_tiled = [
        ((d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1])) / img_area * 100.0
        for d in dets_tiled
    ]
    max_area_tiled = max(areas_tiled) if areas_tiled else 0.0

    print("--------------------------------------------------------------------------------")
    print("                             BENCHMARK RESULTS TABLE                            ")
    print("--------------------------------------------------------------------------------")
    print(
        f"{'Pipeline Mode':<20} | {'Tiles':<6} | {'Detections':<11} | {'Inference (ms)':<15} | {'Total (ms)':<11} | {'Max Area %':<10}"
    )
    print("-" * 88)
    print(
        f"{'Tiling OFF (Single)':<20} | {res_no_tiling['slices_count']:<6} | {len(dets_no_tiling):<11} | "
        f"{res_no_tiling['inference_time_ms']:<15.1f} | {res_no_tiling['total_time_ms']:<11.1f} | {max_area_no_tiling:<10.2f}%"
    )
    print(
        f"{'Tiling ON (SAHI)':<20} | {res_tiled['slices_count']:<6} | {len(dets_tiled):<11} | "
        f"{res_tiled['inference_time_ms']:<15.1f} | {res_tiled['total_time_ms']:<11.1f} | {max_area_tiled:<10.2f}%"
    )
    print("--------------------------------------------------------------------------------")
    print("Notes:")
    print(f"- Tiling trigger configurable via Settings.TILING_MIN_DIM (default {detector_no_tiling.tiling_min_dim}px).")
    print(f"- Phone photo downscaling active: longest side capped at {detector_no_tiling.tiling_min_dim}px when > trigger.")
    print("================================================================================\n")


if __name__ == "__main__":
    run_benchmark()

