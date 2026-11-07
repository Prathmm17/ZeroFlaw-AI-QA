"""Evaluation script for false box suppression on real industrial photos.

Tests on:
1. Cast part with cavitation pitting (WhatsApp Image 2026-10-30 at 6.26.03 AM...)
2. Metal pipe with previous 54.22% false box (WhatsApp Image 2026-10-30 at 6.26.04 AM...)

Reports:
- Before/After box count
- Before/After largest box area %
- Verification that no box exceeds 35% area and non-scratch aspect ratios are <= 15:1.
"""

from __future__ import annotations

import sys
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


def evaluate_photos() -> None:
    detector = DefectDetector()

    # Baseline measurements from before the fix (recorded in inspection audit / baseline check)
    baselines = {
        "cast_part": {
            "name": "Cast Part (Cavitation Pitting)",
            "file": "data/uploads/WhatsApp Image 2026-10-30 at 6.26.03 AM_20261030_011330_6eded82a_orig.jpg",
            "before_count": 6,
            "before_largest_area_pct": 24.03,
        },
        "pipe": {
            "name": "Metal Pipe (Shadow & Surface Reflection)",
            "file": "data/uploads/WhatsApp Image 2026-10-30 at 6.26.04 AM_20261030_011449_d82a1133_orig.jpg",
            "before_count": 6,
            "before_largest_area_pct": 54.22,
        },
    }

    print("================================================================================")
    print("           VISIONGUARD QA - FALSE BOX SUPPRESSION EVALUATION REPORT             ")
    print("================================================================================\n")

    summary_rows = []

    for key, info in baselines.items():
        path = PROJECT_ROOT / info["file"]
        if not path.exists():
            print(f"Error: Specimen file not found: {path}")
            continue

        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        img_area = float(h * w)

        print(f"--- SPECIMEN: {info['name']} ---")
        print(f"File: {path.name}")
        print(f"Dimensions: {w} x {h} px ({int(img_area):,} total pixels)")

        res = detector.detect(path)
        dets = res["detections"]

        after_count = len(dets)
        after_areas = []
        after_aspects = []

        print(f"\nAfter Fix Detections ({after_count} kept):")
        for idx, d in enumerate(dets):
            bx1, by1, bx2, by2 = d["bbox"]
            bw = max(0.0, bx2 - bx1)
            bh = max(0.0, by2 - by1)
            area_pct = (bw * bh) / img_area * 100.0
            aspect = max(bw, bh) / max(min(bw, bh), 1e-6)
            after_areas.append(area_pct)
            after_aspects.append((d["class_name"], aspect))

            print(
                f"  [{idx+1}] Class: {d['class_name']:<22} Source: {d['source']:<12} "
                f"Conf: {d['confidence']:.2f}  Area: {area_pct:5.2f}%  Aspect: {aspect:5.2f}:1  "
                f"BBox: [{round(bx1, 1)}, {round(by1, 1)}, {round(bx2, 1)}, {round(by2, 1)}]"
            )

            # Assert constraints
            assert area_pct <= 35.0, f"Box area {area_pct:.2f}% exceeds 35% threshold!"
            if d["class_name"].lower() != "scratches":
                assert aspect <= 15.0, f"Aspect ratio {aspect:.2f}:1 exceeds 15:1 limit for {d['class_name']}!"

        after_largest_area_pct = max(after_areas) if after_areas else 0.0

        summary_rows.append({
            "name": info["name"],
            "before_count": info["before_count"],
            "after_count": after_count,
            "before_largest_area_pct": info["before_largest_area_pct"],
            "after_largest_area_pct": after_largest_area_pct,
            "status": "PASSED (all <= 35% area & <= 15:1 aspect)",
        })
        print()

    print("================================================================================")
    print("                         BEFORE / AFTER COMPARISON TABLE                        ")
    print("================================================================================")
    print(
        f"{'Specimen':<38} | {'Boxes (B -> A)':<16} | {'Max Area % (B -> A)':<22} | {'Verification':<10}"
    )
    print("-" * 92)
    for row in summary_rows:
        count_str = f"{row['before_count']} -> {row['after_count']}"
        area_str = f"{row['before_largest_area_pct']:.2f}% -> {row['after_largest_area_pct']:.2f}%"
        print(f"{row['name']:<38} | {count_str:<16} | {area_str:<22} | {row['status']}")
    print("================================================================================\n")


if __name__ == "__main__":
    evaluate_photos()

