"""Phase 2 Verification Script: Industrial Defect Weights.

Tests real inference on industrial validation samples with fine-tuned
industrial_best.pt weights to ensure genuine metallurgical defect classification
(crazing, inclusion, patches, pitted_surface, rolled_in_scale, scratches)
with zero unknown_XX COCO artifacts.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# NumPy 2.x patch
import numpy as np
if not hasattr(np, "trapz"):
    np.trapz = getattr(np, "trapezoid", None)

import cv2
from ml.detector import DefectDetector
from ml.model_registry import DEFECT_CLASSES, ModelRegistry
from ml.postprocessor import DetectionPostprocessor


def run_industrial_verification() -> bool:
    print("=" * 70)
    print("VISIONGUARD QA — PHASE 2 INDUSTRIAL WEIGHTS VERIFICATION")
    print("=" * 70)

    # 1. Verify weights file exists
    weights_path = PROJECT_ROOT / "models" / "industrial_best.pt"
    if not weights_path.is_file():
        weights_path = PROJECT_ROOT / "models" / "best.pt"

    if not weights_path.is_file():
        print(f"[FAIL] Industrial model weights not found at {weights_path}")
        return False

    print(f"[OK] Found industrial weights at: {weights_path} ({weights_path.stat().st_size / (1024*1024):.2f} MB)")

    # 2. Verify registered classes
    print(f"[INFO] Registered Defect Classes ({len(DEFECT_CLASSES)}): {DEFECT_CLASSES}")
    expected_classes = {"crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches"}
    assert set(DEFECT_CLASSES.values()) == expected_classes, f"Mismatch in defect classes: {DEFECT_CLASSES}"
    print("[OK] Defect classes match NEU-DET benchmark specifications.")

    # 3. Initialize detector
    detector = DefectDetector(
        model_name="industrial_best.pt",
        confidence_threshold=0.05,
        enable_tiling=False,
    )
    postprocessor = DetectionPostprocessor()
    print("[OK] DefectDetector successfully initialized with industrial weights.")

    # 4. Evaluate on validation benchmark specimens
    val_dir = PROJECT_ROOT / "data" / "industrial_benchmark" / "images" / "val"
    if not val_dir.is_dir():
        print(f"[FAIL] Validation image directory not found at {val_dir}")
        return False

    val_images = list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.png"))
    print(f"[INFO] Found {len(val_images)} validation specimens in {val_dir}")

    total_detections = 0
    unknown_detections = 0
    class_detection_counts = {c: 0 for c in expected_classes}
    latencies = []

    for img_path in val_images:
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        result = detector.detect(image)
        latencies.append(result["inference_time_ms"])

        for det in result["detections"]:
            total_detections += 1
            c_name = det["class_name"]
            if c_name.startswith("unknown_") or c_name not in expected_classes:
                unknown_detections += 1
                print(f"  [WARN] Unexpected class: {c_name} (conf: {det['confidence']:.2f})")
            else:
                class_detection_counts[c_name] += 1

    avg_latency = sum(latencies) / max(len(latencies), 1)
    print("\n" + "-" * 70)
    print("DETECTION BENCHMARK RESULTS:")
    print(f"  Total Validation Images Tested : {len(val_images)}")
    print(f"  Total Defect Detections Found  : {total_detections}")
    print(f"  Average Inference Latency      : {avg_latency:.2f} ms")
    print(f"  Unknown / COCO Leaks           : {unknown_detections}")
    print("  Per-Class Detections:")
    for c_name, count in class_detection_counts.items():
        print(f"    - {c_name:<18}: {count} hits")
    print("-" * 70)

    # 5. Assertions for Phase 2 pass
    if unknown_detections > 0:
        print(f"[FAIL] Found {unknown_detections} non-industrial or COCO labels!")
        return False

    if total_detections == 0:
        print("[FAIL] Model produced 0 detections on validation set!")
        return False

    print("\n[SUCCESS] Phase 2 Verification PASSED: Industrial weights loaded, defect classes confirmed, 0 unknown labels.")
    return True


if __name__ == "__main__":
    success = run_industrial_verification()
    sys.exit(0 if success else 1)

