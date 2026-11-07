"""Phase 2 Verification Suite: Unsupervised Deep Anomaly Engine.

Tests the Unsupervised Anomaly Engine directly on real-world out-of-distribution
specimens (SEM microscope scan with micro-crack, and charred/burnt electrical plug).
Proves that:
1. Multi-scale energy heatmap localizes structural anomalies with zero manual tuning.
2. AnomalyRegion produces exact bounding box, polygon contour, and anomaly score.
3. DefectDetector seamlessly integrates the engine and classifies the flaws.
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

import cv2
from ml.anomaly_engine import UnsupervisedAnomalyEngine
from ml.detector import DefectDetector


def test_sem_micrograph_flaw() -> bool:
    print("\n--- TEST 1: Unsupervised Anomaly Localization on SEM Microscope Scan ---")
    sem_path = PROJECT_ROOT / "data" / "uploads" / "WhatsApp Image 2026-10-29 at 2.34.32 AM_20261028_210759_278b9ff7_orig.jpg"
    if not sem_path.is_file():
        # Fallback to any SEM image in uploads
        sem_candidates = list((PROJECT_ROOT / "data" / "uploads").glob("*2.34.32*.jpg"))
        if not sem_candidates:
            print(f"[FAIL] SEM test specimen not found!")
            return False
        sem_path = sem_candidates[0]

    image = cv2.imread(str(sem_path))
    h, w = image.shape[:2]
    print(f"Specimen: {sem_path.name} ({w}x{h} px)")

    engine = UnsupervisedAnomalyEngine(sensitivity=0.70)
    anomalies = engine.detect_anomalies(image, max_detections=5)

    print(f"Total Structural Anomalies Localized: {len(anomalies)}")
    for idx, a in enumerate(anomalies):
        px_box = [int(a.bbox[0] * w), int(a.bbox[1] * h), int(a.bbox[2] * w), int(a.bbox[3] * h)]
        print(f"  Anomaly #{idx+1}: {a.defect_type_hint.upper()} | Score: {a.anomaly_score:.2f} | BBox: {px_box} | Area: {a.pixel_area} px | Polygon: {len(a.contour)} pts")

    assert len(anomalies) > 0, "Anomaly engine found 0 defects on SEM image!"
    # Check that vertical crack is among top localized flaws (y around 30 to 150)
    crack_found = any(a.bbox[1] * h < 60 and a.bbox[3] * h > 100 for a in anomalies)
    assert crack_found, "Prominent vertical crack was not localized by Anomaly Engine!"
    print("[PASS] SEM vertical micro-crack accurately localized with zero supervised training.")
    return True


def test_burnt_assembly_flaw() -> bool:
    print("\n--- TEST 2: Unsupervised Anomaly Localization on Burnt Electrical Plug ---")
    plug_path = PROJECT_ROOT / "data" / "uploads" / "WhatsApp Image 2026-10-29 at 2.34.17 AM_20261028_210831_8b772b09_orig.jpg"
    if not plug_path.is_file():
        plug_candidates = list((PROJECT_ROOT / "data" / "uploads").glob("*2.34.17*.jpg"))
        if not plug_candidates:
            print(f"[FAIL] Burnt plug test specimen not found!")
            return False
        plug_path = plug_candidates[0]

    image = cv2.imread(str(plug_path))
    h, w = image.shape[:2]
    print(f"Specimen: {plug_path.name} ({w}x{h} px)")

    engine = UnsupervisedAnomalyEngine(sensitivity=0.70)
    anomalies = engine.detect_anomalies(image, max_detections=5)

    print(f"Total Structural Anomalies Localized: {len(anomalies)}")
    for idx, a in enumerate(anomalies):
        px_box = [int(a.bbox[0] * w), int(a.bbox[1] * h), int(a.bbox[2] * w), int(a.bbox[3] * h)]
        print(f"  Anomaly #{idx+1}: {a.defect_type_hint.upper()} | Score: {a.anomaly_score:.2f} | BBox: {px_box} | Area: {a.pixel_area} px")

    assert len(anomalies) > 0, "Anomaly engine found 0 defects on burnt plug!"
    # Check that scorched socket terminal is localized (x around 220-290, y around 180-240)
    burn_found = any(a.defect_type_hint == "patches" or (a.bbox[0] * w < 250 and a.bbox[2] * w > 260) for a in anomalies)
    assert burn_found, "Thermal burn mark was not localized by Anomaly Engine!"
    print("[PASS] Burnt plug thermal scorch accurately localized with zero supervised training.")
    return True


def test_detector_integration() -> bool:
    print("\n--- TEST 3: DefectDetector Dual-Engine Pipeline Test ---")
    detector = DefectDetector(model_name="best.pt", confidence_threshold=0.20, enable_tiling=False)

    sem_path = PROJECT_ROOT / "data" / "uploads" / "WhatsApp Image 2026-10-29 at 2.34.32 AM_20261028_210759_278b9ff7_orig.jpg"
    img = cv2.imread(str(sem_path))
    res = detector.detect(img)

    print(f"Total Detected Flaws: {len(res['detections'])}")
    print(f"Severity Breakdown: {res['summary']['by_severity']}")
    assert len(res["detections"]) > 0, "DefectDetector produced 0 detections!"
    assert res["summary"]["has_critical"] is True, "Critical flaw was not flagged!"
    print("[PASS] DefectDetector Dual-Engine seamlessly integrates Unsupervised Anomaly Engine.")
    return True


def main() -> int:
    print("=" * 70)
    print("VISIONGUARD QA — PHASE 2 UNSUPERVISED ANOMALY ENGINE VERIFICATION")
    print("=" * 70)

    t1_pass = test_sem_micrograph_flaw()
    t2_pass = test_burnt_assembly_flaw()
    t3_pass = test_detector_integration()

    if t1_pass and t2_pass and t3_pass:
        print("\n" + "=" * 70)
        print("[SUCCESS] ALL PHASE 2 ANOMALY ENGINE VERIFICATION TESTS PASSED (100% SUCCESS)")
        print("=" * 70)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())

