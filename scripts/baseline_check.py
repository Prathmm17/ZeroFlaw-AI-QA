import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# NumPy 2.x patch
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
from ml.anomaly_engine import UnsupervisedAnomalyEngine

def main():
    detector = DefectDetector()
    engine = UnsupervisedAnomalyEngine(sensitivity=0.70)
    files = [
        ("cast part", "data/uploads/WhatsApp Image 2026-10-30 at 6.26.03 AM_20261030_011330_6eded82a_orig.jpg"),
        ("pipe", "data/uploads/WhatsApp Image 2026-10-30 at 6.26.04 AM_20261030_011449_d82a1133_orig.jpg"),
    ]
    for tag, path in files:
        img = cv2.imread(path)
        h, w = img.shape[:2]
        print(f"=== {tag.upper()} ({w}x{h}) ===")
        # Anomaly engine directly
        anoms = engine.detect_anomalies(img)
        print(f"Direct Anomaly Engine detected {len(anoms)} anomalies:")
        for idx, a in enumerate(anoms):
            ax1, ay1, ax2, ay2 = a.bbox
            bw = (ax2 - ax1) * w
            bh = (ay2 - ay1) * h
            area_pct = (bw * bh) / (w * h) * 100
            aspect = max(bw, bh) / max(min(bw, bh), 1e-6)
            print(f"  Anomaly #{idx+1}: type={a.defect_type_hint}, area_ratio={area_pct:.2f}%, aspect={aspect:.2f}, score={a.anomaly_score:.2f}")

        # Full detector pipeline
        res = detector.detect(path)
        dets = res["detections"]
        print(f"Full Detector pipeline produced {len(dets)} detections:")
        for idx, d in enumerate(dets):
            bx = d["bbox"]
            bw = bx[2] - bx[0]
            bh = bx[3] - bx[1]
            area_pct = (bw * bh) / (w * h) * 100
            aspect = max(bw, bh) / max(min(bw, bh), 1e-6)
            print(f"  Det #{idx+1}: class={d['class_name']}, source={d['source']}, conf={d['confidence']:.2f}, area_pct={area_pct:.2f}%, aspect={aspect:.2f}")

if __name__ == "__main__":
    main()
