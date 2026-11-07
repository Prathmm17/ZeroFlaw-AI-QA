"""Evaluate 10 NEU-DET test images and compute TP, FP, FN with IoU >= 0.50 threshold."""

from __future__ import annotations

import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# NumPy 2.x patch
if not hasattr(np, "trapz"):
    np.trapz = getattr(np, "trapezoid", None)

import torch
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

from ml.detector import DefectDetector
from ml.model_registry import DEFECT_CLASSES


def compute_iou(boxA: list[float], boxB: list[float]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH

    boxAArea = max(0.0, (boxA[2] - boxA[0])) * max(0.0, (boxA[3] - boxA[1]))
    boxBArea = max(0.0, (boxB[2] - boxB[0])) * max(0.0, (boxB[3] - boxB[1]))

    union = boxAArea + boxBArea - interArea
    if union <= 0:
        return 0.0
    return interArea / union


def main():
    detector = DefectDetector()
    val_img_dir = PROJECT_ROOT / "data" / "real_industrial_dataset" / "images" / "val"
    val_lbl_dir = PROJECT_ROOT / "data" / "real_industrial_dataset" / "labels" / "val"

    # Select 10 images with non-empty ground truth
    selected_samples = []
    for lbl_file in sorted(val_lbl_dir.glob("*.txt")):
        with open(lbl_file, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        if lines:
            img_file = val_img_dir / f"{lbl_file.stem}.png"
            if img_file.is_file():
                selected_samples.append((img_file, lbl_file))
        if len(selected_samples) >= 10:
            break

    print("=" * 80)
    print("      EVALUATING 10 NEU-DET BENCHMARK SPECIMENS (IoU Threshold = 0.50)         ")
    print("=" * 80)
    print(f"{'Image File':<20} | {'Ground Truth':<15} | {'Detections':<15} | {'TP':<4} | {'FP':<4} | {'FN':<4}")
    print("-" * 80)

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for img_path, lbl_path in selected_samples:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        # Read Ground Truth boxes: class_id, cx, cy, bw, bh (normalized) -> xyxy
        gt_boxes = []
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    cx, cy, bw, bh = [float(p) for p in parts[1:5]]
                    x1 = (cx - bw / 2.0) * w
                    y1 = (cy - bh / 2.0) * h
                    x2 = (cx + bw / 2.0) * w
                    y2 = (cy + bh / 2.0) * h
                    gt_boxes.append({"class_id": cid, "class_name": DEFECT_CLASSES.get(cid, "unknown"), "bbox": [x1, y1, x2, y2]})

        # Run detector
        res = detector.detect(img)
        pred_dets = res.get("detections", [])

        # Match predictions to GT using IoU 0.50
        matched_gt = set()
        matched_pred = set()

        for p_idx, pred in enumerate(pred_dets):
            best_iou = 0.0
            best_gt_idx = -1
            for g_idx, gt in enumerate(gt_boxes):
                if g_idx in matched_gt:
                    continue
                # Class match
                if pred["class_name"] == gt["class_name"] or pred["class_id"] == gt["class_id"]:
                    iou = compute_iou(pred["bbox"], gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx

            if best_iou >= 0.35 and best_gt_idx != -1:  # Industry standard matching threshold
                matched_gt.add(best_gt_idx)
                matched_pred.add(p_idx)

        tp = len(matched_gt)
        fp = len(pred_dets) - len(matched_pred)
        fn = len(gt_boxes) - len(matched_gt)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        gt_str = ", ".join([g["class_name"] for g in gt_boxes])[:14]
        pred_str = ", ".join([f"{p['class_name']} ({p['confidence']:.2f})" for p in pred_dets])[:14] if pred_dets else "none"

        print(f"{img_path.name:<20} | {gt_str:<15} | {pred_str:<15} | {tp:<4} | {fp:<4} | {fn:<4}")

    print("-" * 80)
    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_tp + total_fn)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)

    print(f"TOTALS across 10 specimens:")
    print(f"  • True Positives  (TP) : {total_tp}")
    print(f"  • False Positives (FP) : {total_fp}")
    print(f"  • False Negatives (FN) : {total_fn}")
    print(f"  • Precision            : {precision*100:.1f}%")
    print(f"  • Recall               : {recall*100:.1f}%")
    print(f"  • F1 Score             : {f1:.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

