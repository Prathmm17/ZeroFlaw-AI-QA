"""Run real inference on an industrial sample and render high-fidelity visualization with defect knowledge."""

from __future__ import annotations

import json
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

from backend.config import settings
from backend.services.knowledge_service import get_all_defect_knowledge, get_defect_knowledge
from ml.detector import DefectDetector


def main():
    print("=" * 70)
    print("VISIONGUARD QA — DEFECT KNOWLEDGE INFERENCE & VISUALIZATION")
    print("=" * 70)

    # 1. Verify Defect Knowledge Base
    all_knowledge = get_all_defect_knowledge()
    print(f"Loaded {len(all_knowledge)} defect knowledge entries:")
    for k, v in all_knowledge.items():
        print(f"  • {v['display_name']} ({v['confidence_label']}) -> Def Severity: {v['default_severity'].upper()}")
    print("-" * 70)

    # 2. Pick a real industrial specimen
    sample_path = PROJECT_ROOT / "data" / "real_industrial_dataset" / "images" / "val" / "ind_val_0481.png"
    if not sample_path.is_file():
        # Fallback to any val image
        val_imgs = list((PROJECT_ROOT / "data" / "real_industrial_dataset" / "images" / "val").glob("*.png"))
        sample_path = val_imgs[0] if val_imgs else PROJECT_ROOT / "data" / "samples" / "sample_scratch_001.png"

    print(f"Testing real inference on: {sample_path.name}")
    img_bgr = cv2.imread(str(sample_path))
    assert img_bgr is not None, f"Could not load {sample_path}"
    h, w = img_bgr.shape[:2]
    print(f"Image Resolution: {w}x{h} px")

    # 3. Initialize DefectDetector
    detector = DefectDetector(confidence_threshold=0.20)
    det_result = detector.detect(img_bgr)

    detections = det_result.get("detections", [])
    print(f"Total Detections Found: {len(detections)}")
    print(f"Inference Latency: {det_result.get('inference_time_ms', 0):.1f} ms")
    print(f"Domain Trust: {det_result.get('domain_trust', 'HIGH')}")
    print("-" * 70)

    if not detections:
        # If no detections at 0.20, pick another image or run on crazing / inclusion sample
        for alt_file in (PROJECT_ROOT / "data" / "real_industrial_dataset" / "images" / "val").glob("*.png"):
            alt_bgr = cv2.imread(str(alt_file))
            alt_res = detector.detect(alt_bgr)
            if alt_res.get("detections"):
                sample_path = alt_file
                img_bgr = alt_bgr
                det_result = alt_res
                detections = alt_res["detections"]
                h, w = img_bgr.shape[:2]
                print(f"Found {len(detections)} detection(s) on alternative sample: {sample_path.name}")
                break

    for i, d in enumerate(detections):
        conf_pct = d['confidence'] * 100
        print(f"\n[Detection #{i+1}]: {d['class_name'].upper()} ({conf_pct:.1f}% conf)")
        print(f"  • Source: {d.get('source', 'yolo')}")
        print(f"  • Severity: {d.get('severity', 'minor').upper()}")
        print(f"  • Bounding Box: {d['bbox']}")
        print(f"  • Definition: {d.get('definition', 'N/A')}")
        print(f"  • Typical Cause: {d.get('cause', 'N/A')}")
        print(f"  • Recommended Action: {d.get('action', 'N/A')}")
        print(f"  • Visual Appearance: {d.get('visual_look', 'N/A')}")
        print(f"  • Failure Risk: {d.get('risk', 'N/A')}")

    # 4. Render High-Fidelity UI Screenshot with DefectViewer highlight simulation
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "defect_knowledge_sample.png"

    # Create composite canvas (1280x720) with Dark Industrial Theme
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    canvas[:] = (22, 13, 9)  # #090d16 industrial-950

    # Draw header bar
    cv2.rectangle(canvas, (0, 0), (1280, 60), (42, 23, 15), -1)  # #0f172a
    cv2.line(canvas, (0, 60), (1280, 60), (85, 65, 51), 1)     # border-slate-800
    cv2.putText(canvas, "VISIONGUARD QA | Defect Inspection & Knowledge Diagnostic", (24, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "ACTIVE MODEL: best.pt  |  TIER 1 EDGE GATE  |  DOMAIN TRUST: HIGH", (780, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (212, 182, 6), 1, cv2.LINE_AA)

    # Left: Image Viewport (640x580)
    img_vp_x, img_vp_y, img_vp_w, img_vp_h = 24, 80, 640, 600
    cv2.rectangle(canvas, (img_vp_x, img_vp_y), (img_vp_x + img_vp_w, img_vp_y + img_vp_h), (25, 20, 15), -1)
    cv2.rectangle(canvas, (img_vp_x, img_vp_y), (img_vp_x + img_vp_w, img_vp_y + img_vp_h), (70, 55, 40), 1)

    # Scale specimen image into viewport
    scale = min((img_vp_w - 20) / w, (img_vp_h - 20) / h)
    disp_w, disp_h = int(w * scale), int(h * scale)
    disp_img = cv2.resize(img_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    off_x = img_vp_x + (img_vp_w - disp_w) // 2
    off_y = img_vp_y + (img_vp_h - disp_h) // 2
    canvas[off_y:off_y + disp_h, off_x:off_x + disp_w] = disp_img

    # Highlight primary defect with Cyan Neon Glow and corner brackets
    if detections:
        prim = detections[0]
        bx1, by1, bx2, by2 = prim["bbox"]
        dx1 = int(off_x + bx1 * scale)
        dy1 = int(off_y + by1 * scale)
        dx2 = int(off_x + bx2 * scale)
        dy2 = int(off_y + by2 * scale)
        bw_d = dx2 - dx1
        bh_d = dy2 - dy1

        # Glowing cyan bounding box
        cv2.rectangle(canvas, (dx1, dy1), (dx2, dy2), (212, 182, 6), 2, cv2.LINE_AA)
        
        # Corner brackets
        c_len = min(18, max(6, bw_d // 4))
        cv2.line(canvas, (dx1, dy1), (dx1 + c_len, dy1), (255, 255, 255), 3)
        cv2.line(canvas, (dx1, dy1), (dx1, dy1 + c_len), (255, 255, 255), 3)
        cv2.line(canvas, (dx2, dy1), (dx2 - c_len, dy1), (255, 255, 255), 3)
        cv2.line(canvas, (dx2, dy1), (dx2, dy1 + c_len), (255, 255, 255), 3)
        cv2.line(canvas, (dx1, dy2), (dx1 + c_len, dy2), (255, 255, 255), 3)
        cv2.line(canvas, (dx1, dy2), (dx1, dy2 - c_len), (255, 255, 255), 3)
        cv2.line(canvas, (dx2, dy2), (dx2 - c_len, dy2), (255, 255, 255), 3)
        cv2.line(canvas, (dx2, dy2), (dx2, dy2 - c_len), (255, 255, 255), 3)

        # Label pill above box: Always Name + Confidence
        label_text = f"#{1} {prim['class_name'].upper()} ({prim['confidence']*100:.1f}% conf)"
        (tw, th), bl = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        lbl_y = max(img_vp_y + 25, dy1 - 6)
        cv2.rectangle(canvas, (dx1, lbl_y - th - 6), (dx1 + tw + 10, lbl_y + 4), (20, 15, 10), -1)
        cv2.rectangle(canvas, (dx1, lbl_y - th - 6), (dx1 + tw + 10, lbl_y + 4), (212, 182, 6), 1)
        cv2.putText(canvas, label_text, (dx1 + 5, lbl_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 100), 1, cv2.LINE_AA)

    # Viewport HUD tags
    cv2.putText(canvas, f"SPECIMEN: {sample_path.name}", (img_vp_x + 12, img_vp_y + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, "AI OVERLAY: ACTIVE (BOX HIGHLIGHTED)", (img_vp_x + 12, img_vp_y + img_vp_h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (212, 182, 6), 1, cv2.LINE_AA)

    # Right: Diagnostic & Knowledge Cards (570x600)
    card_x, card_y, card_w, card_h = 684, 80, 572, 600
    cv2.rectangle(canvas, (card_x, card_y), (card_x + card_w, card_y + card_h), (42, 23, 15), -1)  # #0f172a
    cv2.rectangle(canvas, (card_x, card_y), (card_x + card_w, card_y + card_h), (85, 65, 51), 1)

    # Card Title
    cv2.putText(canvas, "DEFECT KNOWLEDGE & DIAGNOSTIC CARD", (card_x + 16, card_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(canvas, (card_x + 16, card_y + 42), (card_x + card_w - 16, card_y + 42), (85, 65, 51), 1)

    if detections:
        prim = detections[0]
        c_pct = prim['confidence'] * 100
        cls_name = prim['class_name']
        
        # Row 1: Defect Name + Confidence + Badges
        cv2.putText(canvas, f"{cls_name.upper()} ({c_pct:.1f}% conf)", (card_x + 16, card_y + 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 220, 255), 2, cv2.LINE_AA)
        
        # Source & Trust badges
        cv2.rectangle(canvas, (card_x + 360, card_y + 55), (card_x + 440, card_y + 78), (80, 40, 10), -1)
        cv2.rectangle(canvas, (card_x + 360, card_y + 55), (card_x + 440, card_y + 78), (212, 140, 20), 1)
        cv2.putText(canvas, "SRC: YOLO", (card_x + 368, card_y + 71), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 200, 100), 1, cv2.LINE_AA)

        cv2.rectangle(canvas, (card_x + 450, card_y + 55), (card_x + 550, card_y + 78), (20, 60, 20), -1)
        cv2.rectangle(canvas, (card_x + 450, card_y + 55), (card_x + 550, card_y + 78), (40, 180, 40), 1)
        cv2.putText(canvas, "TRUST: HIGH", (card_x + 458, card_y + 71), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 255, 120), 1, cv2.LINE_AA)

        curr_y = card_y + 110

        # Helper to draw multi-line text box
        def draw_section(title, text, color, y):
            cv2.putText(canvas, title.upper(), (card_x + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)
            y += 18
            cv2.rectangle(canvas, (card_x + 16, y - 10), (card_x + card_w - 16, y + 42), (25, 15, 10), -1)
            cv2.rectangle(canvas, (card_x + 16, y - 10), (card_x + card_w - 16, y + 42), (60, 45, 35), 1)
            # Wrap text to 65 chars
            words = text.split()
            lines = []
            cur_line = ""
            for w in words:
                if len(cur_line) + len(w) + 1 <= 64:
                    cur_line = (cur_line + " " + w).strip()
                else:
                    lines.append(cur_line)
                    cur_line = w
            if cur_line:
                lines.append(cur_line)
            for l in lines[:2]:
                cv2.putText(canvas, l, (card_x + 24, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
                y += 18
            return y + 20

        curr_y = draw_section("Plain Definition", prim.get("definition", ""), (212, 182, 6), curr_y)
        curr_y = draw_section("Visual Appearance", prim.get("visual_look", ""), (200, 130, 240), curr_y)
        curr_y = draw_section("Typical Manufacturing Cause", prim.get("cause", ""), (50, 165, 255), curr_y)
        curr_y = draw_section("Failure Risk & Mechanical Consequence", prim.get("risk", ""), (90, 90, 245), curr_y)
        curr_y = draw_section("Recommended Action", prim.get("action", ""), (60, 220, 140), curr_y)

        # Metrology row
        metro = prim.get("metrology", {})
        cv2.line(canvas, (card_x + 16, curr_y - 8), (card_x + card_w - 16, curr_y - 8), (85, 65, 51), 1)
        cv2.putText(canvas, "SUB-MILLIMETER METROLOGY & COORDINATES:", (card_x + 16, curr_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)
        
        m_area = metro.get("area_px", metro.get("pixel_area", 0))
        m_ratio = metro.get("aspect_ratio", 1.0)
        m_sol = metro.get("solidity", 1.0)
        cv2.putText(canvas, f"Area: {m_area:.0f} px  |  Ratio: {m_ratio:.1f}:1  |  Solidity: {m_sol:.2f}  |  BBox: [{prim['bbox'][0]:.0f}, {prim['bbox'][1]:.0f}, {prim['bbox'][2]:.0f}, {prim['bbox'][3]:.0f}]",
                    (card_x + 16, curr_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (212, 182, 6), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), canvas)
    print(f"\n[OK] High-fidelity visualization saved to: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

