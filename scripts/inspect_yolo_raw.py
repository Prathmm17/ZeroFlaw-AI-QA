import sys
sys.path.insert(0, '.')
import cv2
import numpy as np
from pathlib import Path
from ml.detector import DefectDetector
import onnxruntime as ort

img1_path = Path('data/uploads/WhatsApp Image 2026-10-30 at 6.26.03 AM_20261030_062646_ec25dbfb_orig.jpg')
img2_path = Path('data/uploads/WhatsApp Image 2026-10-30 at 6.26.04 AM (1)_20261030_062736_c4552931_orig.jpg')

detector = DefectDetector(model_name='best.onnx', confidence_threshold=0.01)

for p in [img1_path, img2_path]:
    print('=' * 60)
    print('Inspecting:', p.name)
    img = cv2.imread(str(p))
    # Let's inspect raw boxes from ONNX directly
    h, w = img.shape[:2]
    # Letterbox to 640x640
    scale = min(640.0 / w, 640.0 / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
    pad_w = (640 - nw) // 2
    pad_h = (640 - nh) // 2
    canvas[pad_h:pad_h + nh, pad_w:pad_w + nw] = resized
    blob = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, axis=0)

    session = ort.InferenceSession('models/best.onnx', providers=['CPUExecutionProvider'])
    out = session.run(None, {session.get_inputs()[0].name: blob})[0]
    # Shape is [1, 10, 8400]
    preds = out[0] # [10, 8400]
    classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled_in_scale', 'scratches']
    scores = preds[4:, :] # [6, 8400]
    max_scores_per_anchor = scores.max(axis=0) # [8400]

    print(f'Max confidence across entire image: {float(max_scores_per_anchor.max()):.4f}')
    top_anchors = np.where(max_scores_per_anchor > 0.05)[0]
    print(f'Anchors with conf > 0.05: {len(top_anchors)}')
    for a in top_anchors[:10]:
        cls_idx = int(scores[:, a].argmax())
        conf = float(scores[cls_idx, a])
        cx, cy, bw, bh = preds[0:4, a]
        print(f'  Class: {classes[cls_idx]} (conf={conf:.3f}) box_xywh=[{cx:.1f}, {cy:.1f}, {bw:.1f}, {bh:.1f}]')

