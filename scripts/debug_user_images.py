import sys
sys.path.insert(0, '.')
import cv2
from pathlib import Path
from backend.services.detection_service import get_arbitrator, get_detector

img1_path = Path('data/uploads/WhatsApp Image 2026-10-30 at 6.26.03 AM_20261030_062646_ec25dbfb_orig.jpg')
img2_path = Path('data/uploads/WhatsApp Image 2026-10-30 at 6.26.04 AM (1)_20261030_062736_c4552931_orig.jpg')

detector = get_detector()
arbitrator = get_arbitrator()

for p in [img1_path, img2_path]:
    print('=' * 70)
    print('Testing:', p.name)
    img = cv2.imread(str(p))
    print('Image dimensions:', img.shape)

    det_res = detector.detect(img)
    raw_dets = det_res.get('detections', [])
    print(f'Raw YOLO Detections: {len(raw_dets)}')
    for d in raw_dets:
        print(f"   -> {d.get('class_name')} (conf={d.get('confidence', 0):.3f}) bbox={d.get('bbox')}")

    res_fast = arbitrator.arbitrate(img, filename=p.name, deep_verify=False)
    print(f'Arbitrator (deep_verify=False): {len(res_fast.detections)} detections, verdict={res_fast.verdict}')
    for d in res_fast.detections:
        print(f"   -> {d.class_name} (conf={d.confidence:.3f}) source={getattr(d, 'source', None)}")

    res_deep = arbitrator.arbitrate(img, filename=p.name, deep_verify=True)
    print(f'Arbitrator (deep_verify=True): {len(res_deep.detections)} detections, verdict={res_deep.verdict}')
    for d in res_deep.detections:
        print(f"   -> {d.class_name} (conf={d.confidence:.3f}) source={getattr(d, 'source', None)}")
