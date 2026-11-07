import sys
sys.path.insert(0, '.')
from pathlib import Path
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

img1_path = Path('data/uploads/WhatsApp Image 2026-10-30 at 6.26.03 AM_20261030_062646_ec25dbfb_orig.jpg')
img2_path = Path('data/uploads/WhatsApp Image 2026-10-30 at 6.26.04 AM (1)_20261030_062736_c4552931_orig.jpg')

for p in [img1_path, img2_path]:
    print('=' * 70)
    print('Testing API Upload:', p.name)
    with open(p, 'rb') as f:
        # Default upload (deep_verify=False or True)
        res_fast = client.post('/api/detect/upload', files={'file': (p.name, f.read(), 'image/jpeg')})
    print(f'Status Code: {res_fast.status_code}')
    data = res_fast.json()
    print(f"Total Defects Found : {data.get('total_defects')}")
    print(f"Verdict Status      : {data.get('status')}")
    print(f"Domain Trust        : {data.get('domain_trust')}")
    print(f"Domain Warning      : {data.get('domain_warning')}")
    print("Detections:")
    for idx, d in enumerate(data.get('defects', [])):
        print(f"  [{idx+1}] {d.get('class_name')} ({d.get('confidence')*100:.1f}%) | Severity: {d.get('severity')} | Source: {d.get('source')} | bbox: {d.get('bbox')}")

