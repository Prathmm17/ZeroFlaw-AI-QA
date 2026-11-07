import requests
from pathlib import Path

img1_path = Path('data/uploads/WhatsApp Image 2026-10-30 at 6.26.03 AM_20261030_062646_ec25dbfb_orig.jpg')
img2_path = Path('data/uploads/WhatsApp Image 2026-10-30 at 6.26.04 AM (1)_20261030_062736_c4552931_orig.jpg')

for p in [img1_path, img2_path]:
    print('=' * 75)
    print(f'UPLOADING PHOTO TO LIVE SERVER: {p.name}')
    with open(p, 'rb') as f:
        # Default upload (as sent by browser UI)
        res = requests.post(
            'http://127.0.0.1:8000/api/detect/upload',
            files={'file': (p.name, f, 'image/jpeg')},
            params={'deep_verify': True}
        )
    print(f'HTTP Status: {res.status_code}')
    if res.status_code == 201:
        data = res.json()
        print(f"Inspection ID       : {data.get('id')}")
        print(f"Total Defects Found : {data.get('total_defects')}")
        print(f"Inspection Status   : {data.get('status').upper()}")
        print(f"Arbitration Tier    : {data.get('arbitration_tier')}")
        print(f"Domain Trust        : {data.get('domain_trust')}")
        print(f"Domain Warning      : {data.get('domain_warning')}")
        print(f"Latency             : {data.get('inference_time_ms')} ms (total: {data.get('total_time_ms')} ms)")
        print("\nDetected Defects List:")
        for idx, d in enumerate(data.get('defects', [])):
            print(f"  [{idx+1}] {d.get('class_name')} | Conf: {d.get('confidence')*100:.1f}% | Severity: {d.get('severity').upper()} | Source: {d.get('source')} | BBox: {d.get('bbox')}")
    else:
        print('Error:', res.text)

