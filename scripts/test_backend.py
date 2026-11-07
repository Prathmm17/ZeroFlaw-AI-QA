"""Integration test suite verifying all VisionGuard QA backend endpoints and services."""

from __future__ import annotations

import io
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.config import settings
from backend.database import init_db
from backend.main import app

# Ensure DB initialized
init_db()


def run_tests() -> bool:
    print("\n" + "=" * 60, flush=True)
    print("  RUNNING VISIONGUARD BACKEND INTEGRATION TESTS", flush=True)
    print("=" * 60, flush=True)

    with TestClient(app) as client:
        # 1. Health check
        print("\n[1] Testing GET /api/health ...", flush=True)
        resp = client.get("/api/health")
        print("Status:", resp.status_code, "Response:", resp.json(), flush=True)
        assert resp.status_code == 200, f"Health check failed: {resp.text}"
        assert resp.json()["status"] == "healthy"
        print("✓ Health check PASSED", flush=True)

        # 2. List samples
        print("\n[2] Testing GET /api/detect/samples ...", flush=True)
        resp = client.get("/api/detect/samples")
        print("Status:", resp.status_code, "Found samples:", len(resp.json()), flush=True)
        assert resp.status_code == 200
        samples = resp.json()
        assert len(samples) > 0, "Expected at least 1 sample image in data/samples"
        sample_to_test = samples[0]
        print(f"✓ List samples PASSED (testing with '{sample_to_test}')", flush=True)

        # 3. Detect sample image
        print(f"\n[3] Testing POST /api/detect/sample/{sample_to_test} ...", flush=True)
        resp = client.post(f"/api/detect/sample/{sample_to_test}")
        print("Status:", resp.status_code, flush=True)
        assert resp.status_code == 200, f"Sample detection failed: {resp.text}"
        data = resp.json()
        inspection_id = data["id"]
        print(f"Inspection ID: {inspection_id}, Total Defects: {data['total_defects']}, Status: {data['status']}", flush=True)
        print("✓ Sample detection PASSED", flush=True)

        # 4. Upload and detect image
        print("\n[4] Testing POST /api/detect/upload ...", flush=True)
        sample_path = settings.SAMPLES_DIR / sample_to_test
        with open(sample_path, "rb") as f:
            file_bytes = f.read()

        files = {"file": ("test_gear.png", io.BytesIO(file_bytes), "image/png")}
        resp = client.post("/api/detect/upload", files=files, data={"notes": "Integration test inspection"})
        print("Status:", resp.status_code, flush=True)
        assert resp.status_code == 201, f"Upload detection failed: {resp.text}"
        upload_data = resp.json()
        upload_id = upload_data["id"]
        print(f"Uploaded Inspection ID: {upload_id}, Result URL: {upload_data['result_url']}", flush=True)
        print("✓ Upload detection PASSED", flush=True)

        # 5. Get history
        print("\n[5] Testing GET /api/detect/history ...", flush=True)
        resp = client.get("/api/detect/history?page=1&page_size=10")
        print("Status:", resp.status_code, flush=True)
        assert resp.status_code == 200
        history = resp.json()
        assert history["total"] >= 2
        print(f"Total recorded inspections: {history['total']}", flush=True)
        print("✓ History pagination PASSED", flush=True)

        # 6. Get single inspection detail
        print(f"\n[6] Testing GET /api/detect/{inspection_id} ...", flush=True)
        resp = client.get(f"/api/detect/{inspection_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == inspection_id
        print("✓ Inspection detail retrieval PASSED", flush=True)

        # 7. Get analytics overview
        print("\n[7] Testing GET /api/analytics/overview ...", flush=True)
        resp = client.get("/api/analytics/overview")
        assert resp.status_code == 200
        analytics = resp.json()
        print("Analytics overview:", analytics, flush=True)
        assert analytics["total_inspections"] >= 2
        print("✓ Analytics overview PASSED", flush=True)

        # 8. Get analytics trends
        print("\n[8] Testing GET /api/analytics/trends?days=7 ...", flush=True)
        resp = client.get("/api/analytics/trends?days=7")
        assert resp.status_code == 200
        trends = resp.json()["trends"]
        assert len(trends) == 7
        print("✓ Analytics trends PASSED", flush=True)

        # 9. Get alerts
        print("\n[9] Testing GET /api/alerts ...", flush=True)
        resp = client.get("/api/alerts")
        assert resp.status_code == 200
        alerts = resp.json()
        print(f"Total recorded alerts: {len(alerts)}", flush=True)
        print("✓ Alerts retrieval PASSED", flush=True)

        # 10. Test alert acknowledgment if any alert exists
        if len(alerts) > 0:
            alert_id = alerts[0]["id"]
            print(f"\n[10] Testing POST /api/alerts/{alert_id}/acknowledge ...", flush=True)
            resp = client.post(
                f"/api/alerts/{alert_id}/acknowledge",
                json={"acknowledged_by": "QA Lead Tester", "notes": "Reviewed and acknowledged"},
            )
            assert resp.status_code == 200
            assert resp.json()["acknowledged"] is True
            print(f"✓ Alert #{alert_id} acknowledgment PASSED", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("  ALL 10 BACKEND INTEGRATION TESTS PASSED SUCCESSFULLY! ✓", flush=True)
    print("=" * 60 + "\n", flush=True)
    return True


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
