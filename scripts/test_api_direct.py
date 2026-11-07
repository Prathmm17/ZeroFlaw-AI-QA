"""Direct asynchronous integration test for VisionGuard QA backend."""

from __future__ import annotations

import asyncio
import io
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx
from backend.config import settings
from backend.database import init_db
from backend.main import app


async def main() -> None:
    print("\n" + "=" * 60, flush=True)
    print("  RUNNING ASYNC BACKEND INTEGRATION TESTS", flush=True)
    print("=" * 60, flush=True)

    init_db()

    # Use ASGITransport for direct in-process FastAPI testing
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Health check
        print("\n[1] Testing GET /api/health ...", flush=True)
        resp = await client.get("/api/health")
        print("Status:", resp.status_code, "Body:", resp.json(), flush=True)
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"
        print("✓ Health check PASSED", flush=True)

        # 2. List samples
        print("\n[2] Testing GET /api/detect/samples ...", flush=True)
        resp = await client.get("/api/detect/samples")
        print("Status:", resp.status_code, "Samples count:", len(resp.json()), flush=True)
        assert resp.status_code == 200
        samples = resp.json()
        assert len(samples) > 0
        sample_name = samples[0]
        print(f"✓ List samples PASSED (testing with '{sample_name}')", flush=True)

        # 3. Detect sample
        print(f"\n[3] Testing POST /api/detect/sample/{sample_name} ...", flush=True)
        resp = await client.post(f"/api/detect/sample/{sample_name}")
        print("Status:", resp.status_code, flush=True)
        assert resp.status_code == 200
        sample_data = resp.json()
        inspection_id = sample_data["id"]
        print(f"✓ Sample detection PASSED (Inspection #{inspection_id}, defects: {sample_data['total_defects']})", flush=True)

        # 4. Upload file
        print("\n[4] Testing POST /api/detect/upload ...", flush=True)
        sample_path = settings.SAMPLES_DIR / sample_name
        with open(sample_path, "rb") as f:
            file_bytes = f.read()

        files = {"file": ("demo_gear.png", file_bytes, "image/png")}
        resp = await client.post("/api/detect/upload", files=files, data={"notes": "Direct async upload test"})
        print("Status:", resp.status_code, flush=True)
        assert resp.status_code == 201
        upload_data = resp.json()
        print(f"✓ Upload detection PASSED (Inspection #{upload_data['id']})", flush=True)

        # 5. History
        print("\n[5] Testing GET /api/detect/history ...", flush=True)
        resp = await client.get("/api/detect/history?page=1&page_size=10")
        assert resp.status_code == 200
        history = resp.json()
        print(f"History items: {len(history['items'])}, Total: {history['total']}", flush=True)
        assert history["total"] >= 2
        print("✓ History retrieval PASSED", flush=True)

        # 6. Analytics Overview
        print("\n[6] Testing GET /api/analytics/overview ...", flush=True)
        resp = await client.get("/api/analytics/overview")
        assert resp.status_code == 200
        overview = resp.json()
        print("Overview:", overview, flush=True)
        assert overview["total_inspections"] >= 2
        print("✓ Analytics overview PASSED", flush=True)

        # 7. Analytics Trends
        print("\n[7] Testing GET /api/analytics/trends?days=7 ...", flush=True)
        resp = await client.get("/api/analytics/trends?days=7")
        assert resp.status_code == 200
        trends = resp.json()["trends"]
        assert len(trends) == 7
        print("✓ Analytics trends PASSED", flush=True)

        # 8. Alerts
        print("\n[8] Testing GET /api/alerts ...", flush=True)
        resp = await client.get("/api/alerts")
        assert resp.status_code == 200
        alerts = resp.json()
        print(f"Alerts count: {len(alerts)}", flush=True)
        print("✓ Alerts list PASSED", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("  ALL 8 BACKEND API TESTS COMPLETED AND PASSED! ✓", flush=True)
    print("=" * 60 + "\n", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

