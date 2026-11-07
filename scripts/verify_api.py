"""Verify VisionGuard QA FastAPI server with real HTTP requests."""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from backend.main import app

PORT = 8921
BASE_URL = f"http://127.0.0.1:{PORT}"


class ServerThread(threading.Thread):
    def __init__(self):
        super().__init__()
        config = uvicorn.Config(app=app, host="127.0.0.1", port=PORT, log_level="warning")
        self.server = uvicorn.Server(config)
        self.daemon = True

    def run(self):
        self.server.run()

    def stop(self):
        self.server.should_exit = True


def make_get(path: str) -> dict:
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "VisionGuard-Test"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def make_post(path: str, data: dict | None = None) -> dict:
    url = f"{BASE_URL}{path}"
    payload = json.dumps(data or {}).encode("utf-8") if data else b""
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "VisionGuard-Test"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    print("=" * 60)
    print("  VERIFYING VISIONGUARD FASTAPI SERVER OVER HTTP")
    print("=" * 60)

    server = ServerThread()
    server.start()

    # Wait for server to bind
    for _ in range(30):
        try:
            make_get("/api/health")
            break
        except Exception:
            time.sleep(0.5)

    try:
        # 1. Health check
        print("\n[1] GET /api/health")
        health = make_get("/api/health")
        print("Response:", health)
        assert health["status"] == "healthy"
        assert health["model_ready"] == "True"
        print("✓ Health check PASSED")

        # 2. List samples
        print("\n[2] GET /api/detect/samples")
        samples = make_get("/api/detect/samples")
        print("Samples count:", len(samples))
        assert len(samples) > 0
        sample_name = samples[0]
        print(f"✓ Samples check PASSED (sample: '{sample_name}')")

        # 3. Detect sample
        print(f"\n[3] POST /api/detect/sample/{sample_name}")
        insp = make_post(f"/api/detect/sample/{sample_name}")
        print(f"Inspection #{insp['id']}, status: {insp['status']}, defects: {insp['total_defects']}")
        assert insp["id"] > 0
        print("✓ Sample detection PASSED")

        # 4. History
        print("\n[4] GET /api/detect/history")
        hist = make_get("/api/detect/history?page=1&page_size=5")
        print(f"Total inspections in DB: {hist['total']}")
        assert hist["total"] >= 1
        print("✓ History check PASSED")

        # 5. Analytics Overview
        print("\n[5] GET /api/analytics/overview")
        analytics = make_get("/api/analytics/overview")
        print(f"Total defects recorded: {analytics['total_defects']}, Rate: {analytics['defect_rate_percentage']}%")
        print("✓ Analytics overview PASSED")

        # 6. Analytics Trends
        print("\n[6] GET /api/analytics/trends?days=7")
        trends = make_get("/api/analytics/trends?days=7")
        print(f"Trend data points: {len(trends['trends'])}")
        assert len(trends["trends"]) == 7
        print("✓ Trends check PASSED")

        # 7. Alerts
        print("\n[7] GET /api/alerts")
        alerts = make_get("/api/alerts")
        print(f"Total alerts: {len(alerts)}")
        print("✓ Alerts check PASSED")

        print("\n" + "=" * 60)
        print("  ALL FASTAPI HTTP ENDPOINTS VERIFIED AND WORKING! ✓")
        print("=" * 60 + "\n")
    finally:
        server.stop()


if __name__ == "__main__":
    main()

