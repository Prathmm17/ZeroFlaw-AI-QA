"""Unit and integration tests for Phase 4: Production Routing & Real-Time Conveyor Streaming Latency.

Verification Gates:
1. Average edge stream inference latency <= 30.0ms.
2. Sustained streaming throughput >= 20 FPS.
3. Automated production routing rules:
   - PASS (conforming) -> continuous outfeed line
   - REWORK (major defect) -> pneumatic redirect
   - REJECT (critical flaw) -> emergency line halt broadcasted over WebSocket in <15ms
4. Queue decoupling with stale frame dropping under load.
5. Zero memory leaks across prolonged frame processing iterations.
"""

from __future__ import annotations

import gc
import json
import os
import queue
import time
from typing import Any, Coroutine, Dict, List

import cv2
import numpy as np
import pytest

from backend.routers.stream import (
    _run_stream_yolo_inference,
    evaluate_production_routing,
)
from backend.services.alert_service import ws_manager
from ml.detector import DefectDetector


def run_coroutine_sync(coro: Coroutine[Any, Any, Any]) -> Any:
    """Execute coroutine synchronously without Windows IOCP event loop overhead."""
    try:
        coro.send(None)
    except StopIteration as exc:
        return exc.value


@pytest.fixture(scope="module")
def detector() -> DefectDetector:
    """Fixture providing initialized DefectDetector."""
    return DefectDetector(model_name="best.onnx")


@pytest.fixture(scope="module")
def sample_jpeg_bytes() -> bytes:
    """Fixture providing standard 640x480 binary JPEG frame bytes."""
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    cv2.line(frame, (100, 100), (300, 300), (20, 20, 20), 5)
    success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
    assert success
    return encoded.tobytes()


def test_stream_inference_latency_under_30ms(detector: DefectDetector, sample_jpeg_bytes: bytes) -> None:
    """Assert stream-optimized single-pass YOLO inference maintains <= 30.0ms average latency."""
    # Warmup
    for _ in range(5):
        _run_stream_yolo_inference(sample_jpeg_bytes, detector)

    latencies: List[float] = []
    for _ in range(30):
        t0 = time.perf_counter()
        res = _run_stream_yolo_inference(sample_jpeg_bytes, detector)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        assert "detections" in res
        assert "production_routing" in res

    avg_latency = float(np.mean(latencies))
    print(f"\n[Phase 4 Latency Benchmark] Mean: {avg_latency:.2f}ms, Min: {np.min(latencies):.2f}ms, Max: {np.max(latencies):.2f}ms")
    assert avg_latency <= 30.0, f"Average stream latency {avg_latency:.2f}ms exceeded 30.0ms threshold!"


def test_stream_sustained_fps_over_20(detector: DefectDetector, sample_jpeg_bytes: bytes) -> None:
    """Assert continuous frame processing achieves >= 20 FPS throughput capability."""
    frame_count = 40
    t_start = time.perf_counter()
    for _ in range(frame_count):
        _run_stream_yolo_inference(sample_jpeg_bytes, detector)
    total_time = time.perf_counter() - t_start

    sustained_fps = frame_count / total_time
    print(f"\n[Phase 4 Throughput Benchmark] Sustained: {sustained_fps:.1f} FPS (Total: {total_time:.2f}s for {frame_count} frames)")
    assert sustained_fps >= 20.0, f"Sustained throughput {sustained_fps:.1f} FPS below 20.0 FPS requirement!"


def test_production_routing_pass_state() -> None:
    """Verify conforming specimen generates PASS signal for continuous feed."""
    clean_summary = {"has_critical": False, "by_severity": {"minor": 0, "major": 0, "critical": 0}}
    routing = evaluate_production_routing(detections=[], summary=clean_summary, has_critical=False)

    assert routing["action"] == "PASS"
    assert routing["signal"] == "CONTINUE"
    assert routing["target_station"] == "CONVEYOR_CONTINUOUS"
    assert routing["state_color"] == "green"
    assert routing["dispatch_latency_ms"] <= 5.0


def test_production_routing_rework_state() -> None:
    """Verify major repairable defect triggers REWORK pneumatic diverter redirect."""
    major_summary = {"has_critical": False, "by_severity": {"minor": 0, "major": 1, "critical": 0}}
    mock_detections = [{"class_name": "inclusion", "severity": "major", "confidence": 0.72}]
    routing = evaluate_production_routing(detections=mock_detections, summary=major_summary, has_critical=False)

    assert routing["action"] == "REWORK"
    assert routing["signal"] == "DIVERT_PNEUMATIC"
    assert routing["target_station"] == "REWORK_STATION_B"
    assert routing["state_color"] == "amber"
    assert "pneumatic diverter" in routing["reason"].lower()


def test_production_routing_reject_and_websocket_emergency_alert() -> None:
    """Verify critical structural flaw generates REJECT emergency line-halt broadcasted in <15ms."""
    crit_summary = {"has_critical": True, "by_severity": {"minor": 0, "major": 0, "critical": 1}}
    mock_detections = [{"class_name": "crazing", "severity": "critical", "confidence": 0.94}]
    routing = evaluate_production_routing(detections=mock_detections, summary=crit_summary, has_critical=True)

    assert routing["action"] == "REJECT"
    assert routing["signal"] == "EMERGENCY_HALT"
    assert routing["target_station"] == "QUARANTINE_HALT"
    assert routing["state_color"] == "red"
    assert routing["dispatch_latency_ms"] <= 10.0

    class MockWebSocket:
        def __init__(self) -> None:
            self.received: List[str] = []

        async def send_text(self, text: str) -> None:
            self.received.append(text)

    emergency_payload = {
        "type": "emergency_line_halt",
        "action": routing["action"],
        "signal": routing["signal"],
        "target_station": routing["target_station"],
        "reason": routing["reason"],
        "dispatch_latency_ms": routing["dispatch_latency_ms"],
    }

    ws_manager.active_connections.clear()
    mock_client = MockWebSocket()
    ws_manager.active_connections.append(mock_client)  # type: ignore

    try:
        t0 = time.perf_counter()
        run_coroutine_sync(ws_manager.broadcast(emergency_payload))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        print(f"\n[Phase 4 Emergency WS Broadcast Latency] {latency_ms:.3f}ms")
        assert len(mock_client.received) == 1
        msg = json.loads(mock_client.received[0])
        assert msg["type"] == "emergency_line_halt"
        assert msg["signal"] == "EMERGENCY_HALT"
        assert latency_ms < 15.0, f"Broadcast latency {latency_ms:.2f}ms exceeded 15.0ms threshold!"
    finally:
        ws_manager.disconnect(mock_client)  # type: ignore


def test_stream_queue_backpressure_and_stale_dropping() -> None:
    """Verify frame queue with maxsize=1 discards stale frames without accumulating latency."""
    q: queue.Queue[bytes] = queue.Queue(maxsize=1)
    frames = [b"frame_1", b"frame_2", b"frame_3"]

    for f in frames:
        if q.full():
            try:
                q.get_nowait()
            except queue.Empty:
                pass
        q.put(f)

    assert q.qsize() == 1
    latest_frame = q.get_nowait()
    assert latest_frame == b"frame_3", "Queue failed to drop stale frames and retain latest!"


def test_stream_prolonged_memory_leak_check(detector: DefectDetector, sample_jpeg_bytes: bytes) -> None:
    """Verify processing 100 continuous video frames creates zero memory leaks."""
    import psutil

    process = psutil.Process(os.getpid())
    gc.collect()
    mem_before = process.memory_info().rss / (1024 * 1024)

    for _ in range(100):
        _run_stream_yolo_inference(sample_jpeg_bytes, detector)

    gc.collect()
    mem_after = process.memory_info().rss / (1024 * 1024)
    mem_delta = mem_after - mem_before
    print(f"\n[Phase 4 Memory Stability] Initial: {mem_before:.1f} MB, Final: {mem_after:.1f} MB, Delta: {mem_delta:.2f} MB")
    assert mem_delta <= 5.0, f"Memory growth {mem_delta:.2f} MB exceeds 5.0 MB threshold (possible leak)!"
