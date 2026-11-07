"""WebSocket endpoint for real-time video stream defect detection.

Architectural highlights:
- YOLO-only inference (no morphological anomaly engine, no VLM, no heavy overlays).
- Binary JPEG frame ingestion over WebSocket for minimal network payload.
- Asynchronous Queue(maxsize=1) ensures decoupling and drops stale frames.
- Dedicated ThreadPoolExecutor runs inference with ONNX Runtime across all CPU threads.
- Non-blocking background database persistence triggered only on verdict state changes.
- Sends lightweight detection JSON only (no base64 re-encoding).
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.config import settings
from backend.services.alert_service import ws_manager
from backend.services.detection_service import get_detector
from backend.utils.image_utils import decode_base64_to_bytes
from ml.detector import DefectDetector

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Real-time Stream & WebSockets"])

# Global dedicated executors for stream pipeline
inference_executor = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="stream_onnx_worker",
)
db_executor = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="stream_db_worker",
)


def evaluate_production_routing(
    detections: List[Dict[str, Any]],
    summary: Dict[str, Any],
    has_critical: bool,
) -> Dict[str, Any]:
    """Execute high-speed industrial routing rules for conveyor automated dispatch.

    Rules:
    - PASS: Zero defects or minor flaws within tolerance -> continuous feed.
    - REWORK: Major repairable defect -> pneumatic arm redirects to offline rework.
    - REJECT: Catastrophic structural flaw -> immediate line-halt signal broadcasted.
    """
    t_route_start = time.perf_counter()
    by_sev = summary.get("by_severity", {})
    has_major = by_sev.get("major", 0) > 0

    if has_critical:
        action = "REJECT"
        signal = "EMERGENCY_HALT"
        target_station = "QUARANTINE_HALT"
        reason = "Catastrophic structural flaw detected — line quarantined."
        state_color = "red"
    elif has_major:
        action = "REWORK"
        signal = "DIVERT_PNEUMATIC"
        target_station = "REWORK_STATION_B"
        reason = "Major repairable defect detected — pneumatic diverter engaged."
        state_color = "amber"
    else:
        action = "PASS"
        signal = "CONTINUE"
        target_station = "CONVEYOR_CONTINUOUS"
        reason = "Specimen conforming to tolerances — continuous feed authorized."
        state_color = "green"

    dispatch_latency_ms = (time.perf_counter() - t_route_start) * 1000.0

    return {
        "action": action,
        "signal": signal,
        "target_station": target_station,
        "reason": reason,
        "state_color": state_color,
        "dispatch_latency_ms": round(dispatch_latency_ms, 3),
    }


def _run_stream_yolo_inference(
    frame_bytes: bytes,
    detector: DefectDetector,
    confidence_threshold: Optional[float] = None,
    ambient_boost: bool = False,
) -> Dict[str, Any]:
    """Execute high-speed single-pass YOLO inference in an executor worker thread."""
    nparr = np.frombuffer(frame_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode binary frame bytes into image array")

    # Run stream-optimized YOLO-only detection (<30ms) with optional dynamic sensitivity & CLAHE boost
    res = detector.detect(
        img,
        deep_verify=False,
        yolo_only=True,
        confidence_threshold=confidence_threshold,
        ambient_boost=ambient_boost,
    )
    detections: List[Dict[str, Any]] = res.get("detections", [])
    summary: Dict[str, Any] = res.get("summary", {})
    inference_ms = float(res.get("inference_time_ms", 0.0))

    has_critical = bool(summary.get("has_critical", False))
    by_sev = summary.get("by_severity", {})
    has_major = by_sev.get("major", 0) > 0
    has_minor = by_sev.get("minor", 0) > 0

    routing = evaluate_production_routing(detections, summary, has_critical)

    if has_critical:
        max_sev = "critical"
        verdict = "rejected"
    elif has_major:
        max_sev = "major"
        verdict = "flagged"
    elif has_minor:
        max_sev = "minor"
        verdict = "flagged"
    else:
        max_sev = "none"
        verdict = "passed"

    return {
        "detections": detections,
        "summary": summary,
        "inference_time_ms": inference_ms,
        "max_severity": max_sev,
        "has_critical": has_critical,
        "verdict": verdict,
        "production_routing": routing,
        "raw_img": img,
    }


def _persist_verdict_change(
    img: np.ndarray,
    detections: List[Dict[str, Any]],
    verdict: str,
    max_severity: str,
    inference_time_ms: float,
) -> None:
    """Persist inspection record and defect entities to SQLite in a background worker."""
    try:
        import uuid
        from backend.database import DefectRecord, Inspection, SessionLocal
        from backend.services.alert_service import AlertService

        with SessionLocal() as db:
            timestamp_slug = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
            unique_id = uuid.uuid4().hex[:8]
            filename = f"stream_{timestamp_slug}_{unique_id}.jpg"

            orig_path = settings.UPLOADS_DIR / filename
            result_path = settings.RESULTS_DIR / filename

            cv2.imwrite(str(orig_path), img)
            cv2.imwrite(str(result_path), img)

            crit_count = sum(1 for d in detections if d.get("severity") == "critical")
            maj_count = sum(1 for d in detections if d.get("severity") == "major")
            min_count = sum(1 for d in detections if d.get("severity") == "minor")

            inspection = Inspection(
                timestamp=datetime.datetime.now(datetime.timezone.utc),
                source="stream",
                filename=filename,
                original_path=str(orig_path),
                result_path=str(result_path),
                total_defects=len(detections),
                critical_count=crit_count,
                major_count=maj_count,
                minor_count=min_count,
                max_severity=max_severity,
                status=verdict,
                inference_time_ms=inference_time_ms,
                total_time_ms=inference_time_ms,
                notes=f"Stream state-change: transitioned to {verdict.upper()}",
                arbitration_tier="tier1_edge",
                arbitration_status="EDGE_CONFIRMED" if len(detections) > 0 else "CLEAN_PASS",
                domain_trust="HIGH",
            )
            db.add(inspection)
            db.commit()
            db.refresh(inspection)

            for det in detections:
                bbox = det["bbox"]
                defect_rec = DefectRecord(
                    inspection_id=inspection.id,
                    class_id=det["class_id"],
                    class_name=det["class_name"],
                    confidence=det["confidence"],
                    severity=det["severity"],
                    bbox_x1=bbox[0],
                    bbox_y1=bbox[1],
                    bbox_x2=bbox[2],
                    bbox_y2=bbox[3],
                    area_ratio=det.get("area_ratio", 0.0),
                    description=det.get("description", ""),
                    source=det.get("source", "yolo"),
                )
                db.add(defect_rec)

            db.commit()

            # Process quality alerts
            AlertService.process_inspection_alerts(db, inspection, detections)
            logger.info(
                "Stream verdict state-change saved to DB — inspection #%d (verdict=%s, defects=%d)",
                inspection.id,
                verdict,
                len(detections),
            )
    except Exception as exc:
        logger.error("Failed to persist stream state change to database: %s", exc)


@router.websocket("/ws/alerts")
async def websocket_alerts_endpoint(websocket: WebSocket) -> None:
    """Persistent WebSocket channel for real-time quality alert notifications."""
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except Exception:
                pass
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as exc:
        logger.warning("WebSocket alert connection error: %s", exc)
        ws_manager.disconnect(websocket)


@router.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket) -> None:
    """Bidirectional streaming WebSocket for high-speed video defect detection.

    Ingestion:
        Accepts binary JPEG bytes (captured at <= 640px, JPEG quality 0.6).
        Also backwards compatible with JSON {"frame": "<base64>"}.

    Inference:
        Executes YOLO-only forward pass on ONNX model in an executor thread.
        Decoupled asyncio.Queue(maxsize=1) discards stale frames when under load.

    Emission:
        Returns detection JSON only (no base64 image overhead).
        Writes to DB in the background only when the verdict state changes.
    """
    await websocket.accept()
    detector = get_detector()
    loop = asyncio.get_running_loop()
    logger.info("WebSocket video stream client connected.")

    # Queue of size 1 holding the latest frame
    frame_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1)
    last_verdict: Optional[str] = None
    prev_time = time.perf_counter()
    stop_event = asyncio.Event()

    active_conf_threshold: Optional[float] = None
    active_ambient_boost: bool = False

    async def frame_receiver() -> None:
        nonlocal active_conf_threshold, active_ambient_boost
        try:
            while not stop_event.is_set():
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break

                frame_bytes: Optional[bytes] = None
                if "bytes" in message and message["bytes"]:
                    frame_bytes = message["bytes"]
                elif "text" in message and message["text"]:
                    try:
                        data = json.loads(message["text"])
                        if data.get("type") == "ping":
                            await websocket.send_text(json.dumps({"type": "pong"}))
                            continue
                        if data.get("type") == "config":
                            if "confidence_threshold" in data:
                                active_conf_threshold = float(data["confidence_threshold"])
                            if "ambient_boost" in data:
                                active_ambient_boost = bool(data["ambient_boost"])
                            await websocket.send_text(json.dumps({
                                "type": "config_ack",
                                "confidence_threshold": active_conf_threshold,
                                "ambient_boost": active_ambient_boost,
                            }))
                            continue
                        if data.get("frame"):
                            frame_bytes = decode_base64_to_bytes(data["frame"])
                    except Exception:
                        continue

                if frame_bytes:
                    # Drop stale frame if queue already holds one
                    if frame_queue.full():
                        try:
                            frame_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    await frame_queue.put(frame_bytes)
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.warning("Stream frame receiver error: %s", exc)
        finally:
            stop_event.set()

    async def frame_processor() -> None:
        nonlocal last_verdict, prev_time
        try:
            while not stop_event.is_set():
                try:
                    frame_bytes = await asyncio.wait_for(frame_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue

                # Run inference in dedicated executor thread using all CPU threads
                try:
                    res = await loop.run_in_executor(
                        inference_executor,
                        _run_stream_yolo_inference,
                        frame_bytes,
                        detector,
                        active_conf_threshold,
                        active_ambient_boost,
                    )
                except Exception as exc:
                    logger.error("Error in stream inference: %s", exc)
                    continue

                now = time.perf_counter()
                fps = round(1.0 / max(0.001, now - prev_time), 1)
                prev_time = now

                verdict = res["verdict"]
                # Write to DB in the background only when the verdict changes
                if last_verdict is None or verdict != last_verdict:
                    last_verdict = verdict
                    loop.run_in_executor(
                        db_executor,
                        _persist_verdict_change,
                        res["raw_img"],
                        res["detections"],
                        verdict,
                        res["max_severity"],
                        res["inference_time_ms"],
                    )

                routing = res.get("production_routing", {})
                # Immediate line-halt broadcast if critical flaw detected (<10ms)
                if res["has_critical"] or routing.get("action") == "REJECT":
                    emergency_payload = {
                        "type": "emergency_line_halt",
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "action": "REJECT",
                        "signal": "EMERGENCY_HALT",
                        "target_station": "QUARANTINE_HALT",
                        "reason": routing.get("reason", "Catastrophic structural flaw detected"),
                        "total_defects": len(res["detections"]),
                        "max_severity": res["max_severity"],
                        "dispatch_latency_ms": routing.get("dispatch_latency_ms", 0.0),
                    }
                    asyncio.create_task(ws_manager.broadcast(emergency_payload))

                response_payload = {
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "inference_time_ms": round(res["inference_time_ms"], 1),
                    "fps": fps,
                    "total_defects": len(res["detections"]),
                    "max_severity": res["max_severity"],
                    "has_critical": res["has_critical"],
                    "verdict": verdict,
                    "production_routing": routing,
                    "defects": res["detections"],
                }

                await websocket.send_text(json.dumps(response_payload))
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.warning("Stream frame processor error: %s", exc)
        finally:
            stop_event.set()

    recv_task = asyncio.create_task(frame_receiver())
    proc_task = asyncio.create_task(frame_processor())

    done, pending = await asyncio.wait(
        [recv_task, proc_task],
        return_when=asyncio.FIRST_COMPLETED,
    )
    stop_event.set()
    for task in pending:
        task.cancel()
    logger.info("WebSocket video stream client disconnected.")
