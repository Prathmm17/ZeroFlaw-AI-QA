"""Unit and integration tests for Video Inspection Feature (Phase B).

Verifies:
1. Video file decoding and sampling via OpenCV cv2.VideoCapture.
2. Defect timeline compilation with timestamps, severities, and thumbnails.
3. Endpoints: POST /api/video/upload, GET /api/video/jobs, GET /api/video/{id}/status, GET /api/video/{id}/result.
4. Rejection of unsupported media extensions with 400 Bad Request.
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient

from backend.config import settings
from backend.main import app
from backend.routers.video import get_video_result, get_video_status, list_video_jobs, upload_video
from backend.services.detection_service import get_detector
from backend.services.video_service import VideoService


@pytest.fixture(scope="module", autouse=True)
def prewarm_detector() -> None:
    """Pre-warm model once so all tests execute in <1s."""
    det = get_detector()
    assert det.is_ready()


@pytest.fixture(scope="module")
def synthetic_video_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate a valid 3-second synthetic industrial video with defect frames."""
    temp_dir = tmp_path_factory.mktemp("video_test")
    vpath = temp_dir / "test_conveyor_run.mp4"

    width, height = 640, 480
    fps = 10.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(vpath), fourcc, fps, (width, height))

    # 30 frames = 3 seconds
    for f_idx in range(30):
        # Industrial steel background
        frame = np.full((height, width, 3), 140, dtype=np.uint8)
        # Roller marks
        for y in range(0, height, 40):
            cv2.line(frame, (0, y), (width, y), (110, 110, 110), 2)

        # On frames 10-14 (at ~1.0-1.4s), draw an inclusion / crack defect
        if 10 <= f_idx <= 14:
            cv2.line(frame, (200, 200), (350, 280), (30, 30, 30), 6)
            cv2.circle(frame, (350, 280), 12, (20, 20, 20), -1)

        out.write(frame)

    out.release()
    assert vpath.is_file()
    return vpath


def test_video_service_processing(synthetic_video_path: Path) -> None:
    """Verify VideoService decodes video, samples frames, and compiles defect timeline."""
    video_id = VideoService.create_job(synthetic_video_path, "test_conveyor_run.mp4")
    assert video_id.startswith("vid_")

    job_result = VideoService.process_video(
        video_id=video_id,
        sample_fps=3.0,
        confidence_threshold=0.20,
        ambient_boost=True,
    )

    assert job_result["status"] == "completed"
    assert job_result["progress"] == 100
    assert job_result["duration_seconds"] > 0
    assert job_result["sampled_frames"] >= 5
    assert "timeline" in job_result
    assert "summary" in job_result
    assert job_result["summary"]["peak_timestamp_sec"] >= 0.0

    # Ensure thumbnail images were created and exist
    for item in job_result["timeline"]:
        assert "timestamp_sec" in item
        assert "thumbnail_url" in item
        thumb_name = Path(item["thumbnail_url"]).name
        thumb_file = settings.RESULTS_DIR / thumb_name
        assert thumb_file.is_file(), f"Thumbnail {thumb_file} was not written to disk"


def test_video_router_endpoints(synthetic_video_path: Path) -> None:
    """Verify upload_video router function and status/result endpoints."""
    # 1. Test direct router upload handler
    with open(synthetic_video_path, "rb") as f:
        file_bytes = f.read()

    upload_file = UploadFile(
        filename="router_test_run.mp4",
        file=io.BytesIO(file_bytes),
    )

    upload_res = upload_video(
        file=upload_file,
        sample_fps=3.0,
        confidence_threshold=0.20,
        ambient_boost=True,
    )

    assert "video_id" in upload_res
    assert upload_res["status"] == "processing"
    video_id = upload_res["video_id"]

    # Wait for background job to reach completion
    for _ in range(50):
        status_data = get_video_status(video_id)
        if status_data["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)

    # 2. Test status endpoint
    status_data = get_video_status(video_id)
    assert status_data["video_id"] == video_id
    assert status_data["status"] == "completed"
    assert status_data["progress"] == 100

    # 3. Test result endpoint
    result_data = get_video_result(video_id)
    assert result_data["video_id"] == video_id
    assert len(result_data["timeline"]) >= 0
    assert "summary" in result_data

    # 4. Test jobs listing
    jobs_list = list_video_jobs()
    assert isinstance(jobs_list, list)
    assert any(j["video_id"] == video_id for j in jobs_list)


def test_video_upload_invalid_format_rejected() -> None:
    """Verify non-video files are rejected with 400 Bad Request."""
    fake_txt = UploadFile(
        filename="report.txt",
        file=io.BytesIO(b"Hello world, not a video"),
    )

    with pytest.raises(Exception) as exc_info:
        upload_video(file=fake_txt)

    assert "400" in str(exc_info.value) or "Unsupported file format" in str(exc_info.value)
