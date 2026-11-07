"""FastAPI router for video file upload and defect inspection.

Provides:
- POST /api/video/upload: Upload industrial video file and trigger background defect scanning.
- GET  /api/video/jobs: List historical video inspection runs.
- GET  /api/video/{video_id}/status: Poll processing progress and stage.
- GET  /api/video/{video_id}/result: Retrieve comprehensive timestamped defect timeline.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile, status

from backend.config import settings
from backend.services.video_service import VideoService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/video", tags=["Video Defect Inspection"])

video_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="video_analysis_worker")

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _run_background_video_analysis(
    video_id: str,
    sample_fps: float,
    confidence_threshold: float,
    ambient_boost: bool,
) -> None:
    """Worker task executing video frame decoding and YOLO defect inference."""
    try:
        VideoService.process_video(
            video_id=video_id,
            sample_fps=sample_fps,
            confidence_threshold=confidence_threshold,
            ambient_boost=ambient_boost,
        )
    except Exception as exc:
        logger.error("Background video processing failed for %s: %s", video_id, exc)


@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
def upload_video(
    file: UploadFile = File(...),
    sample_fps: float = Query(3.0, ge=0.5, le=10.0, description="Frames sampled per second"),
    confidence_threshold: float = Query(0.25, ge=0.10, le=0.90, description="Minimum confidence threshold"),
    ambient_boost: bool = Query(True, description="Enable CLAHE contrast enhancement for ambient video"),
) -> Dict[str, Any]:
    """Upload an industrial video specimen and launch defect analysis."""
    filename = file.filename or "specimen_video.mp4"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    unique_slug = f"video_{uuid.uuid4().hex[:8]}{ext}"
    dest_path = settings.UPLOADS_DIR / unique_slug

    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        logger.error("Failed to save uploaded video %s: %s", filename, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save video on server: {exc}",
        ) from exc
    finally:
        file.file.close()

    video_id = VideoService.create_job(dest_path, filename)

    # Dispatch to background worker thread pool
    video_executor.submit(
        _run_background_video_analysis,
        video_id=video_id,
        sample_fps=sample_fps,
        confidence_threshold=confidence_threshold,
        ambient_boost=ambient_boost,
    )

    return {
        "video_id": video_id,
        "filename": filename,
        "status": "processing",
        "progress": 5,
        "message": "Video uploaded successfully. Defect inspection initiated.",
    }


@router.get("/jobs")
def list_video_jobs() -> List[Dict[str, Any]]:
    """List all past and present video inspection jobs."""
    return VideoService.list_jobs()


@router.get("/{video_id}/status")
def get_video_status(video_id: str) -> Dict[str, Any]:
    """Poll processing status, progress percentage, and current analysis stage."""
    job = VideoService.get_job(video_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video job '{video_id}' not found",
        )

    return {
        "video_id": job["video_id"],
        "filename": job["filename"],
        "status": job["status"],
        "progress": job["progress"],
        "current_stage": job["current_stage"],
        "sampled_frames": job.get("sampled_frames", 0),
        "total_defects": job.get("total_defects_found", 0),
        "max_severity": job.get("max_severity", "none"),
        "error": job.get("error"),
    }


@router.get("/{video_id}/result")
def get_video_result(video_id: str) -> Dict[str, Any]:
    """Retrieve full video inspection results with clickable defect timeline."""
    job = VideoService.get_job(video_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video job '{video_id}' not found",
        )

    return job
