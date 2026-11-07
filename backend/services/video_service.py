"""Video Processing Service for VisionGuard QA.

Provides:
1. Video file frame-by-frame decoding using OpenCV cv2.VideoCapture.
2. Efficient frame sampling (default 3-5 FPS) for sub-5s inspection latency.
3. High-throughput YOLO defect scanning with dynamic sensitivity thresholding.
4. Annotated thumbnail generation for all detected flaw timestamps.
5. Interactive timeline compilation with clickable timestamps and severity heatmaps.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from backend.config import settings
from backend.services.detection_service import get_detector

logger = logging.getLogger(__name__)

# In-memory job registry for video inspection tasks
_jobs_lock = threading.Lock()
_video_jobs: Dict[str, Dict[str, Any]] = {}


class VideoService:
    """Service orchestrating industrial video defect inspection."""

    @classmethod
    def create_job(cls, video_path: Path, filename: str) -> str:
        """Register a new video inspection job."""
        video_id = f"vid_{uuid.uuid4().hex[:10]}"
        job_data = {
            "video_id": video_id,
            "filename": filename,
            "video_path": str(video_path),
            "video_url": f"/data/uploads/{video_path.name}",
            "status": "queued",
            "progress": 0,
            "current_stage": "Enqueued for analysis",
            "duration_seconds": 0.0,
            "fps": 0.0,
            "total_frames": 0,
            "sampled_frames": 0,
            "total_defects_found": 0,
            "max_severity": "none",
            "timeline": [],
            "summary": {
                "by_class": {},
                "by_severity": {"minor": 0, "major": 0, "critical": 0},
                "peak_timestamp_sec": 0.0,
            },
            "error": None,
            "created_at": time.time(),
        }

        with _jobs_lock:
            _video_jobs[video_id] = job_data

        logger.info("Video inspection job %s created for file: %s", video_id, filename)
        return video_id

    @classmethod
    def get_job(cls, video_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve job data by ID."""
        with _jobs_lock:
            job = _video_jobs.get(video_id)
            if job:
                return dict(job)
        return None

    @classmethod
    def list_jobs(cls) -> List[Dict[str, Any]]:
        """List all video inspection jobs ordered by recency."""
        with _jobs_lock:
            jobs = list(_video_jobs.values())
        return sorted(jobs, key=lambda x: x.get("created_at", 0), reverse=True)

    @classmethod
    def process_video(
        cls,
        video_id: str,
        sample_fps: float = 3.0,
        confidence_threshold: Optional[float] = 0.25,
        ambient_boost: bool = True,
    ) -> Dict[str, Any]:
        """Execute frame-by-frame defect scanning on the specified video file."""
        job = cls.get_job(video_id)
        if not job:
            raise ValueError(f"Job {video_id} not found")

        video_path = Path(job["video_path"])
        if not video_path.is_file():
            with _jobs_lock:
                _video_jobs[video_id]["status"] = "failed"
                _video_jobs[video_id]["error"] = f"Video file not found at {video_path}"
            raise FileNotFoundError(f"Video file not found at {video_path}")

        detector = get_detector()

        with _jobs_lock:
            _video_jobs[video_id]["status"] = "processing"
            _video_jobs[video_id]["current_stage"] = "Opening video stream..."
            _video_jobs[video_id]["progress"] = 5

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            err_msg = f"Failed to open video file: {video_path.name}"
            with _jobs_lock:
                _video_jobs[video_id]["status"] = "failed"
                _video_jobs[video_id]["error"] = err_msg
            raise RuntimeError(err_msg)

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            native_fps = float(cap.get(cv2.CAP_PROP_FPS))
            if native_fps <= 0.0 or np.isnan(native_fps):
                native_fps = 30.0

            duration_sec = total_frames / native_fps if total_frames > 0 else 0.0
            frame_interval = max(1, int(round(native_fps / max(0.5, sample_fps))))

            with _jobs_lock:
                _video_jobs[video_id]["fps"] = round(native_fps, 2)
                _video_jobs[video_id]["duration_seconds"] = round(duration_sec, 2)
                _video_jobs[video_id]["total_frames"] = total_frames
                _video_jobs[video_id]["current_stage"] = "Scanning video frames for surface defects..."

            timeline: List[Dict[str, Any]] = []
            class_counts: Dict[str, int] = {}
            sev_counts: Dict[str, int] = {"minor": 0, "major": 0, "critical": 0}
            total_defects = 0
            sampled_count = 0
            frame_idx = 0
            peak_defect_sec = 0.0
            max_flaws_in_single_frame = 0

            sev_rank = {"none": 0, "minor": 1, "major": 2, "critical": 3}
            highest_sev_rank = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    sampled_count += 1
                    t_sec = round(frame_idx / native_fps, 2)

                    # Run fast YOLO inspection
                    res = detector.detect(
                        frame,
                        deep_verify=False,
                        yolo_only=True,
                        confidence_threshold=confidence_threshold,
                        ambient_boost=ambient_boost,
                    )

                    detections = res.get("detections", [])

                    if detections:
                        frame_max_sev = "minor"
                        for d in detections:
                            s = d.get("severity", "minor")
                            c = d.get("class_name", "defect")
                            class_counts[c] = class_counts.get(c, 0) + 1
                            if s in sev_counts:
                                sev_counts[s] += 1
                            if sev_rank.get(s, 0) > sev_rank.get(frame_max_sev, 0):
                                frame_max_sev = s
                            if sev_rank.get(s, 0) > highest_sev_rank:
                                highest_sev_rank = sev_rank.get(s, 0)

                        total_defects += len(detections)

                        if len(detections) > max_flaws_in_single_frame:
                            max_flaws_in_single_frame = len(detections)
                            peak_defect_sec = t_sec

                        # Generate annotated thumbnail preview
                        thumb = frame.copy()
                        for det in detections:
                            bx1, by1, bx2, by2 = [int(v) for v in det["bbox"]]
                            d_sev = det.get("severity", "minor")
                            color = (
                                (59, 130, 246)
                                if d_sev == "minor"
                                else (16, 185, 129)
                                if d_sev == "passed"
                                else (11, 158, 245)
                                if d_sev == "major"
                                else (94, 63, 244)
                            )
                            # BGR color: critical = red, major = amber, minor = sky
                            bgr_col = (
                                (68, 68, 244)
                                if d_sev == "critical"
                                else (11, 158, 245)
                                if d_sev == "major"
                                else (246, 130, 59)
                            )
                            cv2.rectangle(thumb, (bx1, by1), (bx2, by2), bgr_col, 2)
                            label_str = f"{det.get('class_name', 'defect').upper()} {int(det.get('confidence', 0)*100)}%"
                            cv2.putText(
                                thumb,
                                label_str,
                                (bx1, max(18, by1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (255, 255, 255),
                                1,
                                cv2.LINE_AA,
                            )

                        thumb_filename = f"{video_id}_thumb_{frame_idx}.jpg"
                        thumb_path = settings.RESULTS_DIR / thumb_filename
                        cv2.imwrite(str(thumb_path), thumb)

                        timeline.append(
                            {
                                "timestamp_sec": t_sec,
                                "frame_idx": frame_idx,
                                "defect_count": len(detections),
                                "max_severity": frame_max_sev,
                                "defects": detections,
                                "thumbnail_url": f"/data/results/{thumb_filename}",
                            }
                        )

                    # Update progress
                    calc_progress = min(95, int(10 + (frame_idx / max(1, total_frames)) * 85))
                    with _jobs_lock:
                        _video_jobs[video_id]["progress"] = calc_progress
                        _video_jobs[video_id]["sampled_frames"] = sampled_count
                        _video_jobs[video_id]["total_defects_found"] = total_defects

                frame_idx += 1

            rank_to_sev = {0: "none", 1: "minor", 2: "major", 3: "critical"}
            overall_max_sev = rank_to_sev.get(highest_sev_rank, "none")

            with _jobs_lock:
                _video_jobs[video_id]["status"] = "completed"
                _video_jobs[video_id]["progress"] = 100
                _video_jobs[video_id]["current_stage"] = "Inspection completed"
                _video_jobs[video_id]["sampled_frames"] = sampled_count
                _video_jobs[video_id]["total_defects_found"] = total_defects
                _video_jobs[video_id]["max_severity"] = overall_max_sev
                _video_jobs[video_id]["timeline"] = timeline
                _video_jobs[video_id]["summary"] = {
                    "by_class": class_counts,
                    "by_severity": sev_counts,
                    "peak_timestamp_sec": peak_defect_sec,
                }

            logger.info(
                "Video %s analysis finished: %d frames sampled, %d defects found, max severity: %s",
                video_id,
                sampled_count,
                total_defects,
                overall_max_sev,
            )
            return cls.get_job(video_id) or {}

        except Exception as exc:
            logger.exception("Error processing video %s: %s", video_id, exc)
            with _jobs_lock:
                _video_jobs[video_id]["status"] = "failed"
                _video_jobs[video_id]["error"] = str(exc)
            raise
        finally:
            cap.release()

