"""Zero-Shot Gemini Flash Visual Grounding & Semantic Defect Localization Engine.

Uses Gemini 2.5/3.8 Flash multimodal perception to visually ground, localize,
and classify complex or unseen manufacturing defects with exact bounding coordinates
and failure physics, with a deterministic offline fallback for air-gapped environments.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GroundedDefect:
    """A visually grounded defect prediction from the VLM or Anomaly CV engine.

    Attributes:
        box_2d: Normalized bounding box [ymin, xmin, ymax, xmax] in [0.0, 1.0].
        label: Semantic defect description (e.g. "unclassified_anomaly", "thermal_burn").
        confidence: Detection confidence rating [0.0, 1.0].
        severity: Industrial risk tier ("critical", "major", "minor").
        failure_mechanism: Detailed physics-of-failure mechanism.
        source: Detection source: "vlm" (Gemini API) or "anomaly_cv" (OpenCV fallback).
    """

    box_2d: Tuple[float, float, float, float]
    label: str
    confidence: float
    severity: str
    failure_mechanism: str
    source: str = "vlm"


class GeminiVisualGrounder:
    """Multimodal Zero-Shot Defect Grounding Engine."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._client = None
        self._sdk_type = None

        if self.api_key:
            try:
                # Try google.genai client
                try:
                    from google import genai
                    self._client = genai.Client(api_key=self.api_key)
                    self._sdk_type = "genai"
                    logger.info("Initialized GeminiVisualGrounder via google-genai Client.")
                except ImportError:
                    import google.generativeai as genai
                    genai.configure(api_key=self.api_key)
                    self._client = genai.GenerativeModel("gemini-2.5-flash")
                    self._sdk_type = "generativeai"
                    logger.info("Initialized GeminiVisualGrounder via google-generativeai.")
            except Exception as exc:
                logger.warning("Could not initialize Gemini API client: %s. Using deterministic fallback.", exc)
                self._client = None

    def is_api_active(self) -> bool:
        """Return True if an active Gemini client is configured."""
        return self._client is not None

    def ground_defects(
        self,
        image: np.ndarray,
        material_hint: str = "Industrial Component Specimen",
    ) -> List[GroundedDefect]:
        """Perform zero-shot visual grounding on the image to locate defects.

        Args:
            image: Image array (H, W, 3) in BGR.
            material_hint: Contextual material or part description.

        Returns:
            List of GroundedDefect objects with normalized bounding boxes and semantic labels.
        """
        if self._client:
            try:
                return self._query_gemini_grounding(image, material_hint)
            except Exception as exc:
                logger.warning("Gemini visual grounding query failed (%s); falling back to deterministic localizer.", exc)

        return self._deterministic_fallback_grounding(image)

    def _query_gemini_grounding(self, image: np.ndarray, material_hint: str) -> List[GroundedDefect]:
        """Execute multimodal prompt on Gemini Flash to ground defects."""
        h, w = image.shape[:2]
        _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64_data = base64.b64encode(buffer).decode("utf-8")

        prompt = f"""You are an expert Manufacturing QA and Optical Metrology Engineer.
Examine this industrial specimen image ({material_hint}).
Your task is to detect and visually ground any physical manufacturing defects, including:
- Hairline cracks, fatigue fractures, or micro-voids
- Thermal burns, scorched plastic, or charring
- Scratches, tooling gouges, or mechanical abrasion
- Surface pitting, inclusions, or cavitation

Return a strict JSON list of detected defects. Each item MUST have:
{{
  "box_2d": [ymin, xmin, ymax, xmax],  // normalized floats between 0.0 and 1.0
  "label": "crazing | scratches | patches | pitted_surface | inclusion | thermal_burn",
  "confidence": float between 0.60 and 0.99,
  "severity": "critical | major | minor",
  "failure_mechanism": "detailed physical mechanism description"
}}

If the specimen is completely normal with zero flaws, return an empty list: []
Return ONLY raw JSON, without markdown formatting."""

        if self._sdk_type == "genai":
            response = self._client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64_data}},
                    prompt,
                ],
            )
            raw_text = response.text
        else:
            img_part = {"mime_type": "image/jpeg", "data": buffer.tobytes()}
            response = self._client.generate_content([img_part, prompt])
            raw_text = response.text

        clean_json = raw_text.strip()
        if clean_json.startswith("```"):
            clean_json = clean_json.split("\n", 1)[1]
            if clean_json.endswith("```"):
                clean_json = clean_json.rsplit("\n", 1)[0]

        data = json.loads(clean_json)
        results: List[GroundedDefect] = []

        for item in data:
            box = item.get("box_2d", [0, 0, 0, 0])
            # Ensure box format is [ymin, xmin, ymax, xmax] normalized
            ymin = max(0.0, min(1.0, float(box[0])))
            xmin = max(0.0, min(1.0, float(box[1])))
            ymax = max(ymin, min(1.0, float(box[2])))
            xmax = max(xmin, min(1.0, float(box[3])))

            # Normalize label
            raw_label = item.get("label", "patches").lower()
            if "burn" in raw_label or "scorch" in raw_label:
                mapped_label = "patches"
            elif "crack" in raw_label or "fissure" in raw_label:
                mapped_label = "crazing"
            elif "scratch" in raw_label or "gouge" in raw_label:
                mapped_label = "scratches"
            elif "pit" in raw_label:
                mapped_label = "pitted_surface"
            else:
                mapped_label = raw_label

            results.append(
                GroundedDefect(
                    box_2d=(ymin, xmin, ymax, xmax),
                    label=mapped_label,
                    confidence=float(item.get("confidence", 0.85)),
                    severity=item.get("severity", "critical").lower(),
                    failure_mechanism=item.get("failure_mechanism", "Surface morphological irregularity"),
                    source="vlm",
                )
            )

        return results

    def _deterministic_fallback_grounding(self, image: np.ndarray) -> List[GroundedDefect]:
        """Deterministic physical saliency fallback when offline or no API key."""
        from ml.anomaly_engine import UnsupervisedAnomalyEngine

        engine = UnsupervisedAnomalyEngine(sensitivity=0.72)
        anomalies = engine.detect_anomalies(image, max_detections=4)
        results: List[GroundedDefect] = []

        for anom in anomalies:
            # anom.bbox is (x1, y1, x2, y2) normalized
            x1, y1, x2, y2 = anom.bbox
            box_2d = (y1, x1, y2, x2)  # [ymin, xmin, ymax, xmax]

            # Out-of-domain / CV fallback: label as "unclassified_anomaly" (never invent a class name)
            label = "unclassified_anomaly"
            sev = "critical" if anom.anomaly_score >= 0.75 or anom.pixel_area > 500 else "major"
            mechanism = "Unclassified surface anomaly detected via local unsupervised morphological gradient filter"

            results.append(
                GroundedDefect(
                    box_2d=box_2d,
                    label=label,
                    confidence=float(anom.anomaly_score),
                    severity=sev,
                    failure_mechanism=mechanism,
                    source="anomaly_cv",
                )
            )

        return results
