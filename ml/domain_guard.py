"""Domain Guard & Visual Out-of-Distribution Assessment Engine.

Computes physical optical domain metrics (mean color saturation, texture/Laplacian variance)
combined with the maximum supervised YOLO confidence to certify whether an inspected
specimen matches the training domain (flat grayscale steel plates) or represents
an out-of-domain 3D industrial specimen.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DOMAIN_WARNING_MESSAGE = (
    "Image differs from the training domain (flat steel surface). Result may be unreliable."
)


@dataclass
class DomainGuardAssessment:
    """Domain trust certificate and diagnostic metrics.

    Attributes:
        trust: Trust level certification ("HIGH" or "LOW").
        warning: Warning message if trust is LOW, else None.
        color_saturation: Mean HSV color saturation [0.0, 255.0].
        texture_variance: Laplacian contrast variance.
        max_yolo_confidence: Maximum confidence emitted by YOLO for this image.
        reason: Human-readable diagnostic rationale.
    """

    trust: str
    warning: Optional[str]
    color_saturation: float
    texture_variance: float
    max_yolo_confidence: float
    reason: str


class DomainGuard:
    """Certifies whether an image complies with the flat steel training distribution."""

    def __init__(
        self,
        saturation_threshold: float = 12.0,
        texture_var_threshold: float = 350.0,
        min_yolo_conf_threshold: float = 0.40,
    ) -> None:
        self.saturation_threshold = saturation_threshold
        self.texture_var_threshold = texture_var_threshold
        self.min_yolo_conf_threshold = min_yolo_conf_threshold

    def assess(
        self,
        image: np.ndarray,
        max_yolo_confidence: float = 0.0,
    ) -> DomainGuardAssessment:
        """Assess optical domain compliance of an industrial specimen image.

        Args:
            image: BGR or Grayscale input image array.
            max_yolo_confidence: Highest confidence score from supervised YOLO detections.

        Returns:
            DomainGuardAssessment dataclass with trust level ("HIGH" | "LOW") and metrics.
        """
        if image is None or image.size == 0:
            return DomainGuardAssessment(
                trust="LOW",
                warning=DOMAIN_WARNING_MESSAGE,
                color_saturation=0.0,
                texture_variance=0.0,
                max_yolo_confidence=max_yolo_confidence,
                reason="Empty or invalid image array",
            )

        # 1. Color Saturation (HSV space)
        if len(image.shape) == 3 and image.shape[2] == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            color_saturation = float(np.mean(hsv[:, :, 1]))
        else:
            color_saturation = 0.0

        # 2. Texture / Laplacian Variance
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = float(lap.var())

        # 3. Decision Boundary Evaluation
        # Flat steel specimens in NEU-DET are monochrome (saturation < 10)
        # with well-defined defect features recognized by YOLO.
        reasons = []
        is_low_trust = False

        # Strong out-of-domain natural color photography (e.g. phone photos, painted/colored components)
        if color_saturation > 25.0:
            is_low_trust = True
            reasons.append(
                f"high color saturation ({color_saturation:.1f} > 25.0)"
            )

        # Strong out-of-domain 3D geometry / non-planar shadow gradients
        if texture_variance > 700.0:
            is_low_trust = True
            reasons.append(
                f"complex 3D/lighting texture variance ({texture_variance:.1f} > 700.0)"
            )

        if color_saturation > self.saturation_threshold and max_yolo_confidence < self.min_yolo_conf_threshold:
            is_low_trust = True
            reasons.append(
                f"high color saturation ({color_saturation:.1f} > {self.saturation_threshold:.1f})"
            )

        if texture_variance > self.texture_var_threshold and max_yolo_confidence < self.min_yolo_conf_threshold:
            is_low_trust = True
            reasons.append(
                f"complex 3D/lighting texture variance ({texture_variance:.1f} > {self.texture_var_threshold:.1f})"
            )

        if max_yolo_confidence < 0.20 and (color_saturation > 8.0 or texture_variance > 200.0):
            is_low_trust = True
            reasons.append(
                f"supervised model confidence collapsed ({max_yolo_confidence:.2f} < 0.20)"
            )

        if is_low_trust:
            trust = "LOW"
            warning = DOMAIN_WARNING_MESSAGE
            reason_str = "Out-of-domain: " + ", ".join(reasons)
            logger.info("DomainGuard evaluated trust=LOW — %s", reason_str)
        else:
            trust = "HIGH"
            warning = None
            reason_str = "In-domain planar steel surface confirmed"
            logger.debug("DomainGuard evaluated trust=HIGH — %s", reason_str)

        return DomainGuardAssessment(
            trust=trust,
            warning=warning,
            color_saturation=round(color_saturation, 2),
            texture_variance=round(texture_variance, 2),
            max_yolo_confidence=round(max_yolo_confidence, 4),
            reason=reason_str,
        )

