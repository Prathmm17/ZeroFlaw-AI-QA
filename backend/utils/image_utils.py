"""Image utility functions for VisionGuard QA.

Provides base64 conversions, file saving, and image format helpers.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def encode_image_to_base64(image: np.ndarray, format_ext: str = ".jpg", quality: int = 85) -> str:
    """Encode an OpenCV BGR image array into a base64 JPEG/PNG string."""
    try:
        encode_params = []
        if format_ext.lower() in [".jpg", ".jpeg"]:
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        elif format_ext.lower() == ".png":
            encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 6]

        success, buffer = cv2.imencode(format_ext, image, encode_params)
        if not success:
            raise ValueError(f"Failed to encode image with format {format_ext}")

        base64_str = base64.b64encode(buffer).decode("utf-8")
        mime = "image/png" if format_ext.lower() == ".png" else "image/jpeg"
        return f"data:{mime};base64,{base64_str}"
    except Exception as exc:
        logger.error("Error encoding image to base64: %s", exc)
        raise


def decode_base64_to_bytes(base64_str: str) -> bytes:
    """Decode a base64 data URI or raw string into raw image bytes."""
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    return base64.b64decode(base64_str)


def decode_base64_to_image(base64_str: str) -> np.ndarray:
    """Decode a base64 data URI or raw string into an OpenCV BGR image array."""
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]

        image_data = base64.b64decode(base64_str)
        np_arr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("cv2.imdecode returned None; invalid image data")
        return image
    except Exception as exc:
        logger.error("Error decoding base64 string to image: %s", exc)
        raise ValueError(f"Could not decode base64 image: {exc}") from exc


def generate_thumbnail(image: np.ndarray, max_dim: int = 240) -> np.ndarray:
    """Resize image to fit within max_dim while preserving aspect ratio."""
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image.copy()

    scale = max_dim / float(max(h, w))
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

