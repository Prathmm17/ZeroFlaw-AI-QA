"""Image preprocessing module for the VisionGuard QA defect detection pipeline.

Handles loading, resizing (with letterboxing), normalizing, and validating
images before they are fed into a YOLO-based defect detection model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Preprocesses images for YOLO-based defect detection inference.

    Supports loading images from file paths, raw bytes, or numpy arrays,
    resizing with aspect-ratio-preserving letterboxing, and pixel-value
    normalization.

    Attributes:
        target_size: Target (width, height) for the YOLO model input.
    """

    # Reasonable bounds for image dimensions (in pixels).
    _MIN_DIM: int = 8
    _MAX_DIM: int = 16384

    def __init__(self, target_size: tuple[int, int] = (640, 640)) -> None:
        """Initialise the preprocessor.

        Args:
            target_size: Desired output size as ``(width, height)``.
                         Defaults to ``(640, 640)`` which is the standard
                         YOLO input resolution.
        """
        if (
            not isinstance(target_size, tuple)
            or len(target_size) != 2
            or not all(isinstance(d, int) and d > 0 for d in target_size)
        ):
            raise ValueError(
                f"target_size must be a tuple of two positive integers, got {target_size}"
            )

        self.target_size: tuple[int, int] = target_size
        self._original_hw: tuple[int, int] | None = None
        logger.info("ImagePreprocessor initialised with target_size=%s", target_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_image(self, source: Union[str, Path, np.ndarray, bytes]) -> np.ndarray:
        """Load an image from a file path, numpy array, or raw bytes.

        The loaded image is always returned in BGR colour order (OpenCV
        default).  The original height and width are stored in
        ``_original_hw`` for later retrieval via
        :meth:`get_original_dimensions`.

        Args:
            source: One of:
                * ``str`` or ``Path`` – filesystem path to an image file.
                * ``np.ndarray`` – an already-loaded image array.
                * ``bytes`` – raw image bytes (e.g. from an HTTP response).

        Returns:
            BGR ``np.ndarray`` with dtype ``uint8``.

        Raises:
            FileNotFoundError: If *source* is a path that does not exist.
            ValueError: If the image cannot be decoded or the source type
                is unsupported.
        """
        try:
            if isinstance(source, (str, Path)):
                image = self._load_from_path(Path(source))
            elif isinstance(source, np.ndarray):
                image = self._load_from_array(source)
            elif isinstance(source, bytes):
                image = self._load_from_bytes(source)
            else:
                raise ValueError(
                    f"Unsupported source type: {type(source).__name__}. "
                    "Expected str, Path, np.ndarray, or bytes."
                )
        except (FileNotFoundError, ValueError):
            # Re-raise known errors without wrapping.
            raise
        except Exception as exc:
            logger.error("Unexpected error while loading image: %s", exc)
            raise ValueError(f"Failed to load image: {exc}") from exc

        self._original_hw = (image.shape[0], image.shape[1])
        logger.debug(
            "Image loaded – original dimensions (h×w): %d×%d",
            *self._original_hw,
        )
        return image

    def resize(self, image: np.ndarray) -> np.ndarray:
        """Resize an image with aspect-ratio-preserving letterboxing.

        The image is scaled so that it fits within :attr:`target_size`
        without distortion.  Any remaining border is filled with a neutral
        gray ``(114, 114, 114)`` – the standard YOLO letterbox colour.

        Args:
            image: BGR ``np.ndarray`` to resize.

        Returns:
            Letterboxed BGR ``np.ndarray`` of shape
            ``(target_h, target_w, channels)`` with dtype ``uint8``.

        Raises:
            ValueError: If *image* fails validation.
        """
        if not self.validate_image(image):
            raise ValueError("Cannot resize an invalid image.")

        target_w, target_h = self.target_size
        img_h, img_w = image.shape[:2]

        # Compute the uniform scale factor.
        scale = min(target_w / img_w, target_h / img_h)
        new_w = int(round(img_w * scale))
        new_h = int(round(img_h * scale))

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create the canvas and paste the resized image centred.
        channels = image.shape[2] if image.ndim == 3 else 1
        if image.ndim == 3:
            canvas = np.full(
                (target_h, target_w, channels), 114, dtype=np.uint8
            )
        else:
            canvas = np.full((target_h, target_w), 114, dtype=np.uint8)

        pad_top = (target_h - new_h) // 2
        pad_left = (target_w - new_w) // 2

        canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized

        logger.debug(
            "Resized %dx%d → %dx%d (scale=%.4f, pad_top=%d, pad_left=%d)",
            img_w,
            img_h,
            target_w,
            target_h,
            scale,
            pad_top,
            pad_left,
        )
        return canvas

    def get_letterbox_params(
        self, image_shape: tuple[int, ...]
    ) -> tuple[float, int, int]:
        """Calculate letterbox scaling and padding parameters for an image shape.

        Args:
            image_shape: (height, width) or (height, width, channels).

        Returns:
            Tuple of (scale, pad_left, pad_top).
        """
        target_w, target_h = self.target_size
        img_h, img_w = int(image_shape[0]), int(image_shape[1])
        scale = min(target_w / img_w, target_h / img_h)
        new_w = int(round(img_w * scale))
        new_h = int(round(img_h * scale))
        pad_left = (target_w - new_w) // 2
        pad_top = (target_h - new_h) // 2
        return float(scale), int(pad_left), int(pad_top)

    @staticmethod
    def inverse_letterbox_coords(
        bbox: tuple[float, float, float, float],
        scale: float,
        pad_left: int,
        pad_top: int,
        orig_shape: tuple[int, ...],
    ) -> tuple[float, float, float, float]:
        """Map a bounding box from letterbox space back to original image space.

        x = (x_lb - pad_left) / scale
        y = (y_lb - pad_top) / scale
        Then clip to image bounds [0, orig_w] and [0, orig_h].

        Args:
            bbox: (x1_lb, y1_lb, x2_lb, y2_lb) in letterbox space.
            scale: Scaling factor applied during letterboxing.
            pad_left: Horizontal padding in pixels added during letterboxing.
            pad_top: Vertical padding in pixels added during letterboxing.
            orig_shape: (height, width) of the original unpadded image.

        Returns:
            Tuple of (x1, y1, x2, y2) mapped to original image bounds.
        """
        img_h, img_w = float(orig_shape[0]), float(orig_shape[1])
        x1_lb, y1_lb, x2_lb, y2_lb = bbox
        x1 = max(0.0, min(img_w, (x1_lb - pad_left) / scale))
        y1 = max(0.0, min(img_h, (y1_lb - pad_top) / scale))
        x2 = max(0.0, min(img_w, (x2_lb - pad_left) / scale))
        y2 = max(0.0, min(img_h, (y2_lb - pad_top) / scale))
        return (round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2))

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize pixel values to the ``[0, 1]`` float32 range.

        Args:
            image: Input ``np.ndarray`` (any dtype).

        Returns:
            ``np.ndarray`` with dtype ``float32`` and values in ``[0, 1]``.

        Raises:
            ValueError: If *image* fails validation.
        """
        if not self.validate_image(image):
            raise ValueError("Cannot normalize an invalid image.")

        normalized: np.ndarray = image.astype(np.float32) / 255.0
        logger.debug("Image normalized to float32 [0, 1].")
        return normalized

    def preprocess(self, source: Union[str, Path, np.ndarray, bytes]) -> np.ndarray:
        """Run the full preprocessing pipeline: load → resize.

        Normalization is intentionally omitted because YOLO performs its
        own internal normalization.

        Args:
            source: Any source accepted by :meth:`load_image`.

        Returns:
            Letterboxed BGR ``np.ndarray`` with dtype ``uint8``, sized to
            :attr:`target_size`, ready for YOLO inference.
        """
        image = self.load_image(source)
        image = self.resize(image)
        logger.info(
            "Preprocessing complete – output shape: %s, dtype: %s",
            image.shape,
            image.dtype,
        )
        return image

    def validate_image(self, image: np.ndarray) -> bool:
        """Check whether an image array is valid for processing.

        Validation criteria:
        * Not ``None`` and is an ``np.ndarray``.
        * Has 2 (grayscale) or 3 (colour) dimensions.
        * Height and width are within ``[_MIN_DIM, _MAX_DIM]``.
        * The array is not empty (``size > 0``).

        Args:
            image: Array to validate.

        Returns:
            ``True`` if the image passes all checks, ``False`` otherwise.
        """
        if image is None or not isinstance(image, np.ndarray):
            logger.warning("Validation failed: image is None or not an ndarray.")
            return False

        if image.size == 0:
            logger.warning("Validation failed: image array is empty.")
            return False

        if image.ndim not in (2, 3):
            logger.warning(
                "Validation failed: unexpected ndim=%d (expected 2 or 3).",
                image.ndim,
            )
            return False

        h, w = image.shape[:2]
        if not (self._MIN_DIM <= h <= self._MAX_DIM) or not (
            self._MIN_DIM <= w <= self._MAX_DIM
        ):
            logger.warning(
                "Validation failed: dimensions %dx%d outside allowed range [%d, %d].",
                w,
                h,
                self._MIN_DIM,
                self._MAX_DIM,
            )
            return False

        return True

    def get_original_dimensions(self) -> tuple[int, int] | None:
        """Return the dimensions of the last loaded image before preprocessing.

        Returns:
            ``(height, width)`` tuple, or ``None`` if no image has been
            loaded yet.
        """
        return self._original_hw

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_from_path(self, path: Path) -> np.ndarray:
        """Load an image from a filesystem path.

        Args:
            path: Path to the image file.

        Returns:
            BGR ``np.ndarray``.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If OpenCV cannot decode the file.
        """
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to decode image file: {path}")

        logger.debug("Loaded image from path: %s", path)
        return image

    def _load_from_array(self, array: np.ndarray) -> np.ndarray:
        """Validate and return an image that is already a numpy array.

        Args:
            array: Existing image array.

        Returns:
            The same array (after validation).

        Raises:
            ValueError: If the array is not a valid image.
        """
        if not self.validate_image(array):
            raise ValueError("Provided numpy array is not a valid image.")

        logger.debug("Loaded image from numpy array, shape=%s", array.shape)
        return array

    def _load_from_bytes(self, raw: bytes) -> np.ndarray:
        """Decode an image from raw bytes.

        Args:
            raw: Raw image bytes (JPEG, PNG, etc.).

        Returns:
            BGR ``np.ndarray``.

        Raises:
            ValueError: If the bytes cannot be decoded into an image.
        """
        if not raw:
            raise ValueError("Cannot decode image from empty bytes.")

        buf = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image from bytes.")

        logger.debug("Loaded image from bytes (%d bytes).", len(raw))
        return image
