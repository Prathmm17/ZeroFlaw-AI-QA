"""Unit tests for forward and inverse letterbox coordinate mapping.

Validates that non-square images (landscape 1000x600, portrait 600x1000, and square 800x800)
map forward to letterbox space and back to original coordinates within 2px tolerance.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# NumPy 2.x patch
if not hasattr(np, "trapz"):
    np.trapz = getattr(np, "trapezoid", None)

import torch
# PyTorch 2.6 weights_only patch
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

from ml.postprocessor import DetectionPostprocessor
from ml.preprocessor import ImagePreprocessor


def make_dummy_yolo_result(boxes_xyxy: list[list[float]], confs: list[float], cls_ids: list[int]):
    """Construct a lightweight mock of Ultralytics Results object."""
    class SingleBox:
        def __init__(self, xyxy, conf, cls_id):
            self.xyxy = torch.tensor([xyxy], dtype=torch.float32)
            self.conf = torch.tensor([conf], dtype=torch.float32)
            self.cls = torch.tensor([cls_id], dtype=torch.float32)

    class DummyBoxes:
        def __init__(self, xyxy_list, conf_list, cls_list):
            self._boxes = [
                SingleBox(b, c, cl) for b, c, cl in zip(xyxy_list, conf_list, cls_list)
            ]

        def __iter__(self):
            return iter(self._boxes)

        def __len__(self):
            return len(self._boxes)

    boxes = DummyBoxes(boxes_xyxy, confs, cls_ids)
    return SimpleNamespace(boxes=boxes, orig_shape=(640, 640))


@pytest.mark.parametrize(
    "width,height,rect",
    [
        # 1. Landscape 1000x600 (width=1000, height=600)
        (1000, 600, (150.0, 120.0, 480.0, 360.0)),
        # 2. Portrait 600x1000 (width=600, height=1000)
        (600, 1000, (80.0, 220.0, 350.0, 750.0)),
        # 3. Square 800x800 (width=800, height=800)
        (800, 800, (100.0, 100.0, 500.0, 600.0)),
    ],
)
def test_preprocessor_inverse_letterbox(width: int, height: int, rect: Tuple[float, float, float, float]):
    """Test mathematical roundtrip precision of ImagePreprocessor inverse_letterbox_coords."""
    preprocessor = ImagePreprocessor(target_size=(640, 640))
    scale, pad_left, pad_top = preprocessor.get_letterbox_params((height, width))

    x1_orig, y1_orig, x2_orig, y2_orig = rect

    # Forward letterbox transformation
    x1_lb = x1_orig * scale + pad_left
    y1_lb = y1_orig * scale + pad_top
    x2_lb = x2_orig * scale + pad_left
    y2_lb = y2_orig * scale + pad_top

    # Inverse letterbox transformation
    x1_inv, y1_inv, x2_inv, y2_inv = preprocessor.inverse_letterbox_coords(
        (x1_lb, y1_lb, x2_lb, y2_lb),
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        orig_shape=(height, width),
    )

    # Assert recovery is accurate within 2.0 pixels
    assert abs(x1_inv - x1_orig) <= 2.0, f"x1 mismatch: orig={x1_orig}, recovered={x1_inv}"
    assert abs(y1_inv - y1_orig) <= 2.0, f"y1 mismatch: orig={y1_orig}, recovered={y1_inv}"
    assert abs(x2_inv - x2_orig) <= 2.0, f"x2 mismatch: orig={x2_orig}, recovered={x2_inv}"
    assert abs(y2_inv - y2_orig) <= 2.0, f"y2 mismatch: orig={y2_orig}, recovered={y2_inv}"


@pytest.mark.parametrize(
    "width,height,rect",
    [
        (1000, 600, (200.0, 150.0, 600.0, 450.0)),
        (600, 1000, (120.0, 180.0, 450.0, 800.0)),
        (640, 640, (50.0, 50.0, 300.0, 300.0)),
    ],
)
def test_postprocessor_inverse_mapping(width: int, height: int, rect: Tuple[float, float, float, float]):
    """Test that DetectionPostprocessor.process maps YOLO letterbox boxes back to original bounds."""
    preprocessor = ImagePreprocessor(target_size=(640, 640))
    scale, pad_left, pad_top = preprocessor.get_letterbox_params((height, width))

    x1_orig, y1_orig, x2_orig, y2_orig = rect

    # Forward letterbox box
    x1_lb = x1_orig * scale + pad_left
    y1_lb = y1_orig * scale + pad_top
    x2_lb = x2_orig * scale + pad_left
    y2_lb = y2_orig * scale + pad_top

    mock_result = make_dummy_yolo_result(
        boxes_xyxy=[[x1_lb, y1_lb, x2_lb, y2_lb]],
        confs=[0.88],
        cls_ids=[0],
    )

    postprocessor = DetectionPostprocessor(min_confidence=0.20)
    detections = postprocessor.process(
        mock_result,
        image_shape=(height, width),
        letterbox_params=(scale, pad_left, pad_top),
    )

    assert len(detections) == 1
    det = detections[0]
    bx1, by1, bx2, by2 = det.bbox

    assert abs(bx1 - x1_orig) <= 2.0, f"bx1 mismatch: {bx1} vs {x1_orig}"
    assert abs(by1 - y1_orig) <= 2.0, f"by1 mismatch: {by1} vs {y1_orig}"
    assert abs(bx2 - x2_orig) <= 2.0, f"bx2 mismatch: {bx2} vs {x2_orig}"
    assert abs(by2 - y2_orig) <= 2.0, f"by2 mismatch: {by2} vs {y2_orig}"
