"""VisionGuard QA — ML Pipeline Package.

Provides defect detection, preprocessing, postprocessing,
high-resolution slicing/tiling, and model management for manufacturing quality assurance.
"""

# NumPy 2.x compatibility patch for ultralytics metrics (np.trapz -> np.trapezoid)
import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = getattr(np, "trapezoid", None)

from ml.anomaly_engine import AnomalyRegion, UnsupervisedAnomalyEngine
from ml.detector import DefectDetector
from ml.model_registry import ModelRegistry
from ml.postprocessor import DetectionPostprocessor
from ml.preprocessor import ImagePreprocessor
from ml.segmenter import DefectMetrology, DefectSegmenter
from ml.tiler import ImageSlice, ImageTiler

__all__ = [
    "DefectDetector",
    "UnsupervisedAnomalyEngine",
    "AnomalyRegion",
    "ImagePreprocessor",
    "DetectionPostprocessor",
    "ModelRegistry",
    "DefectSegmenter",
    "DefectMetrology",
    "ImageTiler",
    "ImageSlice",
]
