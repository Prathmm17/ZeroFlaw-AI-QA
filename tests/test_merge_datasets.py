"""Unit tests for dataset merging, class harmonization (dent addition), and Colab notebook validation."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import nbformat

from ml.merge_datasets import (
    DEFAULT_CLASS_MAPPING,
    DEFAULT_TARGET_CLASSES,
    DatasetMerger,
    run_dummy_test,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_default_target_classes_contains_dent():
    """Verify DEFAULT_TARGET_CLASSES includes all 7 classes including 'dent' at index 6."""
    assert len(DEFAULT_TARGET_CLASSES) == 7
    assert DEFAULT_TARGET_CLASSES[6] == "dent"
    assert DEFAULT_TARGET_CLASSES[0] == "crazing"
    assert DEFAULT_TARGET_CLASSES[5] == "scratches"


def test_class_mapping_json_file_validity():
    """Verify data/extra/class_mapping.json exists and defines dent mappings."""
    mapping_file = PROJECT_ROOT / "data" / "extra" / "class_mapping.json"
    assert mapping_file.is_file(), f"Missing {mapping_file}"
    with open(mapping_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "target_classes" in data
    assert data["target_classes"]["6"] == "dent"
    assert "name_to_target_id" in data
    assert data["name_to_target_id"]["dent"] == 6
    assert data["name_to_target_id"]["punching_dent"] == 6


def test_dataset_merger_remapping_logic():
    """Verify DatasetMerger re-indexes extra class names and ids to unified schema."""
    merger = DatasetMerger()

    # Direct name mapping
    assert merger.map_class_id(0, source_name="dent") == 6
    assert merger.map_class_id(10, source_name="punching_dent") == 6
    assert merger.map_class_id(2, source_name="scratch") == 5
    assert merger.map_class_id(4, source_name="pit") == 3
    assert merger.map_class_id(99, source_name="unknown_flaw") is None


def test_merge_datasets_dummy_execution():
    """Run full merger execution on dummy dataset and verify remapping and dataset.yaml."""
    success = run_dummy_test()
    assert success is True


def test_train_colab_notebook_validity_and_contents():
    """Validate that ml/train_colab.ipynb is a valid Jupyter Notebook with all requested parameters."""
    nb_path = PROJECT_ROOT / "ml" / "train_colab.ipynb"
    assert nb_path.is_file(), f"Notebook not found at {nb_path}"

    # 1. Official nbformat schema validation
    nb = nbformat.read(str(nb_path), as_version=4)
    nbformat.validate(nb)
    assert nb.nbformat == 4
    assert len(nb.cells) >= 10

    # 2. Check full source content for required specifications
    full_source = ""
    for cell in nb.cells:
        full_source += "\n" + "".join(cell.source)

    # YOLOv8s option
    assert "yolov8s.pt" in full_source, "Notebook missing 'yolov8s.pt' model option"
    # imgsz 640
    assert "640" in full_source, "Notebook missing 640 image size"
    # 100 epochs
    assert "100" in full_source, "Notebook missing 100 epochs"
    # patience 30
    assert "patience" in full_source and "30" in full_source, "Notebook missing patience 30"
    # mosaic
    assert "mosaic" in full_source.lower(), "Notebook missing mosaic augmentation"
    # hsv
    assert "hsv_h" in full_source and "hsv_s" in full_source, "Notebook missing HSV augmentations"
    # flips
    assert "fliplr" in full_source and "flipud" in full_source, "Notebook missing flip augmentations"
    # mixup 0.1
    assert "mixup" in full_source and "0.1" in full_source, "Notebook missing mixup 0.1"
    # cosine lr
    assert "cos_lr" in full_source, "Notebook missing cosine LR"
    # oversampling for weak classes
    assert "oversample" in full_source.lower() or "oversampling" in full_source.lower(), "Notebook missing oversampling logic"
    # validation and outputs/metrics.json
    assert "metrics.json" in full_source, "Notebook missing outputs/metrics.json"
    assert "model.val(" in full_source, "Notebook missing model.val validation call"
    # export onnx
    assert "format='onnx'" in full_source or 'format="onnx"' in full_source, "Notebook missing ONNX export"

