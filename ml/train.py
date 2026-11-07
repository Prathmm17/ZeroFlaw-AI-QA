"""Training script for VisionGuard QA defect detection.

Fine-tunes a YOLOv8n model on the synthetic defect dataset.  Handles
dataset validation, ``dataset.yaml`` creation, training execution, and
model artifact copying so the rest of the pipeline always finds the best
weights at ``<project_root>/models/best.pt``.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

import yaml
from ultralytics import YOLO

from ml.model_registry import DEFECT_CLASSES, ModelRegistry

logger: logging.Logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = ModelRegistry.get_project_root()

_MIN_TRAINING_IMAGES: int = 10


# ------------------------------------------------------------------
# Dataset helpers
# ------------------------------------------------------------------


def create_dataset_yaml(data_dir: Path) -> Path:
    """Create or verify the ``dataset.yaml`` file required by YOLO training.

    The file is written to *data_dir*/``dataset.yaml`` with absolute paths
    so that training works regardless of the current working directory.

    Args:
        data_dir: Root directory of the dataset.  Expected sub-structure::

            data_dir/
              images/
                train/
                val/
              labels/
                train/
                val/

    Returns:
        Absolute path to the generated ``dataset.yaml`` file.

    Raises:
        FileNotFoundError: If *data_dir* does not exist.
    """
    data_dir = Path(data_dir).resolve()

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {data_dir}"
        )

    class_names: dict[int, str] = dict(DEFECT_CLASSES)

    dataset_config: dict = {
        "path": str(data_dir),
        "train": str(data_dir / "images" / "train"),
        "val": str(data_dir / "images" / "val"),
        "nc": len(class_names),
        "names": class_names,
    }

    yaml_path: Path = data_dir / "dataset.yaml"

    try:
        with yaml_path.open("w", encoding="utf-8") as fh:
            yaml.dump(dataset_config, fh, default_flow_style=False, sort_keys=False)
        logger.info("Dataset YAML written to %s", yaml_path)
    except OSError as exc:
        logger.exception("Failed to write dataset YAML")
        raise RuntimeError(
            f"Could not write dataset.yaml to {yaml_path}: {exc}"
        ) from exc

    return yaml_path


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------


def validate_dataset(data_dir: Path) -> bool:
    """Validate that a YOLO-format dataset is complete and well-formed.

    Performs the following checks:

    1. ``images/train`` and ``labels/train`` directories exist and are
       non-empty.
    2. At least *_MIN_TRAINING_IMAGES* training images are present.
    3. Every label file contains lines in valid YOLO format
       (``class_id cx cy w h``, with ``class_id`` as an integer and all
       coordinate values as floats in [0, 1]).
    4. Class IDs fall within the range defined by ``DEFECT_CLASSES``.

    Args:
        data_dir: Root of the dataset directory.

    Returns:
        ``True`` if the dataset passes all checks, ``False`` otherwise.
        Warnings for individual issues are emitted via the logger.
    """
    data_dir = Path(data_dir).resolve()
    valid: bool = True

    # --- directory existence ---
    required_dirs: list[Path] = [
        data_dir / "images" / "train",
        data_dir / "labels" / "train",
    ]

    for dir_path in required_dirs:
        if not dir_path.is_dir():
            logger.error("Required directory missing: %s", dir_path)
            valid = False

    if not valid:
        return False

    # --- minimum image count ---
    image_extensions: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    train_images: list[Path] = [
        p
        for p in (data_dir / "images" / "train").iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    ]

    if len(train_images) < _MIN_TRAINING_IMAGES:
        logger.error(
            "Insufficient training images: found %d, need at least %d",
            len(train_images),
            _MIN_TRAINING_IMAGES,
        )
        valid = False

    logger.info("Training images found: %d", len(train_images))

    # --- optional val split check (warn only) ---
    val_images_dir: Path = data_dir / "images" / "val"
    if val_images_dir.is_dir():
        val_count: int = sum(
            1
            for p in val_images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in image_extensions
        )
        if val_count == 0:
            logger.warning("Validation image directory exists but is empty: %s", val_images_dir)
        else:
            logger.info("Validation images found: %d", val_count)
    else:
        logger.warning("No validation image directory found at %s", val_images_dir)

    # --- label format validation ---
    label_dir: Path = data_dir / "labels" / "train"
    label_files: list[Path] = [
        p for p in label_dir.iterdir() if p.is_file() and p.suffix == ".txt"
    ]

    if not label_files:
        logger.error("No label files found in %s", label_dir)
        return False

    logger.info("Training label files found: %d", len(label_files))

    max_class_id: int = max(DEFECT_CLASSES.keys())
    issues_found: int = 0
    _MAX_ISSUES_LOGGED: int = 20

    for label_path in label_files:
        try:
            lines: list[str] = label_path.read_text(encoding="utf-8").strip().splitlines()
        except OSError as exc:
            logger.warning("Cannot read label file %s: %s", label_path.name, exc)
            issues_found += 1
            continue

        for line_no, raw_line in enumerate(lines, start=1):
            line: str = raw_line.strip()
            if not line:
                continue

            parts: list[str] = line.split()
            if len(parts) != 5:
                if issues_found < _MAX_ISSUES_LOGGED:
                    logger.warning(
                        "Label %s line %d: expected 5 values, got %d",
                        label_path.name,
                        line_no,
                        len(parts),
                    )
                issues_found += 1
                valid = False
                continue

            # class_id must be a non-negative integer
            try:
                class_id: int = int(parts[0])
            except ValueError:
                if issues_found < _MAX_ISSUES_LOGGED:
                    logger.warning(
                        "Label %s line %d: class_id '%s' is not an integer",
                        label_path.name,
                        line_no,
                        parts[0],
                    )
                issues_found += 1
                valid = False
                continue

            if class_id < 0 or class_id > max_class_id:
                if issues_found < _MAX_ISSUES_LOGGED:
                    logger.warning(
                        "Label %s line %d: class_id %d outside valid range 0–%d",
                        label_path.name,
                        line_no,
                        class_id,
                        max_class_id,
                    )
                issues_found += 1
                valid = False
                continue

            # x_center, y_center, width, height must be floats in [0, 1]
            for i, coord_name in enumerate(
                ["x_center", "y_center", "width", "height"], start=1
            ):
                try:
                    val: float = float(parts[i])
                except ValueError:
                    if issues_found < _MAX_ISSUES_LOGGED:
                        logger.warning(
                            "Label %s line %d: %s '%s' is not a float",
                            label_path.name,
                            line_no,
                            coord_name,
                            parts[i],
                        )
                    issues_found += 1
                    valid = False
                    break

                if not 0.0 <= val <= 1.0:
                    if issues_found < _MAX_ISSUES_LOGGED:
                        logger.warning(
                            "Label %s line %d: %s=%.6f outside [0, 1]",
                            label_path.name,
                            line_no,
                            coord_name,
                            val,
                        )
                    issues_found += 1
                    valid = False
                    break

    if issues_found > 0:
        logger.warning("Total label issues found: %d", issues_found)
        if issues_found > _MAX_ISSUES_LOGGED:
            logger.warning(
                "(Only the first %d issues were logged individually.)",
                _MAX_ISSUES_LOGGED,
            )

    if valid:
        logger.info("Dataset validation passed ✓")
    else:
        logger.error("Dataset validation failed — see warnings above")

    return valid


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------


def train_model(
    data_dir: Optional[Path] = None,
    epochs: int = 30,
    batch_size: int = 16,
    image_size: int = 640,
    device: str = "cpu",
    project_dir: Optional[Path] = None,
) -> Path:
    """Fine-tune YOLOv8n on the VisionGuard defect dataset.

    Args:
        data_dir: Root of the YOLO dataset directory.  Defaults to
            ``<project_root>/data/training``.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        image_size: Input image resolution (square).
        device: Compute device (``"cpu"``, ``"0"``, ``"cuda:0"``, …).
        project_dir: Directory under which ultralytics saves run
            artefacts.  Defaults to ``<project_root>/runs``.

    Returns:
        Absolute path to the copied ``best.pt`` model file located at
        ``<project_root>/models/best.pt``.

    Raises:
        FileNotFoundError: If the dataset directory or the resulting
            ``best.pt`` file cannot be found.
        RuntimeError: If training fails for any reason.
    """
    if data_dir is None:
        data_dir = PROJECT_ROOT / "data" / "training"
    data_dir = Path(data_dir).resolve()

    if project_dir is None:
        project_dir = PROJECT_ROOT / "runs"
    project_dir = Path(project_dir).resolve()

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}.  "
            "Run the synthetic data generator first."
        )

    # --- dataset yaml ---
    yaml_path: Path = create_dataset_yaml(data_dir)

    # --- load base model ---
    logger.info("Loading base YOLOv8n model for fine-tuning …")
    try:
        model: YOLO = YOLO("yolov8n.pt")
    except Exception as exc:
        logger.exception("Failed to load base YOLOv8n model")
        raise RuntimeError(f"Cannot load base model: {exc}") from exc

    # --- train ---
    logger.info(
        "Starting training — epochs=%d  batch=%d  imgsz=%d  device=%s",
        epochs,
        batch_size,
        image_size,
        device,
    )

    try:
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            batch=batch_size,
            imgsz=image_size,
            device=device,
            project=str(project_dir),
            name="visionguard",
            exist_ok=True,
            pretrained=True,
            verbose=True,
        )
    except Exception as exc:
        logger.exception("Training failed")
        raise RuntimeError(f"YOLOv8 training failed: {exc}") from exc

    # --- locate best.pt produced by training ---
    trained_best: Path = project_dir / "visionguard" / "weights" / "best.pt"

    if not trained_best.is_file():
        # Fallback: try last.pt when best.pt was not saved
        trained_last: Path = project_dir / "visionguard" / "weights" / "last.pt"
        if trained_last.is_file():
            logger.warning(
                "best.pt not found; using last.pt instead from %s", trained_last
            )
            trained_best = trained_last
        else:
            raise FileNotFoundError(
                f"Training completed but no weight file found at "
                f"{trained_best} or {trained_last}. "
                "Check the training logs for errors."
            )

    # --- copy to models/ ---
    models_dir: Path = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    dest: Path = models_dir / "best.pt"

    try:
        shutil.copy2(str(trained_best), str(dest))
        logger.info("Best model copied to %s", dest)
    except OSError as exc:
        logger.exception("Failed to copy best model to models directory")
        raise RuntimeError(
            f"Could not copy model to {dest}: {exc}"
        ) from exc

    return dest.resolve()


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the training script.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed ``argparse.Namespace``.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8n on the VisionGuard defect dataset.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Mini-batch size (default: 16).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Input image resolution (default: 640).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Compute device, e.g. "cpu", "0", "cuda:0" (default: "cpu").',
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to the dataset root (default: <project>/data/training).",
    )

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point for CLI invocation.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    args: argparse.Namespace = _parse_args(argv)

    data_dir: Path = (
        Path(args.data_dir).resolve()
        if args.data_dir is not None
        else PROJECT_ROOT / "data" / "training"
    )

    # --- validate ---
    logger.info("Validating dataset at %s …", data_dir)
    if not validate_dataset(data_dir):
        logger.error(
            "Dataset validation failed.  Fix the issues above before training."
        )
        sys.exit(1)

    # --- train ---
    try:
        model_path: Path = train_model(
            data_dir=data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            device=args.device,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("Training aborted: %s", exc)
        sys.exit(1)

    # --- summary ---
    print("\n" + "=" * 60)
    print("  VisionGuard Training Complete")
    print("=" * 60)
    print(f"  Model saved to : {model_path}")
    print(f"  Epochs          : {args.epochs}")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  Image size      : {args.image_size}")
    print(f"  Device          : {args.device}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
