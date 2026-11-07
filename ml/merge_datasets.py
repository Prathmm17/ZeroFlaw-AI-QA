"""Dataset merger and class harmonization pipeline for VisionGuard QA.

Merges supplementary industrial defect datasets (such as GC10-DET, Severstal Steel,
or NEU-DET) into the VisionGuard multi-domain dataset with cross-dataset class
re-indexing and ontology alignment, including addition of novel classes like 'dent'.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("merge_datasets")

# Default Target Ontology for VisionGuard QA with 7 classes (6 base + dent)
DEFAULT_TARGET_CLASSES: Dict[int, str] = {
    0: "crazing",
    1: "inclusion",
    2: "patches",
    3: "pitted_surface",
    4: "rolled_in_scale",
    5: "scratches",
    6: "dent",
}

# Default Class Mapping for GC10-DET / Severstal / Industrial Benchmarks
DEFAULT_CLASS_MAPPING: Dict[str, Any] = {
    "target_classes": {
        "0": "crazing",
        "1": "inclusion",
        "2": "patches",
        "3": "pitted_surface",
        "4": "rolled_in_scale",
        "5": "scratches",
        "6": "dent",
    },
    "name_to_target_id": {
        "crazing": 0,
        "crack": 0,
        "micro_crack": 0,
        "crease": 0,
        "inclusion": 1,
        "slag_inclusion": 1,
        "slag": 1,
        "patches": 2,
        "patch": 2,
        "surface_oxidation": 2,
        "scab": 2,
        "pitted_surface": 3,
        "pit": 3,
        "pitting": 3,
        "rolled_pit": 3,
        "rolled_in_scale": 4,
        "scale": 4,
        "mill_scale": 4,
        "scratches": 5,
        "scratch": 5,
        "abrasion": 5,
        "dent": 6,
        "punching_dent": 6,
        "mechanical_dent": 6,
        "dimple": 6,
    },
    "id_to_target_id": {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
    },
}


class DatasetMerger:
    """Manages merging external YOLO-formatted datasets into a consolidated dataset."""

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        mapping_config: Optional[Dict[str, Any]] = None,
        mapping_file: Optional[Path] = None,
    ) -> None:
        self.base_dir = Path(base_dir) if base_dir else PROJECT_ROOT / "data" / "real_industrial_dataset"
        self.mapping = self._load_mapping(mapping_config, mapping_file)
        self.target_classes = {
            int(k): v for k, v in self.mapping.get("target_classes", DEFAULT_TARGET_CLASSES).items()
        }

    def _load_mapping(
        self,
        config: Optional[Dict[str, Any]],
        file_path: Optional[Path],
    ) -> Dict[str, Any]:
        if config:
            return config
        if file_path and Path(file_path).is_file():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Failed reading mapping file %s: %s; using default mapping.", file_path, e)
        return DEFAULT_CLASS_MAPPING

    def map_class_id(
        self,
        source_id: int,
        source_name: Optional[str] = None,
        source_class_names: Optional[Dict[int, str]] = None,
    ) -> Optional[int]:
        """Map a source dataset class ID to the unified target class ID."""
        # 1. Check if source class name is provided or resolvable
        if not source_name and source_class_names and source_id in source_class_names:
            source_name = source_class_names[source_id]

        if source_name:
            norm_name = source_name.lower().strip().replace("-", "_").replace(" ", "_")
            name_map = self.mapping.get("name_to_target_id", {})
            if norm_name in name_map:
                return int(name_map[norm_name])

        # 2. Check direct ID mapping
        id_map = self.mapping.get("id_to_target_id", {})
        str_id = str(source_id)
        if str_id in id_map:
            return int(id_map[str_id])

        # 3. Direct passthrough if in target range
        if source_id in self.target_classes:
            return source_id

        return None

    def merge(
        self,
        extra_dir: Path,
        output_dir: Path,
        extra_prefix: str = "extra",
        copy_base: bool = True,
        extra_names: Optional[Dict[int, str]] = None,
    ) -> Dict[str, Any]:
        """Merge base dataset and extra dataset into output_dir.

        Args:
            extra_dir: Path to directory containing images/ and labels/ of the extra dataset.
            output_dir: Target destination path for unified dataset.
            extra_prefix: Filename prefix to prevent collisions.
            copy_base: Whether to include images/labels from base_dir.
            extra_names: Optional class names mapping for source IDs in extra_dir.

        Returns:
            Dictionary with merge metrics and bounding box counts per class.
        """
        extra_dir = Path(extra_dir).resolve()
        output_dir = Path(output_dir).resolve()

        if not extra_dir.is_dir():
            raise FileNotFoundError(f"Extra dataset directory does not exist: {extra_dir}")

        for split in ("train", "val"):
            (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        stats = {
            "base_copied_images": 0,
            "base_copied_labels": 0,
            "extra_merged_images": 0,
            "extra_merged_labels": 0,
            "total_boxes_remapped": 0,
            "boxes_dropped_unmapped": 0,
            "per_class_box_counts": {name: 0 for name in self.target_classes.values()},
        }

        # Step 1: Copy base dataset if requested
        if copy_base and self.base_dir.is_dir():
            logger.info("Transferring base dataset from %s to %s...", self.base_dir, output_dir)
            for split in ("train", "val"):
                base_img_dir = self.base_dir / "images" / split
                base_lbl_dir = self.base_dir / "labels" / split

                if base_img_dir.is_dir():
                    for img_file in base_img_dir.glob("*.*"):
                        if img_file.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
                            dest = output_dir / "images" / split / img_file.name
                            shutil.copyfile(img_file, dest)
                            stats["base_copied_images"] += 1

                if base_lbl_dir.is_dir():
                    for lbl_file in base_lbl_dir.glob("*.txt"):
                        dest = output_dir / "labels" / split / lbl_file.name
                        shutil.copyfile(lbl_file, dest)
                        stats["base_copied_labels"] += 1
                        # Count existing boxes
                        with open(lbl_file, "r", encoding="utf-8") as lf:
                            for line in lf:
                                parts = line.strip().split()
                                if parts:
                                    try:
                                        cid = int(parts[0])
                                        if cid in self.target_classes:
                                            stats["per_class_box_counts"][self.target_classes[cid]] += 1
                                    except ValueError:
                                        pass

        # Step 2: Merge extra dataset with class remapping
        logger.info("Merging extra dataset from %s (prefix=%s)...", extra_dir, extra_prefix)
        for split in ("train", "val"):
            ex_img_dir = extra_dir / "images" / split
            ex_lbl_dir = extra_dir / "labels" / split

            if not ex_img_dir.is_dir():
                # Check flat fallback without split
                if (extra_dir / "images").is_dir():
                    ex_img_dir = extra_dir / "images"
                    ex_lbl_dir = extra_dir / "labels"
                else:
                    continue

            for img_path in ex_img_dir.glob("*.*"):
                if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
                    continue

                stem = img_path.stem
                new_stem = f"{extra_prefix}_{stem}"
                new_img_name = f"{new_stem}{img_path.suffix.lower()}"
                dest_img = output_dir / "images" / split / new_img_name
                shutil.copyfile(img_path, dest_img)
                stats["extra_merged_images"] += 1

                # Locate corresponding label file
                lbl_path = ex_lbl_dir / f"{stem}.txt"
                dest_lbl = output_dir / "labels" / split / f"{new_stem}.txt"

                remapped_lines: List[str] = []
                if lbl_path.is_file():
                    with open(lbl_path, "r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                try:
                                    src_id = int(parts[0])
                                    tgt_id = self.map_class_id(src_id, source_class_names=extra_names)
                                    if tgt_id is not None and tgt_id in self.target_classes:
                                        remapped_lines.append(f"{tgt_id} {' '.join(parts[1:])}")
                                        stats["total_boxes_remapped"] += 1
                                        cname = self.target_classes[tgt_id]
                                        stats["per_class_box_counts"][cname] += 1
                                    else:
                                        stats["boxes_dropped_unmapped"] += 1
                                except ValueError:
                                    stats["boxes_dropped_unmapped"] += 1

                with open(dest_lbl, "w", encoding="utf-8") as out_f:
                    if remapped_lines:
                        out_f.write("\n".join(remapped_lines) + "\n")
                stats["extra_merged_labels"] += 1

        # Step 3: Write unified dataset.yaml
        yaml_content = (
            f"# VisionGuard QA - Harmonized Multi-Domain Dataset (Including Dent Class)\n"
            f"path: {str(output_dir).replace(os.sep, '/')}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"nc: {len(self.target_classes)}\n"
            f"names:\n"
        )
        for cid in sorted(self.target_classes.keys()):
            yaml_content += f"  {cid}: {self.target_classes[cid]}\n"

        yaml_path = output_dir / "dataset.yaml"
        with open(yaml_path, "w", encoding="utf-8") as yf:
            yf.write(yaml_content)

        logger.info("Generated unified dataset YAML at: %s", yaml_path)
        return stats


def run_dummy_test() -> bool:
    """Create a tiny dummy extra dataset, merge it, and assert class remapping correctness."""
    dummy_dir = PROJECT_ROOT / "data" / "extra" / "dummy_test_dataset"
    out_dir = PROJECT_ROOT / "data" / "extra" / "merged_test_output"

    try:
        if dummy_dir.exists():
            shutil.rmtree(dummy_dir)
        if out_dir.exists():
            shutil.rmtree(out_dir)

        # 1. Build tiny dummy structure
        for split in ("train", "val"):
            (dummy_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (dummy_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        # Create dummy image files (100x100 white PNGs)
        import cv2
        dummy_img = np.full((100, 100, 3), 200, dtype=np.uint8)

        # Image 1: Contains a "dent" flaw (mapped to class 6)
        cv2.imwrite(str(dummy_dir / "images" / "train" / "part_dent_01.png"), dummy_img)
        with open(dummy_dir / "labels" / "train" / "part_dent_01.txt", "w") as f:
            # Source class 0 in extra dataset will be mapped to 'dent' (class 6)
            f.write("0 0.500 0.500 0.200 0.200\n")

        # Image 2: Contains a "scratch" (mapped to class 5) and "dent" (mapped to class 6)
        cv2.imwrite(str(dummy_dir / "images" / "val" / "part_scratch_dent_02.png"), dummy_img)
        with open(dummy_dir / "labels" / "val" / "part_scratch_dent_02.txt", "w") as f:
            f.write("1 0.300 0.400 0.150 0.150\n")  # class 1 -> scratch (5)
            f.write("0 0.700 0.800 0.100 0.100\n")  # class 0 -> dent (6)

        # 2. Define custom mapping for this dummy dataset
        mapping_data = {
            "target_classes": DEFAULT_TARGET_CLASSES,
            "id_to_target_id": {
                "0": 6,  # 0 -> dent
                "1": 5,  # 1 -> scratches
            },
            "name_to_target_id": {
                "dent": 6,
                "scratch": 5,
            },
        }

        mapping_file = dummy_dir / "class_mapping.json"
        with open(mapping_file, "w", encoding="utf-8") as f:
            json.dump(mapping_data, f, indent=2)

        # 3. Execute Merger without base copy (isolated test)
        merger = DatasetMerger(mapping_config=mapping_data)
        stats = merger.merge(
            extra_dir=dummy_dir,
            output_dir=out_dir,
            extra_prefix="dummy",
            copy_base=False,
            extra_names={0: "dent", 1: "scratch"},
        )

        # 4. Assertions
        train_lbl = out_dir / "labels" / "train" / "dummy_part_dent_01.txt"
        val_lbl = out_dir / "labels" / "val" / "dummy_part_scratch_dent_02.txt"
        yaml_file = out_dir / "dataset.yaml"

        assert train_lbl.is_file(), "Merged train label file missing"
        assert val_lbl.is_file(), "Merged val label file missing"
        assert yaml_file.is_file(), "Unified dataset.yaml missing"

        # Verify class remapping in train_lbl: class must be 6 (dent)
        with open(train_lbl) as f:
            content = f.read().strip()
            assert content.startswith("6 "), f"Expected class 6 (dent), got: {content}"

        # Verify class remapping in val_lbl: must contain 5 (scratches) and 6 (dent)
        with open(val_lbl) as f:
            lines = [l.strip() for l in f if l.strip()]
            assert len(lines) == 2, f"Expected 2 boxes, got {len(lines)}"
            classes = {int(l.split()[0]) for l in lines}
            assert classes == {5, 6}, f"Expected classes {5, 6}, got: {classes}"

        # Verify dataset.yaml has dent class
        with open(yaml_file) as f:
            yaml_txt = f.read()
            assert "6: dent" in yaml_txt, "dataset.yaml missing '6: dent'"
            assert "nc: 7" in yaml_txt, "dataset.yaml missing 'nc: 7'"

        assert stats["total_boxes_remapped"] == 3
        assert stats["per_class_box_counts"]["dent"] == 2
        assert stats["per_class_box_counts"]["scratches"] == 1

        print("=" * 70)
        print("DATASET MERGER TEST (DUMMY RUN) PASSED!")
        print(f"  • Remapped Boxes: {stats['total_boxes_remapped']}")
        print(f"  • Dent Boxes: {stats['per_class_box_counts']['dent']}")
        print(f"  • Scratch Boxes: {stats['per_class_box_counts']['scratches']}")
        print(f"  • Unified YAML: {yaml_file}")
        print("=" * 70)
        return True

    finally:
        # Clean up dummy test files
        if dummy_dir.exists():
            shutil.rmtree(dummy_dir)
        if out_dir.exists():
            shutil.rmtree(out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisionGuard QA Dataset Merger")
    parser.add_argument("--base-dir", type=str, default="", help="Path to base dataset")
    parser.add_argument("--extra-dir", type=str, default="", help="Path to extra dataset (GC10-DET or Severstal)")
    parser.add_argument("--mapping-json", type=str, default="", help="Path to class mapping JSON file")
    parser.add_argument("--output-dir", type=str, default="", help="Destination merged dataset directory")
    parser.add_argument("--prefix", type=str, default="extra", help="Prefix for merged images/labels")
    parser.add_argument("--test-dummy", action="store_true", help="Run self-contained dummy test")
    args = parser.parse_args()

    if args.test_dummy or not args.extra_dir:
        success = run_dummy_test()
        sys.exit(0 if success else 1)

    base = Path(args.base_dir) if args.base_dir else None
    mapping_path = Path(args.mapping_json) if args.mapping_json else None
    out = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "data" / "harmonized_dataset"

    merger = DatasetMerger(base_dir=base, mapping_file=mapping_path)
    res = merger.merge(extra_dir=Path(args.extra_dir), output_dir=out, extra_prefix=args.prefix)
    print("\nDataset merge finished successfully! Summary:")
    print(json.dumps(res, indent=2))
