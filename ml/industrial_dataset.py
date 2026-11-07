"""Industrial Surface Defect Benchmark Dataset Generator (NEU-DET Format).

Synthesizes high-fidelity metallurgical defects for hot-rolled and cold-rolled steel:
0: crazing          (micro-crack networks)
1: inclusion        (embedded non-metallic slag/impurities)
2: patches          (surface oxidation/coating irregularities)
3: pitted_surface   (chemical/mechanical erosion pinholes)
4: rolled_in_scale  (dark jagged mill-scale bands)
5: scratches        (abrasive directional tool marks)
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import yaml

from ml.model_registry import DEFECT_CLASSES, ModelRegistry

logger = logging.getLogger(__name__)


def generate_steel_substrate(width: int = 640, height: int = 640) -> np.ndarray:
    """Generate realistic metallic steel substrate with rolled grain texture."""
    base_val = random.randint(140, 185)
    noise = np.random.normal(0, random.uniform(4.0, 7.0), (height, width)).astype(np.float32)
    surface = np.clip(base_val + noise, 0, 255).astype(np.uint8)

    # Add rolling direction grain lines
    grain_direction_y = random.choice([True, False])
    if grain_direction_y:
        for y in range(0, height, random.randint(4, 12)):
            val = random.randint(-12, 12)
            surface[y : min(height, y + 2), :] = np.clip(
                surface[y : min(height, y + 2), :].astype(np.int16) + val, 0, 255
            ).astype(np.uint8)

    # 3-channel BGR
    return cv2.merge([surface, surface, surface])


def draw_crazing(img: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
    """Draw micro-crack networks (crazing) across a localized stress area."""
    h, w = img.shape[:2]
    cx = random.randint(int(w * 0.2), int(w * 0.8))
    cy = random.randint(int(h * 0.2), int(h * 0.8))
    radius = random.randint(40, 90)

    color = (random.randint(30, 50), random.randint(30, 50), random.randint(30, 50))
    pts = []

    # Spiderweb branching
    num_branches = random.randint(4, 7)
    for b in range(num_branches):
        angle = (b / num_branches) * 2 * np.pi + random.uniform(-0.3, 0.3)
        curr_x, curr_y = cx, cy
        pts.append((curr_x, curr_y))
        dist = 0
        while dist < radius:
            step = random.randint(8, 18)
            dist += step
            next_x = int(curr_x + step * np.cos(angle) + random.randint(-4, 4))
            next_y = int(curr_y + step * np.sin(angle) + random.randint(-4, 4))
            next_x = max(2, min(w - 3, next_x))
            next_y = max(2, min(h - 3, next_y))
            cv2.line(img, (curr_x, curr_y), (next_x, next_y), color, 1, cv2.LINE_AA)
            pts.append((next_x, next_y))
            curr_x, curr_y = next_x, next_y

    all_x = [p[0] for p in pts]
    all_y = [p[1] for p in pts]
    x1, x2 = max(0, min(all_x) - 4), min(w, max(all_x) + 4)
    y1, y2 = max(0, min(all_y) - 4), min(h, max(all_y) + 4)

    # YOLO bbox normalized [class_id, x_center, y_center, width, height]
    bbox = (0, (x1 + x2) / (2.0 * w), (y1 + y2) / (2.0 * h), (x2 - x1) / float(w), (y2 - y1) / float(h))
    return img, [bbox]


def draw_inclusion(img: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
    """Draw non-metallic slag inclusion embedded into matrix."""
    h, w = img.shape[:2]
    cx = random.randint(int(w * 0.15), int(w * 0.85))
    cy = random.randint(int(h * 0.15), int(h * 0.85))
    axes = (random.randint(12, 35), random.randint(8, 20))
    angle = random.randint(0, 180)

    # Dark core
    cv2.ellipse(img, (cx, cy), axes, angle, 0, 360, (random.randint(25, 45), random.randint(25, 45), random.randint(25, 45)), -1)
    # Lighter halo
    cv2.ellipse(img, (cx, cy), (axes[0] + 3, axes[1] + 3), angle, 0, 360, (190, 190, 190), 1)

    max_dim = max(axes) + 4
    x1, x2 = max(0, cx - max_dim), min(w, cx + max_dim)
    y1, y2 = max(0, cy - max_dim), min(h, cy + max_dim)

    bbox = (1, (x1 + x2) / (2.0 * w), (y1 + y2) / (2.0 * h), (x2 - x1) / float(w), (y2 - y1) / float(h))
    return img, [bbox]


def draw_patches(img: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
    """Draw irregular surface oxidation / coating patches."""
    h, w = img.shape[:2]
    cx = random.randint(int(w * 0.2), int(w * 0.8))
    cy = random.randint(int(h * 0.2), int(h * 0.8))
    pw = random.randint(50, 110)
    ph = random.randint(40, 90)

    overlay = img.copy()
    cv2.ellipse(overlay, (cx, cy), (pw // 2, ph // 2), random.randint(0, 180), 0, 360, (random.randint(90, 120), random.randint(90, 120), random.randint(90, 120)), -1)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

    x1, x2 = max(0, cx - pw // 2), min(w, cx + pw // 2)
    y1, y2 = max(0, cy - ph // 2), min(h, cy + ph // 2)

    bbox = (2, (x1 + x2) / (2.0 * w), (y1 + y2) / (2.0 * h), (x2 - x1) / float(w), (y2 - y1) / float(h))
    return img, [bbox]


def draw_pitted_surface(img: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
    """Draw cluster of corrosion pits / erosion pinholes."""
    h, w = img.shape[:2]
    cx = random.randint(int(w * 0.2), int(w * 0.8))
    cy = random.randint(int(h * 0.2), int(h * 0.8))
    cluster_radius = random.randint(25, 60)

    num_pits = random.randint(12, 28)
    pts = []
    for _ in range(num_pits):
        px = int(cx + random.gauss(0, cluster_radius / 2))
        py = int(cy + random.gauss(0, cluster_radius / 2))
        px = max(2, min(w - 3, px))
        py = max(2, min(h - 3, py))
        r = random.randint(1, 4)
        cv2.circle(img, (px, py), r, (random.randint(30, 60), random.randint(30, 60), random.randint(30, 60)), -1)
        # Highlight rim
        cv2.circle(img, (px + 1, py + 1), r, (195, 195, 195), 1)
        pts.append((px, py))

    all_x = [p[0] for p in pts]
    all_y = [p[1] for p in pts]
    x1, x2 = max(0, min(all_x) - 4), min(w, max(all_x) + 4)
    y1, y2 = max(0, min(all_y) - 4), min(h, max(all_y) + 4)

    bbox = (3, (x1 + x2) / (2.0 * w), (y1 + y2) / (2.0 * h), (x2 - x1) / float(w), (y2 - y1) / float(h))
    return img, [bbox]


def draw_rolled_in_scale(img: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
    """Draw dark jagged mill scale bands pressed during hot rolling."""
    h, w = img.shape[:2]
    x1 = random.randint(int(w * 0.1), int(w * 0.6))
    y1 = random.randint(int(h * 0.2), int(h * 0.8))
    bw = random.randint(80, 180)
    bh = random.randint(15, 45)

    pts = [
        (x1, y1),
        (x1 + bw // 3, y1 + random.randint(-8, 8)),
        (x1 + (2 * bw) // 3, y1 + random.randint(-8, 8)),
        (x1 + bw, y1 + bh),
        (x1 + (2 * bw) // 3, y1 + bh + random.randint(-5, 5)),
        (x1 + bw // 3, y1 + bh + random.randint(-5, 5)),
    ]
    cv2.fillPoly(img, [np.array(pts, np.int32)], (random.randint(40, 70), random.randint(40, 70), random.randint(40, 70)))

    x2 = min(w, x1 + bw)
    y2 = min(h, y1 + bh + 10)

    bbox = (4, (x1 + x2) / (2.0 * w), (y1 + y2) / (2.0 * h), (x2 - x1) / float(w), (y2 - y1) / float(h))
    return img, [bbox]


def draw_scratches(img: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
    """Draw linear abrasive tooling scratches with specular edge highlights."""
    h, w = img.shape[:2]
    sx = random.randint(int(w * 0.1), int(w * 0.7))
    sy = random.randint(int(h * 0.1), int(h * 0.7))
    length = random.randint(60, 160)
    angle = random.uniform(0.1, np.pi * 0.8)

    ex = int(sx + length * np.cos(angle))
    ey = int(sy + length * np.sin(angle))
    ex = max(2, min(w - 3, ex))
    ey = max(2, min(h - 3, ey))

    # Dark groove
    cv2.line(img, (sx, sy), (ex, ey), (40, 40, 40), 2, cv2.LINE_AA)
    # Highlight reflection along edge
    cv2.line(img, (sx + 1, sy + 1), (ex + 1, ey + 1), (210, 210, 210), 1, cv2.LINE_AA)

    x1, x2 = min(sx, ex) - 3, max(sx, ex) + 3
    y1, y2 = min(sy, ey) - 3, max(sy, ey) + 3
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)

    bbox = (5, (x1 + x2) / (2.0 * w), (y1 + y2) / (2.0 * h), (x2 - x1) / float(w), (y2 - y1) / float(h))
    return img, [bbox]


DEFECT_GENERATORS = [
    draw_crazing,
    draw_inclusion,
    draw_patches,
    draw_pitted_surface,
    draw_rolled_in_scale,
    draw_scratches,
]


def generate_industrial_dataset(
    output_dir: Path | None = None,
    num_train: int = 120,
    num_val: int = 30,
    img_size: int = 640,
) -> Path:
    """Generate industrial surface defect dataset split into train/val."""
    root = output_dir or (ModelRegistry.get_project_root() / "data" / "industrial_benchmark")
    root = Path(root).resolve()

    train_img_dir = root / "images" / "train"
    val_img_dir = root / "images" / "val"
    train_lbl_dir = root / "labels" / "train"
    val_lbl_dir = root / "labels" / "val"

    for d in (train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir):
        d.mkdir(parents=True, exist_ok=True)

    logger.info("Generating industrial benchmark at: %s", root)

    for split_name, count, img_dir, lbl_dir in [
        ("train", num_train, train_img_dir, train_lbl_dir),
        ("val", num_val, val_img_dir, val_lbl_dir),
    ]:
        for i in range(count):
            img = generate_steel_substrate(img_size, img_size)
            bboxes = []

            # 15% negative defect-free samples for robust false-positive rejection
            is_defect_free = random.random() < 0.15
            if not is_defect_free:
                num_defects = random.randint(1, 2)
                for _ in range(num_defects):
                    generator = random.choice(DEFECT_GENERATORS)
                    img, box_list = generator(img)
                    bboxes.extend(box_list)

            file_stem = f"neu_steel_{split_name}_{i:04d}"
            img_path = img_dir / f"{file_stem}.png"
            lbl_path = lbl_dir / f"{file_stem}.txt"

            cv2.imwrite(str(img_path), img)

            # Write YOLO format: class x_center y_center width height
            with open(lbl_path, "w", encoding="utf-8") as f:
                for b in bboxes:
                    cls_id, xc, yc, bw, bh = b
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    # Generate dataset.yaml
    yaml_config = {
        "path": str(root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "nc": len(DEFECT_CLASSES),
        "names": {int(k): v for k, v in DEFECT_CLASSES.items()},
    }

    yaml_path = root / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_config, f, sort_keys=False)

    logger.info("Industrial benchmark dataset created successfully at %s", yaml_path)
    return yaml_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = generate_industrial_dataset()
    print("Dataset generated at:", p)
