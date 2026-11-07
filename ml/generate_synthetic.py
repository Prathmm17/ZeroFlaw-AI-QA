"""Generate realistic synthetic manufacturing defect images with YOLO-format annotations.

This module creates synthetic training data for defect detection models by
procedurally generating base surface textures and overlaying various defect
types (cracks, scratches, dents, stains, chips) with accurate bounding box
annotations in YOLO format.

Typical usage:
    python -m ml.generate_synthetic
"""

import logging
import math
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Defect class mapping
DEFECT_CLASSES: dict[int, str] = {
    0: "crack",
    1: "scratch",
    2: "dent",
    3: "stain",
    4: "chip",
}

# Type alias for a single YOLO annotation row
YoloAnnotation = tuple[int, float, float, float, float]


def _clamp(value: int, lo: int, hi: int) -> int:
    """Clamp an integer value between lo and hi (inclusive).

    Args:
        value: The value to clamp.
        lo: Minimum allowed value.
        hi: Maximum allowed value.

    Returns:
        Clamped integer.
    """
    return max(lo, min(hi, value))


def _bbox_from_points(
    points: list[tuple[int, int]], img_w: int, img_h: int, padding: int = 2
) -> tuple[float, float, float, float]:
    """Compute normalised YOLO bbox (x_center, y_center, w, h) from pixel points.

    Args:
        points: List of (x, y) pixel coordinates defining the defect region.
        img_w: Image width in pixels.
        img_h: Image height in pixels.
        padding: Extra pixels added around the tight bounding box.

    Returns:
        Tuple of (x_center_norm, y_center_norm, w_norm, h_norm) in [0, 1].
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x_min = _clamp(min(xs) - padding, 0, img_w - 1)
    x_max = _clamp(max(xs) + padding, 0, img_w - 1)
    y_min = _clamp(min(ys) - padding, 0, img_h - 1)
    y_max = _clamp(max(ys) + padding, 0, img_h - 1)

    cx = (x_min + x_max) / 2.0 / img_w
    cy = (y_min + y_max) / 2.0 / img_h
    bw = (x_max - x_min) / img_w
    bh = (y_max - y_min) / img_h

    # Ensure bbox stays within [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    bw = max(0.001, min(1.0, bw))
    bh = max(0.001, min(1.0, bh))

    return cx, cy, bw, bh


# ---------------------------------------------------------------------------
# Base surface generators
# ---------------------------------------------------------------------------


def generate_base_surface(
    width: int = 640, height: int = 640, surface_type: str = "metal"
) -> np.ndarray:
    """Create a base surface texture image.

    Args:
        width: Output image width in pixels.
        height: Output image height in pixels.
        surface_type: One of ``"metal"``, ``"circuit"``, or ``"plastic"``.

    Returns:
        BGR numpy array of shape (height, width, 3).

    Raises:
        ValueError: If *surface_type* is not recognised.
    """
    if surface_type == "metal":
        return _surface_metal(width, height)
    if surface_type == "circuit":
        return _surface_circuit(width, height)
    if surface_type == "plastic":
        return _surface_plastic(width, height)
    raise ValueError(
        f"Unknown surface_type '{surface_type}'. "
        "Choose from 'metal', 'circuit', 'plastic'."
    )


def _surface_metal(width: int, height: int) -> np.ndarray:
    """Generate a metallic surface with gradient and Gaussian noise."""
    base_val = random.randint(140, 190)
    gradient = np.linspace(base_val - 20, base_val + 20, height, dtype=np.float32)
    surface = np.tile(gradient[:, None], (1, width)).astype(np.float32)

    noise = np.random.normal(0, 6, (height, width)).astype(np.float32)
    surface = np.clip(surface + noise, 0, 255).astype(np.uint8)

    # Fine horizontal streaks typical of brushed metal
    for _ in range(random.randint(5, 15)):
        y = random.randint(0, height - 1)
        streak_val = random.randint(-15, 15)
        thickness = random.randint(1, 3)
        y_end = min(y + thickness, height)
        surface[y:y_end, :] = np.clip(
            surface[y:y_end, :].astype(np.int16) + streak_val, 0, 255
        ).astype(np.uint8)

    bgr = cv2.merge([surface, surface, surface])
    # Slight warm tint
    tint = np.zeros_like(bgr, dtype=np.int16)
    tint[:, :, 2] = random.randint(0, 8)  # red channel
    tint[:, :, 1] = random.randint(0, 4)  # green channel
    bgr = np.clip(bgr.astype(np.int16) + tint, 0, 255).astype(np.uint8)
    return bgr


def _surface_circuit(width: int, height: int) -> np.ndarray:
    """Generate a PCB-like dark green surface with line patterns."""
    base_b = random.randint(25, 45)
    base_g = random.randint(65, 100)
    base_r = random.randint(20, 40)
    surface = np.full((height, width, 3), (base_b, base_g, base_r), dtype=np.uint8)

    # Add fine noise
    noise = np.random.normal(0, 3, (height, width, 3)).astype(np.int16)
    surface = np.clip(surface.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Draw trace-like lines
    line_color_light = (base_b + 40, base_g + 50, base_r + 30)
    for _ in range(random.randint(8, 25)):
        orientation = random.choice(["h", "v"])
        if orientation == "h":
            y = random.randint(0, height - 1)
            x1 = random.randint(0, width // 3)
            x2 = random.randint(width // 3, width - 1)
            cv2.line(surface, (x1, y), (x2, y), line_color_light, 1)
        else:
            x = random.randint(0, width - 1)
            y1 = random.randint(0, height // 3)
            y2 = random.randint(height // 3, height - 1)
            cv2.line(surface, (x, y1), (x, y2), line_color_light, 1)

    # Small pads
    for _ in range(random.randint(5, 15)):
        cx = random.randint(20, width - 20)
        cy = random.randint(20, height - 20)
        r = random.randint(3, 6)
        cv2.circle(surface, (cx, cy), r, line_color_light, -1)

    return surface


def _surface_plastic(width: int, height: int) -> np.ndarray:
    """Generate a light, slightly textured plastic surface."""
    base_val = random.randint(200, 235)
    surface = np.full((height, width), base_val, dtype=np.uint8)

    # Very subtle noise
    noise = np.random.normal(0, 3, (height, width)).astype(np.int16)
    surface = np.clip(surface.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    bgr = cv2.merge([surface, surface, surface])

    # Slight colour cast
    tint_b = random.randint(-5, 5)
    tint_g = random.randint(-5, 5)
    tint_r = random.randint(-5, 5)
    tint = np.full_like(bgr, (tint_b, tint_g, tint_r), dtype=np.int16)
    bgr = np.clip(bgr.astype(np.int16) + tint, 0, 255).astype(np.uint8)
    return bgr


# ---------------------------------------------------------------------------
# Defect generators
# ---------------------------------------------------------------------------


def add_crack(
    image: np.ndarray,
) -> tuple[np.ndarray, list[YoloAnnotation]]:
    """Draw a realistic crack on the image.

    Cracks are rendered as thin jagged polylines with brownish/dark colour
    and a slight Gaussian blur to soften edges.

    Args:
        image: BGR image array (modified in-place and returned).

    Returns:
        Tuple of (modified image, list of YOLO annotations) where each
        annotation is ``(class_id, x_center, y_center, w, h)`` normalised.
    """
    img = image.copy()
    h, w = img.shape[:2]

    # Random start point (avoid very edge to keep bbox valid)
    margin = int(min(w, h) * 0.05)
    x = random.randint(margin, w - margin)
    y = random.randint(margin, h - margin)

    num_segments = random.randint(8, 20)
    angle = random.uniform(0, 2 * math.pi)
    step_len = random.randint(8, 25)

    points: list[tuple[int, int]] = [(x, y)]
    for _ in range(num_segments):
        angle += random.uniform(-0.6, 0.6)
        dx = int(step_len * math.cos(angle) + random.randint(-3, 3))
        dy = int(step_len * math.sin(angle) + random.randint(-3, 3))
        x = _clamp(x + dx, 0, w - 1)
        y = _clamp(y + dy, 0, h - 1)
        points.append((x, y))

    # Dark brownish colour with slight variation
    color = (
        random.randint(15, 45),
        random.randint(15, 40),
        random.randint(25, 55),
    )
    thickness = random.randint(1, 2)

    pts_arr = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts_arr], isClosed=False, color=color, thickness=thickness)

    # Optional thin secondary branch
    if random.random() < 0.4:
        branch_start_idx = random.randint(2, len(points) - 2)
        bx, by = points[branch_start_idx]
        branch_pts: list[tuple[int, int]] = [(bx, by)]
        b_angle = angle + random.uniform(-1.2, 1.2)
        for _ in range(random.randint(3, 7)):
            b_angle += random.uniform(-0.5, 0.5)
            bx = _clamp(bx + int(step_len * 0.6 * math.cos(b_angle)), 0, w - 1)
            by = _clamp(by + int(step_len * 0.6 * math.sin(b_angle)), 0, h - 1)
            branch_pts.append((bx, by))
        b_arr = np.array(branch_pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [b_arr], isClosed=False, color=color, thickness=1)
        points.extend(branch_pts)

    # Slight blur along crack region
    cx, cy, bw, bh = _bbox_from_points(points, w, h, padding=4)
    x1 = max(0, int((cx - bw / 2) * w) - 2)
    y1 = max(0, int((cy - bh / 2) * h) - 2)
    x2 = min(w, int((cx + bw / 2) * w) + 2)
    y2 = min(h, int((cy + bh / 2) * h) + 2)
    roi = img[y1:y2, x1:x2]
    if roi.size > 0:
        img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (3, 3), 0.5)

    annotation: YoloAnnotation = (0, cx, cy, bw, bh)
    return img, [annotation]


def add_scratch(
    image: np.ndarray,
) -> tuple[np.ndarray, list[YoloAnnotation]]:
    """Draw a scratch on the image.

    Scratches are rendered as mostly-straight thin lines that are lighter
    than the surrounding surface, with slight positional randomness.

    Args:
        image: BGR image array.

    Returns:
        Tuple of (modified image, list of YOLO annotations).
    """
    img = image.copy()
    h, w = img.shape[:2]

    margin = int(min(w, h) * 0.05)
    x1 = random.randint(margin, w - margin)
    y1 = random.randint(margin, h - margin)

    length = random.randint(60, int(min(w, h) * 0.6))
    angle = random.uniform(0, math.pi)

    x2 = _clamp(x1 + int(length * math.cos(angle)), 0, w - 1)
    y2 = _clamp(y1 + int(length * math.sin(angle)), 0, h - 1)

    # Build a slightly wobbly polyline between start and end
    num_pts = random.randint(5, 12)
    points: list[tuple[int, int]] = []
    for i in range(num_pts + 1):
        t = i / num_pts
        px = int(x1 + t * (x2 - x1) + random.randint(-2, 2))
        py = int(y1 + t * (y2 - y1) + random.randint(-2, 2))
        px = _clamp(px, 0, w - 1)
        py = _clamp(py, 0, h - 1)
        points.append((px, py))

    # Lighter colour (scratch exposes underlying material)
    avg_brightness = int(np.mean(img[y1, _clamp(x1, 0, w - 1)]))
    bright_offset = random.randint(30, 70)
    val = min(255, avg_brightness + bright_offset)
    color = (val, val, val)

    thickness = random.randint(1, 2)
    pts_arr = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts_arr], isClosed=False, color=color, thickness=thickness)

    cx, cy, bw, bh = _bbox_from_points(points, w, h, padding=3)
    annotation: YoloAnnotation = (1, cx, cy, bw, bh)
    return img, [annotation]


def add_dent(
    image: np.ndarray,
) -> tuple[np.ndarray, list[YoloAnnotation]]:
    """Draw a dent on the image.

    Dents are rendered as elliptical darker regions with a subtle radial
    gradient to simulate depth.

    Args:
        image: BGR image array.

    Returns:
        Tuple of (modified image, list of YOLO annotations).
    """
    img = image.copy()
    h, w = img.shape[:2]

    radius_x = random.randint(15, int(w * 0.12))
    radius_y = random.randint(15, int(h * 0.12))
    cx = random.randint(radius_x + 5, w - radius_x - 5)
    cy = random.randint(radius_y + 5, h - radius_y - 5)
    angle_rot = random.randint(0, 180)

    # Create a gradient mask for the dent
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (cx, cy), (radius_x, radius_y), angle_rot, 0, 360, 1.0, -1)

    # Soften edges with blur
    ksize = max(3, (min(radius_x, radius_y) // 2) * 2 + 1)
    mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

    # Darken the surface under the mask
    darken_amount = random.uniform(0.15, 0.35)
    darkened = (img.astype(np.float32) * (1.0 - mask[:, :, None] * darken_amount))
    img = np.clip(darkened, 0, 255).astype(np.uint8)

    # Compute bounding box from ellipse extremes
    cos_a = math.cos(math.radians(angle_rot))
    sin_a = math.sin(math.radians(angle_rot))
    # Tight axis-aligned bounding box of a rotated ellipse
    ex = math.sqrt(radius_x**2 * cos_a**2 + radius_y**2 * sin_a**2)
    ey = math.sqrt(radius_x**2 * sin_a**2 + radius_y**2 * cos_a**2)

    pts = [
        (int(cx - ex), int(cy - ey)),
        (int(cx + ex), int(cy + ey)),
    ]
    bcx, bcy, bw, bh = _bbox_from_points(pts, w, h, padding=3)
    annotation: YoloAnnotation = (2, bcx, bcy, bw, bh)
    return img, [annotation]


def add_stain(
    image: np.ndarray,
) -> tuple[np.ndarray, list[YoloAnnotation]]:
    """Draw an irregular stain/discolouration on the image.

    Stains are contour-based irregular blobs with colour shift applied
    inside the boundary.

    Args:
        image: BGR image array.

    Returns:
        Tuple of (modified image, list of YOLO annotations).
    """
    img = image.copy()
    h, w = img.shape[:2]

    # Generate irregular blob via random polar-coordinate contour
    cx = random.randint(int(w * 0.15), int(w * 0.85))
    cy = random.randint(int(h * 0.15), int(h * 0.85))
    base_radius = random.randint(20, int(min(w, h) * 0.12))

    num_vertices = random.randint(10, 20)
    contour_pts: list[tuple[int, int]] = []
    for i in range(num_vertices):
        theta = 2 * math.pi * i / num_vertices
        r = base_radius * random.uniform(0.6, 1.4)
        px = _clamp(cx + int(r * math.cos(theta)), 0, w - 1)
        py = _clamp(cy + int(r * math.sin(theta)), 0, h - 1)
        contour_pts.append((px, py))

    contour_arr = np.array(contour_pts, dtype=np.int32)

    # Build mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [contour_arr], 255)

    # Blur the mask for soft edges
    k = max(5, base_radius // 2 * 2 + 1)
    mask_f = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (k, k), 0)

    # Stain colour: brownish / yellowish / rust-like
    stain_colour = np.array(
        [
            random.randint(30, 80),   # B
            random.randint(60, 130),  # G
            random.randint(100, 180), # R
        ],
        dtype=np.float32,
    )
    blend_strength = random.uniform(0.25, 0.50)

    stain_layer = np.full_like(img, stain_colour, dtype=np.float32)
    blended = (
        img.astype(np.float32) * (1 - mask_f[:, :, None] * blend_strength)
        + stain_layer * mask_f[:, :, None] * blend_strength
    )
    img = np.clip(blended, 0, 255).astype(np.uint8)

    bcx, bcy, bw, bh = _bbox_from_points(contour_pts, w, h, padding=3)
    annotation: YoloAnnotation = (3, bcx, bcy, bw, bh)
    return img, [annotation]


def add_chip(
    image: np.ndarray,
) -> tuple[np.ndarray, list[YoloAnnotation]]:
    """Draw a chip (small edge damage) on the image.

    Chips are small irregular lighter spots with hard edges, typically
    found near corners or edges of a component.

    Args:
        image: BGR image array.

    Returns:
        Tuple of (modified image, list of YOLO annotations).
    """
    img = image.copy()
    h, w = img.shape[:2]

    # Place chips near edges more often
    if random.random() < 0.6:
        # Near an edge
        edge = random.choice(["top", "bottom", "left", "right"])
        margin = int(min(w, h) * 0.08)
        if edge == "top":
            cx = random.randint(margin, w - margin)
            cy = random.randint(5, margin)
        elif edge == "bottom":
            cx = random.randint(margin, w - margin)
            cy = random.randint(h - margin, h - 5)
        elif edge == "left":
            cx = random.randint(5, margin)
            cy = random.randint(margin, h - margin)
        else:
            cx = random.randint(w - margin, w - 5)
            cy = random.randint(margin, h - margin)
    else:
        cx = random.randint(int(w * 0.1), int(w * 0.9))
        cy = random.randint(int(h * 0.1), int(h * 0.9))

    chip_radius = random.randint(6, 18)

    # Irregular polygon for the chip
    num_v = random.randint(5, 9)
    chip_pts: list[tuple[int, int]] = []
    for i in range(num_v):
        theta = 2 * math.pi * i / num_v
        r = chip_radius * random.uniform(0.5, 1.3)
        px = _clamp(cx + int(r * math.cos(theta)), 0, w - 1)
        py = _clamp(cy + int(r * math.sin(theta)), 0, h - 1)
        chip_pts.append((px, py))

    contour = np.array(chip_pts, dtype=np.int32)

    # Lighter fill (exposed material) with hard edge
    avg_brightness = int(np.mean(img[cy, cx]))
    fill_val = min(255, avg_brightness + random.randint(40, 90))
    fill_color = (fill_val, fill_val, fill_val)

    cv2.fillPoly(img, [contour], fill_color)

    # Dark border to simulate chipped edge shadow
    border_color = (
        max(0, fill_val - random.randint(60, 100)),
        max(0, fill_val - random.randint(60, 100)),
        max(0, fill_val - random.randint(60, 100)),
    )
    cv2.polylines(img, [contour], isClosed=True, color=border_color, thickness=1)

    bcx, bcy, bw, bh = _bbox_from_points(chip_pts, w, h, padding=3)
    annotation: YoloAnnotation = (4, bcx, bcy, bw, bh)
    return img, [annotation]


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

# Map of defect adders keyed by class id
_DEFECT_ADDERS = {
    0: add_crack,
    1: add_scratch,
    2: add_dent,
    3: add_stain,
    4: add_chip,
}


def _generate_single_image(
    image_size: int,
) -> tuple[np.ndarray, list[YoloAnnotation], bool]:
    """Generate one synthetic image with random defects.

    Args:
        image_size: Width and height of the square image.

    Returns:
        Tuple of (image, annotations_list, is_negative) where
        *is_negative* is True when the image intentionally has no defects.
    """
    surface_type = random.choice(["metal", "circuit", "plastic"])
    img = generate_base_surface(image_size, image_size, surface_type)

    # ~15 % chance of defect-free negative example
    if random.random() < 0.15:
        return img, [], True

    num_defects = random.randint(1, 3)
    all_annotations: list[YoloAnnotation] = []

    for _ in range(num_defects):
        defect_id = random.choice(list(_DEFECT_ADDERS.keys()))
        adder = _DEFECT_ADDERS[defect_id]
        img, annots = adder(img)
        all_annotations.extend(annots)

    return img, all_annotations, False


def _write_annotations(label_path: Path, annotations: list[YoloAnnotation]) -> None:
    """Write YOLO-format annotations to a text file.

    Args:
        label_path: Destination ``.txt`` file path.
        annotations: List of (class_id, x_c, y_c, w, h) tuples.
    """
    lines: list[str] = []
    for cls_id, xc, yc, bw, bh in annotations:
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    label_path.write_text("\n".join(lines), encoding="utf-8")


def generate_dataset(
    output_dir: Path,
    num_train: int = 150,
    num_val: int = 30,
    image_size: int = 640,
) -> Path:
    """Generate a full YOLO-format training dataset of synthetic defects.

    Creates the following directory structure inside *output_dir*::

        output_dir/
          dataset.yaml
          images/
            train/  (num_train images)
            val/    (num_val images)
          labels/
            train/  (corresponding .txt files)
            val/    (corresponding .txt files)

    Args:
        output_dir: Root directory for the generated dataset.
        num_train: Number of training images to generate.
        num_val: Number of validation images to generate.
        image_size: Width and height (square) for every image.

    Returns:
        Path to the generated ``dataset.yaml`` file.
    """
    splits: dict[str, int] = {"train": num_train, "val": num_val}

    for split_name, count in splits.items():
        img_dir = output_dir / "images" / split_name
        lbl_dir = output_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        positive_count = 0
        negative_count = 0

        for idx in range(count):
            img, annotations, is_negative = _generate_single_image(image_size)
            fname = f"synth_{split_name}_{idx:05d}"

            img_path = img_dir / f"{fname}.png"
            lbl_path = lbl_dir / f"{fname}.txt"

            cv2.imwrite(str(img_path), img)

            if annotations:
                _write_annotations(lbl_path, annotations)
                positive_count += 1
            else:
                # Write empty label file for negative images (YOLO convention)
                lbl_path.write_text("", encoding="utf-8")
                negative_count += 1

        logger.info(
            "Split '%s': %d images (%d positive, %d negative)",
            split_name,
            count,
            positive_count,
            negative_count,
        )

    # Write dataset.yaml
    dataset_config: dict = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {int(k): v for k, v in DEFECT_CLASSES.items()},
    }

    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as fh:
        yaml.dump(dataset_config, fh, default_flow_style=False, sort_keys=False)

    logger.info("Dataset YAML written to %s", yaml_path)
    return yaml_path


def generate_sample_images(
    output_dir: Path, num_images: int = 10
) -> list[Path]:
    """Generate a set of sample images for demonstration purposes.

    Each sample showcases a single defect type drawn on a random surface
    so that every defect class is represented at least once (when
    *num_images* >= 5).

    Args:
        output_dir: Directory in which to save the sample images.
        num_images: Total number of sample images to create.

    Returns:
        List of paths to the generated sample images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    defect_ids = list(_DEFECT_ADDERS.keys())

    for idx in range(num_images):
        surface_type = random.choice(["metal", "circuit", "plastic"])
        img = generate_base_surface(640, 640, surface_type)

        # Ensure each defect class appears at least once
        if idx < len(defect_ids):
            defect_id = defect_ids[idx]
        else:
            defect_id = random.choice(defect_ids)

        adder = _DEFECT_ADDERS[defect_id]
        img, _ = adder(img)

        fname = f"sample_{DEFECT_CLASSES[defect_id]}_{idx:03d}.png"
        img_path = output_dir / fname
        cv2.imwrite(str(img_path), img)
        paths.append(img_path)
        logger.info("Sample image saved: %s", img_path)

    return paths


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    dataset_dir = PROJECT_ROOT / "data" / "synthetic"
    samples_dir = PROJECT_ROOT / "data" / "samples"

    logger.info("=== Generating synthetic training dataset ===")
    yaml_path = generate_dataset(dataset_dir, num_train=150, num_val=30)
    logger.info("Dataset config: %s", yaml_path)

    logger.info("=== Generating sample images ===")
    sample_paths = generate_sample_images(samples_dir, num_images=10)

    print("\n" + "=" * 60)
    print("Synthetic dataset generation complete!")
    print(f"  Dataset YAML : {yaml_path}")
    print(f"  Train images : {dataset_dir / 'images' / 'train'}")
    print(f"  Val images   : {dataset_dir / 'images' / 'val'}")
    print(f"  Sample images: {samples_dir} ({len(sample_paths)} files)")
    print(f"  Classes      : {DEFECT_CLASSES}")
    print("=" * 60)
