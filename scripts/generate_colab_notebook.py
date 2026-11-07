"""Generate valid, production-ready Google Colab training notebook ml/train_colab.ipynb."""

import json
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_NOTEBOOK = PROJECT_ROOT / "ml" / "train_colab.ipynb"


def make_markdown_cell(text: str) -> dict:
    lines = [l + "\n" for l in text.splitlines()]
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]  # Notebook standard for last line
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": lines,
    }


def make_code_cell(code: str) -> dict:
    lines = [l + "\n" for l in code.splitlines()]
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def build_notebook() -> dict:
    cells = []

    # Cell 1: Markdown Title
    cells.append(make_markdown_cell(
        "# VisionGuard QA — Google Colab Production Training Pipeline\n"
        "\n"
        "This notebook implements end-to-end training of **YOLOv8s** (or YOLOv8n) for real-time manufacturing defect localization and classification on the VisionGuard QA industrial benchmark.\n"
        "\n"
        "### Training Specifications:\n"
        "- **Model**: YOLOv8s (`yolov8s.pt`, 11.2M params) with option for YOLOv8n (`yolov8n.pt`)\n"
        "- **Input Resolution**: `640x640` px\n"
        "- **Epochs & Early Stopping**: 100 epochs, patience 30\n"
        "- **Learning Rate Schedule**: Cosine Annealing (`cos_lr=True`, `lr0=0.01`, `lrf=0.01`)\n"
        "- **Augmentations**: Mosaic (`1.0`), Mixup (`0.1`), HSV (`h=0.015, s=0.7, v=0.4`), Random Flips (`fliplr=0.5, flipud=0.5`)\n"
        "- **Class Balancing**: Offline Oversampling for underrepresented/weak defect classes\n"
        "- **Validation & Export**: Automated post-training `val`, `outputs/metrics.json` generation, and dynamic ONNX export."
    ))

    # Cell 2: Check Hardware
    cells.append(make_code_cell(
        "# 1. Verify GPU Acceleration\n"
        "!nvidia-smi"
    ))

    # Cell 3: Install Dependencies
    cells.append(make_code_cell(
        "# 2. Install Required Dependencies\n"
        "!pip install -q ultralytics onnx onnxruntime pyyaml opencv-python-headless albumentations"
    ))

    # Cell 4: Markdown Environment Setup
    cells.append(make_markdown_cell(
        "### 3. Mount Google Drive or Clone Repository\n"
        "Set up the workspace and dataset paths."
    ))

    # Cell 5: Setup Workspace & Dataset
    cells.append(make_code_cell(
        "import os\n"
        "import shutil\n"
        "from pathlib import Path\n"
        "\n"
        "# Define paths\n"
        "WORKSPACE_ROOT = Path('/content/VisionGuard')\n"
        "if not WORKSPACE_ROOT.exists():\n"
        "    WORKSPACE_ROOT = Path('/content')\n"
        "\n"
        "OUTPUTS_DIR = WORKSPACE_ROOT / 'outputs'\n"
        "MODELS_DIR = WORKSPACE_ROOT / 'models'\n"
        "OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)\n"
        "MODELS_DIR.mkdir(parents=True, exist_ok=True)\n"
        "\n"
        "print(f\"Workspace directory: {WORKSPACE_ROOT}\")\n"
        "print(f\"Outputs directory: {OUTPUTS_DIR}\")"
    ))

    # Cell 6: Markdown Hyperparameters & Model Configuration
    cells.append(make_markdown_cell(
        "### 4. Training Configuration & Model Selection\n"
        "Select **YOLOv8s** (recommended for production accuracy) or **YOLOv8n** (ultra-lightweight)."
    ))

    # Cell 7: Hyperparameters Code
    cells.append(make_code_cell(
        "# Training Hyperparameters\n"
        "MODEL_VARIANT = 'yolov8s.pt'  # Options: 'yolov8s.pt' (production) or 'yolov8n.pt' (edge fast)\n"
        "IMG_SIZE = 640\n"
        "EPOCHS = 100\n"
        "PATIENCE = 30\n"
        "BATCH_SIZE = 16\n"
        "LR0 = 0.01\n"
        "LRF = 0.01\n"
        "COS_LR = True\n"
        "MOSAIC = 1.0\n"
        "MIXUP = 0.1\n"
        "HSV_H = 0.015\n"
        "HSV_S = 0.7\n"
        "HSV_V = 0.4\n"
        "FLIPLR = 0.5\n"
        "FLIPUD = 0.5\n"
        "\n"
        "print(f\"Model Variant: {MODEL_VARIANT}\")\n"
        "print(f\"Resolution: {IMG_SIZE}x{IMG_SIZE}\")\n"
        "print(f\"Epochs: {EPOCHS} (patience={PATIENCE})\")\n"
        "print(f\"Mixup: {MIXUP}, Mosaic: {MOSAIC}, Cosine LR: {COS_LR}\")"
    ))

    # Cell 8: Markdown Oversampling Section
    cells.append(make_markdown_cell(
        "### 5. Weak Class Imbalance Analysis & Oversampling Engine\n"
        "Defect detection datasets in manufacturing often suffer from severe class imbalance (e.g. microscopic crazing or rare inclusions appear less frequently than scratches). "
        "This cell analyzes annotation frequency and replicates minority class training samples to balance representation."
    ))

    # Cell 9: Oversampling Implementation Code
    cells.append(make_code_cell(
        "import yaml\n"
        "from collections import Counter\n"
        "\n"
        "def balance_dataset_classes(data_yaml_path: str, oversample_threshold_ratio: float = 0.5):\n"
        "    with open(data_yaml_path, 'r') as f:\n"
        "        data_cfg = yaml.safe_load(f)\n"
        "\n"
        "    root_path = Path(data_cfg.get('path', '.'))\n"
        "    train_img_dir = root_path / data_cfg['train']\n"
        "    train_lbl_dir = root_path / str(data_cfg['train']).replace('images', 'labels')\n"
        "\n"
        "    if not train_lbl_dir.exists():\n"
        "        train_lbl_dir = root_path / 'labels' / 'train'\n"
        "\n"
        "    # Count class occurrences in train split\n"
        "    class_counts = Counter()\n"
        "    img_to_classes = {}\n"
        "\n"
        "    for lbl_file in train_lbl_dir.glob('*.txt'):\n"
        "        stem = lbl_file.stem\n"
        "        with open(lbl_file, 'r') as f:\n"
        "            classes_in_file = set()\n"
        "            for line in f:\n"
        "                parts = line.strip().split()\n"
        "                if parts:\n"
        "                    cid = int(parts[0])\n"
        "                    class_counts[cid] += 1\n"
        "                    classes_in_file.add(cid)\n"
        "            if classes_in_file:\n"
        "                img_to_classes[stem] = classes_in_file\n"
        "\n"
        "    names = data_cfg['names']\n"
        "    print(\"--- Class Distribution Before Oversampling ---\")\n"
        "    for cid in sorted(names.keys()):\n"
        "        print(f\"  Class {cid} ({names[cid]}): {class_counts[cid]} bounding boxes\")\n"
        "\n"
        "    if not class_counts:\n"
        "        return data_yaml_path\n"
        "\n"
        "    max_count = max(class_counts.values())\n"
        "    target_min = int(max_count * oversample_threshold_ratio)\n"
        "\n"
        "    oversampled_count = 0\n"
        "    for cid, cnt in class_counts.items():\n"
        "        if cnt < target_min:\n"
        "            needed = target_min - cnt\n"
        "            candidate_stems = [s for s, clist in img_to_classes.items() if cid in clist]\n"
        "            if not candidate_stems:\n"
        "                continue\n"
        "            print(f\"Oversampling weak class {cid} ({names[cid]}): +{needed} instances needed...\")\n"
        "            idx = 0\n"
        "            while needed > 0:\n"
        "                stem = candidate_stems[idx % len(candidate_stems)]\n"
        "                # Find image file\n"
        "                orig_img = None\n"
        "                for ext in ('.png', '.jpg', '.jpeg', '.bmp'):\n"
        "                    cand = train_img_dir / f\"{stem}{ext}\"\n"
        "                    if cand.exists():\n"
        "                        orig_img = cand\n"
        "                        break\n"
        "                orig_lbl = train_lbl_dir / f\"{stem}.txt\"\n"
        "\n"
        "                if orig_img and orig_lbl.exists():\n"
        "                    new_stem = f\"os_c{cid}_{idx}_{stem}\"\n"
        "                    shutil.copyfile(orig_img, train_img_dir / f\"{new_stem}{orig_img.suffix}\")\n"
        "                    shutil.copyfile(orig_lbl, train_lbl_dir / f\"{new_stem}.txt\")\n"
        "                    oversampled_count += 1\n"
        "                    needed -= 1\n"
        "                idx += 1\n"
        "\n"
        "    print(f\"[SUCCESS] Added {oversampled_count} oversampled minority class training pairs.\")\n"
        "    return data_yaml_path\n"
        "\n"
        "# Specify active dataset YAML path\n"
        "DATA_YAML = '/content/VisionGuard/data/real_industrial_dataset/dataset.yaml'\n"
        "if not Path(DATA_YAML).exists():\n"
        "    DATA_YAML = 'data/real_industrial_dataset/dataset.yaml'\n"
        "\n"
        "if Path(DATA_YAML).exists():\n"
        "    balance_dataset_classes(DATA_YAML)\n"
        "else:\n"
        "    print(f\"Dataset YAML not found at {DATA_YAML}; ensure dataset is uploaded or cloned.\")"
    ))

    # Cell 10: Markdown Model Training
    cells.append(make_markdown_cell(
        "### 6. Execute YOLOv8s Multi-Domain Fine-Tuning\n"
        "Train with Cosine LR schedule, Mosaic 1.0, Mixup 0.1, and photometric HSV jitter."
    ))

    # Cell 11: Training Execution Code
    cells.append(make_code_cell(
        "from ultralytics import YOLO\n"
        "\n"
        "# Initialize model\n"
        "print(f\"Loading base weights: {MODEL_VARIANT}...\")\n"
        "model = YOLO(MODEL_VARIANT)\n"
        "\n"
        "# Train model\n"
        "train_results = model.train(\n"
        "    data=DATA_YAML,\n"
        "    epochs=EPOCHS,\n"
        "    patience=PATIENCE,\n"
        "    imgsz=IMG_SIZE,\n"
        "    batch=BATCH_SIZE,\n"
        "    cos_lr=COS_LR,\n"
        "    lr0=LR0,\n"
        "    lrf=LRF,\n"
        "    mosaic=MOSAIC,\n"
        "    mixup=MIXUP,\n"
        "    hsv_h=HSV_H,\n"
        "    hsv_s=HSV_S,\n"
        "    hsv_v=HSV_V,\n"
        "    fliplr=FLIPLR,\n"
        "    flipud=FLIPUD,\n"
        "    project='runs/train_colab',\n"
        "    name='visionguard_colab',\n"
        "    exist_ok=True,\n"
        "    save=True,\n"
        "    plots=True,\n"
        "    verbose=True,\n"
        ")\n"
        "\n"
        "print(\"\\n[SUCCESS] Model training complete!\")"
    ))

    # Cell 12: Markdown Validation & Metrics JSON
    cells.append(make_markdown_cell(
        "### 7. Post-Training Validation Evaluation & `outputs/metrics.json`\n"
        "Run comprehensive evaluation across the validation split, extract precision, recall, mAP50, mAP50-95, and per-class AP, and record to `outputs/metrics.json`."
    ))

    # Cell 13: Validation Code & metrics.json
    cells.append(make_code_cell(
        "import json\n"
        "import numpy as np\n"
        "\n"
        "# Run validation\n"
        "print(\"Evaluating best weights on validation split...\")\n"
        "val_results = model.val(\n"
        "    data=DATA_YAML,\n"
        "    imgsz=IMG_SIZE,\n"
        "    split='val',\n"
        "    save_json=True,\n"
        "    plots=True,\n"
        ")\n"
        "\n"
        "# Extract metrics\n"
        "map50 = float(val_results.box.map50)\n"
        "map50_95 = float(val_results.box.map)\n"
        "precision = float(val_results.box.mp)\n"
        "recall = float(val_results.box.mr)\n"
        "\n"
        "print(f\"\\nOverall Results:\")\n"
        "print(f\"  mAP@50:     {map50:.4f}\")\n"
        "print(f\"  mAP@50-95:  {map50_95:.4f}\")\n"
        "print(f\"  Precision:  {precision:.4f}\")\n"
        "print(f\"  Recall:     {recall:.4f}\")\n"
        "\n"
        "per_class = {}\n"
        "for cid, cname in val_results.names.items():\n"
        "    ap50 = float(val_results.box.maps[cid]) if hasattr(val_results.box, 'maps') and len(val_results.box.maps) > cid else 0.0\n"
        "    per_class[cname] = {\n"
        "        \"class_id\": int(cid),\n"
        "        \"ap50\": round(ap50, 4),\n"
        "    }\n"
        "\n"
        "metrics_data = {\n"
        "    \"model\": MODEL_VARIANT,\n"
        "    \"epochs\": EPOCHS,\n"
        "    \"imgsz\": IMG_SIZE,\n"
        "    \"mAP50\": round(map50, 4),\n"
        "    \"mAP50_95\": round(map50_95, 4),\n"
        "    \"precision\": round(precision, 4),\n"
        "    \"recall\": round(recall, 4),\n"
        "    \"per_class\": per_class,\n"
        "}\n"
        "\n"
        "metrics_file = OUTPUTS_DIR / 'metrics.json'\n"
        "with open(metrics_file, 'w') as f:\n"
        "    json.dump(metrics_data, f, indent=2)\n"
        "\n"
        "print(f\"\\n[SUCCESS] Wrote evaluation telemetry to: {metrics_file}\")\n"
        "print(json.dumps(metrics_data, indent=2))"
    ))

    # Cell 14: Markdown ONNX Export
    cells.append(make_markdown_cell(
        "### 8. Export to ONNX with Dynamic Axes\n"
        "Export the fine-tuned PyTorch checkpoint to ONNX format for low-latency CPU and edge deployment."
    ))

    # Cell 15: ONNX Export Code
    cells.append(make_code_cell(
        "import onnx\n"
        "import onnxruntime as ort\n"
        "\n"
        "print(\"Exporting trained model to ONNX (dynamic=True, opset=12)...\")\n"
        "onnx_path = model.export(\n"
        "    format='onnx',\n"
        "    imgsz=IMG_SIZE,\n"
        "    dynamic=True,\n"
        "    opset=12,\n"
        "    simplify=False,\n"
        ")\n"
        "\n"
        "exported_onnx = Path(onnx_path)\n"
        "dest_onnx = MODELS_DIR / 'best.onnx'\n"
        "shutil.copyfile(exported_onnx, dest_onnx)\n"
        "print(f\"[SUCCESS] Copied active ONNX model to: {dest_onnx}\")\n"
        "\n"
        "# Verify ONNX model validity\n"
        "onnx_model = onnx.load(str(dest_onnx))\n"
        "onnx.checker.check_model(onnx_model)\n"
        "\n"
        "# Test ONNX Runtime session\n"
        "session = ort.InferenceSession(str(dest_onnx), providers=['CPUExecutionProvider'])\n"
        "input_meta = session.get_inputs()[0]\n"
        "output_meta = session.get_outputs()[0]\n"
        "print(f\"ONNX Verification PASSED:\")\n"
        "print(f\"  Input:  {input_meta.name} -> {input_meta.shape}\")\n"
        "print(f\"  Output: {output_meta.name} -> {output_meta.shape}\")"
    ))

    # Cell 16: Markdown Conclusion & Download
    cells.append(make_markdown_cell(
        "### 9. Packaging Artifacts for VisionGuard QA Deployment\n"
        "Copies best weights to `models/best.pt` and creates a zip file of weights and metrics."
    ))

    # Cell 17: Archive & Download Code
    cells.append(make_code_cell(
        "best_pt = Path('runs/train_colab/visionguard_colab/weights/best.pt')\n"
        "if best_pt.exists():\n"
        "    shutil.copyfile(best_pt, MODELS_DIR / 'best.pt')\n"
        "    print(f\"Updated primary checkpoint at: {MODELS_DIR / 'best.pt'}\")\n"
        "\n"
        "# Create deployable zip bundle\n"
        "!zip -q -r /content/visionguard_trained_artifacts.zip models/ outputs/ runs/train_colab/visionguard_colab/confusion_matrix.png\n"
        "print(\"Artifact bundle created: /content/visionguard_trained_artifacts.zip\")"
    ))

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": [],
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    return notebook_dict


def main():
    nb = build_notebook()
    with open(OUTPUT_NOTEBOOK, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"[OK] Generated {OUTPUT_NOTEBOOK} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
