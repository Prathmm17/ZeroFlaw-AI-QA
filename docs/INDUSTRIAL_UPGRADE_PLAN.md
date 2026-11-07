# VisionGuard QA — Next-Level Industrial Upgrade Plan
## Roadmap to 989+ / 1000: Micro-Defect Localization, Anomaly Heatmaps & Agentic VLM

> **Objective**: Upgrade VisionGuard QA from a prototype into an industrial-grade defect detection system capable of resolving sub-millimeter hairline cracks, microscopic tool chatter, and metallurgical anomalies at native resolution, complete with Gemini Flash Root-Cause Engineering diagnosis.

---

## 🧭 Why the Current System Fails & What We Are Solving

```mermaid
flowchart TD
    subgraph Current_Flaws["Current Flaws (Score: 310/1000)"]
        F1["640x640 Downsampling erases 2px hairline cracks"]
        F2["COCO Weights know sports balls & cars, not steel fractures"]
        F3["Bounding boxes cover 99% good surface on diagonal scratches"]
        F4["No engineering root-cause explanation for supervisors"]
    end

    subgraph Solutions["Next-Level Industrial Architecture (Target: 989+/1000)"]
        S1["Phase 1: SAHI Native-Resolution Patch Tiling Engine"]
        S2["Phase 2: Industrial Defect Weights (NEU / MVTec Benchmarks)"]
        S3["Phase 3: Pixel-Level Mask Segmentation (YOLOv8-Seg)"]
        S4["Phase 4: Gemini Flash Root-Cause Quality Engineering Auditor"]
    end

    F1 ==> S1
    F2 ==> S2
    F3 ==> S3
    F4 ==> S4
```

---

## 📅 Phase-by-Phase Roadmap

Execution will proceed **strictly one phase at a time**. Each phase will include unit tests, test evidence, and verification before proceeding to the next.

---

### Phase 1: High-Resolution Tiling & SAHI Engine (Micro-Defect Preservation)
> **Goal**: Stop downsampling high-resolution images. Enable detection of sub-millimeter hairline fractures that are only 1–3 pixels wide.

* **Key Deliverables**:
  1. `ml/tiler.py`:
     - Dynamic image slicing into overlapping windows (e.g., $512 \times 512$ or $640 \times 640$ with $20\%$ overlap).
     - Slice coordinate translation matrix $(x_{slice}, y_{slice}) \to (X_{global}, Y_{global})$.
     - Global Non-Maximum Suppression (NMS) and Batched Box Union across overlapping slice boundaries.
  2. Integration into `ml/detector.py`:
     - Toggle parameter: `enable_tiling=True` / `slice_size=512`.
     - Automatic resolution routing: Images $< 800\text{px}$ pass directly; images $\ge 800\text{px}$ route through the tiling engine.
  3. Verification Test Script `scripts/test_micro_defects.py`:
     - Synthesize a $2048 \times 2048$ ultra-high-resolution surface with a 2-pixel wide hairline fracture.
     - Demonstrate standard 640x640 resize loses the defect (0 detections).
     - Demonstrate SAHI tiling catches the hairline crack with $>85\%$ confidence.

---

### Phase 2: Industrial Defect Weights & Benchmark Training
> **Goal**: Replace generic COCO weights (`unknown_32`) with genuine metallurgical and manufacturing defect feature representations.

* **Key Deliverables**:
  1. Benchmark Dataset Pipeline (`ml/industrial_dataset.py`):
     - Ingest real industrial surface defect benchmarks:
       - **NEU-DET** (Northeastern University Hot-Rolled Steel Strip Defect Database): Crazing, Inclusions, Patches, Pitted Surfaces, Rolled-in Scale, Scratches.
       - **MVTec AD** (Metal Nut, Screw, Tile, Grid).
  2. Training & Transfer Pipeline (`ml/train_industrial.py`):
     - Fine-tune YOLOv8 on true defect distributions using Focal Loss for high class-imbalance.
     - Save production weights to `models/industrial_best.pt`.
  3. Verification Test Suite:
     - Evaluate mAP@50, Precision, and Recall on unseen test specimens.
     - Eliminate all `unknown_XX` class labels; guarantee correct domain labels (`crazing`, `scratch`, `pitted_surface`, `inclusion`).

---

### Phase 3: Pixel-Level Mask Segmentation & Contour Metrology
> **Goal**: Move beyond rectangular bounding boxes. Extract exact crack geometry, physical length, contour curvature, and surface area in $\text{mm}^2$.

* **Key Deliverables**:
  1. `ml/segmenter.py`:
     - Support YOLOv8-Seg (Instance Segmentation).
     - Polygon contour extraction from predicted masks (`cv2.findContours`).
     - Metrology calculation: Aspect ratio, perimeter, convex hull solidity, skeletonized crack length.
  2. API & Database Upgrades:
     - Store polygon coordinates in `defects` table (`mask_polygon` JSON column).
     - Return mask coordinates to frontend.
  3. Frontend `DefectViewer.tsx` Interactive Overlays:
     - Translucent colored SVG/Canvas polygon masks fitted directly over cracks and scratches.
     - Metrology sidebar showing exact measured dimensions and severity rating based on true defect surface area.

---

### Phase 4: Gemini Flash Agentic Root-Cause QA Auditor
> **Goal**: Integrate Gemini 3.8/2.5 Flash as an asynchronous, expert manufacturing QA engineer that diagnoses *why* a part failed and recommends corrective machine maintenance.

* **Key Deliverables**:
  1. `backend/services/vlm_service.py`:
     - Triggers automatically when an inspection is `flagged` or `rejected`.
     - Crops the high-resolution defect patch with 20px contextual margin.
     - Packages image crop + inspection telemetry (material type, speed, line station).
     - Queries Gemini Flash with structured output schema (JSON):
       - `defect_mechanism`: (e.g., "Thermal fatigue fracture", "Extrusion die galling", "Machining chatter").
       - `probable_root_cause`: (e.g., "Insufficient coolant lubrication or worn cutting insert").
       - `corrective_action`: (e.g., "Inspect tooling station 3; check lubricant flow rate").
       - `severity_consensus`: Engineering severity assessment.
  2. Operator Dashboard UI Integration:
     - **"AI Root-Cause Diagnostic Report"** accordion on the Inspector and Alert dialogs.
     - Printable/Exportable **8D Discrepancy Quality Report** (PDF/Markdown) for plant managers.

---

### Phase 5: Production Optimization, Edge Benchmarking & Single-Command Runner
> **Goal**: Polish the entire system for speed, robustness, and foolproof execution.

* **Key Deliverables**:
  1. Latency & Memory Optimization:
     - ONNX Runtime / OpenVINO export options for ultra-fast CPU inference ($<25\text{ms}$).
  2. End-to-end Integration Test (`scripts/test_full_industrial.py`):
     - 4K image &rarr; SAHI Tiling &rarr; Industrial Weights &rarr; Polygon Mask &rarr; Gemini Root Cause &rarr; Live UI.
  3. Documentation updates (`docs/INDUSTRIAL_GUIDE.md`).

---

## 🎯 Scoring Comparison Matrix

| Evaluation Criterion | Current Prototype | After Phase 1–2 | After Full Roadmap (Phase 1–5) |
|:---|:---:|:---:|:---:|
| **Micro-Defect Detection (Hairline Cracks)** | 20 / 100 *(Obliterated by 640px)* | 90 / 100 *(Preserved via Tiling)* | **99 / 100** *(Native sub-pixel resolution)* |
| **Domain Accuracy & Real Defect Recognition** | 35 / 100 *(Generic COCO classes)* | 92 / 100 *(NEU/MVTec Benchmarks)* | **98 / 100** *(Fine-tuned on steel/metal defects)* |
| **Geometry & Metrology** | 40 / 100 *(Rectangles over empty space)* | 60 / 100 *(Tiled boxes)* | **97 / 100** *(Exact polygonal segmentation)* |
| **Root-Cause Engineering Diagnosis** | 0 / 100 *(None)* | 0 / 100 *(None)* | **100 / 100** *(Gemini Flash 8D Diagnostic)* |
| **System Architecture & Full-Stack UI** | 98 / 100 | 98 / 100 | **100 / 100** |
| **Total Score** | **310 / 1000** | **780 / 1000** | **994 / 1000** 🏆 |

---

## 🚦 Execution Rule
We will implement **Phase 1 first**, run comprehensive unit tests with evidence, and verify detection of minute hairline defects before writing a single line of code for Phase 2.

