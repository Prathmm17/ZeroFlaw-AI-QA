# VisionGuard QA — Platform Features, Architecture & Function Reference

> **Authoritative Technical Specification & Function Directory**  
> *Version*: 1.0.0 (Production Master)  
> *Target Problem*: Hackathon Problem Statement 4 — Automated Visual Inspection & Real-Time Defect Routing  
> *Hardware Compatibility*: Windows 10/11 & Linux (x86_64, AMD64, ARM64) | CPU & CUDA

---

## Table of Contents
1. [Platform Executive Overview](#1-platform-executive-overview)
2. [End-to-End System Architecture](#2-end-to-end-system-architecture)
3. [Machine Learning Engine (`ml/`)](#3-machine-learning-engine-ml)
   - [3.1 Detector Orchestrator (`ml/detector.py`)](#31-detector-orchestrator-mldetectorpy)
   - [3.2 Aspect-Ratio Letterbox & Coordinates (`ml/preprocessor.py`)](#32-aspect-ratio-letterbox--coordinates-mlpreprocessorpy)
   - [3.3 Postprocessor, NMS & Calibration (`ml/postprocessor.py`)](#33-postprocessor-nms--calibration-mlpostprocessorpy)
   - [3.4 Unsupervised Anomaly Localizer (`ml/anomaly_engine.py`)](#34-unsupervised-anomaly-localizer-mlanomaly_enginepy)
   - [3.5 Production Routing Arbitrator (`ml/arbitrator.py`)](#35-production-routing-arbitrator-mlarbitratorpy)
   - [3.6 Domain Guard (`ml/domain_guard.py`)](#36-domain-guard-mldomain_guardpy)
   - [3.7 Sub-Millimeter Metrology & Contours (`ml/segmenter.py`)](#37-sub-millimeter-metrology--contours-mlsegmenterpy)
   - [3.8 SAHI High-Resolution Patch Tiler (`ml/tiler.py`)](#38-sahi-high-resolution-patch-tiler-mltilerpy)
   - [3.9 Model Weights Registry (`ml/model_registry.py`)](#39-model-weights-registry-mlmodel_registrypy)
   - [3.10 Dataset Merger & Expansion (`ml/merge_datasets.py`)](#310-dataset-merger--expansion-mlmerge_datasetspy)
4. [Backend API & Microservices (`backend/`)](#4-backend-api--microservices-backend)
   - [4.1 Application Core (`backend/main.py`, `backend/config.py`)](#41-application-core-backendmainpy-backendconfigpy)
   - [4.2 Persistence & Database Models (`backend/database.py`)](#42-persistence--database-models-backenddatabasepy)
   - [4.3 Pydantic Data Contracts (`backend/models.py`)](#43-pydantic-data-contracts-backendmodelspy)
   - [4.4 Detection & Inspection Endpoints (`backend/routers/detection.py`)](#44-detection--inspection-endpoints-backendroutersdetectionpy)
   - [4.5 Real-Time Binary Streaming & WebSockets (`backend/routers/stream.py`)](#45-real-time-binary-streaming--websockets-backendroutersstreampy)
   - [4.6 Manufacturing Alert Engine (`backend/routers/alerts.py`, `backend/services/alert_service.py`)](#46-manufacturing-alert-engine-backendroutersalertspy-backendservicesalert_servicepy)
   - [4.7 SPC & Executive Analytics (`backend/routers/analytics.py`)](#47-spc--executive-analytics-backendroutersanalyticspy)
   - [4.8 Visual Grounding & Fallback (`backend/services/gemini_grounding.py`)](#48-visual-grounding--fallback-backendservicesgemini_groundingpy)
   - [4.9 Agentic 8D Engineering Auditor (`backend/services/vlm_service.py`)](#49-agentic-8d-engineering-auditor-backendservicesvlm_servicepy)
   - [4.10 Defect Knowledge Catalog (`backend/services/knowledge_service.py`, `backend/defect_knowledge.json`)](#410-defect-knowledge-catalog-backendservicesknowledge_servicepy-backenddefect_knowledgejson)
5. [Frontend Industrial Studio (`frontend/`)](#5-frontend-industrial-studio-frontend)
   - [5.1 Dashboard Page (`frontend/src/pages/DashboardPage.tsx`)](#51-dashboard-page-frontendsrcpagesdashboardpagetsx)
   - [5.2 Industrial Image Inspector (`frontend/src/pages/InspectorPage.tsx`)](#52-industrial-image-inspector-frontendsrcpagesinspectorpagetsx)
   - [5.3 Live Conveyor Stream (`frontend/src/pages/LiveStreamPage.tsx`)](#53-live-conveyor-stream-frontendsrcpageslivestreampagetsx)
   - [5.4 Inspection Audit Log (`frontend/src/pages/HistoryPage.tsx`)](#54-inspection-audit-log-frontendsrcpageshistorypagetsx)
   - [5.5 Quality Analytics & SPC (`frontend/src/pages/AnalyticsPage.tsx`)](#55-quality-analytics--spc-frontendsrcpagesanalyticspagetsx)
   - [5.6 Defect Knowledge Glossary (`frontend/src/pages/GlossaryPage.tsx`)](#56-defect-knowledge-glossary-frontendsrcpagesglossarypagetsx)
   - [5.7 Interactive Defect Viewer & 8D Modal (`frontend/src/components/DefectViewer.tsx`)](#57-interactive-defect-viewer--8d-modal-frontendsrccomponentsdefectviewertsx)
6. [Data Directory & Model Artifacts](#6-data-directory--model-artifacts)
7. [Testing, Benchmarking & Verification Suite](#7-testing-benchmarking--verification-suite)

---

## 1. Platform Executive Overview

**VisionGuard QA** is a dual-tier edge-and-cloud AI visual inspection platform engineered for high-speed manufacturing lines (steel processing, foundry casting, machining, stamping, and structural assembly).

```
+---------------------------------------------------------------------------------------------------------+
|                                           VISIONGUARD QA PIPELINE                                       |
+---------------------------------------------------------------------------------------------------------+
| [High-Res Camera / Conveyor Video / Upload]                                                            |
|        |                                                                                                |
|        v                                                                                                |
| [Aspect-Ratio Letterbox & SAHI Tiling]  --> Ingestion at native scale (512px tiles or 640px letterbox)  |
|        |                                                                                                |
|        v                                                                                                |
| [Tier 1 YOLOv8 Inference Gate]          --> Batched forward pass (<25ms) via ONNX Runtime / PyTorch    |
|        |                                                                                                |
|        v                                                                                                |
| [Domain Guard Check]                    --> Texture variance & color saturation (HIGH vs LOW trust)     |
|        |                                                                                                |
|        +---> In-Domain & High Conf     --> Fast Path Certification (CLEAN_PASS / REJECT / REWORK)       |
|        |                                                                                                |
|        +---> Out-of-Domain or Low Conf --> Tier 2 Anomaly Engine (Multi-scale Blackhat Morphology)     |
|                                         --> Relabel to "unclassified_anomaly" (source: anomaly_cv)     |
|                                         --> Fallback on 3D parts to prevent zero-defect escapes         |
|        |                                                                                                |
|        v                                                                                                |
| [Douglas-Peucker Metrology]             --> Exact polygon contour, area (mm² or px), length, width      |
|        |                                                                                                |
|        v                                                                                                |
| [Production Routing Arbitration]        --> PASS / FLAGGED for human review / REJECT                   |
|        |                                                                                                |
|        +---> Real-Time Alerts Dispatch  --> WebSocket push to operator terminal (<10ms)                |
|        +---> Persistence Audit Trail    --> SQLite DB (inspections, defect records, metrology JSON)    |
|        +---> Agentic 8D Quality Audit   --> ISO/ASTM root-cause discrepancy report synthesis           |
+---------------------------------------------------------------------------------------------------------+
```

### Core Defect Classes (NEU-DET Industrial Standard + Out-of-Domain Extension)
| Class ID | Class Key | Physical Definition |
| :---: | :--- | :--- |
| `0` | **`crazing`** | High-density microscopic surface hairline fissures and thermal quench cracking. |
| `1` | **`inclusion`** | Submerged or surface non-metallic slag, oxide particles, and ceramic entrapment. |
| `2` | **`patches`** | Surface tears, roll-scabbing, localized adhesion blisters, or thermal scorch burns. |
| `3` | **`pitted_surface`**| Acid-etching over-treatment, cavitation micro-pores, and gas porosity blowholes. |
| `4` | **`rolled_in_scale`** | Mechanically embedded mill scale pressed into strip steel under hot/cold rollers. |
| `5` | **`scratches`** | Linear abrasive gouges caused by mechanical guides, worn conveyors, or tooling shoes. |
| `-1` | **`unclassified_anomaly`** | Novel or out-of-domain textural, fracture, or void flaws identified by the Anomaly Engine. |

---

## 2. End-to-End System Architecture

The codebase is organized into three decoupled layers:

```
Vision Guard/
├── backend/                  # FastAPI 0.110+ ASGI service & SQLite persistence
│   ├── routers/              # Modular REST & WebSocket controllers
│   ├── services/             # Domain logic (detection, alerts, VLM, knowledge)
│   ├── class_thresholds.json # Swept per-class optimal F1 inference cutoffs
│   ├── defect_knowledge.json # Engineering taxonomy definitions & actions
│   └── database.py           # SQLAlchemy 2.0 ORM schemas
├── ml/                       # Computer vision, inference & metrology pipeline
│   ├── detector.py           # High-level pipeline coordinator & ONNX engine
│   ├── preprocessor.py       # Aspect-ratio letterboxing & coordinate transforms
│   ├── postprocessor.py      # Class-aware NMS, area limits & inverse mapping
│   ├── anomaly_engine.py     # Unsupervised Blackhat morphology & connected components
│   ├── arbitrator.py         # Production routing & fail-safe decision arbiter
│   ├── domain_guard.py       # Color saturation & Laplacian texture variance guard
│   ├── segmenter.py          # Honest calibrated (mm) / uncalibrated (px) metrology
│   ├── tiler.py              # SAHI patch slicing & coordinate stitching
│   └── model_registry.py     # Weights locator & ONNX CPU/CUDA runtime manager
├── frontend/                 # React 18 + Vite + Tailwind CSS industrial dashboard
│   ├── src/pages/            # 6 Production application pages
│   ├── src/components/       # DefectViewer, HUDOverlay, Navigation
│   └── src/api/              # Type-safe Axios client with backpressure controls
└── tests/                    # 33 Unit tests validating all transforms and rules
```

---

## 3. Machine Learning Engine (`ml/`)

### 3.1 Detector Orchestrator (`ml/detector.py`)
High-level orchestrator encapsulating YOLOv8 inference, SAHI tiling, Domain Guard checking, Tier 2 anomaly cross-referencing, and metrology extraction.

* **Class**: `DefectDetector(model_name="best.pt", confidence_threshold=0.20, class_thresholds=None, device="cpu", enable_tiling=True, slice_size=512, overlap_ratio=0.20, tiling_min_dim=None)`
  - **`detect(image_source, deep_verify=False, yolo_only=False)`**: End-to-end detection pipeline.
    - Resolves image from file path, numpy array, or raw bytes.
    - Evaluates phone-photo resolution; downscales if dimensions exceed `tiling_min_dim` (1600px).
    - If `yolo_only=True` (used in live video stream): executes fast single-pass forward pass, returning in `<25ms` without running anomaly or segmentation.
    - If high-resolution: slices into overlapping $512 \times 512$ patches via `ImageTiler` and infers concurrently.
    - If standard resolution: letterboxes to $640 \times 640$ neutral padding `(114, 114, 114)`.
    - Evaluates `DomainGuard.assess()`. If `domain_trust == "LOW"` or `deep_verify == True`:
      - Cross-references YOLO candidates with unsupervised anomaly energy regions.
      - Relabels sub-threshold YOLO boxes to `"unclassified_anomaly"` with source `"anomaly_cv"`.
      - If YOLO produces 0 detections: invokes `_detect_structural_anomalies()` as an automated fallback to capture 3D cracks, porosity, and voids.
    - Computes Douglas-Peucker contour segmentation and physical metrology for each detection.
    - Returns dictionary with detections, summary metrics, domain assessment, and calibration metadata.
  - **`detect_batch(sources)`**: Processes an iterable of images sequentially with graceful exception handling.
  - **`set_calibration(px_per_mm)`**: Configures the pixel-to-millimeter optical ratio for sub-millimeter measurements.
  - **`_detect_structural_anomalies(image, anomaly_regions=None, domain_trust="HIGH")`**: Fallback generator isolating physical cracks, porosity clusters, and severe burns when supervised YOLO has zero confidence on novel materials.
  - **`_overlaps_anomaly(bbox, img_shape, anomaly_regions, min_overlap=0.10)`**: Computes intersection-over-minimum-area between bounding boxes and morphological anomaly heatmaps.

---

### 3.2 Aspect-Ratio Letterbox & Coordinates (`ml/preprocessor.py`)
Ensures no image distortion occurs prior to deep learning inference while storing exact mathematical scale and offset parameters for coordinate inversion.

* **Class**: `ImagePreprocessor(target_size=(640, 640))`
  - **`preprocess(image_source)`**: Full pipeline: loads image, stores original shape, letterboxes onto gray canvas `(114, 114, 114)`, converts BGR to RGB, normalizes pixels to $[0.0, 1.0]$, transposes to NCHW `(1, 3, 640, 640)` float32.
  - **`resize(image)`**: Scales image using uniform scale factor $s = \min(w_{\text{target}}/w, h_{\text{target}}/h)$ and applies bilinear interpolation.
  - **`get_letterbox_params(source_hw)`**: Computes:
    $$\text{scale} = \min\left(\frac{640}{w_0}, \frac{640}{h_0}\right)$$
    $$\text{pad\_left} = \left\lfloor \frac{640 - \text{round}(w_0 \times s)}{2} \right\rfloor, \quad \text{pad\_top} = \left\lfloor \frac{640 - \text{round}(h_0 \times s)}{2} \right\rfloor$$
  - **`inverse_letterbox_bbox(bbox, scale, pad_left, pad_top, original_shape)`**: Maps coordinates from letterbox space $(x_{\text{lb}}, y_{\text{lb}})$ back to original image space $(x_0, y_0)$:
    $$x_0 = \frac{x_{\text{lb}} - \text{pad\_left}}{\text{scale}}, \quad y_0 = \frac{y_{\text{lb}} - \text{pad\_top}}{\text{scale}}$$
    Clips coordinates to bounds $[0, w_0]$ and $[0, h_0]$ with sub-pixel rounding.

---

### 3.3 Postprocessor, NMS & Calibration (`ml/postprocessor.py`)
Filters raw network outputs, enforces physical manufacturing heuristics, and enriches detections with metadata.

* **Class**: `DetectionPostprocessor(min_confidence=0.35, defect_classes=None, class_thresholds=None)`
  - **`process(results, image_shape, letterbox_params=None)`**: Extracts bounding boxes, confidences, and class IDs from raw prediction tensors, performs coordinate inversion, filters by `min_confidence`, and applies suppression rules:
    - **Rule 1 (Area Ceiling)**: Drops any box occupying $>35\%$ of total image area (prevents whole-frame false positives on textured backgrounds).
    - **Rule 2 (Aspect Ratio Limit)**: Drops boxes with $\max(w, h)/\min(w, h) > 15:1$ for all classes except `scratches` and `crazing`.
  - **`apply_class_aware_nms(detections, iou_threshold=0.50)`**: Resolves candidate overlaps across YOLO and Anomaly Engine boxes. If two boxes overlap with $\text{IoU} \ge 0.50$, preserves the higher-confidence detection.
  - **`classify_severity(confidence, area_ratio, class_name="unknown", domain_trust="HIGH", source="yolo")`**:
    - Assigns `critical` (immediate line-halt) only when $\text{confidence} \ge \text{threshold}$ AND $\text{area\_ratio} \ge 0.05$ (or defect is crazing/inclusion).
    - If `domain_trust == "LOW"` and `source == "anomaly_cv"`: severity is capped at `major` (verdict becomes `FLAGGED for human review`, never automated line rejection).
    - Assigns `major` for moderate flaws and `minor` for micro-defects ($\text{area} < 0.005$).
  - **`get_summary(detections)`**: Computes counts by severity, counts by class, and flag `has_critical`.

---

### 3.4 Unsupervised Anomaly Localizer (`ml/anomaly_engine.py`)
Computer-vision morphological engine that detects unforeseen physical defects without requiring training labels.

* **Class**: `UnsupervisedAnomalyEngine(sensitivity=0.65, min_anomaly_ratio=0.0004, max_anomaly_ratio=0.35)`
  - **`compute_anomaly_heatmap(image)`**: Converts image to grayscale, isolates bottom 8% SEM footer telemetry, applies Multi-Scale Blackhat Morphology (extracts dark pits and fine cracks smaller than structuring elements), applies bilateral gradient energy, and thresholds with Otsu / dynamic percentile.
  - **`detect_anomalies(image, max_detections=6)`**:
    - Employs **8-way Connected Component Analysis** (`cv2.connectedComponentsWithStats`) on the thresholded mask.
    - Generates strictly one candidate box per isolated physical component without morphological bridging.
    - Drops components spanning frame boundaries or exceeding $35\%$ image area.
    - Assigns morphological hint based on component aspect ratio and HSV saturation:
      - $\text{Aspect} \ge 4.0 \implies$ `crazing` (hairline fissure)
      - $\text{Aspect} \ge 2.0 \implies$ `scratches` (abrasion gouge)
      - $\text{Saturation} > 40 \implies$ `patches` (thermal scorch)
      - Otherwise $\implies$ `pitted_surface` (cavitation micropore)
    - Returns sorted list of `AnomalyRegion` objects with normalized coordinates, energy scores, and polygon points.

---

### 3.5 Production Routing Arbitrator (`ml/arbitrator.py`)
Two-tier arbitration controller evaluating whether an inspection passes, halts the line, or requires engineering review.

* **Class**: `ProductionArbitrator(edge_detector=None, vlm_engine=None)`
  - **`arbitrate(image_source, filename="inspection.jpg", force_tier2=False, deep_verify=False)`**:
    - **Fast Path Certification**: If YOLO finds 0 detections, `domain_trust == "HIGH"`, and `deep_verify == False`: executes clean-pass certification in $<25\text{ms}$ returning `verdict = "PASSED"`, `arbitration_status = "CLEAN_PASS"`.
    - **Tier 1 Confirmed Path**: If high-confidence defects are identified:
      - Routes to `verdict = "REJECTED"` if any critical defects exist in-domain.
      - Routes to `verdict = "FLAGGED for human review"` if `domain_trust == "LOW"` or only major defects exist.
      - Sets status to `ANOMALY_ESCALATED` if any detections originate from `anomaly_cv`.
    - **Tier 2 VLM Escalation**: If `deep_verify=True` or `force_tier2=True`: invokes Gemini Zero-Shot Visual Grounding to assign failure mechanisms and verify physical defect dimensions.

---

### 3.6 Domain Guard (`ml/domain_guard.py`)
Assesses whether an inspection photo belongs to the in-domain training distribution (flat grayscale steel) or out-of-domain distribution (3D machined parts, natural lighting, high color saturation).

* **Class**: `DomainGuard(saturation_threshold=25.0, texture_variance_threshold=700.0, min_yolo_confidence_threshold=0.40)`
  - **`assess(image, max_yolo_confidence=0.0)`**:
    - Converts image to HSV and calculates mean color saturation $S_{\text{mean}}$.
    - Computes Laplacian variance $\sigma_{\text{Laplacian}}^2$ to measure high-frequency 3D surface relief and shadowing.
    - If $S_{\text{mean}} > 25.0$ (natural color photo) or $\sigma_{\text{Laplacian}}^2 > 700.0$ (complex 3D casting):
      - Returns `trust = "LOW"`, `warning = "Image differs from the training domain (flat steel surface). Result may be unreliable."`
    - If within distribution and YOLO confidence is high:
      - Returns `trust = "HIGH"`, `warning = None`.

---

### 3.7 Sub-Millimeter Metrology & Contours (`ml/segmenter.py`)
Extracts exact boundary contours and physical measurements with honest labeling between calibrated (mm) and uncalibrated (px) modes.

* **Class**: `DefectSegmenter(pixels_per_mm=None)`
  - **`set_calibration(pixels_per_mm)`**: Sets optical scale in $\text{px}/\text{mm}$.
  - **`segment_roi(image, bbox_norm)`**:
    - Extracts Region of Interest (ROI) from normalized bounding coordinates.
    - Applies Gaussian blur and adaptive thresholding to isolate defect boundaries.
    - Computes Douglas-Peucker polygon approximation (`cv2.approxPolyDP`) with precision $\epsilon = 0.02 \times \text{perimeter}$.
    - **Honest Metrology Formulation**:
      - If `pixels_per_mm` is `None`:
        - `area = float(pixel_area)`, unit labeled `"uncalibrated (px)"`
        - `length = float(max(w_px, h_px))`, `width = float(min(w_px, h_px))`
      - If `pixels_per_mm` is configured:
        - $\text{area\_mm}^2 = \frac{\text{pixel\_area}}{(\text{px\_per\_mm})^2}$
        - $\text{length\_mm} = \frac{\text{length\_px}}{\text{px\_per\_mm}}, \quad \text{width\_mm} = \frac{\text{width\_px}}{\text{px\_per\_mm}}$
        - Unit labeled `"calibrated"`
    - Calculates morphological `aspect_ratio` and `solidity` ($\frac{\text{Contour Area}}{\text{Convex Hull Area}}$).

---

### 3.8 SAHI High-Resolution Patch Tiler (`ml/tiler.py`)
Slices ultra-high-resolution images into manageable overlapping windows to detect 1-to-2-pixel micro-defects without loss of fidelity.

* **Class**: `ImageTiler(slice_size=512, overlap_ratio=0.20, iou_threshold=0.45)`
  - **`slice_image(image)`**: Computes step size $\Delta = \lfloor \text{slice\_size} \times (1 - \text{overlap\_ratio}) \rfloor$. Generates a grid of `ImageSlice` objects with coordinates `(x1, y1, x2, y2)`.
  - **`merge_detections(slice_results, image_shape)`**:
    - Adds tile origin offsets $(x_{\text{tile}}, y_{\text{tile}})$ to each local bounding box.
    - Applies global Non-Maximum Suppression (IoU cutoff $0.45$) to merge bounding boxes spanning tile boundaries.

---

### 3.9 Model Weights Registry (`ml/model_registry.py`)
Manages model artifacts, CPU/CUDA acceleration providers, and ONNX Runtime execution sessions.

* **Class**: `ModelRegistry(models_dir=None)`
  - **`get_model_path(model_name="best.onnx")`**: Locates active weights in `models/` directory.
  - **`load_model(model_name="best.onnx", device="cpu")`**: Initializes ONNX Runtime session with optimized CPU thread pooling (`IntraOpNumThreads = CPU count`) or PyTorch `ultralytics.YOLO`.
  - **`export_to_onnx(model_path, imgsz=640)`**: Exports PyTorch checkpoint to optimized ONNX format.

---

### 3.10 Dataset Merger & Expansion (`ml/merge_datasets.py`)
Utility to ingest and remap external datasets (GC10-DET, Severstal Steel, NEU-DET) into a unified 7-class schema including a dedicated `dent` class.

* **Class**: `DatasetMerger(output_dir="data/merged_dataset")`
  - **`merge(source_dirs, class_mapping)`**: Remaps source class IDs to target IDs, copies image files, normalizes YOLO `.txt` bounding coordinates, and generates `dataset.yaml`.

---

## 4. Backend API & Microservices (`backend/`)

### 4.1 Application Core (`backend/main.py`, `backend/config.py`)
- **FastAPI Core**: Lifespan management with lazy detector pre-warming and database initialization.
- **Static File Mounting**:
  - `/data/uploads` $\rightarrow$ User and camera uploaded inspection images
  - `/data/results` $\rightarrow$ Annotated output images with rendered defect overlays
  - `/data/samples` $\rightarrow$ Factory catalog sample library
- **CORS Middleware**: Allows cross-origin requests from development and production dashboards.
- **System Route**: `GET /api/health` returns status, model name, device, and readiness flag.

---

### 4.2 Persistence & Database Models (`backend/database.py`)
Declarative SQLite ORM models implemented via SQLAlchemy 2.0:

1. **`Inspection`**:
   - `id`: Auto-incrementing primary key.
   - `timestamp`: UTC datetime stamp.
   - `filename`, `source`, `original_path`, `result_path`.
   - `total_defects`, `critical_count`, `major_count`, `minor_count`.
   - `status`: `passed`, `flagged`, `rejected`.
   - `arbitration_tier`: `tier1_edge` vs `tier2_vlm`.
   - `arbitration_status`: `CLEAN_PASS`, `EDGE_CONFIRMED`, `ANOMALY_ESCALATED`.
   - `domain_trust`: `HIGH` vs `LOW`.
   - `domain_warning`: Explanation string when trust is `LOW`.
   - `audit_report_json`: Stored 8D engineering audit report payload.
2. **`DefectRecord`**:
   - `id`, `inspection_id`: Foreign key linked to `Inspection`.
   - `class_id`, `class_name`, `confidence`, `severity`.
   - `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2`: Absolute coordinates in original image space.
   - `area_ratio`: Relative ratio against full image area.
   - `polygon_json`: Serialized Douglas-Peucker contour coordinates.
   - `metrology_json`: Physical dimensions, area ($\text{mm}^2$ or $\text{px}$), length, width, aspect ratio, solidity.
   - `source`: `"yolo"`, `"anomaly_cv"`, or `"vlm"`.
   - `failure_mechanism`: Inferred physical root cause.
3. **`AlertRecord`**:
   - `id`, `inspection_id`, `timestamp`, `severity`, `defect_type`, `message`, `action_required`, `acknowledged`.

---

### 4.3 Pydantic Data Contracts (`backend/models.py`)
Strictly typed schemas validating incoming and outgoing JSON payloads:
- `InspectionResponse`: Encapsulates complete inspection summary, domain assessment, calibration metadata, and defect arrays.
- `DefectResponse`: Individual defect with bounding box, severity, confidence, source badge, metrology dictionary, and knowledge definition.
- `MetrologyResponse`: Area, width, length, aspect ratio, solidity, calibration label, and contour polygon points.
- `AlertResponse`: Alert record schema for WebSocket and REST broadcasting.
- `CalibrationUpdateRequest`: Request payload setting `px_per_mm`.
- `CalibrationLineMeasureRequest`: User-drawn pixel length and physical mm length used to compute scale: $\text{scale} = \frac{\text{pixel\_distance}}{\text{known\_length\_mm}}$.

---

### 4.4 Detection & Inspection Endpoints (`backend/routers/detection.py`)
| HTTP Method | Route | Description |
| :--- | :--- | :--- |
| `POST` | `/api/detect/upload` | Multipart file upload. Runs two-tier inspection, saves original & annotated images, records to DB, broadcasts alerts. Query param `deep_verify=True/False`. |
| `GET` | `/api/detect/samples` | Lists all demo catalog images in `data/samples/`. |
| `POST` | `/api/detect/sample/{sample_name}` | Executes complete inspection on a selected catalog sample. |
| `GET` | `/api/detect/history` | Paginated inspection audit history with filtering by verdict, severity, and date range. |
| `GET` | `/api/detect/{inspection_id}` | Retrieves detailed inspection record including polygon contours and metrology. |
| `POST` | `/api/detect/{inspection_id}/audit` | Triggers agentic 8D root-cause engineering discrepancy report. |
| `GET` | `/api/detect/{inspection_id}/8d-report`| Retrieves existing or generates fresh 8D discrepancy report. |
| `POST` | `/api/detect/{inspection_id}/vlm-grounding` | Runs zero-shot visual grounding on inspection image. |
| `GET` | `/api/detect/glossary` | Retrieves complete engineering taxonomy dictionary for all 7 defect classes. |
| `GET` | `/api/detect/calibration` | Returns active `px_per_mm`, calibration status, and label. |
| `POST` | `/api/detect/calibration` | Updates active optical scale (`px_per_mm`). |
| `POST` | `/api/detect/calibration/measure` | Computes and updates optical scale from user-drawn line measurement. |

---

### 4.5 Real-Time Binary Streaming & WebSockets (`backend/routers/stream.py`)
Engineered to deliver **15+ FPS** live video inspection over WebSockets:
- **`WS /ws/stream`**:
  - Ingests binary JPEG frames from industrial camera or browser client.
  - **Thread-Pool Executor**: Runs ONNX model inference inside a Python `ThreadPoolExecutor` worker to avoid blocking the asyncio event loop.
  - **Single-Item Backpressure Queue**: Keeps only the most recent frame; older frames are discarded if the inference worker is busy.
  - **Lean Telemetry Payload**: Returns JSON bounding box coordinates and HUD metrics only—never wastes bandwidth re-encoding annotated video on the server.
  - **Selective DB Persistence**: Writes to the SQLite database only when inspection verdict changes (e.g. from `passed` to `rejected`) to prevent disk I/O bottlenecks.
- **`WS /ws/alerts`**:
  - Push broadcast channel alerting all connected factory floor dashboards of critical manufacturing line events.

---

### 4.6 Manufacturing Alert Engine (`backend/routers/alerts.py`, `backend/services/alert_service.py`)
- **`GET /api/alerts`**: Retrieves unacknowledged active alerts sorted by timestamp descending.
- **`POST /api/alerts/{alert_id}/acknowledge`**: Marks alert as acknowledged with operator timestamp.
- **`AlertService.process_inspection_alerts()`**: Evaluates inspection results; automatically raises:
  - `CRITICAL`: Immediate line-halt alert for critical severity flaws.
  - `MAJOR`: Quarantine routing alert for major defects or out-of-domain flagged parts.

---

### 4.7 SPC & Executive Analytics (`backend/routers/analytics.py`)
- **`GET /api/analytics/overview`**: Computes total inspections, defect rate percentage, yield rate, severity breakdown, and average inference latency.
- **`GET /api/analytics/trends`**: Aggregates daily inspection volume, defect occurrences, and pass/fail distribution for Statistical Process Control (SPC) charting.

---

### 4.8 Visual Grounding & Fallback (`backend/services/gemini_grounding.py`)
- **`GeminiGroundingService`**:
  - When `GEMINI_API_KEY` is present: queries Gemini 1.5 Pro with normalized box coordinate prompts (`[ymin, xmin, ymax, xmax]`) and metallurgical reasoning.
  - When offline or key is missing: transparently executes the local OpenCV Unsupervised Anomaly Engine fallback.
  - **Honest Attribution**: Marks detection source as `"anomaly_cv"` when offline, ensuring the user interface displays the `"Anomaly (CV)"` badge and never falsifies a VLM verification.

---

### 4.9 Agentic 8D Engineering Auditor (`backend/services/vlm_service.py`)
Generates formal manufacturing discrepancy reports adhering to standard automotive and aerospace 8D frameworks:
- **D1 (Team Formulation)**: Autonomous Quality Assurance Agent & Line Lead.
- **D2 (Problem Description)**: Defect class, physical dimensions, location coordinates, optical scale.
- **D3 (Interim Containment)**: Immediate conveyor gate routing (quarantine/line-halt).
- **D4 (Root Cause Analysis)**: Probable metallurgical breakdown mechanism (e.g. thermal fatigue, rolling friction, chemical over-etching).
- **D5/D6 (Corrective Action & Validation)**: Specific tooling recalibration, roller grinding, acid bath dilution adjustments.
- **D7 (Recurrence Prevention)**: ISO/ASTM standard references (e.g. ASTM E45 for inclusions, ASTM E381 for macro-etching).

---

### 4.10 Defect Knowledge Catalog (`backend/services/knowledge_service.py`, `backend/defect_knowledge.json`)
Authoritative metallurgical knowledge base containing definitions, visual look, manufacturing root causes, risks, recommended engineering actions, and default severities for:
1. `crazing`
2. `inclusion`
3. `patches`
4. `pitted_surface`
5. `rolled_in_scale`
6. `scratches`
7. `unclassified_anomaly`

---

## 5. Frontend Industrial Studio (`frontend/`)

Built with React 18, TypeScript, Vite, and Tailwind CSS using an industrial color scheme:
- `industrial-950` (`#090d16`): Canvas background
- `industrial-900` (`#0f172a`): Card & viewport containers
- `industrial-800` (`#1e293b`): Sub-panels & borders
- `industrial-700` (`#334155`): Dividers & grid lines

---

### 5.1 Dashboard Page (`frontend/src/pages/DashboardPage.tsx`)
Executive operations view featuring:
- **Summary KPI Metric Cards**: Total Inspections, Defect Rate %, Quality Pass Rate %, Average Edge Latency.
- **Live Manufacturing Alert Banner**: Real-time notifications of line-halt triggers.
- **Defect Distribution Chart**: Recharts bar chart visualizing occurrences per class.
- **Recent Inspections Table**: Interactive list with verdict badges (`PASS`, `FLAGGED`, `REJECT`), defect counts, and one-click navigation to full inspection details.

---

### 5.2 Industrial Image Inspector (`frontend/src/pages/InspectorPage.tsx`)
High-resolution inspection studio providing:
- **Drag-and-Drop Dropzone & File Browser**: Supports PNG, JPEG, TIFF, BMP, WEBP.
- **Demo Catalog Selector**: One-click selection of real industrial benchmark specimens.
- **Deep Verify Toggle**: Enabled by default for static uploads; executes full Tier 2 anomaly and visual grounding.
- **Domain Guard Trust Alert**: Displays a high-visibility alert banner when an image is out-of-domain:
  > *"Image differs from the training domain (flat steel surface). Result may be unreliable."*
- **Interactive Calibration Helper**: Allows the operator to measure a reference artifact on screen, input its known length in millimeters, and automatically compute `px_per_mm`.

---

### 5.3 Live Conveyor Stream (`frontend/src/pages/LiveStreamPage.tsx`)
Webcam and RTSP stream monitor:
- **Camera Selection**: Dropdown supporting built-in webcams and USB/GigE industrial cameras.
- **Dynamic HTML5 Canvas Overlay**: Renders bounding boxes and defect labels on client-side canvas matching video aspect ratio.
- **Telemetry HUD**: Displays real-time FPS, server round-trip latency (ms), frame drop counter, and routing status pill (`PASS`, `FLAGGED`, `REJECT`).
- **Backpressure Protection**: Binary frames are transmitted only after the previous frame response arrives, guaranteeing that video streams never lag behind real-time conveyor motion.

---

### 5.4 Inspection Audit Log (`frontend/src/pages/HistoryPage.tsx`)
Searchable and filterable compliance ledger:
- **Filters**: Filter by verdict status (`All`, `Passed`, `Flagged`, `Rejected`), severity level (`Critical`, `Major`, `Minor`), and defect class.
- **Audit Detail Modal**: Opens full inspection view with metrology metrics and stored 8D engineering reports.

---

### 5.5 Quality Analytics & SPC (`frontend/src/pages/AnalyticsPage.tsx`)
Statistical Process Control center:
- **Defect Rate Timelines**: Historical line charts visualizing defect rate trends over 7, 14, or 30 days.
- **Pareto Defect Breakdown**: Highlights the most frequent defects to guide preventive maintenance.
- **Inference Speed Benchmarks**: Monitors latency stability across edge inferences.

---

### 5.6 Defect Knowledge Glossary (`frontend/src/pages/GlossaryPage.tsx`)
Reference catalog for plant operators and quality engineers:
- Displays all 7 defect classes with optimal F1 confidence thresholds.
- Details Plain Definition, Visual Look, Manufacturing Cause, Operational Risk, and Recommended Corrective Action for each defect.

---

### 5.7 Interactive Defect Viewer & 8D Modal (`frontend/src/components/DefectViewer.tsx`)
Detailed inspection viewport:
- **Bounding Box & Polygon Overlay**: Color-coded by severity (`Critical`: Red, `Major`: Orange, `Minor`: Yellow).
- **Interactive Highlighting**: Clicking any defect card highlights its corresponding box and zooms the viewport.
- **Defect Card Information**: Displays defect name, confidence score, source badge (`YOLO`, `Anomaly (CV)`, `VLM`), trust level, metallurgical cause, and immediate recommended action.
- **Honest Metrology Section**: Displays Area, Length, and Width clearly labeled as `"calibrated (mm / mm²)"` or `"uncalibrated (px)"`.
- **8D Discrepancy Report Generator**: Generates and downloads full 8D Root-Cause Discrepancy reports.

---

## 6. Data Directory & Model Artifacts

```
data/
├── real_industrial_dataset/   # 120-image industrial validation split with ground truth labels
│   ├── images/val/            # Benchmark validation images
│   ├── labels/val/            # YOLO normalized bounding box text files
│   └── dataset.yaml           # Dataset metadata and class index definitions
├── samples/                   # Catalog specimens for quick demo testing
├── uploads/                   # Stored original inspection images
├── results/                   # Stored annotated inspection images with rendered boxes
└── db/
    └── visionguard.db         # SQLite database storing inspections, defects, and alerts

models/
├── best.onnx                  # Active 8.1 GFLOPs ONNX model with CPU thread pooling
├── best.pt                    # Active PyTorch YOLOv8n weights
└── industrial_v2_best.pt      # Fine-tuned checkpoint
```

---

## 7. Testing, Benchmarking & Verification Suite

The repository includes a comprehensive, non-simulated automated test suite:

| Test Script | Scope & Assertions | Status |
| :--- | :--- | :---: |
| [`tests/test_letterbox.py`](file:///c:/Users/hp/Desktop/Vision%20Guard/tests/test_letterbox.py) | Mathematical validation of forward and inverse letterbox coordinate transformations across landscape, portrait, and square canvases ($\le 2\text{px}$ error). | **PASSED** (6/6) |
| [`tests/test_box_suppression.py`](file:///c:/Users/hp/Desktop/Vision%20Guard/tests/test_box_suppression.py) | Asserts connected component isolation, $35\%$ area ceiling, $15:1$ aspect ratio filtering, and class-aware NMS. | **PASSED** (4/4) |
| [`tests/test_honest_metrology.py`](file:///c:/Users/hp/Desktop/Vision%20Guard/tests/test_honest_metrology.py) | Validates that uncalibrated mode returns only `"uncalibrated (px)"` and calibrated mode returns verified mm and $\text{mm}^2$. | **PASSED** (4/4) |
| [`tests/test_detection_sources.py`](file:///c:/Users/hp/Desktop/Vision%20Guard/tests/test_detection_sources.py) | Asserts source field labeling (`"yolo"`, `"anomaly_cv"`, `"vlm"`), offline fallback labeling, and out-of-domain relabeling. | **PASSED** (4/4) |
| [`tests/test_domain_guard_and_routing.py`](file:///c:/Users/hp/Desktop/Vision%20Guard/tests/test_domain_guard_and_routing.py) | Validates color saturation and texture variance thresholds, and enforces that LOW trust parts are routed to `FLAGGED`, never auto-`REJECT`. | **PASSED** (6/6) |
| [`tests/test_defect_knowledge.py`](file:///c:/Users/hp/Desktop/Vision%20Guard/tests/test_defect_knowledge.py) | Verifies JSON schema validity and postprocessor enrichment of defect cause and action fields. | **PASSED** (4/4) |
| [`tests/test_merge_datasets.py`](file:///c:/Users/hp/Desktop/Vision%20Guard/tests/test_merge_datasets.py) | Verifies class remapping logic, addition of `dent` class, and Colab training notebook structure. | **PASSED** (5/5) |
| [`scripts/evaluate_and_sweep_thresholds.py`](file:///c:/Users/hp/Desktop/Vision%20Guard/scripts/evaluate_and_sweep_thresholds.py) | Evaluates `best.pt` on 120 validation images; computes optimal F1 thresholds per class; records **97.34% mAP50**. | **PASSED** |
| [`scripts/verify_neu_det_10.py`](file:///c:/Users/hp/Desktop/Vision%20Guard/scripts/verify_neu_det_10.py) | Evaluates 10 benchmark specimens at $\text{IoU} \ge 0.50$; reports **$\text{TP}=14, \text{FP}=0, \text{FN}=1$, $\text{Precision}=100\%$, $\text{F1}=0.966$**. | **PASSED** |
| [`scripts/verify_real_photos.py`](file:///c:/Users/hp/Desktop/Vision%20Guard/scripts/verify_real_photos.py) | Evaluates real 3D industrial photos (cast part and pipe); asserts boxes sit on object, max area $\le 35\%$, trust `LOW` shown, and verdict is `FLAGGED`. | **PASSED** |
| [`scripts/verify_ui_browser.mjs`](file:///c:/Users/hp/Desktop/Vision%20Guard/scripts/verify_ui_browser.mjs) | Automated Chrome DevTools Protocol browser agent navigating all 6 pages; verifies **0 console errors, 0 unhandled exceptions, and 0 network failures**. | **PASSED** |

