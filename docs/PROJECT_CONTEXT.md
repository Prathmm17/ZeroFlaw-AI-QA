# VisionGuard QA — Project Context & System Architecture

## 1. Project
* **Name**: VisionGuard QA
* **Domain**: Real-Time AI Quality Assurance & Micro-Defect Localization for Manufacturing Lines
* **Context**: Hackathon Problem Statement 4 — Automated Visual Inspection & Routing

---

## 2. Goal
Detect, locate, segment, and classify microscopic and macroscopic manufacturing defects from high-resolution static images and real-time industrial video camera streams (GigE / USB / RTSP) with sub-millimeter metrology. Automatically arbitrate and route each inspected component to:
* **PASS**: Nominal component within engineering tolerance (Zero Defect Escape).
* **REWORK**: Major defect detected; routed to offline rework / quarantine station.
* **REJECT**: Critical defect detected; triggers automated line-halt and operator alert.

---

## 3. Dataset and Classes
Configuration source: [`data/real_industrial_dataset/dataset.yaml`](file:///c:/Users/hp/Desktop/Vision%20Guard/data/real_industrial_dataset/dataset.yaml) and [`data/industrial_benchmark/dataset.yaml`](file:///c:/Users/hp/Desktop/Vision%20Guard/data/industrial_benchmark/dataset.yaml)

* **Defect Classes (6 total)**:
  * `0`: **`crazing`** (Microscopic stress fractures & thermal quench fatigue cracking)
  * `1`: **`inclusion`** (Submerged non-metallic slag / ceramic foreign particle entrapment)
  * `2`: **`patches`** (Localized roll tearing, surface adhesion, and thermal scorch burns)
  * `3`: **`pitted_surface`** (Cavitation pitting, chemical over-etching, and micropores)
  * `4`: **`rolled_in_scale`** (Embedded mill oxide scale from high-pressure hot/cold rolling)
  * `5`: **`scratches`** (Mechanical guide shoe abrasion and handling gouges)

---

## 4. Model & Inference Preprocessing
* **Active Weights Path**: [`models/best.pt`](file:///c:/Users/hp/Desktop/Vision%20Guard/models/best.pt) (symlinked / copied from fine-tuned [`models/industrial_v2_best.pt`](file:///c:/Users/hp/Desktop/Vision%20Guard/models/industrial_v2_best.pt) and [`models/industrial_best.pt`](file:///c:/Users/hp/Desktop/Vision%20Guard/models/industrial_best.pt))
* **Architecture Type**: Ultralytics YOLOv8n (PyTorch format, 3.0M parameters, 168 layers)
* **Input Resolution**: $640 \times 640$ pixels (RGB/BGR float32 normalized $[0.0, 1.0]$)
* **Inference Preprocessing Pipeline**:
  * **Aspect-Ratio Preserving Letterbox**: Implemented in [`ml/preprocessor.py`](file:///c:/Users/hp/Desktop/Vision%20Guard/ml/preprocessor.py). Scales input using uniform factor $s = \min(640/w, 640/h)$, resizes with bilinear interpolation, and centers onto neutral gray padding `(114, 114, 114)`.
  * **SAHI Patch Tiling Engine**: Implemented in [`ml/tiler.py`](file:///c:/Users/hp/Desktop/Vision%20Guard/ml/tiler.py). Activated automatically when image dimensions exceed $800\text{px}$. Cuts canvases into $512 \times 512$ pixel patches with $20\%$ overlap, infers at native 1:1 scale to preserve 1-to-2-pixel micro-cracks, and merges candidate bounding boxes via global Non-Maximum Suppression (IoU threshold $0.45$).

---

## 5. Pipeline Stages & Implementation Files
Execution flows sequentially through the following real architectural stages:

1. **Input Ingestion & SAHI Slicing**:
   * File: [`ml/preprocessor.py`](file:///c:/Users/hp/Desktop/Vision%20Guard/ml/preprocessor.py) & [`ml/tiler.py`](file:///c:/Users/hp/Desktop/Vision%20Guard/ml/tiler.py)
   * Decodes image/frame bytes, evaluates resolution, slices into tiles if high-res, or applies letterboxing.
2. **Tier 1 (Fast Edge Gate) YOLOv8 Inference**:
   * File: [`ml/detector.py`](file:///c:/Users/hp/Desktop/Vision%20Guard/ml/detector.py) & [`ml/postprocessor.py`](file:///c:/Users/hp/Desktop/Vision%20Guard/ml/postprocessor.py)
   * Batched forward pass on patches or single pass, post-processes raw bounding boxes, filters by confidence ($\ge 0.20$), and categorizes initial severity (`critical`, `major`, `minor`).
3. **Unsupervised Saliency & Structural Anomaly Localizer**:
   * File: [`ml/anomaly_engine.py`](file:///c:/Users/hp/Desktop/Vision%20Guard/ml/anomaly_engine.py)
   * Multi-scale Blackhat morphology and HSV thermal scorch detection with automatic SEM telemetry footer suppression.
4. **Two-Tier Production Arbitration & Confidence Calibration**:
   * File: [`ml/arbitrator.py`](file:///c:/Users/hp/Desktop/Vision%20Guard/ml/arbitrator.py)
   * High-speed screening ($<25\text{ms}$ on clean parts). If an anomaly is present or YOLO confidence is uncertain ($0.15 \le \text{conf} \le 0.50$), escalates to Tier 2 VLM reasoning.
5. **Tier 2 (Zero-Shot VLM) Visual Grounding**:
   * File: [`backend/services/gemini_grounding.py`](file:///c:/Users/hp/Desktop/Vision%20Guard/backend/services/gemini_grounding.py)
   * Emits normalized bounding coordinates `box_2d: [ymin, xmin, ymax, xmax]`, semantically classifies out-of-domain flaws, and attaches physical failure mechanisms. Operates offline via local anomaly fallback if API key is absent.
6. **Sub-Millimeter Metrology & Polygon Segmentation**:
   * File: [`ml/segmenter.py`](file:///c:/Users/hp/Desktop/Vision%20Guard/ml/segmenter.py)
   * Segments bounding box regions of interest into exact boundary contours (Douglas-Peucker polygon approximation) and calculates physical surface area ($\text{mm}^2$), length, width, aspect ratio, and solidity.
7. **Agentic 8D Quality Root-Cause Audit**:
   * File: [`backend/services/vlm_service.py`](file:///c:/Users/hp/Desktop/Vision%20Guard/backend/services/vlm_service.py)
   * Synthesizes formal 8D engineering discrepancy reports mapping defects to ASTM/ISO standards, suspect tooling stations, D3 containment actions, and D5/D6 corrective maintenance.
8. **Manufacturing Routing & Real-Time Alert Dispatch**:
   * File: [`backend/services/alert_service.py`](file:///c:/Users/hp/Desktop/Vision%20Guard/backend/services/alert_service.py)
   * Evaluates overall inspection severity; routes to `PASSED`, `FLAGGED`, or `REJECTED`; broadcasts real-time critical alert payloads over WebSocket to operator dashboards.
9. **Persistence & Audit Trail**:
   * File: [`backend/database.py`](file:///c:/Users/hp/Desktop/Vision%20Guard/backend/database.py) & [`backend/services/detection_service.py`](file:///c:/Users/hp/Desktop/Vision%20Guard/backend/services/detection_service.py)
   * Stores inspection records, defect bounding coordinates, polygon masks, metrology JSON, and 8D reports into SQLite (`data/db/visionguard.db`).

---

## 6. Backend
* **Framework**: FastAPI (Python 3.10+ / 3.13 compatible), Uvicorn ASGI server, SQLAlchemy 2.0 ORM, SQLite DB, WebSockets, Pydantic v2 schemas.
* **Core Source Files**:
  * [`backend/main.py`](file:///c:/Users/hp/Desktop/Vision%20Guard/backend/main.py): Entry point, CORS middleware, lifespan manager, static file mount (`/data`).
  * [`backend/config.py`](file:///c:/Users/hp/Desktop/Vision%20Guard/backend/config.py): BaseSettings configuration and environmental path management.
  * [`backend/database.py`](file:///c:/Users/hp/Desktop/Vision%20Guard/backend/database.py): SQLite declarative models (`Inspection`, `DefectRecord`, `AlertRecord`) and auto-migration logic.
  * [`backend/models.py`](file:///c:/Users/hp/Desktop/Vision%20Guard/backend/models.py): Pydantic request/response validation schemas.
* **All Registered Endpoints**:
  * **System Endpoints**:
    * `GET /`: API welcome and documentation link
    * `GET /api/health`: Health status, model readiness, and compute device
  * **Detection Endpoints** ([`backend/routers/detection.py`](file:///c:/Users/hp/Desktop/Vision%20Guard/backend/routers/detection.py)):
    * `POST /api/detect/upload`: Upload image, run two-tier detection, persist, route alerts
    * `GET /api/detect/samples`: List available sample images in `data/samples/`
    * `POST /api/detect/sample/{sample_name}`: Run full inspection on a selected catalog sample
    * `GET /api/detect/history`: Retrieve paginated inspection history
    * `GET /api/detect/{inspection_id}`: Retrieve detailed inspection record with metrology
    * `POST /api/detect/{inspection_id}/audit`: Trigger 8D root-cause engineering audit
    * `GET /api/detect/{inspection_id}/8d-report`: Retrieve or generate 8D quality discrepancy report
    * `POST /api/detect/{inspection_id}/vlm-grounding`: Trigger zero-shot visual grounding on inspection image
  * **Alerts Endpoints** ([`backend/routers/alerts.py`](file:///c:/Users/hp/Desktop/Vision%20Guard/backend/routers/alerts.py)):
    * `GET /api/alerts`: List unacknowledged/recent critical and major alerts
    * `POST /api/alerts/{alert_id}/acknowledge`: Mark an alert as acknowledged with operator stamp
  * **Analytics Endpoints** ([`backend/routers/analytics.py`](file:///c:/Users/hp/Desktop/Vision%20Guard/backend/routers/analytics.py)):
    * `GET /api/analytics/overview`: Defect counts, defect rate %, severity breakdown, average latency
    * `GET /api/analytics/trends`: Historical inspection quality trend series for SPC charts
  * **WebSocket Endpoints** ([`backend/routers/websocket.py`](file:///c:/Users/hp/Desktop/Vision%20Guard/backend/routers/websocket.py)):
    * `WS /ws/alerts`: Real-time push broadcast for manufacturing line-halt alerts
    * `WS /ws/stream`: Live video streaming endpoint ingesting frames and returning annotated HUD stream

---

## 7. Frontend
* **Framework**: React 18, TypeScript, Vite, Tailwind CSS, Lucide React icons, Recharts data visualization.
* **Active Pages & Components**:
  1. **Dashboard** ([`frontend/src/pages/DashboardPage.tsx`](file:///c:/Users/hp/Desktop/Vision%20Guard/frontend/src/pages/DashboardPage.tsx)): Executive overview, production status cards, real-time alert feed, defect distribution chart.
  2. **Image Inspector** ([`frontend/src/pages/InspectorPage.tsx`](file:///c:/Users/hp/Desktop/Vision%20Guard/frontend/src/pages/InspectorPage.tsx)): Drag-and-drop file upload, demo sample picker, inspection viewport with overlay toggle, and defect diagnostic table.
  3. **Defect Viewer & 8D Audit** ([`frontend/src/components/DefectViewer.tsx`](file:///c:/Users/hp/Desktop/Vision%20Guard/frontend/src/components/DefectViewer.tsx)): Interactive defect inspector with Tier 1 Edge vs Tier 2 VLM verification badges, polygon overlay toggle, sub-millimeter metrology details, and one-click 8D Root-Cause Discrepancy report generator.
  4. **Live Stream** ([`frontend/src/pages/LiveStreamPage.tsx`](file:///c:/Users/hp/Desktop/Vision%20Guard/frontend/src/pages/LiveStreamPage.tsx)): Real-time webcam / RTSP camera stream inspection with dynamic FPS counter and HUD.
  5. **Inspection Audit** ([`frontend/src/pages/HistoryPage.tsx`](file:///c:/Users/hp/Desktop/Vision%20Guard/frontend/src/pages/HistoryPage.tsx)): Filterable, searchable audit log of all historical inspections with severity tags.
  6. **Quality Analytics** ([`frontend/src/pages/AnalyticsPage.tsx`](file:///c:/Users/hp/Desktop/Vision%20Guard/frontend/src/pages/AnalyticsPage.tsx)): Statistical Process Control (SPC) analytics, defect rate percentages, and timeline trends.
* **Theme Colors & Fonts** ([`frontend/tailwind.config.js`](file:///c:/Users/hp/Desktop/Vision%20Guard/frontend/tailwind.config.js)):
  * **Industrial Palette**:
    * `industrial-950`: `#090d16` (Deep canvas background)
    * `industrial-900`: `#0f172a` (Card and panel background)
    * `industrial-800`: `#1e293b` (Sub-panel and border)
    * `industrial-700`: `#334155` (Borders and dividers)
    * `industrial-600`: `#475569` (Muted text / icons)
    * `industrial-500`: `#64748b` (Secondary labels)
  * **Brand Accents**:
    * `brand-blue`: `#0284c7` (Primary interactive)
    * `brand-cyan`: `#06b6d4` (Active status & metrics)
    * `brand-emerald`: `#10b981` (Passed / Nominal quality)
    * `brand-amber`: `#f59e0b` (Major flaw / Warning)
    * `brand-rose`: `#f43f5e` (Critical alert / Line Halt)
  * **Typography**:
    * Monospace: `JetBrains Mono`, `Fira Code`, `monospace` (telemetry, bounding box coordinates, metrology)
    * Sans-serif: `Inter`, system UI font stack (body prose and titles)

---

## 8. Known Problems & Technical Debt
1. **Domain Gap Between Training & Real-World Photography**:
   * The underlying supervised YOLO model was fine-tuned on flat steel (NEU-DET) and synthetic planar textures.
   * On real-world 3D camera images (e.g. electrical plugs, molded plastics) with shadows, perspective, and specular highlights, raw YOLO feature map confidence collapses below $0.15$.
2. **Missing Inverse Letterbox Coordinate Mapping**:
   * In single-pass non-tiled inference, manual letterbox padding `(pad_left, pad_top)` is added in `ml/detector.py`, but `postprocessor.py` reads `box.xyxy` in letterbox space without subtracting padding or dividing by scale. This displaces bounding boxes on non-square images.
3. **Large False Bounding Boxes from Texture Gradient Noise**:
   * Classical morphological filters in the anomaly engine can bridge neighboring surface irregularities, generating large bounding rectangles that encompass both the flaw and adjacent nominal surface.
4. **`[VLM]` Labels Generated by Local Heuristic Fallback**:
   * When `GEMINI_API_KEY` is not present, `gemini_grounding.py` routes to an offline OpenCV Blackhat/HSV filter. Bounding boxes are tagged `[VLM]` in the UI even though they were generated by classical image processing rather than a true vision-language model.
5. **Webcam Streaming Latency**:
   * Browser-to-server WebSocket streaming re-encodes full JPEG frames on CPU, resulting in a latency of $80\text{ms} - 180\text{ms}$ per frame when tiling is active.
6. **Defect Explanations Missing in Base Model**:
   * Standard YOLO outputs only class IDs and probabilities without explaining the physical root cause; rich explanations currently rely entirely on Tier 2 VLM synthesis.

---

## 9. Operating Rules
1. **Real Inference Only**: Never fabricate, hardcode, or simulate defect detections, coordinates, or confidence scores. All outputs must originate from real model forward passes or deterministic mathematical algorithms.
2. **Deterministic Fallbacks**: If external API services (such as Gemini) are unreachable or unconfigured, the system must degrade gracefully to local offline processing without raising unhandled exceptions or halting inspection pipelines.
3. **Cross-Platform Compatibility**: Every script, runner, and path resolution must remain fully compatible with both Windows (PowerShell/CMD, backslashes) and Linux/macOS (Bash, forward slashes).
4. **Pinned Dependency Hygiene**: Keep core libraries pinned and compatible across Python 3.10 through 3.13 (specifically handling NumPy 2.x `np.trapz` removal and PyTorch 2.6+ `weights_only` checkpoint unpickling).

