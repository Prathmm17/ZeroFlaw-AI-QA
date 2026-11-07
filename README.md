# ZeroFlaw AI-QA: Real-Time Edge Industrial Defect Detection & Automated Quality Assurance System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-ONNX%20Edge%20Runtime-00FFFF?style=for-the-badge)](https://ultralytics.com)
[![React 18](https://img.shields.io/badge/React-18.3+-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.5+-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![Vite](https://img.shields.io/badge/Vite-5.4+-646CFF?style=for-the-badge&logo=vite&logoColor=white)](https://vitejs.dev)

> **ZeroFlaw AI-QA** is an edge-first industrial computer vision inspection platform engineered for automated manufacturing quality assurance. Combining ultra-fast ONNX-accelerated YOLOv8 inference, sub-pixel dimensional metrology, CLAHE adaptive ambient illumination boosting, an automated 5-point Conformance Matrix, and real-time conveyor video streaming over WebSockets.

---

## 🌟 Key Highlights & Engineering Features

### 1. Dual-Tier Inspection Architecture
- **Tier 1 (Sub-150ms Edge Fast Gate)**: Runs INT8/FP32 quantized YOLOv8 ONNX runtime directly on local industrial edge hardware (CPU/GPU) with zero internet dependency.
- **Tier 2 (Deep Metrology & VLM Arbitration)**: Escalates ambiguous, low-confidence, or novel anomalies through sub-pixel contour analysis and visual language arbitration for zero false escapes.

### 2. Five Critical Conformance Matrix (Automated Release / Reject)
- **CRIT-01: Zero Structural Flaws**: Catastrophic structural cracks, deep crazing, and ruptures trigger an immediate conveyor halt.
- **CRIT-02: Major Surface Flaw Containment**: Zero repairable inclusions or roll-in scale exceeding boundaries.
- **CRIT-03: Defect Area Density (< 2.0% Threshold)**: Cumulative flaw area percentage calculation across workpiece surface.
- **CRIT-04: Sub-Pixel Dimensional Metrology**: Interactive pixel-to-millimeter reference calibration scale measuring defect dimensions with high geometric precision.
- **CRIT-05: Material Matrix Structural Saliency**: Detection of morphological anomalies and thermal disruptions.

### 3. Real-Time Conveyor Video Inspection & Live Streaming
- Full-duplex WebSocket streaming (`ws://localhost:8000/api/detect/stream`) processing 15+ FPS.
- Physical webcam input support or built-in simulated industrial conveyor simulation.
- Aspect-ratio preservation with intelligent letterbox/pillarbox coordinate re-projection.
- Frame-by-frame industrial video scanner with interactive timeline scrubber.

### 4. Enterprise Audit Trail & Report Generation
- SQLite persistent inspection telemetry database.
- ISO 9001:2015 Quality Certificate browser preview.
- Automotive 8D Corrective Action PDF report export via ReportLab.

### 5. Electric Violet Aurora Glassmorphism UI
- Inspired by futuristic high-contrast dark industrial consoles.
- Obsidian midnight base with electric violet aurora light rays, translucent frosted glass cards, and high-legibility telemetry displays.

---

## 🏗️ System Architecture

```
                                [ Industrial Camera / Video Feed ]
                                                │
                                                ▼
                                [ CLAHE Ambient Boost Preprocessor ]
                                                │
                        ┌───────────────────────┴───────────────────────┐
                        ▼                                               ▼
          [ Tier 1: YOLOv8 ONNX Edge ]                     [ Tier 2: VLM / Sub-Pixel ]
         (Low Latency <100ms Inference)                   (Deep Metrology & Arbitration)
                        │                                               │
                        └───────────────────────┬───────────────────────┘
                                                ▼
                                [ 5-Point Conformance Matrix ]
                                (Pass / Flagged / Line Halt Reject)
                                                │
                        ┌───────────────────────┴───────────────────────┐
                        ▼                                               ▼
         [ WebSocket Live Dashboard ]                    [ 8D Audit PDF & SQLite DB ]
```

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm

### 1. Clone the Repository
```bash
git clone https://github.com/Prathmm17/ZeroFlaw-AI-QA.git
cd ZeroFlaw-AI-QA
```

### 2. Backend Setup
```bash
# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start backend FastAPI server
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
API documentation available at: `http://127.0.0.1:8000/docs`

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
Open your browser at: `http://127.0.0.1:5173`

---

## 🧪 Automated Testing & Verification
```bash
# Run backend test suite
pytest tests/ -v
```

---

## 📂 Project Structure
```
ZeroFlaw-AI-QA/
├── backend/
│   ├── main.py                     # FastAPI application & WebSocket endpoints
│   ├── models.py                   # Pydantic schemas & telemetry models
│   ├── database.py                 # SQLite database engine & session
│   ├── routers/                    # REST API route handlers
│   └── services/                   # Conformance, 8D report & arbitration services
├── frontend/
│   ├── src/
│   │   ├── components/             # Glassmorphic HUD, DefectViewer, StatsCard
│   │   ├── pages/                  # Dashboard, Inspector, Stream, Audit Trail
│   │   └── api/                    # Axios API client & WebSocket adapters
│   ├── package.json
│   └── vite.config.ts
├── models/
│   ├── best.onnx                   # Production quantized YOLOv8 ONNX weights
│   └── best.pt                     # PyTorch model checkpoint
├── data/
│   └── samples/                    # Demo synthetic factory specimens
├── tests/                          # Automated pytest test suites
└── README.md
```

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

