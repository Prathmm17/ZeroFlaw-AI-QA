#!/usr/bin/env bash
set -e

echo "====================================================================="
echo "          VISIONGUARD QA -- REAL-TIME INDUSTRIAL AI SYSTEM"
echo "====================================================================="
echo ""

# 1. Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 could not be found."
    exit 1
fi

# 2. Check Models
if [ ! -f "models/best.pt" ]; then
    if [ -f "models/industrial_v2_best.pt" ]; then
        cp -f "models/industrial_v2_best.pt" "models/best.pt"
    elif [ -f "models/industrial_best.pt" ]; then
        cp -f "models/industrial_best.pt" "models/best.pt"
    else
        echo "[INFO] Downloading base model weights..."
        python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
        mkdir -p models
        cp -f yolov8n.pt models/best.pt
    fi
fi

echo "[OK] Verified model weights."

# 3. Start Backend in background
echo "[1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ..."
python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

# 4. Start Frontend
echo "[2/2] Starting Vite Frontend on http://127.0.0.1:5173 ..."
npm --prefix frontend run dev -- --host 127.0.0.1 --port 5173 &
FRONTEND_PID=$!

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true" EXIT

echo ""
echo "====================================================================="
echo "VisionGuard QA is active!"
echo "  * Web Dashboard : http://127.0.0.1:5173"
echo "  * Backend API   : http://127.0.0.1:8000/docs"
echo "====================================================================="
wait

