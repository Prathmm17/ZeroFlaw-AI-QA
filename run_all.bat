@echo off
setlocal enabledelayedexpansion

echo =====================================================================
echo           VISIONGUARD QA -- REAL-TIME INDUSTRIAL AI SYSTEM
echo =====================================================================
echo.

:: 1. Verify Python
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    pause
    exit /b 1
)

:: 2. Verify Models
if not exist "models\best.pt" (
    if exist "models\industrial_v2_best.pt" (
        copy /Y "models\industrial_v2_best.pt" "models\best.pt" >nul
    ) else if exist "models\industrial_best.pt" (
        copy /Y "models\industrial_best.pt" "models\best.pt" >nul
    ) else (
        echo [INFO] Downloading base model weights...
        python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
        if not exist "models" mkdir "models"
        copy /Y "yolov8n.pt" "models\best.pt" >nul
    )
)

echo [OK] Verified model weights.

:: 3. Start Backend Server
echo [1/2] Starting FastAPI Industrial Backend on http://127.0.0.1:8000 ...
start "VisionGuard Backend" cmd /k "python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"

:: 4. Start Frontend Server
echo [2/2] Starting React Vite Frontend on http://127.0.0.1:5173 ...
start "VisionGuard Frontend" cmd /k "npm.cmd --prefix frontend run dev -- --host 127.0.0.1 --port 5173"

:: Wait 3 seconds then open browser
timeout /t 3 >nul
echo.
echo Opening VisionGuard Dashboard in browser...
start http://127.0.0.1:5173

echo.
echo =====================================================================
echo VisionGuard QA is running!
echo   * Web Dashboard : http://127.0.0.1:5173
echo   * Backend API   : http://127.0.0.1:8000/docs
echo =====================================================================
echo Keep the backend and frontend terminal windows open.
pause

