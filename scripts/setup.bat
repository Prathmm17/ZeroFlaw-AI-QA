@echo off
TITLE VisionGuard QA - Automated Setup
echo ======================================================================
echo          VisionGuard QA - Automated Environment Setup
echo ======================================================================
echo.

set PROJECT_ROOT=%~dp0..
cd /d "%PROJECT_ROOT%"

echo [1/4] Installing Python Dependencies from requirements.txt ...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo Error installing Python dependencies.
    pause
    exit /b 1
)

echo.
echo [2/4] Generating Synthetic Defect Dataset & Factory Samples ...
python ml\generate_synthetic.py
if %ERRORLEVEL% NEQ 0 (
    echo Error generating synthetic defect data.
    pause
    exit /b 1
)

echo.
echo [3/4] Installing Frontend NPM Packages ...
cd /d "%PROJECT_ROOT%\frontend"
call npm.cmd install
if %ERRORLEVEL% NEQ 0 (
    echo Error installing Node dependencies.
    pause
    exit /b 1
)

echo.
echo [4/4] Building Frontend Production Distribution ...
call npm.cmd run build
if %ERRORLEVEL% NEQ 0 (
    echo Error building frontend bundle.
    pause
    exit /b 1
)

cd /d "%PROJECT_ROOT%"
echo.
echo ======================================================================
echo  VisionGuard QA Setup Completed Successfully!
echo  Run 'scripts\run_all.bat' to launch the system.
echo ======================================================================
pause

