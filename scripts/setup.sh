#!/usr/bin/env bash
# VisionGuard QA - Automated Setup for Linux and macOS
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "======================================================================"
echo "         VisionGuard QA - Automated Environment Setup"
echo "======================================================================"
echo ""

echo "[1/4] Installing Python Dependencies..."
pip install -r requirements.txt

echo "[2/4] Generating Synthetic Manufacturing Defect Dataset..."
python ml/generate_synthetic.py

echo "[3/4] Installing Frontend NPM Packages..."
(cd frontend && npm install)

echo "[4/4] Building Frontend Production Distribution..."
(cd frontend && npm run build)

echo ""
echo "======================================================================"
echo " Setup complete! Execute './scripts/run_all.sh' to launch."
echo "======================================================================"

