# run_all.ps1 — Native Windows PowerShell runner for Welcome to Civilization pipeline
# Usage: .\run_all.ps1

$ErrorActionPreference = "Stop"

$PYTHON = ".venv\Scripts\python.exe"
$PIP = ".venv\Scripts\pip.exe"
$PYTEST = ".venv\Scripts\pytest.exe"

Write-Host "=== 1. Setup ===" -ForegroundColor Cyan
if (-not (Test-Path ".venv")) {
    python -m venv .venv
}
& $PIP install --upgrade pip
& $PIP install -r requirements.txt

Write-Host "=== 2. Download / Raw Save ===" -ForegroundColor Cyan
& $PYTHON scripts\01_download_data.py

Write-Host "=== 3. Preprocess & Splits ===" -ForegroundColor Cyan
& $PYTHON scripts\02_preprocess.py

Write-Host "=== 4. EDA ===" -ForegroundColor Cyan
& $PYTHON scripts\03_eda.py

Write-Host "=== 5. Build Reports ===" -ForegroundColor Cyan
& $PYTHON scripts\04_build_report.py

Write-Host "=== 6. Test Suite ===" -ForegroundColor Cyan
& $PYTEST tests\ -v

Write-Host "=== ALL COMPLETED SUCCESSFULLY ===" -ForegroundColor Green
