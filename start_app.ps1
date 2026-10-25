# start_app.ps1 — Convenient full-stack launcher for Welcome to Civilization v2
# Usage: .\start_app.ps1

$ErrorActionPreference = "Stop"
$rootDir = $PSScriptRoot

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Welcome to Civilization v2 — Starting Full-Stack App" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Check Python virtual environment
$pythonPath = Join-Path $rootDir ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonPath)) {
    Write-Host "Error: Python virtual environment not found at $pythonPath" -ForegroundColor Red
    exit 1
}

# 2. Check frontend directory
$frontendDir = Join-Path $rootDir "frontend"
if (-not (Test-Path $frontendDir)) {
    Write-Host "Error: Frontend directory not found at $frontendDir" -ForegroundColor Red
    exit 1
}

# 3. Launch Backend in a dedicated window
Write-Host "[1/2] Launching FastAPI Backend on http://localhost:8000..." -ForegroundColor Yellow
$backendCmd = "cd '$rootDir'; `$Host.UI.RawUI.WindowTitle = 'WTC — FastAPI Backend (Port 8000)'; & '$pythonPath' -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd

# 4. Launch Frontend in a dedicated window
Write-Host "[2/2] Launching Vite React Frontend on http://localhost:5173..." -ForegroundColor Yellow
$frontendCmd = "cd '$frontendDir'; `$Host.UI.RawUI.WindowTitle = 'WTC — Vite Frontend (Port 5173)'; npm run dev"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCmd

Write-Host ""
Write-Host "Both services have been launched in dedicated console windows!" -ForegroundColor Green
Write-Host "  • Backend API:      http://localhost:8000 (Docs: http://localhost:8000/docs)" -ForegroundColor White
Write-Host "  • Frontend Web App: http://localhost:5173" -ForegroundColor White
Write-Host ""

# Optional: Open browser after brief wait
Start-Sleep -Seconds 3
Start-Process "http://localhost:5173"
