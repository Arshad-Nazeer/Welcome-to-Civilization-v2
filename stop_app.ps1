# stop_app.ps1 — Gracefully stops running uvicorn and vite processes
# Usage: .\stop_app.ps1

Write-Host "Stopping Welcome to Civilization services..." -ForegroundColor Yellow

# Stop uvicorn python processes
Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*uvicorn*" } | Stop-Process -Force -ErrorAction SilentlyContinue
Get-Process node -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*vite*" } | Stop-Process -Force -ErrorAction SilentlyContinue

Write-Host "All backend and frontend dev server processes stopped." -ForegroundColor Green
