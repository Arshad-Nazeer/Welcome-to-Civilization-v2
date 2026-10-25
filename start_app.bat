@echo off
title Welcome to Civilization Launcher
echo ==========================================================
echo   Welcome to Civilization v2 — Starting Full-Stack App
echo ==========================================================

set ROOT_DIR=%~dp0
set PYTHON_EXE=%ROOT_DIR%.venv\Scripts\python.exe

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python environment not found at %PYTHON_EXE%
    pause
    exit /b 1
)

echo [1/2] Starting FastAPI Backend on port 8000...
start "WTC - FastAPI Backend (Port 8000)" powershell -NoExit -Command "cd '%ROOT_DIR%'; & '%PYTHON_EXE%' -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload"

echo [2/2] Starting Vite React Frontend on port 5173...
start "WTC - Vite Frontend (Port 5173)" powershell -NoExit -Command "cd '%ROOT_DIR%frontend'; npm run dev"

echo.
echo Both servers have been launched in dedicated console windows!
echo   - Backend:  http://localhost:8000/docs
echo   - Frontend: http://localhost:5173
echo.

timeout /t 3 /nobreak >nul
start http://localhost:5173
