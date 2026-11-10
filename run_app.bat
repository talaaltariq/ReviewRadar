@echo off
title ReviewRadar - Starting Application
echo ========================================================
echo               ReviewRadar Launcher
echo ========================================================
echo.

cd /d "%~dp0"

echo [1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "ReviewRadar - Backend (FastAPI)" cmd /k "cd /d "%~dp0backend" && uvicorn main:app --reload --port 8000"

echo [2/2] Starting React + Vite Frontend on http://localhost:5173 ...
start "ReviewRadar - Frontend (Vite)" cmd /k "cd /d "%~dp0frontend" && npm run dev"

echo.
echo ========================================================
echo Both servers have been launched in separate windows!
echo - Backend:  http://127.0.0.1:8000
echo - Frontend: http://localhost:5173
echo.
echo To stop the app, close both opened terminal windows.
echo ========================================================
timeout /t 3 >nul
start http://localhost:5173

