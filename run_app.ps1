# ReviewRadar Single-Script Launcher (PowerShell)
$root = $PSScriptRoot

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "               ReviewRadar Launcher" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Start Backend in new window
Write-Host "[1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\backend'; Write-Host '--- ReviewRadar Backend (FastAPI :8000) ---' -ForegroundColor Green; uvicorn main:app --reload --port 8000"

# 2. Start Frontend in new window
Write-Host "[2/2] Starting React + Vite Frontend on http://localhost:5173 ..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\frontend'; Write-Host '--- ReviewRadar Frontend (Vite :5173) ---' -ForegroundColor Cyan; npm run dev"

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Both servers launched in their own windows!" -ForegroundColor Yellow
Write-Host "Backend API:  http://127.0.0.1:8000"
Write-Host "Frontend App: http://localhost:5173"
Write-Host ""
Write-Host "Opening your browser in 3 seconds..." -ForegroundColor DarkGray
Write-Host "========================================================" -ForegroundColor Cyan

Start-Sleep -Seconds 3
Start-Process "http://localhost:5173"

