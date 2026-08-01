@echo off
title GateGuard AI - Startup Script

:: 1. Navigate to your project directory (optional if bat is already in the folder)
cd /d "%~dp0"

:: 2. Start the FastAPI server in a new minimized window
echo Starting Backend Server...
start /min cmd /k "python -m uvicorn main:app --reload"

:: 3. Wait for 5 seconds to ensure the server is live before opening the browser
echo Waiting for server to initialize...
timeout /t 5 /nobreak >nul

:: 4. Open the frontend index.html in the default browser
echo Opening Frontend Dashboard...
start "" "login.html"

echo System is running!
pause