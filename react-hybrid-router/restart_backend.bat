@echo off
echo 🔄 Restarting React Hybrid Router Backend
echo.

REM Find and kill process on port 8000
echo 🔍 Finding process on port 8000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000.*LISTENING"') do (
    echo 🛑 Stopping process %%a
    taskkill /F /PID %%a >nul 2>&1
)

echo.
echo ✅ Port 8000 cleared
echo.

REM Wait a moment
timeout /t 2 /nobreak >nul

REM Activate virtual environment and start backend
echo 🚀 Starting backend server...
cd ..
call .venv\Scripts\activate.bat
cd react-hybrid-router

REM Clear Python cache
echo 🧹 Clearing Python cache...
if exist "__pycache__" rd /s /q "__pycache__"
if exist "..\modules\__pycache__" (
    del /f /q "..\modules\__pycache__\*hybrid_router*.pyc" 2>nul
)

echo.
echo 🌟 Starting FastAPI backend on port 8000...
uvicorn backend_api:app --reload --host 0.0.0.0 --port 8000
