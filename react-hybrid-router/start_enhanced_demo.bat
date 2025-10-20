@echo off
echo 🚀 Starting Enhanced Hybrid AI Router Demo
echo.
echo This will start:
echo   - FastAPI Backend (http://localhost:8000)
echo   - React Frontend (http://localhost:3000)
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if Node.js is available
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js is not installed or not in PATH
    pause
    exit /b 1
)

echo ✅ Prerequisites checked
echo.

REM Install Python dependencies if needed
cd ..
if not exist ".venv\" (
    echo 📦 Creating virtual environment...
    python -m venv .venv
)

echo 🔧 Activating virtual environment...
call .venv\Scripts\activate.bat

echo 📦 Installing Python dependencies...
pip install -r requirements.txt

REM Install React dependencies if needed
cd react-hybrid-router
if not exist "node_modules\" (
    echo 📦 Installing React dependencies...
    npm install
)

echo.
echo 🌟 Starting servers...
echo.

REM Start FastAPI backend in background
cd ..
start "FastAPI Backend" cmd /k ".venv\Scripts\activate.bat && python backend_api.py"

REM Wait a moment for backend to start
timeout /t 3 /nobreak >nul

REM Start React frontend
cd react-hybrid-router
start "React Frontend" cmd /k "npm start"

echo.
echo 🎉 Demo started successfully!
echo.
echo 🌐 React Frontend: http://localhost:3000
echo 🔧 FastAPI Backend: http://localhost:8000
echo 📖 API Documentation: http://localhost:8000/docs
echo.
echo Press any key to exit...
pause >nul