@echo off
REM Startup script for Hybrid LLM Router Demo
echo =====================================
echo   Hybrid LLM Router Demo Launcher
echo =====================================
echo.

REM Check if virtual environment is activated in parent directory
if not exist "..\\.venv\\" (
    echo Warning: No virtual environment detected in parent directory!
    echo Please create and activate your virtual environment first:
    echo   cd ..
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo.
    pause
    exit /b 1
)

echo 🔧 Installing/updating Python dependencies...
cd ..
call .venv\Scripts\activate.bat
pip install -r requirements.txt

echo.
echo Installing/updating React dependencies...
cd react-hybrid-router
npm install

echo.
echo 🚀 Starting FastAPI backend server...
cd ..
start "Hybrid Router API" cmd /k ".venv\Scripts\activate.bat && python api_server.py"

echo ⏳ Waiting for backend to initialize...
timeout /t 5 /nobreak >nul

echo.
echo 🌐 Starting React frontend...
cd react-hybrid-router
start "React Frontend" cmd /k "npm start"

echo.
echo ✅ Both servers are starting!
echo.
echo 📍 Access points:
echo   • React Frontend: http://localhost:3000
echo   • FastAPI Backend: http://localhost:8080
echo   • API Documentation: http://localhost:8080/docs
echo.
echo 🔍 Debug mode is enabled. Check console logs for detailed information.
echo.
echo Press any key to return to command prompt...
pause >nul