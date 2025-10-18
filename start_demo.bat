@echo off
REM Startup script for Hybrid LLM Router Demo
echo =====================================
echo   Hybrid LLM Router Demo Launcher
echo =====================================
echo.

REM Check if virtual environment is activated
if "%VIRTUAL_ENV%"=="" (
    echo Warning: No virtual environment detected!
    echo Please activate your virtual environment first:
    echo   .venv\Scripts\activate
    echo.
    pause
    exit /b 1
)

echo 🔧 Installing/updating Python dependencies...
pip install -r requirements.txt

echo.
echo   Installing/updating React dependencies...
cd react-hybrid-router
npm install
cd ..

echo.
echo �🚀 Starting FastAPI backend server...
start "Hybrid Router API" cmd /k "python api_server.py"

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