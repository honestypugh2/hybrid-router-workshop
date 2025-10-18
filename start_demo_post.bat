@echo off
REM Startup script for Hybrid LLM Router Demo
echo =====================================
echo   Hybrid LLM Router Demo Launcher
echo =====================================
echo.


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