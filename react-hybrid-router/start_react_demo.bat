@echo off
echo ========================================
echo  Hybrid LLM Router React Demo Startup
echo ========================================
echo.

REM Check if we're in the right directory by looking for parent directory structure
if not exist "..\\api_server.py" (
    echo ERROR: Please run this script from the react-hybrid-router directory
    echo The parent directory should contain api_server.py
    pause
    exit /b 1
)

echo Starting Python backend...
echo.

REM Start Python backend in a new window from parent directory
cd ..
start "Backend Server" cmd /k "python start_react_demo.py"
cd react-hybrid-router

echo.
echo Backend is starting in a separate window...
echo Frontend will open automatically in your browser
echo.
echo Note: You may see Azure AI Foundry 404 warnings - this is expected
echo The system will use Azure OpenAI fallback automatically
echo.
echo Press any key to continue or Ctrl+C to cancel...
pause > nul