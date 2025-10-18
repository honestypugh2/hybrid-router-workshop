@echo off
REM Hybrid AI Router - React Setup Script for Windows

echo 🚀 Setting up Hybrid AI Router React Application...

REM Check if Node.js is installed
where node >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Node.js is not installed. Please install Node.js 16.0 or higher.
    echo    Download from: https://nodejs.org/
    pause
    exit /b 1
)

echo ✅ Node.js found: 
node -v

REM Navigate to React app directory
cd react-hybrid-router

REM Install dependencies
echo 📦 Installing dependencies...
call npm install

REM Check if installation was successful
if %errorlevel% equ 0 (
    echo ✅ Dependencies installed successfully!
    echo.
    echo 🎯 Quick Start:
    echo    npm start          - Start development server
    echo    npm run build      - Build for production  
    echo    npm test           - Run tests
    echo.
    echo 📝 The app will be available at: http://localhost:3000
    echo.
    echo 🔧 Configuration:
    echo    - Update backend URL in src/services/api.ts
    echo    - Create .env file for environment variables
    echo.
    pause
) else (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)