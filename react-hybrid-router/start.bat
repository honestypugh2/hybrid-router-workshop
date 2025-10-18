@echo off
REM Hybrid AI Router - React App Quick Start Script (Windows)
REM This script sets up and runs the React.js + TypeScript frontend

echo 🚀 Hybrid AI Router - React Setup & Start
echo ==========================================

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js is not installed. Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

echo ✅ Node.js version: 
node --version

REM Navigate to React app directory
cd /d "%~dp0"
echo 📁 Current directory: %CD%

REM Check if package.json exists
if not exist "package.json" (
    echo ❌ package.json not found. Make sure you're in the React app directory.
    pause
    exit /b 1
)

echo 📦 Installing dependencies...
call npm install

if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies. Please check your internet connection and try again.
    pause
    exit /b 1
)

echo ✅ Dependencies installed successfully!

REM Create .env file if it doesn't exist
if not exist ".env" (
    echo ⚙️  Creating environment configuration...
    (
        echo # React App Configuration
        echo REACT_APP_API_BASE_URL=http://localhost:8000
        echo REACT_APP_ENABLE_MOCK_API=true
        echo GENERATE_SOURCEMAP=false
    ) > .env
    echo ✅ Environment file created (.env)
)

echo.
echo 🔧 Setup Complete! Starting development server...
echo.
echo 📍 The app will open at: http://localhost:3000
echo 🔄 The page will reload when you make edits
echo 🛑 Press Ctrl+C to stop the development server
echo.

REM Start the development server
call npm start

pause