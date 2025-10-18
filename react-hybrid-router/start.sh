#!/bin/bash

# Hybrid AI Router - React App Quick Start Script
# This script sets up and runs the React.js + TypeScript frontend

echo "🚀 Hybrid AI Router - React Setup & Start"
echo "=========================================="

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js from https://nodejs.org/"
    exit 1
fi

echo "✅ Node.js version: $(node --version)"

# Navigate to React app directory
cd "$(dirname "$0")"
echo "📁 Current directory: $(pwd)"

# Check if package.json exists
if [ ! -f "package.json" ]; then
    echo "❌ package.json not found. Make sure you're in the React app directory."
    exit 1
fi

echo "📦 Installing dependencies..."
npm install

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies. Please check your internet connection and try again."
    exit 1
fi

echo "✅ Dependencies installed successfully!"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️  Creating environment configuration..."
    cat > .env << EOL
# React App Configuration
REACT_APP_API_BASE_URL=http://localhost:8000
REACT_APP_ENABLE_MOCK_API=true
GENERATE_SOURCEMAP=false
EOL
    echo "✅ Environment file created (.env)"
fi

echo ""
echo "🔧 Setup Complete! Starting development server..."
echo ""
echo "📍 The app will open at: http://localhost:3000"
echo "🔄 The page will reload when you make edits"
echo "🛑 Press Ctrl+C to stop the development server"
echo ""

# Start the development server
npm start