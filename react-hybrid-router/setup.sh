#!/bin/bash

# Hybrid AI Router - React Setup Script
echo "🚀 Setting up Hybrid AI Router React Application..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16.0 or higher."
    echo "   Download from: https://nodejs.org/"
    exit 1
fi

# Check Node.js version
NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 16 ]; then
    echo "❌ Node.js version 16.0 or higher is required. Current version: $(node -v)"
    exit 1
fi

echo "✅ Node.js version: $(node -v)"

# Navigate to React app directory
cd react-hybrid-router

# Install dependencies
echo "📦 Installing dependencies..."
npm install

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully!"
    echo ""
    echo "🎯 Quick Start:"
    echo "   npm start          - Start development server"
    echo "   npm run build      - Build for production"
    echo "   npm test           - Run tests"
    echo ""
    echo "📝 The app will be available at: http://localhost:3000"
    echo ""
    echo "🔧 Configuration:"
    echo "   - Update backend URL in src/services/api.ts"
    echo "   - Create .env file for environment variables"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi