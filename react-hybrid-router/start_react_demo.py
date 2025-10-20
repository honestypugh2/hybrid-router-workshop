#!/usr/bin/env python3
"""
Startup script for the Hybrid LLM Router React Demo
Handles environment setup and graceful error handling
Run from react-hybrid-router directory
"""

import os
import sys
import subprocess
import time
import signal
from pathlib import Path
import threading
import webbrowser

def check_python_env():
    """Check if we're in the correct Python environment"""
    try:
        import dotenv
        print("✅ Python environment appears to be set up correctly")
        return True
    except ImportError:
        print("❌ Missing dependencies. Please run:")
        print("   cd ..")
        print("   pip install -r requirements.txt")
        print("   Or activate your virtual environment first")
        return False

def install_deps():
    """Install Python dependencies if needed"""
    try:
        print("📦 Installing Python dependencies...")
        parent_dir = Path(__file__).parent.parent
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=parent_dir)
        print("✅ Python dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def start_backend():
    """Start the FastAPI backend server"""
    try:
        print("🚀 Starting FastAPI backend...")
        
        # Navigate to parent directory for imports
        parent_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(parent_dir))
        os.chdir(parent_dir)
        
        # Load environment
        from dotenv import load_dotenv
        load_dotenv()
        
        # Import and run the API server
        import uvicorn
        from api_server import app
        
        # Start server in a separate thread
        def run_server():
            uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        print("✅ Backend started on http://localhost:8080")
        print("📖 API docs available at http://localhost:8080/docs")
        return True
        
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        print("💡 This might be due to Azure AI Foundry 404 errors, which are expected if Foundry isn't deployed")
        print("💡 The system should still work with Azure OpenAI fallback")
        return False

def start_frontend():
    """Start the React frontend"""
    try:
        print("🌐 Starting React frontend...")
        react_dir = Path(__file__).parent  # Current directory should be react-hybrid-router
        
        if not react_dir.exists():
            print("❌ React directory not found")
            return False
        
        # Check if node_modules exists
        if not (react_dir / "node_modules").exists():
            print("📦 Installing React dependencies...")
            subprocess.check_call(["npm", "install"], cwd=react_dir)
        
        # Start React dev server
        process = subprocess.Popen(
            ["npm", "start"], 
            cwd=react_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        
        print("✅ React frontend starting...")
        print("🌐 Frontend will be available at http://localhost:3000")
        
        # Wait a moment and open browser
        time.sleep(3)
        webbrowser.open("http://localhost:3000")
        
        return process
        
    except Exception as e:
        print(f"❌ Failed to start frontend: {e}")
        return None

def main():
    """Main startup function"""
    print("🎯 Hybrid LLM Router React Demo Startup")
    print("=" * 50)
    
    # Check if we're in the right directory
    parent_dir = Path(__file__).parent.parent
    if not (parent_dir / "api_server.py").exists():
        print("❌ Please ensure the parent directory contains api_server.py")
        print("❌ Run this script from the react-hybrid-router directory")
        return
    
    # Check Python environment
    if not check_python_env():
        if not install_deps():
            print("❌ Cannot continue without dependencies")
            return
    
    # Start backend
    if not start_backend():
        print("⚠️ Backend failed to start, but continuing with frontend...")
        print("💡 You may see Azure AI Foundry 404 warnings - this is expected")
    
    # Give backend time to start
    time.sleep(2)
    
    # Start frontend
    frontend_process = start_frontend()
    
    if frontend_process:
        print("\n🎉 Demo is starting up!")
        print("📍 Backend: http://localhost:8080")
        print("📍 Frontend: http://localhost:3000")
        print("📖 API Docs: http://localhost:8080/docs")
        print("\n💡 Note: Azure AI Foundry 404 warnings are expected if Foundry isn't deployed")
        print("💡 The system will use Azure OpenAI fallback automatically")
        print("\nPress Ctrl+C to stop both servers")
        
        try:
            # Wait for frontend process
            frontend_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            frontend_process.terminate()
            print("✅ Demo stopped")
    else:
        print("❌ Failed to start the demo")

if __name__ == "__main__":
    main()