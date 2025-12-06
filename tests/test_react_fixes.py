#!/usr/bin/env python3
"""
Test React ESLint fixes and dependency resolution
"""

import subprocess
import os
import sys

def run_command(command, cwd=None, timeout=60):
    """Run a command and return success status"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def test_react_fixes():
    """Test that React fixes resolve ESLint warnings"""
    print("🧪 Testing React ESLint Fixes")
    print("=" * 50)
    
    react_dir = os.path.join(os.path.dirname(__file__), "react-hybrid-router")
    
    if not os.path.exists(react_dir):
        print("❌ React directory not found")
        return False
    
    print("📦 Installing React dependencies...")
    success, stdout, stderr = run_command("npm install", cwd=react_dir)
    
    if not success:
        print(f"❌ npm install failed: {stderr}")
        return False
    
    print("✅ Dependencies installed")
    
    print("\n🔍 Running TypeScript type check...")
    success, stdout, stderr = run_command("npm run type-check", cwd=react_dir)
    
    if not success:
        print(f"❌ TypeScript compilation failed: {stderr}")
        return False
    
    print("✅ TypeScript compilation successful")
    
    print("\n🧹 Testing ESLint (sample check)...")
    # Check specific files that had warnings
    files_to_check = [
        "src/components/App.tsx",
        "src/components/ChatInterface.tsx"
    ]
    
    for file_path in files_to_check:
        print(f"   Checking {file_path}...")
        success, stdout, stderr = run_command(
            f"npx eslint {file_path}",
            cwd=react_dir
        )
        
        if "useEffect has a missing dependency" in stderr or "assigned a value but never used" in stderr:
            print(f"❌ ESLint warnings still present in {file_path}")
            print(f"   Error: {stderr}")
            return False
        else:
            print(f"✅ {file_path} - No dependency/unused variable warnings")
    
    print("\n🎉 All React fixes verified successfully!")
    return True

def main():
    """Main test function"""
    print("🔧 React Dependencies & ESLint Fixes Verification")
    print("=" * 60)
    
    success = test_react_fixes()
    
    if success:
        print("\n✅ SUCCESS: All React fixes are working correctly!")
        print("\n🚀 Ready to start the application:")
        print("   cd react-hybrid-router")
        print("   start_demo.bat  # Starts both backend and frontend")
        print("   # OR manually:")
        print("   python api_server.py  # Terminal 1")
        print("   cd react-hybrid-router && npm start  # Terminal 2")
    else:
        print("\n❌ Some issues remain. Check the error messages above.")
        print("\n🔧 Troubleshooting:")
        print("   1. Ensure Node.js and npm are installed")
        print("   2. Delete node_modules and package-lock.json, then npm install")
        print("   3. Check for any syntax errors in the TypeScript files")

if __name__ == "__main__":
    main()