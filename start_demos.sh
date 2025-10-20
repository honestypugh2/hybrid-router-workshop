#!/bin/bash
echo "================================"
echo " Hybrid LLM Router Demo Scripts"
echo "================================"
echo
echo "All demo scripts are located in the react-hybrid-router directory."
echo
echo "Available demos:"
echo
echo "1. Enhanced Demo (Recommended) - Dual backend support"
echo "   cd react-hybrid-router"
echo "   ./start_enhanced_demo.bat (Windows) or equivalent shell script"
echo
echo "2. Basic Demo - Original backend compatibility"
echo "   cd react-hybrid-router"
echo "   ./start_demo.bat (Windows) or equivalent shell script"
echo
echo "3. React Demo with Python startup"
echo "   cd react-hybrid-router"
echo "   python start_react_demo.py"
echo
echo "4. Using npm scripts (from react-hybrid-router directory):"
echo "   npm run demo-enhanced    # Enhanced demo"
echo "   npm run demo            # Basic demo"
echo "   npm run demo-react      # React demo"
echo "   npm run demo-python     # Python startup"
echo
echo "Changing to react-hybrid-router directory..."
echo
cd react-hybrid-router
echo "✅ Now in react-hybrid-router directory."
echo "Available demo scripts:"
echo
ls -la start_* 2>/dev/null
echo
read -p "Press Enter to continue..."