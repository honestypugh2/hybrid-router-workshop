@echo off
echo ================================
echo  Hybrid LLM Router Demo Scripts
echo ================================
echo.
echo All demo scripts are located in the react-hybrid-router directory.
echo.
echo Available demos:
echo.
echo 1. Enhanced Demo (Recommended) - Dual backend support
echo    cd react-hybrid-router
echo    start_enhanced_demo.bat
echo.
echo 2. Basic Demo - Original backend compatibility
echo    cd react-hybrid-router  
echo    start_demo.bat
echo.
echo 3. React Demo with Python startup
echo    cd react-hybrid-router
echo    start_react_demo.bat
echo.
echo 4. Python startup script
echo    cd react-hybrid-router
echo    python start_react_demo.py
echo.
echo Changing to react-hybrid-router directory...
echo.
cd react-hybrid-router
echo ✅ Now in react-hybrid-router directory.
echo Choose your demo script:
echo.
dir start_*.bat start_*.py 2>nul
echo.
echo 💡 Tip: You can also use npm scripts:
echo   npm run demo-enhanced    # Enhanced demo
echo   npm run demo            # Basic demo
echo   npm run demo-react      # React demo
echo   npm run demo-python     # Python startup
echo.
pause