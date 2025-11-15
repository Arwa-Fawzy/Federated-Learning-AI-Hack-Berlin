@echo off
echo ================================
echo SenorMatics Dashboard Launcher
echo ================================
echo.
echo Starting Streamlit dashboard...
echo.

cd /d "%~dp0"
streamlit run app.py

pause

