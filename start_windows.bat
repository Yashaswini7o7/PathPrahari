@echo off
cd /d "%~dp0prototype"
echo Installing the Python packages PathPrahari needs (only slow the first time)...
python -m pip install -r requirements.txt
echo.
python run_demo.py
echo.
pause
