#!/bin/sh
# Runs the PathPrahari prototype and opens the dashboard when it finishes.
cd "$(dirname "$0")/prototype" || exit 1
echo "Installing the Python packages PathPrahari needs (only slow the first time)..."
python3 -m pip install -r requirements.txt
echo
python3 run_demo.py
