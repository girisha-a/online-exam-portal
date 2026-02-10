#!/bin/bash
# Exit on error
set -e

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Generating ML models..."
python ml/behavior_model.py
python ml/performance_model.py

echo "Build completed successfully!"
