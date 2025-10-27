#!/bin/bash
# Startup script for Predictive Maintenance System

echo "======================================"
echo "AI Predictive Maintenance System"
echo "======================================"
echo ""

# Check if models exist
if [ ! -d "models" ] || [ -z "$(ls -A models 2>/dev/null)" ]; then
    echo "No trained models found!"
    echo "Training models first..."
    python train_lightweight.py
    echo ""
fi

echo "Starting Streamlit Dashboard..."
echo "Access the app at: http://localhost:8502"
echo ""

streamlit run app.py --server.port 8502 --server.headless true