"""Quick test script to verify system functionality."""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from tensorflow.keras.models import load_model

print("="*60)
print("Testing Predictive Maintenance System")
print("="*60)
print()

# Test 1: Check directory structure
print("Test 1: Checking directory structure...")
dirs = ['data', 'models', 'results', 'logs']
for d in dirs:
    exists = Path(d).exists()
    status = "✓" if exists else "✗"
    print(f"  {status} {d}/")
print()

# Test 2: Check data files
print("Test 2: Checking data files...")
data_files = ['train_FD001.txt', 'test_FD001.txt', 'RUL_FD001.txt']
for f in data_files:
    exists = Path(f'data/{f}').exists()
    status = "✓" if exists else "✗"
    print(f"  {status} data/{f}")
print()

# Test 3: Check trained models
print("Test 3: Checking trained models...")
model_files = {
    'random_forest_model.pkl': 'Random Forest',
    'xgboost_model.json': 'XGBoost',
    'lstm_model.h5': 'LSTM'
}

for filename, name in model_files.items():
    path = Path(f'models/{filename}')
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  ✓ {name}: {size_mb:.2f} MB")
    else:
        print(f"  ✗ {name}: Not found")
print()

# Test 4: Load and test models
print("Test 4: Loading models...")
try:
    # Load Random Forest
    rf_model = joblib.load('models/random_forest_model.pkl')
    print("  ✓ Random Forest loaded successfully")
    
    # Load LSTM
    lstm_model = load_model('models/lstm_model.h5', compile=False)
    print("  ✓ LSTM loaded successfully")
    
    print()
    print("Test 5: Running inference test...")
    
    # Create dummy data
    X_dummy = np.random.randn(1, 50, 34)  # 1 sample, 50 timesteps, 34 features
    X_flat = X_dummy.reshape(1, -1)
    
    # Test Random Forest
    rf_pred = rf_model.predict(X_flat)
    rf_proba = rf_model.predict_proba(X_flat)
    print(f"  ✓ RandomForest prediction: Status={rf_pred[0]}, Confidence={rf_proba[0].max():.2f}")
    
    # Test LSTM
    lstm_pred = lstm_model.predict(X_dummy, verbose=0)
    print(f"  ✓ LSTM prediction: RUL={lstm_pred[0][0]:.2f} cycles")
    
except Exception as e:
    print(f"  ✗ Error loading models: {e}")

print()
print("="*60)
print("System Test Complete!")
print("="*60)
print()
print("To start the dashboard, run:")
print("  streamlit run app.py")
print()
print("Or use the startup script:")
print("  bash start_app.sh")
print()