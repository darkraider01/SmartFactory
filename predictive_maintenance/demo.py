"""Demo script showing complete usage workflow."""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

print("""
╔════════════════════════════════════════════════════════════════╗
║  AI-Powered Predictive Maintenance System - Demo              ║
║  Demonstrating Complete Workflow                              ║
╚════════════════════════════════════════════════════════════════╝
""")

# Step 1: Generate Sample Data
print("\n📊 Step 1: Generating Sample Sensor Data...")
print("-" * 60)

np.random.seed(42)
n_timesteps = 100

data = {
    'unit_id': [1] * n_timesteps,
    'cycle': range(1, n_timesteps + 1),
    'os1': np.random.uniform(-0.5, 0.5, n_timesteps),
    'os2': np.random.uniform(-0.5, 0.5, n_timesteps),
    'os3': np.ones(n_timesteps) * 100
}

# Generate 21 sensor readings with degradation pattern
for i in range(1, 22):
    base = np.random.uniform(0.4, 0.6)
    degradation = np.linspace(0, 0.3, n_timesteps)  # Simulates wear
    noise = np.random.normal(0, 0.05, n_timesteps)
    data[f's{i}'] = base + degradation + noise

df = pd.DataFrame(data)

print(f"✓ Generated {len(df)} timesteps of sensor data")
print(f"✓ Machine ID: {df['unit_id'].iloc[0]}")
print(f"✓ Cycles: {df['cycle'].min()} to {df['cycle'].max()}")
print(f"\nSample data (first 3 rows):")
print(df[['unit_id', 'cycle', 's1', 's2', 's3', 's7', 's11']].head(3).to_string(index=False))

# Step 2: Load Trained Models
print("\n\n🤖 Step 2: Loading Pre-trained Models...")
print("-" * 60)

try:
    rf_model = joblib.load('models/random_forest_model.pkl')
    print("✓ Random Forest Classifier loaded")
    
    xgb_path = Path('models/xgboost_model.json')
    if xgb_path.exists():
        print("✓ XGBoost Classifier available")
    
    lstm_model = load_model('models/lstm_model.h5', compile=False)
    print("✓ LSTM RUL Predictor loaded")
    
    print("\n✓ All models ready for inference!")
except Exception as e:
    print(f"✗ Error loading models: {e}")
    print("Please run: python train_lightweight.py")
    exit(1)

# Step 3: Preprocess Data
print("\n\n⚙️ Step 3: Preprocessing Data...")
print("-" * 60)

# Select feature columns
sensor_cols = [f's{i}' for i in range(1, 22)]
op_settings = ['os1', 'os2', 'os3']
feature_cols = sensor_cols + op_settings

# Normalize features
scaler = MinMaxScaler()
df_normalized = df.copy()
df_normalized[feature_cols] = scaler.fit_transform(df[feature_cols])

print(f"✓ Normalized {len(feature_cols)} features")
print(f"✓ Using MinMax scaling (0-1 range)")

# Create sequences for LSTM (use last 50 timesteps)
sequence_length = 50
X = df_normalized[feature_cols].values[-sequence_length:]
X_seq = X.reshape(1, sequence_length, len(feature_cols))
X_flat = X_seq.reshape(1, -1)  # For RF/XGB

print(f"✓ Created sequence: {X_seq.shape}")

# Step 4: Make Predictions
print("\n\n🔮 Step 4: Generating Predictions...")
print("-" * 60)

# Random Forest prediction
rf_pred = rf_model.predict(X_flat)[0]
rf_proba = rf_model.predict_proba(X_flat)[0]

status_labels = {0: 'Healthy', 1: 'Warning', 2: 'Critical'}
rf_status = status_labels[rf_pred]
rf_confidence = rf_proba.max() * 100

print(f"\nRandom Forest Classifier:")
print(f"  Status: {rf_status}")
print(f"  Confidence: {rf_confidence:.1f}%")
print(f"  Probabilities: Healthy={rf_proba[0]:.2f}, Warning={rf_proba[1]:.2f}, Critical={rf_proba[2]:.2f}")

# LSTM prediction
lstm_pred = lstm_model.predict(X_seq, verbose=0)[0][0]

print(f"\nLSTM RUL Regressor:")
print(f"  Predicted RUL: {lstm_pred:.1f} cycles")

if lstm_pred > 100:
    rul_status = "Healthy - Continue normal operation"
    rul_icon = "🟢"
elif lstm_pred > 50:
    rul_status = "Warning - Schedule inspection soon"
    rul_icon = "🟡"
else:
    rul_status = "Critical - Immediate maintenance required"
    rul_icon = "🔴"

print(f"  Health Status: {rul_icon} {rul_status}")

# Step 5: Generate Recommendations
print("\n\n💡 Step 5: Maintenance Recommendations...")
print("-" * 60)

if rf_pred == 0:  # Healthy
    recommendations = [
        "✅ Machine is operating normally",
        "📅 Schedule routine inspection in next maintenance window",
        "📊 Continue monitoring sensor readings",
        "🔧 No immediate action required"
    ]
elif rf_pred == 1:  # Warning
    recommendations = [
        "⚠️ Machine showing early signs of degradation",
        "🔍 Perform detailed inspection within 1-2 weeks",
        "📦 Order replacement parts as precaution",
        "📈 Increase monitoring frequency to daily",
        "📝 Document any unusual vibrations or noises"
    ]
else:  # Critical
    recommendations = [
        "🚨 URGENT: Machine failure imminent",
        "⏸️ Consider stopping operation for maintenance",
        "🔧 Schedule immediate maintenance intervention",
        "📦 Ensure replacement parts are available",
        "👥 Alert maintenance team immediately",
        "📋 Prepare work order and safety procedures"
    ]

if lstm_pred < 30:
    recommendations.append(f"⏱️ Estimated {int(lstm_pred)} cycles remaining - plan replacement soon")
elif lstm_pred < 75:
    recommendations.append(f"📆 Estimated {int(lstm_pred)} cycles remaining - plan maintenance")

for rec in recommendations:
    print(f"  {rec}")

# Step 6: Summary Report
print("\n\n📋 Step 6: Summary Report")
print("=" * 60)

print(f"""
Machine ID: {df['unit_id'].iloc[0]}
Total Cycles Analyzed: {len(df)}
Last Cycle Number: {df['cycle'].iloc[-1]}

PREDICTION SUMMARY:
------------------
Health Status: {rul_icon} {rf_status}
Confidence: {rf_confidence:.1f}%
Remaining Useful Life: {lstm_pred:.0f} cycles

RECOMMENDED ACTION:
-------------------
""")

if rf_pred == 0:
    print("Continue normal operation with routine monitoring.")
elif rf_pred == 1:
    print("Schedule inspection within 1-2 weeks. Monitor closely.")
else:
    print("IMMEDIATE ACTION REQUIRED. Stop and inspect machine.")

print("\n" + "=" * 60)

# Step 7: Save Results
print("\n\n💾 Step 7: Saving Results...")
print("-" * 60)

results = {
    'unit_id': df['unit_id'].iloc[0],
    'timestamp': pd.Timestamp.now().isoformat(),
    'cycles_analyzed': len(df),
    'rf_status': rf_status,
    'rf_confidence': float(rf_confidence),
    'lstm_rul': float(lstm_pred),
    'recommendations': recommendations
}

results_df = pd.DataFrame([results])
results_file = 'results/demo_prediction_results.csv'
results_df.to_csv(results_file, index=False)

print(f"✓ Results saved to: {results_file}")
print(f"✓ You can review the prediction anytime")

# Final Message
print("""

╔════════════════════════════════════════════════════════════════╗
║  Demo Complete! ✨                                            ║
║                                                                ║
║  Next Steps:                                                   ║
║  1. Launch the dashboard: streamlit run app.py                ║
║  2. Upload your own sensor data                                ║
║  3. Get real-time predictions and recommendations              ║
║                                                                ║
║  For help: See USAGE_GUIDE.md                                  ║
╚════════════════════════════════════════════════════════════════╝
""")

print("\n🚀 Ready to start? Run: streamlit run app.py\n")