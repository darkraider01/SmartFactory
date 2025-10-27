"""Lightweight training script optimized for memory constraints."""

import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
import joblib
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import callbacks
import tensorflow as tf

from data_preprocessing import DataPreprocessor
from model_training import XGBoostModel, LSTMModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Limit TensorFlow memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def train_xgboost_lightweight(X_train, y_train, X_test, y_test):
    """Train XGBoost with memory optimization."""
    logger.info("Training XGBoost (lightweight)...")
    
    # Subsample for memory efficiency
    sample_size = min(8000, len(X_train))
    indices = np.random.choice(len(X_train), sample_size, replace=False)
    X_train_sub = X_train[indices]
    y_train_sub = y_train[indices]
    
    # Reshape sequences to 2D
    X_train_flat = X_train_sub.reshape(X_train_sub.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=8,
        learning_rate=0.1,
        random_state=42,
        tree_method='hist',
        eval_metric='mlogloss'
    )
    
    model.fit(X_train_flat, y_train_sub, verbose=False)
    
    # Evaluate
    predictions = model.predict(X_test_flat)
    accuracy = (predictions == y_test).mean()
    
    logger.info(f"XGBoost Accuracy: {accuracy:.4f}")
    
    # Save
    model.save_model('models/xgboost_model.json')
    logger.info("XGBoost model saved")
    
    return {'accuracy': float(accuracy)}

def train_lstm_lightweight(X_train, y_train, X_test, y_test):
    """Train LSTM with reduced complexity."""
    logger.info("Training LSTM (lightweight)...")
    
    # Subsample for memory efficiency
    sample_size = min(10000, len(X_train))
    indices = np.random.choice(len(X_train), sample_size, replace=False)
    X_train_sub = X_train[indices]
    y_train_sub = y_train[indices]
    
    # Build smaller LSTM
    model = Sequential([
        LSTM(32, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Train with fewer epochs
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        X_train_sub, y_train_sub,
        validation_split=0.2,
        epochs=20,
        batch_size=128,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Evaluate
    predictions = model.predict(X_test, verbose=0).flatten()
    mse = np.mean((y_test - predictions) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - predictions))
    
    logger.info(f"LSTM RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    # Save
    model.save('models/lstm_model.h5')
    logger.info("LSTM model saved")
    
    return {'rmse': float(rmse), 'mae': float(mae)}

def main():
    """Main lightweight training pipeline."""
    logger.info("Starting Lightweight Training Pipeline")
    
    # Load preprocessed data
    logger.info("Loading preprocessed data...")
    preprocessor = DataPreprocessor()
    data = preprocessor.prepare_data_pipeline(download=False)
    
    X_train = data['X_train_seq']
    y_train_class = data['y_train_class']
    y_train_rul = data['y_train_rul']
    X_test = data['X_test_seq']
    y_test_class = data['y_test_class']
    y_test_rul = data['y_test_rul']
    
    logger.info(f"Data loaded: {X_train.shape}, {X_test.shape}")
    
    results = {}
    
    # Train XGBoost
    try:
        xgb_metrics = train_xgboost_lightweight(X_train, y_train_class, X_test, y_test_class)
        results['xgboost'] = xgb_metrics
    except Exception as e:
        logger.error(f"XGBoost training failed: {e}")
    
    # Train LSTM
    try:
        lstm_metrics = train_lstm_lightweight(X_train, y_train_rul, X_test, y_test_rul)
        results['lstm'] = lstm_metrics
    except Exception as e:
        logger.error(f"LSTM training failed: {e}")
    
    # Save results
    results_file = f'results/training_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Training complete! Results saved to {results_file}")
    logger.info(f"Results: {results}")

if __name__ == "__main__":
    main()
