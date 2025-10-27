"""Model training module for predictive maintenance.

Implements multiple ML/DL models:
- RandomForestClassifier: Machine health status classification
- XGBoostClassifier: Enhanced fault prediction
- LSTM: Remaining Useful Life (RUL) regression
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
import logging
import yaml

# Traditional ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, mean_absolute_error
import xgboost as xgb

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class RandomForestModel:
    """Random Forest Classifier for machine health status prediction."""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize Random Forest model.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        params = config['models']['random_forest']
        self.model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            random_state=params['random_state'],
            n_jobs=params['n_jobs']
        )
        self.model_name = "RandomForest"
        logger.info(f"Initialized {self.model_name} classifier")
    
    def train(self, X_train, y_train):
        """Train the Random Forest model.
        
        Args:
            X_train: Training features (2D array)
            y_train: Training labels (1D array)
        """
        logger.info(f"Training {self.model_name} on {X_train.shape[0]} samples...")
        
        # Reshape if sequences
        if len(X_train.shape) == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
        
        self.model.fit(X_train, y_train)
        logger.info(f"{self.model_name} training complete")
    
    def predict(self, X_test):
        """Make predictions.
        
        Args:
            X_test: Test features
            
        Returns:
            predictions: Predicted labels
            probabilities: Class probabilities
        """
        # Reshape if sequences
        if len(X_test.shape) == 3:
            X_test = X_test.reshape(X_test.shape[0], -1)
        
        predictions = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)
        return predictions, probabilities
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: True labels
            
        Returns:
            Dictionary of metrics
        """
        predictions, probabilities = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
        recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(y_test, predictions)
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': conf_matrix.tolist()
        }
        
        logger.info(f"{self.model_name} Evaluation:")
        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1-Score:  {f1:.4f}")
        
        return metrics
    
    def save(self, filepath):
        """Save model to disk.
        
        Args:
            filepath: Path to save model
        """
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from disk.
        
        Args:
            filepath: Path to load model from
        """
        self.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")


class XGBoostModel:
    """XGBoost Classifier for enhanced fault prediction."""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize XGBoost model.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        params = config['models']['xgboost']
        self.model = xgb.XGBClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            random_state=params['random_state'],
            eval_metric='mlogloss'
        )
        self.model_name = "XGBoost"
        logger.info(f"Initialized {self.model_name} classifier")
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        logger.info(f"Training {self.model_name} on {X_train.shape[0]} samples...")
        
        # Reshape if sequences
        if len(X_train.shape) == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
        
        eval_set = None
        if X_val is not None and y_val is not None:
            if len(X_val.shape) == 3:
                X_val = X_val.reshape(X_val.shape[0], -1)
            eval_set = [(X_val, y_val)]
        
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        logger.info(f"{self.model_name} training complete")
    
    def predict(self, X_test):
        """Make predictions.
        
        Args:
            X_test: Test features
            
        Returns:
            predictions: Predicted labels
            probabilities: Class probabilities
        """
        # Reshape if sequences
        if len(X_test.shape) == 3:
            X_test = X_test.reshape(X_test.shape[0], -1)
        
        predictions = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)
        return predictions, probabilities
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: True labels
            
        Returns:
            Dictionary of metrics
        """
        predictions, probabilities = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
        recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(y_test, predictions)
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': conf_matrix.tolist()
        }
        
        logger.info(f"{self.model_name} Evaluation:")
        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1-Score:  {f1:.4f}")
        
        return metrics
    
    def save(self, filepath):
        """Save model to disk.
        
        Args:
            filepath: Path to save model
        """
        self.model.save_model(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from disk.
        
        Args:
            filepath: Path to load model from
        """
        self.model.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")


class LSTMModel:
    """LSTM model for Remaining Useful Life (RUL) prediction."""
    
    def __init__(self, input_shape, config_path="config.yaml"):
        """Initialize LSTM model.
        
        Args:
            input_shape: Tuple of (timesteps, features)
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.params = config['models']['lstm']
        self.input_shape = input_shape
        self.model = None
        self.model_name = "LSTM"
        self.history = None
        
        self._build_model()
        logger.info(f"Initialized {self.model_name} regression model")
    
    def _build_model(self):
        """Build LSTM architecture."""
        self.model = Sequential([
            LSTM(self.params['units'], return_sequences=True, input_shape=self.input_shape),
            Dropout(self.params['dropout']),
            LSTM(self.params['units'] // 2, return_sequences=False),
            Dropout(self.params['dropout']),
            Dense(32, activation='relu'),
            Dense(1)  # Regression output
        ])
        
        optimizer = keras.optimizers.Adam(learning_rate=self.params['learning_rate'])
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        logger.info(f"LSTM architecture built: {self.model.count_params()} parameters")
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the LSTM model.
        
        Args:
            X_train: Training sequences (samples, timesteps, features)
            y_train: Training targets (RUL values)
            X_val: Validation sequences (optional)
            y_val: Validation targets (optional)
        """
        logger.info(f"Training {self.model_name} on {X_train.shape[0]} sequences...")
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=self.params['patience'],
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        # Train
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        logger.info(f"{self.model_name} training complete")
    
    def predict(self, X_test):
        """Make RUL predictions.
        
        Args:
            X_test: Test sequences
            
        Returns:
            predictions: Predicted RUL values
        """
        predictions = self.model.predict(X_test, verbose=0)
        return predictions.flatten()
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance.
        
        Args:
            X_test: Test sequences
            y_test: True RUL values
            
        Returns:
            Dictionary of metrics
        """
        predictions = self.predict(X_test)
        
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        
        # R-squared
        ss_res = np.sum((y_test - predictions) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2_score': float(r2)
        }
        
        logger.info(f"{self.model_name} Evaluation:")
        logger.info(f"  RMSE:     {rmse:.4f}")
        logger.info(f"  MAE:      {mae:.4f}")
        logger.info(f"  R²:       {r2:.4f}")
        
        return metrics
    
    def save(self, filepath):
        """Save model to disk.
        
        Args:
            filepath: Path to save model
        """
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from disk.
        
        Args:
            filepath: Path to load model from
        """
        self.model = load_model(filepath)
        logger.info(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Quick test
    print("Model classes initialized successfully")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"XGBoost version: {xgb.__version__}")