"""Main training pipeline script.

Runs end-to-end preprocessing, model training, and evaluation.
"""

import argparse
import sys
import json
import logging
from pathlib import Path
import numpy as np
from datetime import datetime

from data_preprocessing import DataPreprocessor
from model_training import RandomForestModel, XGBoostModel, LSTMModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def create_directories():
    """Create necessary output directories."""
    dirs = ['data', 'models', 'results', 'logs']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
    logger.info("Output directories created")


def train_classification_models(data, models_dir):
    """Train classification models (RandomForest, XGBoost).
    
    Args:
        data: Preprocessed data dictionary
        models_dir: Directory to save models
        
    Returns:
        Dictionary of model metrics
    """
    results = {}
    
    X_train = data['X_train_seq']
    y_train = data['y_train_class']
    X_test = data['X_test_seq']
    y_test = data['y_test_class']
    
    # 1. Random Forest
    logger.info("\n" + "="*60)
    logger.info("Training Random Forest Classifier")
    logger.info("="*60)
    
    rf_model = RandomForestModel()
    rf_model.train(X_train, y_train)
    rf_metrics = rf_model.evaluate(X_test, y_test)
    rf_model.save(models_dir / 'random_forest_model.pkl')
    results['random_forest'] = rf_metrics
    
    # 2. XGBoost
    logger.info("\n" + "="*60)
    logger.info("Training XGBoost Classifier")
    logger.info("="*60)
    
    # Split for validation
    split_idx = int(0.8 * len(X_train))
    X_train_xgb, X_val_xgb = X_train[:split_idx], X_train[split_idx:]
    y_train_xgb, y_val_xgb = y_train[:split_idx], y_train[split_idx:]
    
    xgb_model = XGBoostModel()
    xgb_model.train(X_train_xgb, y_train_xgb, X_val_xgb, y_val_xgb)
    xgb_metrics = xgb_model.evaluate(X_test, y_test)
    xgb_model.save(models_dir / 'xgboost_model.json')
    results['xgboost'] = xgb_metrics
    
    return results


def train_regression_model(data, models_dir):
    """Train LSTM regression model for RUL prediction.
    
    Args:
        data: Preprocessed data dictionary
        models_dir: Directory to save models
        
    Returns:
        Dictionary of model metrics
    """
    logger.info("\n" + "="*60)
    logger.info("Training LSTM Regression Model")
    logger.info("="*60)
    
    X_train = data['X_train_seq']
    y_train = data['y_train_rul']
    X_test = data['X_test_seq']
    y_test = data['y_test_rul']
    
    # Split for validation
    split_idx = int(0.8 * len(X_train))
    X_train_lstm, X_val_lstm = X_train[:split_idx], X_train[split_idx:]
    y_train_lstm, y_val_lstm = y_train[:split_idx], y_train[split_idx:]
    
    # Build and train LSTM
    input_shape = (X_train.shape[1], X_train.shape[2])
    lstm_model = LSTMModel(input_shape=input_shape)
    lstm_model.train(X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm)
    lstm_metrics = lstm_model.evaluate(X_test, y_test)
    lstm_model.save(models_dir / 'lstm_model.h5')
    
    return {'lstm': lstm_metrics}


def save_results(results, results_dir):
    """Save training results to JSON file.
    
    Args:
        results: Dictionary of all model metrics
        results_dir: Directory to save results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f'training_results_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"\nResults saved to {results_file}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)
    
    for model_name, metrics in results.items():
        logger.info(f"\n{model_name.upper()}:")
        for metric, value in metrics.items():
            if metric != 'confusion_matrix':
                logger.info(f"  {metric}: {value:.4f}")


def main(args):
    """Main training pipeline.
    
    Args:
        args: Command-line arguments
    """
    logger.info("Starting Predictive Maintenance Training Pipeline")
    logger.info(f"Timestamp: {datetime.now()}")
    
    # Create directories
    create_directories()
    models_dir = Path('models')
    results_dir = Path('results')
    
    # Step 1: Data Preprocessing
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Data Preprocessing")
    logger.info("="*60)
    
    preprocessor = DataPreprocessor(config_path=args.config)
    data = preprocessor.prepare_data_pipeline(download=not args.skip_download)
    
    logger.info(f"Training samples: {len(data['X_train_seq'])}")
    logger.info(f"Test samples: {len(data['X_test_seq'])}")
    logger.info(f"Sequence shape: {data['X_train_seq'].shape}")
    
    # Step 2: Train Classification Models
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Training Classification Models")
    logger.info("="*60)
    
    classification_results = train_classification_models(data, models_dir)
    
    # Step 3: Train Regression Model
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Training Regression Model (LSTM)")
    logger.info("="*60)
    
    regression_results = train_regression_model(data, models_dir)
    
    # Combine results
    all_results = {**classification_results, **regression_results}
    
    # Step 4: Save Results
    save_results(all_results, results_dir)
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING PIPELINE COMPLETE!")
    logger.info("="*60)
    logger.info(f"Models saved in: {models_dir}")
    logger.info(f"Results saved in: {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train predictive maintenance models')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip dataset download if already present')
    
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        sys.exit(1)