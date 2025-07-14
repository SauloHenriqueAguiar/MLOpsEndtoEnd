"""
Utility functions for model operations.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json
import joblib
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive model metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }


def validate_model_performance(metrics, thresholds):
    """Validate if model meets performance thresholds."""
    checks = []
    
    if 'min_r2' in thresholds:
        checks.append(metrics.get('r2', 0) >= thresholds['min_r2'])
    
    if 'max_rmse' in thresholds:
        checks.append(metrics.get('rmse', float('inf')) <= thresholds['max_rmse'])
    
    if 'max_mae' in thresholds:
        checks.append(metrics.get('mae', float('inf')) <= thresholds['max_mae'])
    
    if 'max_mape' in thresholds:
        checks.append(metrics.get('mape', float('inf')) <= thresholds['max_mape'])
    
    return all(checks)


def save_model_artifacts(model, encoder, metrics, save_dir):
    """Save model artifacts including metrics and metadata."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = save_path / "random_forest_model.pkl"
    joblib.dump(model, model_path)
    
    # Save encoder
    if encoder is not None:
        encoder_path = save_path / "label_encoder.pkl"
        joblib.dump(encoder, encoder_path)
    
    # Save metrics
    metrics_data = {
        **metrics,
        'saved_at': datetime.now().isoformat(),
        'model_type': 'RandomForestRegressor'
    }
    
    metrics_path = save_path / "model_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    logger.info(f"Model artifacts saved to {save_path}")


def load_model_artifacts(load_dir):
    """Load model artifacts."""
    load_path = Path(load_dir)
    
    # Load model
    model_path = load_path / "random_forest_model.pkl"
    model = joblib.load(model_path)
    
    # Load encoder
    encoder_path = load_path / "label_encoder.pkl"
    encoder = None
    if encoder_path.exists():
        encoder = joblib.load(encoder_path)
    
    # Load metrics
    metrics_path = load_path / "model_metrics.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    
    return model, encoder, metrics