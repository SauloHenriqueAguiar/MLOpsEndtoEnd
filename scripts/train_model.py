#!/usr/bin/env python3
"""
Training script for house price prediction model.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import DataLoader
from src.models.random_forest_model import HousePricePredictor, MLflowTracker
from src.utils.config_loader import load_config
from sklearn.model_selection import train_test_split
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train house price prediction model")
    parser.add_argument("--config", default="config.yaml", help="Configuration file")
    parser.add_argument("--optimize", action="store_true", help="Enable hyperparameter optimization")
    parser.add_argument("--samples", type=int, default=1000, help="Number of synthetic samples")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize data loader
    loader = DataLoader()
    
    # Load or generate data
    try:
        df = loader.load_raw_data()
        logger.info("Loaded existing data")
    except FileNotFoundError:
        df = loader.generate_synthetic_data(args.samples)
        loader.save_data(df, "house_prices.csv")
        logger.info(f"Generated {args.samples} synthetic samples")
    
    # Prepare features and target
    features = config['features']['numeric_features'] + config['features']['categorical_features']
    X = df[features]
    y = df[config['features']['target']]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['data']['split']['test_size'],
        random_state=config['data']['split']['random_state']
    )
    
    # Initialize predictor
    predictor = HousePricePredictor()
    
    # Train model
    logger.info("Starting model training...")
    train_metrics = predictor.train(X_train, y_train, optimize_hyperparameters=args.optimize)
    
    # Evaluate on test set
    test_metrics = predictor.evaluate(X_test, y_test)
    
    # Log to MLflow
    MLflowTracker.log_experiment(
        predictor, 
        {**train_metrics, **{f"test_{k}": v for k, v in test_metrics.items()}},
        experiment_name=config['model']['registry']['experiment_name']
    )
    
    # Save model
    model_path = f"{config['data']['paths']['models']}/random_forest_model.pkl"
    predictor.save_model(model_path)
    
    logger.info(f"Training completed!")
    logger.info(f"Test R²: {test_metrics['r2']:.4f}")
    logger.info(f"Test RMSE: {test_metrics['rmse']:.2f}")
    logger.info(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()