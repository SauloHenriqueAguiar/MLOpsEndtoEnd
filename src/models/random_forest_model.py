
"""
Random Forest model for house price prediction.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import joblib
import mlflow
import mlflow.sklearn
from typing import Dict, Tuple, Any, Optional
import logging
from pathlib import Path
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HousePricePredictor:
    """
    Random Forest model for predicting house prices.
    """
    
    def __init__(self, model_config: Optional[Dict] = None):
        """
        Initialize the predictor with configuration.
        
        Args:
            model_config: Dictionary with model configuration
        """
        self.model = None
        self.label_encoder = None
        self.feature_names = None
        self.model_config = model_config or self._get_default_config()
        
    def _get_default_config(self) -> Dict:
        """Get default model configuration."""
        return {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': -1
        }
    
    def prepare_features(self, df: pd.DataFrame, fit_encoder: bool = True) -> pd.DataFrame:
        """
        Prepare features for training or prediction.
        
        Args:
            df: Input DataFrame
            fit_encoder: Whether to fit the label encoder (True for training)
            
        Returns:
            DataFrame with prepared features
        """
        data = df.copy()
        
        # Handle label encoding for categorical variables
        if 'bairro' in data.columns:
            if fit_encoder:
                self.label_encoder = LabelEncoder()
                data['bairro_encoded'] = self.label_encoder.fit_transform(data['bairro'])
            else:
                if self.label_encoder is None:
                    raise ValueError("Label encoder not fitted. Train the model first.")
                data['bairro_encoded'] = self.label_encoder.transform(data['bairro'])
        
        # Feature engineering
        data['area_per_room'] = data['area'] / (data['quartos'] + 1)
        data['bathroom_ratio'] = data['banheiros'] / data['quartos']
        data['age_squared'] = data['idade'] ** 2
        
        # Select features for model
        feature_columns = [
            'area', 'quartos', 'banheiros', 'idade', 'garagem',
            'bairro_encoded', 'area_per_room', 'bathroom_ratio', 'age_squared'
        ]
        
        self.feature_names = feature_columns
        return data[feature_columns]
    
    def train(self, X: pd.DataFrame, y: pd.Series, optimize_hyperparameters: bool = False) -> Dict:
        """
        Train the Random Forest model.
        
        Args:
            X: Features DataFrame
            y: Target Series
            optimize_hyperparameters: Whether to perform hyperparameter optimization
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Starting model training...")
        
        # Prepare features
        X_processed = self.prepare_features(X, fit_encoder=True)
        
        if optimize_hyperparameters:
            logger.info("Performing hyperparameter optimization...")
            self.model = self._optimize_hyperparameters(X_processed, y)
        else:
            self.model = RandomForestRegressor(**self.model_config)
            self.model.fit(X_processed, y)
        
        # Calculate training metrics
        train_predictions = self.model.predict(X_processed)
        metrics = self._calculate_metrics(y, train_predictions)
        
        logger.info(f"Training completed. R² Score: {metrics['r2']:.4f}")
        return metrics
    
    def _optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
        """
        Optimize hyperparameters using GridSearchCV.
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            Optimized RandomForestRegressor
        """
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestRegressor(
            random_state=self.model_config['random_state'],
            n_jobs=self.model_config['n_jobs']
        )
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {-grid_search.best_score_:.2f}")
        
        return grid_search.best_estimator_
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_processed = self.prepare_features(X, fit_encoder=False)
        return self.model.predict(X_processed)
    
    def predict_single(self, area: float, quartos: int, banheiros: int, 
                      idade: float, garagem: int, bairro: str) -> float:
        """
        Predict price for a single house.
        
        Args:
            area: Area in square meters
            quartos: Number of bedrooms
            banheiros: Number of bathrooms
            idade: Age in years
            garagem: Has garage (0 or 1)
            bairro: Neighborhood name
            
        Returns:
            Predicted price
        """
        # Create DataFrame with single row
        data = pd.DataFrame({
            'area': [area],
            'quartos': [quartos],
            'banheiros': [banheiros],
            'idade': [idade],
            'garagem': [garagem],
            'bairro': [bairro]
        })
        
        prediction = self.predict(data)
        return prediction[0]
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            X: Features DataFrame
            y: True target values
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(X)
        return self._calculate_metrics(y, predictions)
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, model_path: str, save_encoder: bool = True) -> None:
        """
        Save the trained model and encoder.
        
        Args:
            model_path: Path to save the model
            save_encoder: Whether to save the label encoder
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        model_dir = Path(model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save encoder
        if save_encoder and self.label_encoder is not None:
            encoder_path = model_dir / "label_encoder.pkl"
            joblib.dump(self.label_encoder, encoder_path)
            logger.info(f"Label encoder saved to {encoder_path}")
        
        # Save feature names
        feature_path = model_dir / "feature_names.json"
        with open(feature_path, 'w') as f:
            json.dump(self.feature_names, f)
    
    def load_model(self, model_path: str, load_encoder: bool = True) -> None:
        """
        Load a trained model and encoder.
        
        Args:
            model_path: Path to the saved model
            load_encoder: Whether to load the label encoder
        """
        # Load model
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        model_dir = Path(model_path).parent
        
        # Load encoder
        if load_encoder:
            encoder_path = model_dir / "label_encoder.pkl"
            if encoder_path.exists():
                self.label_encoder = joblib.load(encoder_path)
                logger.info(f"Label encoder loaded from {encoder_path}")
        
        # Load feature names
        feature_path = model_dir / "feature_names.json"
        if feature_path.exists():
            with open(feature_path, 'r') as f:
                self.feature_names = json.load(f)
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict:
        """
        Perform cross-validation.
        
        Args:
            X: Features DataFrame
            y: Target Series
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with cross-validation results
        """
        if self.model is None:
            self.model = RandomForestRegressor(**self.model_config)
        
        X_processed = self.prepare_features(X, fit_encoder=True)
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            self.model, X_processed, y, 
            cv=cv, scoring='neg_mean_squared_error'
        )
        
        cv_r2_scores = cross_val_score(
            self.model, X_processed, y, 
            cv=cv, scoring='r2'
        )
        
        return {
            'cv_rmse_mean': np.sqrt(-cv_scores.mean()),
            'cv_rmse_std': np.sqrt(cv_scores.std()),
            'cv_r2_mean': cv_r2_scores.mean(),
            'cv_r2_std': cv_r2_scores.std()
        }


class MLflowTracker:
    """
    MLflow tracking utilities for model experiments.
    """
    
    @staticmethod
    def log_experiment(model: HousePricePredictor, metrics: Dict, 
                      experiment_name: str = "house-price-prediction",
                      run_name: Optional[str] = None) -> None:
        """
        Log experiment to MLflow.
        
        Args:
            model: Trained model
            metrics: Dictionary with metrics
            experiment_name: MLflow experiment name
            run_name: MLflow run name
        """
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=run_name):
            # Log parameters
            if hasattr(model.model, 'get_params'):
                mlflow.log_params(model.model.get_params())
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model.model, "model")
            
            # Log feature importance
            if model.model is not None:
                importance_df = model.get_feature_importance()
                importance_dict = dict(zip(importance_df['feature'], importance_df['importance']))
                mlflow.log_metrics({f"importance_{k}": v for k, v in importance_dict.items()})
            
            logger.info(f"Experiment logged to MLflow: {experiment_name}")


def main():
    """
    Main function for testing the model.
    """
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    from src.data.data_loader import DataLoader
    from sklearn.model_selection import train_test_split
    
    # Load data
    loader = DataLoader()
    try:
        df = loader.load_raw_data()
    except FileNotFoundError:
        df = loader.generate_synthetic_data()
    
    # Prepare data
    features = ['area', 'quartos', 'banheiros', 'idade', 'garagem', 'bairro']
    X = df[features]
    y = df['preco']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    predictor = HousePricePredictor()
    train_metrics = predictor.train(X_train, y_train)
    
    # Evaluate model
    test_metrics = predictor.evaluate(X_test, y_test)
    
    logger.info(f"Test RMSE: {test_metrics['rmse']:.2f}")
    logger.info(f"Test R²: {test_metrics['r2']:.4f}")
    
    # Save model
    predictor.save_model("data/models/random_forest_model.pkl")


if __name__ == "__main__":
    main()
