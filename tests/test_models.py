"""
Tests for model classes and utilities.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
from src.models.random_forest_model import HousePricePredictor, MLflowTracker
from src.models.model_utils import calculate_metrics, validate_model_performance, save_model_artifacts


@pytest.fixture
def sample_data():
    """Sample training data."""
    return pd.DataFrame({
        'area': [100, 150, 80, 200, 120],
        'quartos': [2, 3, 1, 4, 2],
        'banheiros': [1, 2, 1, 3, 2],
        'idade': [5, 10, 2, 15, 8],
        'garagem': [1, 1, 0, 1, 1],
        'bairro': ['Centro', 'Zona Sul', 'Centro', 'Zona Norte', 'Centro']
    }), pd.Series([300000, 500000, 200000, 700000, 350000])


@pytest.fixture
def temp_model_dir():
    """Temporary directory for model artifacts."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestHousePricePredictor:
    """Test HousePricePredictor class."""
    
    def test_init_default_config(self):
        """Test initialization with default config."""
        predictor = HousePricePredictor()
        assert predictor.model is None
        assert predictor.model_config['n_estimators'] == 100
    
    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = {'n_estimators': 50, 'random_state': 123}
        predictor = HousePricePredictor(config)
        assert predictor.model_config['n_estimators'] == 50
    
    def test_prepare_features(self, sample_data):
        """Test feature preparation."""
        X, _ = sample_data
        predictor = HousePricePredictor()
        
        features = predictor.prepare_features(X, fit_encoder=True)
        
        assert 'bairro_encoded' in features.columns
        assert 'area_per_room' in features.columns
        assert 'bathroom_ratio' in features.columns
        assert 'age_squared' in features.columns
        assert len(features.columns) == 9
    
    def test_train_model(self, sample_data):
        """Test model training."""
        X, y = sample_data
        predictor = HousePricePredictor()
        
        metrics = predictor.train(X, y)
        
        assert predictor.model is not None
        assert 'r2' in metrics
        assert 'rmse' in metrics
        assert metrics['r2'] >= 0
    
    def test_predict(self, sample_data):
        """Test model prediction."""
        X, y = sample_data
        predictor = HousePricePredictor()
        predictor.train(X, y)
        
        predictions = predictor.predict(X)
        
        assert len(predictions) == len(X)
        assert all(pred > 0 for pred in predictions)
    
    def test_predict_single(self, sample_data):
        """Test single prediction."""
        X, y = sample_data
        predictor = HousePricePredictor()
        predictor.train(X, y)
        
        prediction = predictor.predict_single(120, 3, 2, 5, 1, 'Centro')
        
        assert isinstance(prediction, (int, float))
        assert prediction > 0
    
    def test_evaluate(self, sample_data):
        """Test model evaluation."""
        X, y = sample_data
        predictor = HousePricePredictor()
        predictor.train(X, y)
        
        metrics = predictor.evaluate(X, y)
        
        assert 'r2' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'mse' in metrics
    
    def test_get_feature_importance(self, sample_data):
        """Test feature importance extraction."""
        X, y = sample_data
        predictor = HousePricePredictor()
        predictor.train(X, y)
        
        importance = predictor.get_feature_importance()
        
        assert len(importance) == 9
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns
    
    def test_save_load_model(self, sample_data, temp_model_dir):
        """Test model saving and loading."""
        X, y = sample_data
        predictor = HousePricePredictor()
        predictor.train(X, y)
        
        model_path = Path(temp_model_dir) / "test_model.pkl"
        predictor.save_model(str(model_path))
        
        # Load model
        new_predictor = HousePricePredictor()
        new_predictor.load_model(str(model_path))
        
        assert new_predictor.model is not None
        assert new_predictor.label_encoder is not None
    
    def test_cross_validate(self, sample_data):
        """Test cross validation."""
        X, y = sample_data
        predictor = HousePricePredictor()
        
        cv_results = predictor.cross_validate(X, y, cv=3)
        
        assert 'cv_rmse_mean' in cv_results
        assert 'cv_r2_mean' in cv_results
        assert cv_results['cv_rmse_mean'] > 0
    
    def test_predict_without_training(self, sample_data):
        """Test prediction without training raises error."""
        X, _ = sample_data
        predictor = HousePricePredictor()
        
        with pytest.raises(ValueError):
            predictor.predict(X)


class TestMLflowTracker:
    """Test MLflowTracker class."""
    
    @patch('src.models.random_forest_model.mlflow')
    def test_log_experiment(self, mock_mlflow, sample_data):
        """Test MLflow experiment logging."""
        X, y = sample_data
        predictor = HousePricePredictor()
        predictor.train(X, y)
        
        metrics = {'rmse': 1000, 'r2': 0.8}
        
        MLflowTracker.log_experiment(predictor, metrics)
        
        mock_mlflow.set_experiment.assert_called_once()
        mock_mlflow.start_run.assert_called_once()


class TestModelUtils:
    """Test model utility functions."""
    
    def test_calculate_metrics(self):
        """Test metrics calculation."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert metrics['rmse'] > 0
    
    def test_validate_model_performance_pass(self):
        """Test model validation with good metrics."""
        metrics = {'r2': 0.8, 'rmse': 30000}
        thresholds = {'min_r2': 0.7, 'max_rmse': 50000}
        
        assert validate_model_performance(metrics, thresholds) is True
    
    def test_validate_model_performance_fail(self):
        """Test model validation with poor metrics."""
        metrics = {'r2': 0.5, 'rmse': 60000}
        thresholds = {'min_r2': 0.7, 'max_rmse': 50000}
        
        assert validate_model_performance(metrics, thresholds) is False
    
    def test_save_model_artifacts(self, temp_model_dir):
        """Test saving model artifacts."""
        model = Mock()
        encoder = Mock()
        metrics = {'r2': 0.8, 'rmse': 1000}
        
        save_model_artifacts(model, encoder, metrics, temp_model_dir)
        
        model_dir = Path(temp_model_dir)
        assert (model_dir / "model_metrics.json").exists()