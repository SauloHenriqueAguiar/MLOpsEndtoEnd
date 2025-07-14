"""
Prediction service for house price prediction model.
"""

import joblib
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class PredictionService:
    """Service class for handling model predictions."""
    
    def __init__(self, model_path: str = "data/models"):
        self.model_path = Path(model_path)
        self.model = None
        self.label_encoder = None
        self.feature_names = None
        self.model_metrics = None
        
    def load_model(self) -> bool:
        """Load model and associated artifacts."""
        try:
            # Load model
            model_file = self.model_path / "random_forest_model.pkl"
            self.model = joblib.load(model_file)
            
            # Load label encoder
            encoder_file = self.model_path / "label_encoder.pkl"
            if encoder_file.exists():
                self.label_encoder = joblib.load(encoder_file)
            
            # Load feature names
            features_file = self.model_path / "feature_names.json"
            if features_file.exists():
                with open(features_file, 'r') as f:
                    self.feature_names = json.load(f)
            
            # Load metrics
            metrics_file = self.model_path / "model_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    self.model_metrics = json.load(f)
            
            logger.info("Model artifacts loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
    
    def prepare_features(self, house_data: Dict) -> np.ndarray:
        """Prepare features for prediction."""
        # Encode neighborhood
        bairro_encoded = 0
        if self.label_encoder and 'bairro' in house_data:
            try:
                bairro_encoded = self.label_encoder.transform([house_data['bairro']])[0]
            except ValueError:
                logger.warning(f"Unknown neighborhood: {house_data['bairro']}")
        
        # Feature engineering
        area_per_room = house_data['area'] / (house_data['quartos'] + 1)
        bathroom_ratio = house_data['banheiros'] / house_data['quartos']
        age_squared = house_data['idade'] ** 2
        
        return np.array([[
            house_data['area'],
            house_data['quartos'],
            house_data['banheiros'],
            house_data['idade'],
            house_data['garagem'],
            bairro_encoded,
            area_per_room,
            bathroom_ratio,
            age_squared
        ]])
    
    def predict(self, house_data: Dict) -> Tuple[float, Optional[Dict[str, float]]]:
        """Make single prediction."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        features = self.prepare_features(house_data)
        prediction = self.model.predict(features)[0]
        
        # Calculate confidence interval
        confidence_interval = None
        if hasattr(self.model, 'estimators_'):
            tree_predictions = [tree.predict(features)[0] for tree in self.model.estimators_]
            std_pred = np.std(tree_predictions)
            confidence_interval = {
                "lower_bound": prediction - 1.96 * std_pred,
                "upper_bound": prediction + 1.96 * std_pred
            }
        
        return float(prediction), confidence_interval
    
    def predict_batch(self, houses_data: List[Dict]) -> List[Tuple[float, Optional[Dict[str, float]]]]:
        """Make batch predictions."""
        return [self.predict(house_data) for house_data in houses_data]
    
    def get_neighborhoods(self) -> List[str]:
        """Get available neighborhoods."""
        if self.label_encoder:
            return self.label_encoder.classes_.tolist()
        return ['Centro', 'Zona Sul', 'Zona Norte', 'Zona Oeste', 'Zona Leste']
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            "model_name": "Random Forest Regressor",
            "model_version": "1.0.0",
            "feature_names": self.feature_names or [],
            "model_metrics": self.model_metrics,
            "is_loaded": self.is_loaded()
        }