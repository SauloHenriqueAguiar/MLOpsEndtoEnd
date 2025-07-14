"""
FastAPI application for house price prediction.
"""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import os
import joblib
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="House Price Prediction API",
    description="MLOps API for predicting house prices using Random Forest",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global variables for model and encoder
model = None
label_encoder = None
feature_names = None


class HouseFeatures(BaseModel):
    """
    Pydantic model for house features input.
    """
    area: float = Field(..., gt=0, description="Area in square meters")
    quartos: int = Field(..., ge=1, le=10, description="Number of bedrooms")
    banheiros: int = Field(..., ge=1, le=6, description="Number of bathrooms")
    idade: float = Field(..., ge=0, le=100, description="Age in years")
    garagem: int = Field(..., ge=0, le=1, description="Has garage (0 or 1)")
    bairro: str = Field(..., description="Neighborhood name")
    
    class Config:
        schema_extra = {
            "example": {
                "area": 120.5,
                "quartos": 3,
                "banheiros": 2,
                "idade": 5.0,
                "garagem": 1,
                "bairro": "Zona Sul"
            }
        }


class PredictionResponse(BaseModel):
    """
    Pydantic model for prediction response.
    """
    predicted_price: float = Field(..., description="Predicted house price in BRL")
    confidence_interval: Optional[Dict[str, float]] = Field(None, description="95% confidence interval")
    input_features: HouseFeatures = Field(..., description="Input features used for prediction")
    prediction_timestamp: str = Field(..., description="Timestamp of prediction")
    model_version: str = Field(..., description="Model version used")


class BatchPredictionRequest(BaseModel):
    """
    Pydantic model for batch prediction request.
    """
    houses: List[HouseFeatures] = Field(..., description="List of house features")


class BatchPredictionResponse(BaseModel):
    """
    Pydantic model for batch prediction response.
    """
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    batch_size: int = Field(..., description="Number of predictions in batch")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class ModelInfo(BaseModel):
    """
    Pydantic model for model information.
    """
    model_name: str
    model_version: str
    feature_names: List[str]
    model_metrics: Optional[Dict[str, float]] = None
    last_trained: Optional[str] = None


async def load_model_artifacts():
    """
    Load model, encoder, and other artifacts.
    """
    global model, label_encoder, feature_names
    
    model_dir = Path(os.getenv("MODEL_PATH", "data/models"))
    
    try:
        # Load model
        model_path = model_dir / "random_forest_model.pkl"
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Load label encoder
        encoder_path = model_dir / "label_encoder.pkl"
        if encoder_path.exists():
            label_encoder = joblib.load(encoder_path)
            logger.info(f"Label encoder loaded from {encoder_path}")
        
        # Load feature names
        feature_names_path = model_dir / "feature_names.json"
        if feature_names_path.exists():
            with open(feature_names_path, 'r') as f:
                feature_names = json.load(f)
        else:
            # Default feature names
            feature_names = [
                'area', 'quartos', 'banheiros', 'idade', 'garagem',
                'bairro_encoded', 'area_per_room', 'bathroom_ratio', 'age_squared'
            ]
        
        logger.info("All model artifacts loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model artifacts: {e}")
        raise


def check_model_loaded():
    """
    Dependency to check if model is loaded.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )


def prepare_features(house_data: HouseFeatures) -> np.ndarray:
    """
    Prepare features for prediction.
    
    Args:
        house_data: House features from request
        
    Returns:
        Numpy array with processed features
    """
    # Create DataFrame
    df = pd.DataFrame([house_data.dict()])
    
    # Encode neighborhood
    if label_encoder is not None:
        try:
            bairro_encoded = label_encoder.transform([house_data.bairro])[0]
        except ValueError:
            # Handle unknown neighborhood
            logger.warning(f"Unknown neighborhood: {house_data.bairro}. Using default encoding.")
            bairro_encoded = 0
    else:
        bairro_encoded = 0
    
    # Feature engineering
    area_per_room = house_data.area / (house_data.quartos + 1)
    bathroom_ratio = house_data.banheiros / house_data.quartos
    age_squared = house_data.idade ** 2
    
    # Create feature array
    features = np.array([[
        house_data.area,
        house_data.quartos,
        house_data.banheiros,
        house_data.idade,
        house_data.garagem,
        bairro_encoded,
        area_per_room,
        bathroom_ratio,
        age_squared
    ]])
    
    return features


@app.on_event("startup")
async def startup_event():
    """
    Load model artifacts on startup.
    """
    logger.info("Starting up House Price Prediction API...")
    await load_model_artifacts()


@app.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "House Price Prediction API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None
    }


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info(model_check: None = Depends(check_model_loaded)):
    """
    Get information about the loaded model.
    """
    model_dir = Path(os.getenv("MODEL_PATH", "data/models"))
    metrics_path = model_dir / "model_metrics.json"
    
    model_metrics = None
    last_trained = None
    
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)
            model_metrics = {
                "rmse": metrics_data.get("final_rmse"),
                "mae": metrics_data.get("final_mae"),
                "r2": metrics_data.get("final_r2")
            }
            last_trained = metrics_data.get("training_date")
    
    return ModelInfo(
        model_name="Random Forest Regressor",
        model_version="1.0.0",
        feature_names=feature_names or [],
        model_metrics=model_metrics,
        last_trained=last_trained
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_price(
    house_data: HouseFeatures,
    model_check: None = Depends(check_model_loaded)
):
    """
    Predict house price for a single house.
    
    Args:
        house_data: House features for prediction
        
    Returns:
        Predicted price and additional information
    """
    try:
        # Prepare features
        features = prepare_features(house_data)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Calculate confidence interval (approximate)
        # Using standard deviation from tree predictions
        if hasattr(model, 'estimators_'):
            tree_predictions = [tree.predict(features)[0] for tree in model.estimators_]
            std_pred = np.std(tree_predictions)
            confidence_interval = {
                "lower_bound": prediction - 1.96 * std_pred,
                "upper_bound": prediction + 1.96 * std_pred
            }
        else:
            confidence_interval = None
        
        return PredictionResponse(
            predicted_price=float(prediction),
            confidence_interval=confidence_interval,
            input_features=house_data,
            prediction_timestamp=datetime.now().isoformat(),
            model_version="1.0.0"
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    batch_request: BatchPredictionRequest,
    model_check: None = Depends(check_model_loaded)
):
    """
    Predict house prices for multiple houses.
    
    Args:
        batch_request: Batch of house features
        
    Returns:
        List of predictions with processing time
    """
    start_time = datetime.now()
    
    try:
        predictions = []
        
        for house_data in batch_request.houses:
            # Prepare features
            features = prepare_features(house_data)
            
            # Make prediction
            prediction = model.predict(features)[0]
            
            # Calculate confidence interval
            if hasattr(model, 'estimators_'):
                tree_predictions = [tree.predict(features)[0] for tree in model.estimators_]
                std_pred = np.std(tree_predictions)
                confidence_interval = {
                    "lower_bound": prediction - 1.96 * std_pred,
                    "upper_bound": prediction + 1.96 * std_pred
                }
            else:
                confidence_interval = None
            
            predictions.append(PredictionResponse(
                predicted_price=float(prediction),
                confidence_interval=confidence_interval,
                input_features=house_data,
                prediction_timestamp=datetime.now().isoformat(),
                model_version="1.0.0"
            ))
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        return BatchPredictionResponse(
            predictions=predictions,
            batch_size=len(batch_request.houses),
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/neighborhoods")
async def get_available_neighborhoods():
    """
    Get list of available neighborhoods.
    """
    if label_encoder is not None:
        neighborhoods = label_encoder.classes_.tolist()
    else:
        # Default neighborhoods
        neighborhoods = ['Centro', 'Zona Sul', 'Zona Norte', 'Zona Oeste', 'Zona Leste']
    
    return {
        "neighborhoods": neighborhoods,
        "total_count": len(neighborhoods)
    }


@app.get("/metrics")
async def get_metrics():
    """
    Get model performance metrics.
    """
    model_dir = Path(os.getenv("MODEL_PATH", "data/models"))
    metrics_path = model_dir / "model_metrics.json"
    
    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Model metrics not found"
        )
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics


if __name__ == "__main__":
    import uvicorn
    
    # Run the app
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )