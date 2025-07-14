"""
Tests for the FastAPI application.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import numpy as np
from src.api.app import app, HouseFeatures


@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)


@pytest.fixture
def sample_house_data():
    """Sample house data for testing."""
    return {
        "area": 120.5,
        "quartos": 3,
        "banheiros": 2,
        "idade": 5.0,
        "garagem": 1,
        "bairro": "Zona Sul"
    }


@pytest.fixture
def mock_model():
    """Mock model fixture."""
    model = Mock()
    model.predict.return_value = np.array([500000.0])
    model.estimators_ = [Mock() for _ in range(10)]
    for estimator in model.estimators_:
        estimator.predict.return_value = np.array([500000.0])
    return model


class TestAPI:
    """Test API endpoints."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        assert "House Price Prediction API" in response.json()["message"]
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    @patch('src.api.app.model')
    def test_predict_endpoint(self, mock_model_global, client, sample_house_data, mock_model):
        """Test prediction endpoint."""
        mock_model_global.return_value = mock_model
        with patch('src.api.app.model', mock_model):
            response = client.post("/predict", json=sample_house_data)
            assert response.status_code == 200
            data = response.json()
            assert "predicted_price" in data
            assert data["predicted_price"] > 0
    
    def test_predict_invalid_data(self, client):
        """Test prediction with invalid data."""
        invalid_data = {"area": -10}
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422
    
    @patch('src.api.app.model')
    def test_batch_predict(self, mock_model_global, client, sample_house_data, mock_model):
        """Test batch prediction endpoint."""
        mock_model_global.return_value = mock_model
        batch_data = {"houses": [sample_house_data, sample_house_data]}
        with patch('src.api.app.model', mock_model):
            response = client.post("/predict/batch", json=batch_data)
            assert response.status_code == 200
            data = response.json()
            assert len(data["predictions"]) == 2
    
    def test_neighborhoods_endpoint(self, client):
        """Test neighborhoods endpoint."""
        response = client.get("/neighborhoods")
        assert response.status_code == 200
        assert "neighborhoods" in response.json()


class TestHouseFeatures:
    """Test HouseFeatures model validation."""
    
    def test_valid_house_features(self, sample_house_data):
        """Test valid house features."""
        house = HouseFeatures(**sample_house_data)
        assert house.area == 120.5
        assert house.quartos == 3
    
    def test_invalid_area(self):
        """Test invalid area validation."""
        with pytest.raises(ValueError):
            HouseFeatures(area=-10, quartos=2, banheiros=1, idade=5, garagem=1, bairro="Centro")
    
    def test_invalid_quartos(self):
        """Test invalid quartos validation."""
        with pytest.raises(ValueError):
            HouseFeatures(area=100, quartos=0, banheiros=1, idade=5, garagem=1, bairro="Centro")