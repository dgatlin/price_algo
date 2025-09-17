"""Tests for service module."""

import pytest
import json
from unittest.mock import Mock, patch
import httpx

from src.service.app import app
from src.service.schemas import PredictionRequest, PredictionResponse, HealthResponse
from src.service.inference import PricingInference
from src.service.auth import create_jwt_token, verify_jwt_token
from src.pricing_rf.config import Config


class TestSchemas:
    """Test Pydantic schemas."""
    
    def test_prediction_request(self):
        """Test prediction request schema."""
        request = PredictionRequest(
            features={"feature1": 1.5, "feature2": "category_a"},
            client_ip="127.0.0.1"
        )
        assert request.features["feature1"] == 1.5
        assert request.client_ip == "127.0.0.1"
    
    def test_prediction_response(self):
        """Test prediction response schema."""
        response = PredictionResponse(
            prediction=100.5,
            model_version="1.0.0",
            confidence=0.95
        )
        assert response.prediction == 100.5
        assert response.model_version == "1.0.0"
        assert response.confidence == 0.95
    
    def test_health_response(self):
        """Test health response schema."""
        response = HealthResponse(
            status="healthy",
            model_loaded=True,
            version="1.0.0"
        )
        assert response.status == "healthy"
        assert response.model_loaded is True
        assert response.version == "1.0.0"


class TestAuth:
    """Test authentication functionality."""
    
    def test_create_jwt_token(self):
        """Test JWT token creation."""
        token = create_jwt_token("test_user", "secret_key")
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_verify_jwt_token(self):
        """Test JWT token verification."""
        token = create_jwt_token("test_user", "secret_key")
        user_id = verify_jwt_token(token, "secret_key")
        assert user_id == "test_user"
    
    def test_verify_invalid_token(self):
        """Test invalid JWT token verification."""
        user_id = verify_jwt_token("invalid_token", "secret_key")
        assert user_id is None


class TestInference:
    """Test inference functionality."""
    
    @patch('src.service.inference.mlflow')
    def test_pricing_inference_init(self, mock_mlflow):
        """Test PricingInference initialization."""
        mock_mlflow_client = Mock()
        mock_mlflow.tracking.MlflowClient.return_value = mock_mlflow_client
        mock_mlflow_client.get_latest_versions.return_value = [Mock()]
        
        # Create a mock config
        mock_config = Mock(spec=Config)
        mock_config.service = Mock()
        mock_config.service.mlflow_tracking_uri = "http://localhost:5000"
        
        inference = PricingInference(mock_config)
        assert inference.config == mock_config
    
    @patch('src.service.inference.mlflow')
    def test_predict(self, mock_mlflow):
        """Test prediction functionality."""
        mock_mlflow_client = Mock()
        mock_mlflow.tracking.MlflowClient.return_value = mock_mlflow_client
        mock_mlflow_client.get_latest_versions.return_value = [Mock()]
        
        # Create a mock config
        mock_config = Mock(spec=Config)
        mock_config.service = Mock()
        mock_config.service.mlflow_tracking_uri = "http://localhost:5000"
        
        inference = PricingInference(mock_config)
        
        # Mock the model and preprocessor
        mock_model = Mock()
        mock_model.predict.return_value = [100.5]
        inference.model = mock_model
        
        mock_preprocessor = Mock()
        mock_preprocessor.transform.return_value = [[1.0, 2.0, 3.0]]
        inference.preprocessor = mock_preprocessor
        
        result = inference.predict({"feature1": 1.0, "feature2": 2.0})
        assert result == 100.5
    
    @patch('src.service.inference.mlflow')
    def test_is_model_loaded(self, mock_mlflow):
        """Test model loaded status."""
        mock_mlflow_client = Mock()
        mock_mlflow.tracking.MlflowClient.return_value = mock_mlflow_client
        mock_mlflow_client.get_latest_versions.return_value = [Mock()]
        
        # Create a mock config
        mock_config = Mock(spec=Config)
        mock_config.service = Mock()
        mock_config.service.mlflow_tracking_uri = "http://localhost:5000"
        
        # Mock the _load_model method to prevent actual loading
        with patch.object(PricingInference, '_load_model'):
            inference = PricingInference(mock_config)
            assert inference.is_model_loaded() is False
            
            # Mock both model and preprocessor as loaded
            inference.model = Mock()
            inference.preprocessor = Mock()
            assert inference.is_model_loaded() is True


class TestAPI:
    """Test FastAPI endpoints - simplified without HTTP client."""
    
    def test_app_creation(self):
        """Test that the FastAPI app is created correctly."""
        assert app is not None
        assert hasattr(app, 'routes')
        assert len(app.routes) > 0
    
    def test_health_endpoint_function(self):
        """Test health check endpoint function directly."""
        from src.service.app import health_check
        import asyncio
        
        # Mock the inference module
        with patch('src.service.app.inference') as mock_inference:
            mock_inference.is_model_loaded.return_value = True
            mock_inference.get_model_version.return_value = "1.0.0"
            
            response = asyncio.run(health_check())
            assert response.status == "healthy"
            assert response.model_loaded is True
            assert response.version == "0.1.0"
    
    def test_model_info_endpoint_function(self):
        """Test model info endpoint function directly."""
        from src.service.app import get_model_info
        import asyncio
        
        # Mock the inference module
        with patch('src.service.app.inference') as mock_inference:
            mock_inference.get_model_name.return_value = "pricing-rf"
            mock_inference.get_model_version.return_value = "1.0.0"
            mock_inference.get_feature_names.return_value = ["feature1", "feature2"]
            mock_inference.is_model_loaded.return_value = True
            
            response = asyncio.run(get_model_info())
            assert response["model_name"] == "pricing-rf"
            assert response["model_version"] == "1.0.0"
            assert response["feature_names"] == ["feature1", "feature2"]
            assert response["is_loaded"] is True
    
    def test_metrics_endpoint_function(self):
        """Test metrics endpoint function directly."""
        from src.service.app import get_metrics
        import asyncio
        
        # Mock the inference module
        with patch('src.service.app.inference') as mock_inference:
            mock_inference.get_metrics.return_value = {"accuracy": 0.95, "mae": 10.5}
            
            response = asyncio.run(get_metrics())
            assert "accuracy" in response
            assert "mae" in response
            assert response["accuracy"] == 0.95


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_prediction_error_handling(self):
        """Test prediction error handling."""
        from src.service.app import predict_price
        import asyncio
        
        # Mock the inference module to raise an exception
        with patch('src.service.app.inference.predict') as mock_predict:
            mock_predict.side_effect = Exception("Model error")
            
            request_data = {
                "features": {"feature1": 1.5, "feature2": "category_a"}
            }
            
            with pytest.raises(Exception):
                asyncio.run(predict_price(request_data))


class TestIntegration:
    """Test integration scenarios."""
    
    def test_app_routes(self):
        """Test that all expected routes are registered."""
        route_paths = [route.path for route in app.routes if hasattr(route, 'path')]
        
        expected_routes = [
            "/health",
            "/model_info", 
            "/metrics",
            "/predict",
            "/predict_batch",
            "/docs",
            "/openapi.json"
        ]
        
        for expected_route in expected_routes:
            assert expected_route in route_paths, f"Route {expected_route} not found"
    
    def test_app_metadata(self):
        """Test app metadata."""
        assert app.title == "Pricing Random Forest API"
        assert app.version == "0.1.0"
        assert app.description is not None