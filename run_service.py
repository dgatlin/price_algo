#!/usr/bin/env python3
"""FastAPI service for pricing Random Forest model."""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import json
from typing import Dict, Any, List

from pricing_rf.config import Config
from service.schemas import PredictionRequest, PredictionResponse, HealthResponse
from monitoring.monitoring_endpoints import router as monitoring_router, initialize_monitoring_service, setup_monitoring
from monitoring.monitoring_config import MonitoringConfig

# Initialize FastAPI app
app = FastAPI(
    title="Pricing Random Forest API",
    description="API for price prediction using Random Forest model",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessor
model = None
preprocessor = None
feature_names = None
model_version = "unknown"

# Initialize monitoring service
monitoring_config = MonitoringConfig(
    reference_data_path="data/processed.parquet",
    drift_threshold=0.05,
    monitoring_frequency=10,  # Check every 10 predictions
    alert_on_drift=True
)
monitoring_service = initialize_monitoring_service(monitoring_config)

# Add monitoring router
app.include_router(monitoring_router)

def load_model():
    """Load the trained model and preprocessor from MLflow."""
    global model, preprocessor, feature_names, model_version
    
    try:
        # Set up MLflow
        config = Config()
        mlflow.set_tracking_uri(config.service.mlflow_tracking_uri)
        
        # Load model
        model_uri = f"models:/{config.service.model_name}/{config.service.model_version}"
        model = mlflow.sklearn.load_model(model_uri)
        
        # Load preprocessor
        preprocessor_uri = f"models:/{config.service.model_name}-preprocessor/{config.service.model_version}"
        preprocessor = mlflow.sklearn.load_model(preprocessor_uri)
        
        # Load feature info
        try:
            with open('feature_info.json', 'r') as f:
                feature_info = json.load(f)
                feature_names = feature_info['feature_names']
        except FileNotFoundError:
            # Fallback: get feature names from preprocessor
            feature_names = preprocessor.get_feature_names_out().tolist()
        
        model_version = config.service.model_version
        
        logging.info(f"Model loaded successfully: {config.service.model_name}")
        logging.info(f"Feature names: {len(feature_names)} features")
        
    except Exception as e:
        logging.error(f"Failed to load model: {str(e)}")
        model = None
        preprocessor = None

def predict_price(features: Dict[str, Any]) -> float:
    """Make a single prediction."""
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Validate required features
        required_features = ['feature1', 'feature2', 'feature3', 'category_feature']
        missing_features = [f for f in required_features if f not in features]
        if missing_features:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required features: {missing_features}. Required: {required_features}"
            )
        
        # Validate categorical feature value
        valid_categories = ['category_a', 'category_b', 'category_c']
        if features['category_feature'] not in valid_categories:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid category_feature: {features['category_feature']}. Must be one of: {valid_categories}"
            )
        
        # Validate numerical features are numeric
        for feature in ['feature1', 'feature2', 'feature3']:
            try:
                float(features[feature])
            except (ValueError, TypeError):
                raise HTTPException(
                    status_code=400,
                    detail=f"Feature '{feature}' must be numeric, got: {features[feature]}"
                )
        
        # Convert features to DataFrame
        df = pd.DataFrame([features])
        
        # Add timestamp if not provided (needed for feature engineering)
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.Timestamp.now()
        
        # Add dummy price for feature engineering (will be ignored in prediction)
        if 'price' not in df.columns:
            df['price'] = 100.0  # Dummy value for feature engineering
        
        # Create all features using the same pipeline as training
        from pricing_rf.features import create_all_features
        from pricing_rf.config import Config
        
        config = Config()
        df_features = create_all_features(
            df,
            config.data.time_column,
            config.data.target_column,
            config.data.categorical_features,
            config.data.numerical_features
        )
        
        # Get the feature columns that the model expects
        feature_columns = [col for col in df_features.columns 
                          if col not in [config.data.time_column, config.data.target_column]]
        
        # Select only the features the model was trained on
        X = df_features[feature_columns]
        
        # Preprocess features
        X_processed = preprocessor.transform(X)
        
        # Make prediction
        prediction = model.predict(X_processed)[0]
        
        return float(prediction)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logging.basicConfig(level=logging.INFO)
    load_model()
    
    # Initialize monitoring service
    try:
        await setup_monitoring(monitoring_config.reference_data_path)
        logging.info("Monitoring service initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize monitoring service: {e}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        version="0.1.0"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict price for given features."""
    try:
        # Monitor for drift
        monitoring_result = await monitoring_service.monitor_prediction(
            request.features,
            prediction_id=f"pred_{hash(str(request.features))}"
        )
        
        prediction = predict_price(request.features)
        
        # Log drift detection
        if monitoring_result.get("drift_detected", False):
            logging.warning(f"Data drift detected in prediction: {monitoring_result}")
        
        return PredictionResponse(
            prediction=prediction,
            model_version=model_version,
            confidence=None
        )
    
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch", response_model=List[PredictionResponse])
async def predict_batch(requests: List[PredictionRequest]):
    """Predict prices for multiple feature sets."""
    try:
        predictions = []
        features_list = []
        
        # Collect features for batch monitoring
        for request in requests:
            features_list.append(request.features)
        
        # Monitor batch for drift
        monitoring_result = await monitoring_service.monitor_batch_predictions(
            features_list,
            batch_id=f"batch_{hash(str(features_list))}"
        )
        
        # Make predictions
        for request in requests:
            prediction = predict_price(request.features)
            
            predictions.append(PredictionResponse(
                prediction=prediction,
                model_version=model_version,
                confidence=None
            ))
        
        # Log drift detection
        if monitoring_result.get("drift_detected", False):
            logging.warning(f"Data drift detected in batch: {monitoring_result}")
        
        return predictions
    
    except Exception as e:
        logging.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model_info")
async def get_model_info():
    """Get information about the loaded model."""
    return {
        "model_name": "pricing-rf",
        "model_version": model_version,
        "feature_names": feature_names or [],
        "is_loaded": model is not None,
        "n_features": len(feature_names) if feature_names else 0
    }

@app.get("/features")
async def get_feature_info():
    """Get feature information for the model."""
    if feature_names is None:
        raise HTTPException(status_code=500, detail="Feature information not available")
    
    return {
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "expected_features": [
            "feature1", "feature2", "feature3", "category_feature"
        ]
    }

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load configuration
    config = Config()
    
    print("Starting Pricing Random Forest API...")
    print(f"MLflow URI: {config.service.mlflow_tracking_uri}")
    print(f"Model: {config.service.model_name}")
    print(f"Version: {config.service.model_version}")
    
    # Start the server
    uvicorn.run(
        "run_service:app",
        host=config.service.host,
        port=config.service.port,
        reload=False,
        log_level="info"
    )
