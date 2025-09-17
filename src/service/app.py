"""FastAPI application for pricing model inference.

This module bootstraps a production-ready API for price prediction using a
Random Forest model, integrates real-time drift monitoring, and exposes
health/metrics endpoints. It wires together configuration, inference,
authentication (optional), CORS, and the monitoring router.

Overview
--------
- **App metadata**: Name, description, version for OpenAPI UI.
- **CORS**: Permissive defaults for ease of local dev (tighten in prod).
- **Config & Inference**: Loads `Config` (pydantic-settings) and `PricingInference`.
- **Monitoring**: Initializes a `MonitoringService` and mounts the `/monitoring` router.
- **Auth (optional)**: JWT/IP checks via dependency injection when enabled.
- **Endpoints**:
  - `GET /health`        → liveness + model-loaded flag
  - `POST /predict`      → single prediction with drift check
  - `POST /predict_batch`→ batch predictions with drift check
  - `GET /model_info`    → model metadata (name, version, features)
  - `GET /metrics`       → model performance metrics (from inference layer)
  - `/monitoring/*`      → status, history, config, check, etc. (mounted router)

Startup
-------
- On application startup, `setup_monitoring()` loads reference data from the
  configured path and builds Alibi Detect–based drift detectors.

Security Notes
--------------
- Auth is off by default. When `SERVICE_ENABLE_AUTH=true`:
  - JWT authentication is enforced via `get_current_user`.
  - Optional IP allowlist enforcement via `verify_ip_address`.
- CORS is configured to allow all origins for convenience; restrict this for
  production (e.g., explicit domains and methods).

Operational Notes
-----------------
- Drift checks are frequency-gated (e.g., every 10 requests) per monitoring
  configuration to keep P50 latency low.
- Prediction responses are not modified by drift results; alerts/logs are
  emitted out-of-band when drift is detected.

Run
---
`python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload`

"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from typing import Dict, Any, List

from .schemas import PredictionRequest, PredictionResponse, HealthResponse
from .inference import PricingInference
from .auth import get_current_user, verify_ip_address
from pricing_rf.config import Config
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
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize configuration and inference
config = Config()
inference = PricingInference(config)

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

# Startup event to initialize monitoring
@app.on_event("startup")
async def startup_event():
    """Initialize monitoring service with reference data on startup.

    Attempts to load reference data from `monitoring_config.reference_data_path`
    and construct drift detectors. Logs success/failure; does not raise.
    """
    try:
        await setup_monitoring(monitoring_config.reference_data_path)
        logging.info("Monitoring service initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize monitoring service: {e}")

# Dependency for authentication (optional)
def get_auth_dependency():
    """Return an auth dependency when enabled, otherwise a no-op.

    Behavior
    --------
    - If `SERVICE_ENABLE_AUTH=true`, returns `Depends(get_current_user)` which
      validates JWTs and makes `user` available to route handlers.
    - Otherwise returns a lambda that does nothing, keeping endpoints open.

    Notes
    -----
    - IP allowlisting is enforced ad hoc within endpoints when auth is enabled.
    """
    if config.service.enable_auth:
        return Depends(get_current_user)
    else:
        return lambda: None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Liveness/readiness probe for the service.

    Returns
    -------
    HealthResponse
        - `status`: "healthy" if the process is running
        - `model_loaded`: whether the inference layer has a model in memory
        - `version`: API semantic version
    """
    return HealthResponse(
        status="healthy",
        model_loaded=inference.is_model_loaded(),
        version="0.1.0"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_price(
    request: PredictionRequest,
    user: Any = Depends(get_auth_dependency)
):
    """Predict a single price and run drift monitoring on the payload.

    Request
    -------
    PredictionRequest
        - `features`: dict of feature name → value
        - `client_ip` (optional): used for IP allowlisting when auth enabled

    Response
    --------
    PredictionResponse
        - `prediction`: float price estimate
        - `model_version`: string version from the inference layer
        - `confidence`: optional/confidence score if provided by the model

    Errors
    ------
    500
        On unexpected inference/validation/monitoring errors (logged).

    Notes
    -----
    - If drift is detected, the event is logged (and optionally alerted), but
      the prediction is still returned to the caller.
    """
    try:
        # Verify IP address if auth is enabled
        if config.service.enable_auth:
            verify_ip_address(request.client_ip)
        
        # Monitor for drift
        monitoring_result = await monitoring_service.monitor_prediction(
            request.features,
            prediction_id=f"pred_{hash(str(request.features))}"
        )
        
        # Make prediction
        prediction = inference.predict(request.features)
        
        # Create response
        response = PredictionResponse(
            prediction=prediction,
            model_version=inference.get_model_version(),
            confidence=inference.get_confidence_score()
        )
        
        # Add monitoring info to logs if drift detected
        if monitoring_result.get("drift_detected", False):
            logging.warning(f"Data drift detected in prediction: {monitoring_result}")
        
        return response
    
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch", response_model=List[PredictionResponse])
async def predict_batch(
    requests: List[PredictionRequest],
    user: Any = Depends(get_auth_dependency)
):
    """Predict prices for a batch of payloads and run a batch drift check.

    Request
    -------
    List[PredictionRequest]
        Each element matches the single-prediction schema.

    Response
    --------
    List[PredictionResponse]
        One response per request, preserving order.

    Notes
    -----
    - A single batch drift check is run across the concatenated features.
    - Drift detection does not alter returned predictions; alerts/logs are
      emitted out-of-band if drift is detected.
    """
    try:
        predictions = []
        features_list = []
        
        # Collect features for batch monitoring
        for request in requests:
            # Verify IP address if auth is enabled
            if config.service.enable_auth:
                verify_ip_address(request.client_ip)
            
            features_list.append(request.features)
        
        # Monitor batch for drift
        monitoring_result = await monitoring_service.monitor_batch_predictions(
            features_list,
            batch_id=f"batch_{hash(str(features_list))}"
        )
        
        # Make predictions
        for request in requests:
            prediction = inference.predict(request.features)
            
            predictions.append(PredictionResponse(
                prediction=prediction,
                model_version=inference.get_model_version(),
                confidence=inference.get_confidence_score()
            ))
        
        # Log drift detection
        if monitoring_result.get("drift_detected", False):
            logging.warning(f"Data drift detected in batch: {monitoring_result}")
        
        return predictions
    
    except Exception as e:
        logging.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model_info")
async def get_model_info(user: Any = Depends(get_auth_dependency)):
    """Return model metadata for operational visibility.

    Returns
    -------
    JSON object containing:
      - `model_name`    : MLflow registry name (from config)
      - `model_version` : current loaded version/alias
      - `feature_names` : list of expected features
      - `is_loaded`     : whether the model is in memory
    """
    return {
        "model_name": config.service.model_name,
        "model_version": inference.get_model_version(),
        "feature_names": inference.get_feature_names(),
        "is_loaded": inference.is_model_loaded()
    }


@app.get("/metrics")
async def get_metrics(user: Any = Depends(get_auth_dependency)):
    """Return training/evaluation metrics reported by the inference layer.

    Notes
    -----
    - The exact metric set depends on how the model was trained and logged.
      Typical fields include WAPE/MAE/RMSE/R², tail MAE, and coverage stats.
    """
    return inference.get_metrics()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "app:app",
        host=config.service.host,
        port=config.service.port,
        reload=config.service.debug
    )

