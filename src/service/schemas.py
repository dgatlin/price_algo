"""Pydantic schemas for request/response models.

These data models define the wire format for the Pricing RF FastAPI service.
They provide concise field descriptions, example payloads for OpenAPI/Swagger,
and stable shapes for clients.

Models
------
- `PredictionRequest`  : Input payload for single/batch predictions.
- `PredictionResponse` : Output payload containing the prediction and metadata.
- `HealthResponse`     : Lightweight service liveness/readiness info.
- `ModelInfo`          : Introspection of the currently loaded model.
- `MetricsResponse`    : Training/evaluation metrics surfaced by the service.

Notes
-----
- All examples are rendered in the auto-generated OpenAPI docs at `/docs`.
- `client_ip` is optional; when auth/IP allowlisting is enabled, the service
  may use it to enforce access policies (falling back to request metadata).
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional


class PredictionRequest(BaseModel):
    """Schema for a price prediction request.

    Fields
    ------
    features :
        Flat mapping of feature name → value. Keys should match the model’s
        expected feature set. Types may be numeric or strings for categoricals.
    client_ip :
        Optional caller IP used for IP allowlisting when enabled.

    Example
    -------
    {
      "features": {
        "feature1": 1.5,
        "feature2": 2.3,
        "feature3": 0.8,
        "category_feature": "category_a"
      },
      "client_ip": "203.0.113.42"
    }
    """
    features: Dict[str, Any] = Field(
        ...,
        description="Feature values for prediction",
        example={
            "feature1": 1.5,
            "feature2": 2.3,
            "feature3": 0.8,
            "category_feature": "category_a"
        }
    )
    client_ip: Optional[str] = Field(
        None,
        description="Client IP address (optional; used for IP allowlisting)"
    )


class PredictionResponse(BaseModel):
    """Schema for a price prediction response.

    Fields
    ------
    prediction :
        Predicted price as a floating-point number (original currency units).
    model_version :
        Model registry version/alias used to produce the result.
    confidence :
        Optional confidence/uncertainty score if provided by the model.

    Example
    -------
    {
      "prediction": 123.45,
      "model_version": "12",
      "confidence": null
    }
    """
    prediction: float = Field(
        ...,
        description="Predicted price value"
    )
    model_version: str = Field(
        ...,
        description="Version of the model used for prediction"
    )
    confidence: Optional[float] = Field(
        None,
        description="Confidence score for the prediction (if available)"
    )


class HealthResponse(BaseModel):
    """Schema for the `/health` endpoint.

    Fields
    ------
    status :
        String status indicator (e.g., 'healthy').
    model_loaded :
        Whether the inference layer has a model in memory.
    version :
        Semantic version of the API.

    Example
    -------
    { "status": "healthy", "model_loaded": true, "version": "0.1.0" }
    """
    status: str = Field(
        ...,
        description="Service status"
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the model is loaded"
    )
    version: str = Field(
        ...,
        description="API version"
    )


class ModelInfo(BaseModel):
    """Schema describing the currently loaded model.

    Fields
    ------
    model_name :
        MLflow registry name for the model.
    model_version :
        Resolved numeric version or alias (e.g., 'Production').
    feature_names :
        Ordered list of feature names expected at inference time.
    is_loaded :
        Whether the model and preprocessor are ready to serve.

    Example
    -------
    {
      "model_name": "pricing-rf",
      "model_version": "Production",
      "feature_names": ["feature1", "feature2", "feature3", "category_feature"],
      "is_loaded": true
    }
    """
    model_name: str = Field(
        ...,
        description="Name of the model"
    )
    model_version: str = Field(
        ...,
        description="Version of the model"
    )
    feature_names: List[str] = Field(
        ...,
        description="List of feature names expected by the model"
    )
    is_loaded: bool = Field(
        ...,
        description="Whether the model is loaded"
    )


class MetricsResponse(BaseModel):
    """Schema for training/evaluation metrics returned by `/metrics`.

    Fields
    ------
    metrics :
        Mapping of metric name → value (e.g., WAPE, MAE, RMSE, R²).
    last_updated :
        ISO-8601 timestamp (UTC) indicating when metrics were last refreshed.

    Example
    -------
    {
      "metrics": { "wape": 7.5, "mae": 12.3, "rmse": 20.1, "r2": 0.91 },
      "last_updated": "2024-06-01T12:00:00Z"
    }
    """
    metrics: Dict[str, float] = Field(
        ...,
        description="Model performance metrics"
    )
    last_updated: str = Field(
        ...,
        description="When metrics were last updated (ISO-8601)"
    )
