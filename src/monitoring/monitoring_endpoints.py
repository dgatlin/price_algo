"""FastAPI router exposing monitoring and drift-detection endpoints.

This module wires the monitoring subsystem into a REST surface under the
`/monitoring` prefix. It provides read endpoints for status and history, plus
write endpoints to trigger drift checks, adjust configuration at runtime, and
toggle/reset the monitoring state.

Overview
--------
- **Status & Health**
  - `GET /monitoring/status` — current state, summary, and active config
  - `GET /monitoring/health` — lightweight health indicator
- **History**
  - `GET /monitoring/drift/history` — recent drift checks and counts
- **On-demand checks**
  - `POST /monitoring/drift/check` — run drift on a single feature payload
  - `POST /monitoring/drift/check_batch` — run drift on a list of payloads
- **Operations**
  - `POST /monitoring/config/update` — partial config update (pydantic-validated)
  - `POST /monitoring/reset` — clear state/history (keeps config)
  - `POST /monitoring/enable` / `POST /monitoring/disable` — toggle monitoring

Notes
-----
- This router assumes an application-level initializer calls
  `initialize_monitoring_service()` and (optionally) `setup_monitoring()` during
  startup. Until then, endpoints will return 503 for service access.
- Authentication/authorization is expected to be enforced at the app level
  (e.g., dependency-injected JWT guard) and is not implemented here.
"""

import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from .monitoring_service import MonitoringService
from .monitoring_config import MonitoringConfig
from .drift_detector import DriftResult


logger = logging.getLogger(__name__)

# Create router for monitoring endpoints
router = APIRouter(prefix="/monitoring", tags=["monitoring"])

# Global monitoring service instance
monitoring_service: Optional[MonitoringService] = None


def get_monitoring_service() -> MonitoringService:
    """Return the initialized `MonitoringService` or raise 503 if missing.

    This dependency is used by all endpoints to access the monitoring
    subsystem. The application must call `initialize_monitoring_service()`
    during startup to set the global instance.
    """
    if monitoring_service is None:
        raise HTTPException(status_code=503, detail="Monitoring service not initialized")
    return monitoring_service


class MonitoringStatusResponse(BaseModel):
    """Schema describing the overall monitoring status and active configuration."""

    monitoring_enabled: bool = Field(..., description="Whether monitoring is currently enabled")
    is_initialized: bool = Field(..., description="Whether the service has loaded reference data and detectors")
    drift_summary: Dict[str, Any] = Field(..., description="Aggregated counts and latest detected drift events")
    config: Dict[str, Any] = Field(..., description="Effective monitoring configuration (serialized)")


class DriftHistoryResponse(BaseModel):
    """Schema for drift history and high-level counts."""

    drift_history: List[Dict[str, Any]] = Field(..., description="Chronological list of drift check results")
    total_checks: int = Field(..., description="Total number of drift checks performed (lifetime or since reset)")
    drift_detected: int = Field(..., description="Number of drift-positive checks")


class MonitoringConfigUpdate(BaseModel):
    """Partial update model for runtime configuration changes.

    Only provided fields are applied; unspecified fields retain existing values.
    """

    drift_threshold: Optional[float] = Field(None, description="p-value threshold for drift detection")
    monitoring_frequency: Optional[int] = Field(None, description="Run drift checks every N predictions")
    alert_on_drift: Optional[bool] = Field(None, description="Emit alerts when drift is detected")
    feature_drift_enabled: Optional[bool] = Field(None, description="Enable per-feature drift checks")


@router.get("/status", response_model=MonitoringStatusResponse)
async def get_monitoring_status(
    service: MonitoringService = Depends(get_monitoring_service)
):
    """Return current monitoring status, summary, and effective configuration.

    Responses
    ---------
    200
        `MonitoringStatusResponse` including enablement flags, a summary of
        drift results, and the active config serialized as a dict.
    500
        Unexpected server error (see logs for details).
    """
    try:
        status = service.get_monitoring_status()
        return MonitoringStatusResponse(**status)
    except Exception as e:
        logger.error(f"Error getting monitoring status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/drift/history", response_model=DriftHistoryResponse)
async def get_drift_history(
    limit: Optional[int] = None,
    service: MonitoringService = Depends(get_monitoring_service)
):
    """Return recent drift events and counters.

    Parameters
    ----------
    limit :
        Optional maximum number of history entries to return (most recent first).

    Responses
    ---------
    200
        `DriftHistoryResponse` containing `drift_history`, `total_checks`, and
        `drift_detected`.
    500
        Unexpected server error.
    """
    try:
        history = service.get_drift_history(limit=limit)
        summary = service.drift_detector.get_drift_summary()

        return DriftHistoryResponse(
            drift_history=history,
            total_checks=summary["total_checks"],
            drift_detected=summary["drift_detected"]
        )
    except Exception as e:
        logger.error(f"Error getting drift history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/drift/check")
async def check_drift(
    features: Dict[str, Any],
    service: MonitoringService = Depends(get_monitoring_service)
):
    """Run drift detection on a single feature payload.

    Request body
    ------------
    `features` : object
        A flat mapping of feature name → value (e.g., the same structure used
        by the pricing model). The monitoring service will preprocess this
        payload to match the reference schema before testing.

    Responses
    ---------
    200
        Drift evaluation result (serializable dict/list of `DriftResult`).
    500
        Unexpected server error.
    """
    try:
        result = await service.monitor_prediction(features)
        return result
    except Exception as e:
        logger.error(f"Error checking drift: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/drift/check_batch")
async def check_drift_batch(
    features_list: List[Dict[str, Any]],
    service: MonitoringService = Depends(get_monitoring_service)
):
    """Run drift detection on a batch of feature payloads.

    Request body
    ------------
    `features_list` : array of objects
        Each element is a feature mapping as described in `/drift/check`.

    Responses
    ---------
    200
        Aggregated drift results for the batch.
    500
        Unexpected server error.
    """
    try:
        result = await service.monitor_batch_predictions(features_list)
        return result
    except Exception as e:
        logger.error(f"Error checking batch drift: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/config/update")
async def update_monitoring_config(
    config_update: MonitoringConfigUpdate,
    service: MonitoringService = Depends(get_monitoring_service)
):
    """Apply a partial update to the monitoring configuration at runtime.

    Behavior
    --------
    - Reads the current `MonitoringConfig`.
    - Applies only fields provided in `config_update`.
    - Reconstructs a new `MonitoringConfig` instance and applies it to the service.

    Responses
    ---------
    200
        JSON with `"message"` and the updated `"config"`.
    500
        Unexpected server error or validation failure.
    """
    try:
        # Get current config
        current_config = service.config

        # Update config with provided values
        config_dict = current_config.dict()
        for field, value in config_update.dict(exclude_unset=True).items():
            if value is not None:
                config_dict[field] = value

        # Create new config
        new_config = MonitoringConfig(**config_dict)

        # Update service
        service.update_config(new_config)

        return {"message": "Configuration updated successfully", "config": new_config.dict()}
    except Exception as e:
        logger.error(f"Error updating monitoring config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset")
async def reset_monitoring(
    service: MonitoringService = Depends(get_monitoring_service)
):
    """Clear monitoring state and historical results (keeps current config)."""
    try:
        service.reset_monitoring()
        return {"message": "Monitoring state reset successfully"}
    except Exception as e:
        logger.error(f"Error resetting monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enable")
async def enable_monitoring(
    service: MonitoringService = Depends(get_monitoring_service)
):
    """Enable monitoring without changing configuration or history."""
    try:
        service.enable_monitoring()
        return {"message": "Monitoring enabled"}
    except Exception as e:
        logger.error(f"Error enabling monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/disable")
async def disable_monitoring(
    service: MonitoringService = Depends(get_monitoring_service)
):
    """Disable monitoring without clearing configuration or history."""
    try:
        service.disable_monitoring()
        return {"message": "Monitoring disabled"}
    except Exception as e:
        logger.error(f"Error disabling monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def monitoring_health(
    service: MonitoringService = Depends(get_monitoring_service)
):
    """Health check for the monitoring subsystem.

    Returns
    -------
    JSON object with:
      - `"status"`: `"healthy"` if initialized, else `"unhealthy"`.
      - `"monitoring_enabled"`: bool
      - `"is_initialized"`: bool
    """
    try:
        status = service.get_monitoring_status()
        return {
            "status": "healthy" if status["is_initialized"] else "unhealthy",
            "monitoring_enabled": status["monitoring_enabled"],
            "is_initialized": status["is_initialized"]
        }
    except Exception as e:
        logger.error(f"Error in monitoring health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def initialize_monitoring_service(config: MonitoringConfig) -> MonitoringService:
    """Create and register the global `MonitoringService` instance.

    Call this during application startup (e.g., in a FastAPI lifespan handler)
    before serving requests to ensure dependencies resolve.

    Parameters
    ----------
    config : MonitoringConfig
        Validated configuration for the monitoring subsystem.

    Returns
    -------
    MonitoringService
        The initialized service (also stored in a module-global).
    """
    global monitoring_service
    monitoring_service = MonitoringService(config)
    return monitoring_service


async def setup_monitoring(reference_data_path: Optional[str] = None) -> bool:
    """Load reference data and initialize detectors (optional startup helper).

    Parameters
    ----------
    reference_data_path :
        Optional path to a CSV/Parquet reference dataset. If omitted, the
        service may use `config.reference_data_path` or defer initialization.

    Returns
    -------
    bool
        True if initialization succeeded, False otherwise.
    """
    global monitoring_service
    if monitoring_service is None:
        logger.error("Monitoring service not initialized")
        return False

    return await monitoring_service.initialize(reference_data_path)
