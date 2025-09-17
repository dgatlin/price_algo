"""Monitoring service integrating drift detection with FastAPI endpoints.

This module provides an application-level service (`MonitoringService`) that
coordinates:
  - Loading reference data and initializing Alibi Detect-based drift detectors.
  - Throttled, inference-time drift checks for single predictions or batches.
  - Result aggregation, history tracking, summaries, and optional alerting.
  - A simple interface consumed by FastAPI routers to expose REST endpoints.

Typical lifecycle
-----------------
1) Construct with a validated `MonitoringConfig`.
2) Call `initialize()` at app startup to load reference data and build detectors.
3) For each request:
   - Optionally gate execution via `should_check_drift()` (handled inside).
   - Run `monitor_prediction()` or `monitor_batch_predictions()`.
4) Expose status/history via `get_monitoring_status()` and `get_drift_history()`.
5) Update settings safely at runtime using `update_config()`.

Notes
-----
- The service is **async-friendly** where IO/alerts may be awaited.
- Frequency gating is delegated to the underlying `DriftDetector`.
- Alert delivery is a stub; integrate with Slack/Email/PagerDuty as needed.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime

from .drift_detector import DriftDetector, DriftResult
from .monitoring_config import MonitoringConfig


logger = logging.getLogger(__name__)


class MonitoringService:
    """Coordinate real-time drift monitoring for model predictions.

    Responsibilities
    ----------------
    - Hold the effective `MonitoringConfig` and a `DriftDetector` instance.
    - Initialize detectors from reference data (CSV/Parquet path).
    - Perform overall and per-feature drift checks on incoming payloads.
    - Maintain rolling history and provide summaries for dashboards/alerts.
    - Allow runtime enable/disable and configuration updates.

    Parameters
    ----------
    config : MonitoringConfig
        Validated configuration controlling detectors, frequency, logging,
        persistence, and alert behavior.

    Attributes
    ----------
    config : MonitoringConfig
        Current configuration (can be updated via `update_config`).
    drift_detector : DriftDetector
        Underlying detector façade used to run the tests.
    is_initialized : bool
        True once reference data is successfully loaded and detectors created.
    monitoring_enabled : bool
        Flag checked before attempting any drift work.
    """
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.drift_detector = DriftDetector(config)
        self.is_initialized = False
        self.monitoring_enabled = True
        
    async def initialize(self, reference_data_path: Optional[str] = None) -> bool:
        """Load reference data and construct detectors.

        Uses the provided `reference_data_path` if given; otherwise falls back
        to `config.reference_data_path`. If neither is available, monitoring is
        disabled and the method returns `False`.

        Parameters
        ----------
        reference_data_path : Optional[str]
            Path to a CSV/Parquet file containing the baseline distribution.

        Returns
        -------
        bool
            True if initialization succeeded; False otherwise.
        """
        try:
            if reference_data_path:
                self.drift_detector.load_reference_data_from_file(reference_data_path)
            elif self.config.reference_data_path:
                self.drift_detector.load_reference_data_from_file(self.config.reference_data_path)
            else:
                logger.warning("No reference data path provided. Monitoring will be disabled.")
                self.monitoring_enabled = False
                return False
            
            self.is_initialized = True
            logger.info("Monitoring service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize monitoring service: {e}")
            self.monitoring_enabled = False
            return False
    
    async def monitor_prediction(
        self, 
        features: Dict[str, Any],
        prediction_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate drift for a single feature payload.

        Applies frequency gating via the underlying detector. If monitoring is
        disabled or uninitialized, returns a minimal response indicating that no
        check was performed.

        Parameters
        ----------
        features : Dict[str, Any]
            Flat mapping from feature name to value (model-ready payload).
        prediction_id : Optional[str]
            Optional identifier to annotate logs/alerts.

        Returns
        -------
        Dict[str, Any]
            A JSON-serializable report with top-level flags, timestamps, and
            nested sections for overall and per-feature results.
        """
        if not self.monitoring_enabled or not self.is_initialized:
            return {"monitoring_enabled": False, "drift_detected": False}
        
        try:
            # Convert features to DataFrame
            df = pd.DataFrame([features])
            
            # Check if we should run drift detection
            if not self.drift_detector.should_check_drift():
                return {"monitoring_enabled": True, "drift_detected": False, "check_skipped": True}
            
            # Detect drift
            drift_results = self.drift_detector.detect_drift(df)
            
            # Check if any drift was detected
            drift_detected = any(result.is_drift for result in drift_results)
            
            # Detect feature-specific drift
            feature_drift_results = self.drift_detector.detect_feature_drift(df)
            feature_drift_detected = any(
                any(result.is_drift for result in results) 
                for results in feature_drift_results.values()
            )
            
            # Prepare monitoring result
            monitoring_result = {
                "monitoring_enabled": True,
                "drift_detected": drift_detected or feature_drift_detected,
                "prediction_id": prediction_id,
                "timestamp": datetime.now().isoformat(),
                "overall_drift": {
                    "detected": drift_detected,
                    "results": [result.to_dict() for result in drift_results]
                },
                "feature_drift": {
                    "detected": feature_drift_detected,
                    "results": {
                        feature: [result.to_dict() for result in results]
                        for feature, results in feature_drift_results.items()
                    }
                }
            }
            
            # Log drift detection
            if drift_detected or feature_drift_detected:
                logger.warning(f"Data drift detected in prediction {prediction_id}")
                if self.config.alert_on_drift:
                    await self._send_drift_alert(monitoring_result)
            
            return monitoring_result
            
        except Exception as e:
            logger.error(f"Error in drift monitoring: {e}")
            return {
                "monitoring_enabled": True,
                "drift_detected": False,
                "error": str(e)
            }
    
    async def monitor_batch_predictions(
        self, 
        features_list: List[Dict[str, Any]],
        batch_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate drift for a batch of feature payloads.

        Parameters
        ----------
        features_list : List[Dict[str, Any]]
            List of flat feature mappings (one per prediction).
        batch_id : Optional[str]
            Optional identifier used for logging and alerting.

        Returns
        -------
        Dict[str, Any]
            A JSON-serializable report including detection flags, batch size,
            timestamps, and the same nested structure used by
            `monitor_prediction` for overall/feature sections.
        """
        if not self.monitoring_enabled or not self.is_initialized:
            return {"monitoring_enabled": False, "drift_detected": False}
        
        try:
            # Convert features to DataFrame
            df = pd.DataFrame(features_list)
            
            # Check if we should run drift detection
            if not self.drift_detector.should_check_drift():
                return {"monitoring_enabled": True, "drift_detected": False, "check_skipped": True}
            
            # Detect drift on the entire batch
            drift_results = self.drift_detector.detect_drift(df)
            
            # Detect feature-specific drift
            feature_drift_results = self.drift_detector.detect_feature_drift(df)
            
            # Check if any drift was detected
            drift_detected = any(result.is_drift for result in drift_results)
            feature_drift_detected = any(
                any(result.is_drift for result in results) 
                for results in feature_drift_results.values()
            )
            
            # Prepare monitoring result
            monitoring_result = {
                "monitoring_enabled": True,
                "drift_detected": drift_detected or feature_drift_detected,
                "batch_id": batch_id,
                "batch_size": len(features_list),
                "timestamp": datetime.now().isoformat(),
                "overall_drift": {
                    "detected": drift_detected,
                    "results": [result.to_dict() for result in drift_results]
                },
                "feature_drift": {
                    "detected": feature_drift_detected,
                    "results": {
                        feature: [result.to_dict() for result in results]
                        for feature, results in feature_drift_results.items()
                    }
                }
            }
            
            # Log drift detection
            if drift_detected or feature_drift_detected:
                logger.warning(f"Data drift detected in batch {batch_id}")
                if self.config.alert_on_drift:
                    await self._send_drift_alert(monitoring_result)
            
            return monitoring_result
            
        except Exception as e:
            logger.error(f"Error in batch drift monitoring: {e}")
            return {
                "monitoring_enabled": True,
                "drift_detected": False,
                "error": str(e)
            }
    
    async def _send_drift_alert(self, monitoring_result: Dict[str, Any]) -> None:
        """Emit an alert when drift is detected (integration hook).

        The default implementation logs a warning with the serialized details.
        Replace or extend with calls to your incident/notification stack
        (Slack, email, PagerDuty, Datadog events, etc.).

        Parameters
        ----------
        monitoring_result : Dict[str, Any]
            The report returned by `monitor_prediction`/`monitor_batch_predictions`.
        """
        try:
            # Log the alert
            logger.warning("DRIFT ALERT: Data drift detected in prediction")
            logger.warning(f"Drift details: {monitoring_result}")
            
            # Here you could integrate with external alerting systems
            # e.g., send to Slack, email, or monitoring dashboard
            
        except Exception as e:
            logger.error(f"Failed to send drift alert: {e}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Return enablement flags, initialization state, summary, and config.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing:
              - `monitoring_enabled`
              - `is_initialized`
              - `drift_summary` (from `DriftDetector.get_drift_summary()`)
              - `config` (serialized via `model_dump()`)
        """
        return {
            "monitoring_enabled": self.monitoring_enabled,
            "is_initialized": self.is_initialized,
            "drift_summary": self.drift_detector.get_drift_summary(),
            "config": self.config.model_dump()
        }
    
    def get_drift_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return serialized drift history entries.

        Parameters
        ----------
        limit : Optional[int]
            Maximum number of most recent entries to return. If `None`, returns
            the full history.

        Returns
        -------
        List[Dict[str, Any]]
            List of JSON-serializable `DriftResult` dicts.
        """
        history = self.drift_detector.drift_history
        if limit:
            history = history[-limit:]
        return [result.to_dict() for result in history]
    
    def reset_monitoring(self) -> None:
        """Clear counters and history (configuration is preserved)."""
        self.drift_detector.prediction_count = 0
        self.drift_detector.drift_history = []
        logger.info("Monitoring state reset")
    
    def enable_monitoring(self) -> None:
        """Enable monitoring checks without altering configuration."""
        self.monitoring_enabled = True
        logger.info("Monitoring enabled")
    
    def disable_monitoring(self) -> None:
        """Disable monitoring checks without altering configuration."""
        self.monitoring_enabled = False
        logger.info("Monitoring disabled")
    
    def update_config(self, new_config: MonitoringConfig) -> None:
        """Replace the active configuration and reinitialize the detector.

        Rebuilds the `DriftDetector` with the new settings. If the service was
        already initialized, schedules an async `initialize()` call to reload
        reference data using the new config.

        Parameters
        ----------
        new_config : MonitoringConfig
            The replacement configuration.
        """
        self.config = new_config
        # Reinitialize detector with new config
        self.drift_detector = DriftDetector(new_config)
        if self.is_initialized:
            # Reinitialize with existing reference data
            asyncio.create_task(self.initialize())
        logger.info("Monitoring configuration updated")
