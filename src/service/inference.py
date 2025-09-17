"""MLflow-backed model loading and inference for the pricing service.

This module implements a thin inference layer that:
  - Connects to an MLflow Tracking/Registry server.
  - Loads a registered Random Forest model and its preprocessing pipeline.
  - Retrieves auxiliary artifacts (feature names, metrics) from the linked run.
  - Exposes single/batch prediction methods used by the FastAPI app.

Design notes
------------
- **Registry URIs**: Models are loaded from the MLflow Model Registry using
  URIs of the form `models:/<name>/<version-or-stage>`. The preprocessor is
  assumed to be registered separately as `<name>-preprocessor`.
- **Feature validation**: If `feature_info.json` is present in run artifacts,
  we enforce that incoming payloads include all expected features.
- **Metrics & metadata**: The class attempts to load `metrics.json` to expose
  service-level metrics via `/metrics` without hard-coding training details.
- **Resilience**: Failures to load auxiliary artifacts (features/metrics) do not
  prevent predictions if the core model and preprocessor load successfully.

Artifacts convention (optional)
-------------------------------
- `feature_info.json`:
    {
      "feature_names": ["feat_a", "feat_b", ...]
    }
- `metrics.json`:
    {
      "wape": 7.5, "mae": 12.3, "...": ...
    }
"""

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional
import json
from pathlib import Path

from pricing_rf.config import Config

# Optional imports used in artifact resolution
from mlflow.tracking import MlflowClient
from mlflow import artifacts as mlflow_artifacts


class PricingInference:
    """Pricing model inference facade backed by MLflow.

    Responsibilities
    ----------------
    - Initialize MLflow tracking from `ServiceConfig`.
    - Load a registered model + preprocessor via registry URIs.
    - Resolve the underlying run to fetch `feature_info.json` and `metrics.json`.
    - Provide `predict` and `predict_batch` with basic feature checks.

    Parameters
    ----------
    config : Config
        Aggregated settings including MLflow URIs and model registry targets.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.preprocessor = None
        self.feature_names: Optional[List[str]] = None
        self.model_version: Optional[str] = None  # resolved numeric version when possible
        self.metrics: Dict[str, Any] = {}
        self._last_metrics_updated: Optional[str] = None
        
        # Configure MLflow client
        mlflow.set_tracking_uri(config.service.mlflow_tracking_uri)
        
        # Load model & preprocessor
        self._load_model()
    
    # ----------------------------
    # Private helpers
    # ----------------------------
    def _load_model(self) -> None:
        """Load model and preprocessor from the MLflow Model Registry.

        Behavior
        --------
        - Loads the model registered under `service.model_name` at the
          requested `service.model_version` (can be a stage alias like
          "Production" or a numeric string).
        - Loads the associated preprocessor from `<model_name>-preprocessor`
          using the same version/stage selector.
        - Attempts to resolve the concrete numeric registry version and run_id
          to fetch artifacts for features/metrics.
        """
        try:
            # Build registry URIs (allow stage aliases like "Production")
            model_uri = f"models:/{self.config.service.model_name}/{self.config.service.model_version}"
            preproc_uri = f"models:/{self.config.service.model_name}-preprocessor/{self.config.service.model_version}"

            # Load model & preprocessor via sklearn flavor
            self.model = mlflow.sklearn.load_model(model_uri)
            self.preprocessor = mlflow.sklearn.load_model(preproc_uri)

            # Try to resolve numeric version + run_id for artifacts
            run_id, resolved_version = self._resolve_run_and_version(
                self.config.service.model_name,
                self.config.service.model_version
            )
            if resolved_version:
                self.model_version = str(resolved_version)

            # Load feature names and metrics (best-effort)
            if run_id:
                self._load_feature_info(run_id)
                self._load_metrics(run_id)

            logging.info(
                f"Loaded model '{self.config.service.model_name}' "
                f"version='{self.model_version or self.config.service.model_version}'"
            )
        except Exception as e:
            logging.error(f"Failed to load model or preprocessor: {str(e)}")
            self.model = None
            self.preprocessor = None

    def _resolve_run_and_version(self, name: str, version_or_stage: str) -> (Optional[str], Optional[int]):
        """Resolve the model `run_id` and numeric `version` from the registry.

        Supports:
          - Stage aliases: "Production", "Staging", "None"
          - Numeric versions: "3", "12"
          - Non-standard values: best-effort fallback to latest Production

        Returns
        -------
        (run_id, version) : Tuple[Optional[str], Optional[int]]
        """
        client = MlflowClient()
        try:
            target = version_or_stage.strip()
            mv = None
            if target.isdigit():
                mv = client.get_model_version(name, target)
            else:
                # Treat as stage alias (case-insensitive)
                stage = target.capitalize()
                latest = client.get_latest_versions(name, stages=[stage])
                mv = latest[0] if latest else None

            # Fallback: try Production
            if mv is None:
                latest_prod = client.get_latest_versions(name, stages=["Production"])
                mv = latest_prod[0] if latest_prod else None

            return (mv.run_id, int(mv.version)) if mv else (None, None)
        except Exception as e:
            logging.warning(f"Could not resolve run/version for '{name}/{version_or_stage}': {e}")
            return (None, None)

    def _load_feature_info(self, run_id: str) -> None:
        """Load `feature_info.json` from the run artifacts (best-effort)."""
        try:
            feature_info_path = mlflow_artifacts.download_artifacts(
                run_id=run_id, artifact_path="feature_info.json"
            )
            with open(feature_info_path, "r") as f:
                payload = json.load(f)
            self.feature_names = payload.get("feature_names")
            if not isinstance(self.feature_names, list):
                raise ValueError("feature_names must be a list")
            logging.info(f"Loaded {len(self.feature_names)} feature names from artifacts.")
        except Exception as e:
            logging.warning(f"Could not load feature info: {str(e)}")
            self.feature_names = None

    def _load_metrics(self, run_id: str) -> None:
        """Load `metrics.json` from the run artifacts (best-effort)."""
        try:
            metrics_path = mlflow_artifacts.download_artifacts(
                run_id=run_id, artifact_path="metrics.json"
            )
            with open(metrics_path, "r") as f:
                self.metrics = json.load(f)
            # Use file modified time as a simple "last updated" indicator
            self._last_metrics_updated = Path(metrics_path).stat().st_mtime_ns and pd.Timestamp.utcnow().isoformat()
            logging.info("Loaded metrics from artifacts.")
        except Exception as e:
            logging.warning(f"No metrics.json found or failed to load: {e}")
            self.metrics = {}
            self._last_metrics_updated = None

    # ----------------------------
    # Public API
    # ----------------------------
    def predict(self, features: Dict[str, Any]) -> float:
        """Return a single price prediction for a feature mapping.

        Steps
        -----
        1) Validate that a model & preprocessor are loaded.
        2) Convert the `features` dict into a single-row DataFrame.
        3) If `feature_names` are known, ensure all required features are present.
        4) Transform via the preprocessor and predict with the model.

        Parameters
        ----------
        features : Dict[str, Any]
            Flat mapping of feature name → value.

        Returns
        -------
        float
            Predicted price.
        """
        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded")
        
        df = pd.DataFrame([features])
        
        # Ensure all required features are present (if we know them)
        if self.feature_names:
            missing = set(self.feature_names) - set(df.columns)
            if missing:
                raise ValueError(f"Missing features: {sorted(missing)}")
            # Reorder to match training order if necessary
            df = df.reindex(columns=self.feature_names, fill_value=np.nan)
        
        X_processed = self.preprocessor.transform(df)
        prediction = self.model.predict(X_processed)[0]
        return float(prediction)
    
    def predict_batch(self, features_list: List[Dict[str, Any]]) -> List[float]:
        """Return predictions for a list of feature mappings.

        Parameters
        ----------
        features_list : List[Dict[str, Any]]
            Each element is a dict compatible with `predict`.

        Returns
        -------
        List[float]
            Predicted prices in the same order as the inputs.
        """
        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded")
        
        df = pd.DataFrame(features_list)

        # If we know the expected feature set, align columns
        if self.feature_names:
            missing = set(self.feature_names) - set(df.columns)
            if missing:
                raise ValueError(f"Missing features in batch: {sorted(missing)}")
            df = df.reindex(columns=self.feature_names, fill_value=np.nan)
        
        X_processed = self.preprocessor.transform(df)
        predictions = self.model.predict(X_processed)
        return predictions.tolist()
    
    def is_model_loaded(self) -> bool:
        """Return True if both model and preprocessor are available."""
        return self.model is not None and self.preprocessor is not None
    
    def get_model_version(self) -> str:
        """Return the resolved numeric model version or configured selector."""
        return self.model_version or str(self.config.service.model_version) or "unknown"
    
    def get_feature_names(self) -> List[str]:
        """Return the known feature names (empty list if unavailable)."""
        return list(self.feature_names) if self.feature_names else []
    
    def get_confidence_score(self) -> Optional[float]:
        """Placeholder for uncertainty/interval scoring (None by default).

        Extend by:
        - Training quantile models and returning interval width/coverage.
        - Using conformal prediction residuals to compute a calibrated score.
        - Ensembling and reporting variance-based heuristics.
        """
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Return cached training/evaluation metrics and a last-updated stamp.

        Returns
        -------
        Dict[str, Any]
            {
              "metrics": {...},         # may be empty if no artifact found
              "last_updated": "<iso>",  # None if unknown
            }
        """
        return {
            "metrics": self.metrics,
            "last_updated": self._last_metrics_updated or "unknown"
        }
    
    def reload_model(self) -> None:
        """Reload the model, preprocessor, and artifacts from the registry."""
        self._load_model()

