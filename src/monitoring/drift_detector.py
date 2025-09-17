"""Drift detection utilities built on Alibi Detect.

This module provides a configurable, production-friendly component for monitoring
data drift at inference time. Incoming feature batches are statistically compared
to a fixed *reference* distribution using one or more detectors from
`alibi-detect` (currently Kolmogorov–Smirnov for univariate drift and
Maximum Mean Discrepancy for multivariate drift).

Key capabilities
----------------
- **Pluggable tests:** KSDrift (univariate) and MMDDrift (multivariate).
- **Flexible inputs:** Accepts `pandas.DataFrame` or `numpy.ndarray`; applies
  simple one-hot encoding for configured categorical features.
- **Runtime control:** Thresholds, frequency gating, and per-feature checks are
  configurable via `MonitoringConfig`.
- **Persistence:** Save/load detectors and configuration to/from disk.
- **Observability:** Structured results with timestamps, rolling history,
  and summaries suitable for alerting, dashboards, or audit logs.

Typical usage
-------------
1) Instantiate `DriftDetector(config)`.
2) Call `set_reference_data(df_or_array)` to establish the baseline window.
3) On a schedule or every Nth request (see `should_check_drift()`), call
   `detect_drift(batch)` or `detect_feature_drift(batch)`.
4) Read `get_drift_summary()` and/or persist history for reporting.

Notes
-----
This component is intended to be called in the **read path** (inference) and
keeps CPU overhead low by allowing coarse frequency controls and lightweight,
batch-oriented tests.
"""

import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from datetime import datetime

from alibi_detect.cd import KSDrift, MMDDrift
from alibi_detect.base import BaseDetector
from alibi_detect.utils.saving import save_detector, load_detector

from .monitoring_config import MonitoringConfig


logger = logging.getLogger(__name__)


class DriftResult:
    """Structured result for a single drift test.

    Parameters
    ----------
    is_drift:
        Whether the test detected drift at the configured significance level.
    drift_score:
        Test-specific distance/statistic (e.g., MMD distance or KS statistic).
    p_value:
        P-value returned by the detector for this batch.
    method:
        Identifier of the test used (e.g., `"ks"` or `"mmd"`).
    feature_name:
        Optional name of the feature when running per-feature checks.
    threshold:
        Significance level used by the detector (e.g., 0.05).

    Attributes
    ----------
    timestamp : datetime
        Time when the result was computed (UTC/local per process settings).

    Methods
    -------
    to_dict()
        Serialize the result to a JSON-friendly dictionary.
    """

    def __init__(
        self,
        is_drift: bool,
        drift_score: float,
        p_value: float,
        method: str,
        feature_name: Optional[str] = None,
        threshold: float = 0.05
    ):
        self.is_drift = is_drift
        self.drift_score = drift_score
        self.p_value = p_value
        self.method = method
        self.feature_name = feature_name
        self.threshold = threshold
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation of the result."""
        return {
            "is_drift": self.is_drift,
            "drift_score": self.drift_score,
            "p_value": self.p_value,
            "method": self.method,
            "feature_name": self.feature_name,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat()
        }

    def __repr__(self) -> str:
        return f"DriftResult(is_drift={self.is_drift}, method={self.method}, p_value={self.p_value:.4f})"


class DriftDetector:
    """High-level drift detection façade using Alibi Detect.

    This class manages reference data, constructs the requested detectors
    (KS/MMD), runs drift checks on incoming batches, and records results and
    summaries for operational use.

    Parameters
    ----------
    config : MonitoringConfig
        Runtime configuration including:
        - `drift_methods` (e.g., `["ks", "mmd"]`)
        - `drift_threshold` (significance level)
        - `categorical_features`, `numerical_features`
        - `monitoring_frequency` (check every N predictions)
        - logging/persistence flags and paths.

    Attributes
    ----------
    reference_data : Optional[np.ndarray]
        The baseline window against which incoming data is compared.
    detectors : Dict[str, BaseDetector]
        Initialized alibi-detect detectors keyed by method name.
    prediction_count : int
        Counter used by `should_check_drift()` to throttle checks.
    drift_history : List[DriftResult]
        Chronological list of results for auditing and summaries.
    """

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.reference_data: Optional[np.ndarray] = None
        self.detectors: Dict[str, BaseDetector] = {}
        self.prediction_count = 0
        self.drift_history: List[DriftResult] = []

        # Initialize detectors
        self._initialize_detectors()

    def _initialize_detectors(self) -> None:
        """Log the configured methods; detectors are built when reference data is set.

        Alibi Detect requires `x_ref` at construction time; therefore actual
        detector instances are created in `_create_detectors()` after the
        reference data has been provided.
        """
        logger.info(f"Drift detection methods configured: {self.config.drift_methods}")

    def set_reference_data(self, data: Union[np.ndarray, pd.DataFrame]) -> None:
        """Set or replace the reference distribution and (re)build detectors.

        Accepts either a `DataFrame` (which will be preprocessed according to
        the configured categorical/numerical feature lists) or a `numpy` array.

        Parameters
        ----------
        data:
            Reference batch used to initialize detectors.

        Raises
        ------
        Exception
            If preprocessing or detector construction fails.
        """
        try:
            if isinstance(data, pd.DataFrame):
                data_processed = self._preprocess_data(data)
                self.reference_data = data_processed.values
            else:
                self.reference_data = data

            # Create and fit detectors with reference data
            self._create_detectors()

            logger.info(f"Set reference data with shape: {self.reference_data.shape}")

        except Exception as e:
            logger.error(f"Failed to set reference data: {e}")
            raise

    def _create_detectors(self) -> None:
        """Instantiate Alibi Detect detectors using the current reference data.

        Builds the detectors specified in `config.drift_methods`. Each detector
        is parameterized with the configured p-value threshold and any
        method-specific kwargs from the config (e.g., kernel settings for MMD).

        Raises
        ------
        Exception
            If detector construction fails.
        """
        try:
            if "ks" in self.config.drift_methods:
                self.detectors["ks"] = KSDrift(
                    x_ref=self.reference_data,
                    p_val=self.config.drift_threshold,
                    **self.config.ks_test_params
                )

            if "mmd" in self.config.drift_methods:
                self.detectors["mmd"] = MMDDrift(
                    x_ref=self.reference_data,
                    p_val=self.config.drift_threshold,
                    **self.config.mmd_test_params
                )

            logger.info(f"Created drift detectors: {list(self.detectors.keys())}")

        except Exception as e:
            logger.error(f"Failed to create drift detectors: {e}")
            raise

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Lightweight preprocessing for drift tests.

        One-hot encodes configured categorical features and retains only
        numerical columns plus generated one-hot columns. This keeps the
        drift tests aligned with the features you consider stable.

        Parameters
        ----------
        data:
            Input features as a `DataFrame`.

        Returns
        -------
        pd.DataFrame
            Processed numeric matrix suitable for the detectors.
        """
        data_processed = data.copy()

        # Handle categorical features
        for cat_feature in self.config.categorical_features:
            if cat_feature in data_processed.columns:
                dummies = pd.get_dummies(data_processed[cat_feature], prefix=cat_feature)
                data_processed = pd.concat([data_processed, dummies], axis=1)
                data_processed = data_processed.drop(columns=[cat_feature])

        # Select only numerical features
        numerical_cols = [col for col in data_processed.columns
                          if col in self.config.numerical_features or
                          col.startswith(tuple(self.config.categorical_features))]

        return data_processed[numerical_cols]

    def detect_drift(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        feature_name: Optional[str] = None
    ) -> List[DriftResult]:
        """Run configured detectors against a batch and return results.

        If `reference_data` has not been set, returns an empty list and logs a
        warning. When a `DataFrame` is provided, it is preprocessed to a numeric
        matrix prior to testing.

        Parameters
        ----------
        data:
            Incoming batch to test for drift.
        feature_name:
            Optional tag indicating a single feature is being tested; useful for
            annotating results when this method is called from
            `detect_feature_drift()`.

        Returns
        -------
        List[DriftResult]
            One result per configured detector.

        Notes
        -----
        This method appends results to `drift_history` and, if enabled by
        configuration, persists the rolling history to disk.
        """
        try:
            if isinstance(data, pd.DataFrame):
                data_processed = self._preprocess_data(data)
                data_array = data_processed.values
            else:
                data_array = data

            if self.reference_data is None:
                logger.warning("No reference data set. Cannot detect drift.")
                return []

            results = []

            for method, detector in self.detectors.items():
                try:
                    drift_result = detector.predict(data_array)

                    is_drift_raw = drift_result['data']['is_drift']
                    distance_raw = drift_result['data']['distance']
                    p_val_raw = drift_result['data']['p_val']

                    if isinstance(is_drift_raw, (int, bool)):
                        is_drift = bool(is_drift_raw)
                    else:
                        is_drift = bool(is_drift_raw[0]) if len(is_drift_raw) > 0 else False

                    if isinstance(distance_raw, (int, float)):
                        drift_score = float(distance_raw)
                    else:
                        drift_score = float(distance_raw[0]) if len(distance_raw) > 0 else 0.0

                    if isinstance(p_val_raw, (int, float)):
                        p_value = float(p_val_raw)
                    else:
                        p_value = float(p_val_raw[0]) if len(p_val_raw) > 0 else 1.0

                    result = DriftResult(
                        is_drift=is_drift,
                        drift_score=drift_score,
                        p_value=p_value,
                        method=method,
                        feature_name=feature_name,
                        threshold=self.config.drift_threshold
                    )

                    results.append(result)

                    if self.config.log_drift_results:
                        logger.info(
                            f"Drift detection - Method: {method}, "
                            f"Feature: {feature_name or 'all'}, "
                            f"Drift: {is_drift}, "
                            f"P-value: {p_value:.4f}"
                        )

                except Exception as e:
                    logger.error(f"Error in {method} drift detection: {e}")
                    continue

            self.drift_history.extend(results)

            if self.config.save_drift_results:
                self._save_drift_results()

            return results

        except Exception as e:
            logger.error(f"Failed to detect drift: {e}")
            return []

    def detect_feature_drift(
        self,
        data: Union[np.ndarray, pd.DataFrame]
    ) -> Dict[str, List[DriftResult]]:
        """Run drift tests independently for each configured numerical feature.

        Parameters
        ----------
        data:
            Incoming batch as `DataFrame` or `ndarray`. When a `DataFrame` is
            provided, preprocessing will be applied.

        Returns
        -------
        Dict[str, List[DriftResult]]
            Mapping from feature name to detector results. Empty dict if
            per-feature drift checks are disabled in configuration.

        Notes
        -----
        This method is useful when you need to localize which feature(s) are
        changing and driving the overall drift signal.
        """
        if not self.config.feature_drift_enabled:
            return {}

        try:
            if isinstance(data, pd.DataFrame):
                data_processed = self._preprocess_data(data)
            else:
                data_processed = pd.DataFrame(data)

            feature_results = {}

            for feature in self.config.numerical_features:
                if feature in data_processed.columns:
                    feature_data = data_processed[[feature]].values
                    results = self.detect_drift(feature_data, feature_name=feature)
                    feature_results[feature] = results

            return feature_results

        except Exception as e:
            logger.error(f"Failed to detect feature drift: {e}")
            return {}

    def should_check_drift(self) -> bool:
        """Return True when a drift check should run, per configured frequency.

        Increments the internal request counter and evaluates
        `prediction_count % monitoring_frequency == 0`.
        """
        self.prediction_count += 1
        return self.prediction_count % self.config.monitoring_frequency == 0

    def get_drift_summary(self) -> Dict[str, Any]:
        """Summarize recent drift activity for reporting and alerts.

        Returns
        -------
        Dict[str, Any]
            Contains total checks, number and rate of drift detections,
            methods used, the latest drift events, and timestamp of the last
            check. Safe to serialize for dashboards.
        """
        if not self.drift_history:
            return {"total_checks": 0, "drift_detected": 0, "methods_used": []}

        total_checks = len(self.drift_history)
        drift_detected = sum(1 for result in self.drift_history if result.is_drift)
        methods_used = list(set(result.method for result in self.drift_history))

        latest_drift = [result for result in self.drift_history[-10:] if result.is_drift]

        return {
            "total_checks": total_checks,
            "drift_detected": drift_detected,
            "drift_rate": drift_detected / total_checks if total_checks > 0 else 0,
            "methods_used": methods_used,
            "latest_drift": [result.to_dict() for result in latest_drift],
            "last_check": self.drift_history[-1].timestamp.isoformat() if self.drift_history else None
        }

    def _save_drift_results(self) -> None:
        """Persist current configuration, history, and summary to disk.

        The output is a JSON file controlled by `config.drift_results_path`
        and contains: the (serialized) config, the full `drift_history`,
        and the `get_drift_summary()` snapshot.
        """
        try:
            results_path = Path(self.config.drift_results_path)
            results_path.parent.mkdir(parents=True, exist_ok=True)

            results_data = {
                "config": self.config.model_dump(),
                "drift_history": [result.to_dict() for result in self.drift_history],
                "summary": self.get_drift_summary()
            }

            with open(results_path, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)

            logger.info(f"Saved drift results to {results_path}")

        except Exception as e:
            logger.error(f"Failed to save drift results: {e}")

    def load_reference_data_from_file(self, file_path: str) -> None:
        """Load reference data from a CSV or Parquet file and initialize detectors.

        Parameters
        ----------
        file_path:
            Path to a `.csv` or `.parquet` file.

        Raises
        ------
        ValueError
            If the file extension is unsupported.
        Exception
            If reading or detector initialization fails.
        """
        try:
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith('.parquet'):
                data = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")

            self.set_reference_data(data)
            logger.info(f"Loaded reference data from {file_path}")

        except Exception as e:
            logger.error(f"Failed to load reference data from {file_path}: {e}")
            raise

    def save_detector(self, file_path: str) -> None:
        """Save detector objects and metadata to a directory.

        A subdirectory per method (e.g., `ks_detector/`, `mmd_detector/`) is
        created alongside a `config.json` with basic metadata.

        Parameters
        ----------
        file_path:
            Directory in which to store detector artifacts.

        Raises
        ------
        Exception
            If persistence fails.
        """
        try:
            detector_path = Path(file_path)
            detector_path.parent.mkdir(parents=True, exist_ok=True)

            for method, detector in self.detectors.items():
                method_path = detector_path / f"{method}_detector"
                save_detector(detector, method_path)

            config_data = {
                "config": self.config.dict(),
                "reference_data_shape": self.reference_data.shape if self.reference_data is not None else None,
                "prediction_count": self.prediction_count
            }

            with open(detector_path / "config.json", 'w') as f:
                json.dump(config_data, f, indent=2)

            logger.info(f"Saved drift detector to {detector_path}")

        except Exception as e:
            logger.error(f"Failed to save drift detector: {e}")
            raise

    def load_detector(self, file_path: str) -> None:
        """Load detector objects and metadata from a directory.

        Relies on `config.drift_methods` to know which subdirectories to load.

        Parameters
        ----------
        file_path:
            Directory containing saved detector artifacts and `config.json`.

        Raises
        ------
        Exception
            If loading fails.
        """
        try:
            detector_path = Path(file_path)

            with open(detector_path / "config.json", 'r') as f:
                config_data = json.load(f)

            for method in self.config.drift_methods:
                method_path = detector_path / f"{method}_detector"
                if method_path.exists():
                    self.detectors[method] = load_detector(method_path)

            self.prediction_count = config_data.get("prediction_count", 0)

            logger.info(f"Loaded drift detector from {detector_path}")

        except Exception as e:
            logger.error(f"Failed to load drift detector: {e}")
            raise
