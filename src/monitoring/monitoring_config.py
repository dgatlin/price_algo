"""Configuration schema for runtime monitoring and drift detection.

This module defines a `pydantic` model (`MonitoringConfig`) that centralizes all
tunable parameters for inference-time monitoring, including data-drift tests,
frequency gating, logging/alerting, and persistence paths. It is designed to be
serialized (e.g., to JSON/YAML), validated on load, and passed into the drift
detector at service startup.

Highlights
----------
- **Detectors**: Choose which statistical tests to run (e.g., KS for univariate,
  MMD for multivariate). Additional method names can be listed for future use.
- **Frequency gating**: Control how often drift checks run (e.g., every Nth
  prediction) to keep latency/CPU overhead predictable.
- **Feature scopes**: Provide explicit lists of categorical and numerical
  features to ensure consistent preprocessing for drift tests.
- **Persistence & logging**: Enable saving structured results to disk and emit
  logs for observability; optional alert flag for downstream hooks.
- **Reference data**: Optionally point to a file containing the baseline window
  used to initialize detectors.

Typical usage
-------------
cfg = MonitoringConfig(
    drift_methods=["ks", "mmd"],
    drift_threshold=0.05,
    categorical_features=["grade", "brand"],
    numerical_features=["pop_report", "last_sale_price", "seller_rating"],
    monitoring_frequency=100,
    drift_results_path="monitoring/drift_results.json",
)

Notes
-----
- This config is **consumed by** the drift detector component, which constructs
  the requested detectors using the provided reference data. If a method is
  listed here but not implemented in the detector, it will be ignored or must
  be handled by downstream code.
- The default feature lists are placeholders—replace them with your model’s
  actual schema to avoid unexpected drops in preprocessing.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np


class MonitoringConfig(BaseModel):
    """Runtime configuration for monitoring and drift detection.

    Fields
    ------
    drift_threshold :
        Significance level (p-value) used by detectors (e.g., 0.05).
    reference_data_path :
        Optional path to reference (baseline) data used to initialize detectors.
    drift_methods :
        Names of drift tests to enable (e.g., `["ks", "mmd", "wasserstein"]`).
        Methods not implemented in your detector will be ignored.
    feature_drift_enabled :
        If True, run per-feature drift checks in addition to overall tests.
    categorical_features :
        Feature names treated as categorical for preprocessing (one-hot).
    numerical_features :
        Feature names treated as numerical for preprocessing/scaling.
    ks_test_params :
        Keyword arguments passed to the KS detector (e.g., `alternative`).
    mmd_test_params :
        Keyword arguments passed to the MMD detector (e.g., kernel/sigma).
    wasserstein_test_params :
        Reserved for Wasserstein distance configuration (if implemented).
    monitoring_frequency :
        Run drift checks every N predictions (e.g., 100 → ~1% of requests).
    alert_on_drift :
        Enable alert signaling when drift is detected (integration-specific).
    log_drift_results :
        Emit structured logs for each drift check outcome.
    save_drift_results :
        Persist config, history, and summary to `drift_results_path`.
    drift_results_path :
        File path used when persisting drift results (JSON).
    """

    # Drift detection thresholds
    drift_threshold: float = Field(
        default=0.05,
        description="Threshold for drift detection (p-value)"
    )

    # Reference data configuration
    reference_data_path: Optional[str] = Field(
        default=None,
        description="Path to reference training data for drift detection"
    )

    # Drift detection methods
    drift_methods: List[str] = Field(
        default=["ks", "mmd"],
        description="List of drift detection methods to use"
    )

    # Feature-specific drift detection
    feature_drift_enabled: bool = Field(
        default=True,
        description="Enable per-feature drift detection"
    )

    # Categorical feature handling
    categorical_features: List[str] = Field(
        default=["category_feature"],
        description="List of categorical feature names"
    )

    # Numerical feature handling
    numerical_features: List[str] = Field(
        default=["feature1", "feature2", "feature3"],
        description="List of numerical feature names"
    )

    # Drift detection parameters
    ks_test_params: Dict[str, Any] = Field(
        default={"alternative": "two-sided"},
        description="Parameters for Kolmogorov-Smirnov test"
    )

    mmd_test_params: Dict[str, Any] = Field(
        default={"kernel": "rbf", "sigma": 1.0},
        description="Parameters for Maximum Mean Discrepancy test"
    )

    wasserstein_test_params: Dict[str, Any] = Field(
        default={},
        description="Parameters for Wasserstein distance test"
    )

    # Monitoring frequency
    monitoring_frequency: int = Field(
        default=100,
        description="Number of predictions before running drift detection"
    )

    # Alerting configuration
    alert_on_drift: bool = Field(
        default=True,
        description="Enable alerts when drift is detected"
    )

    # Logging configuration
    log_drift_results: bool = Field(
        default=True,
        description="Log drift detection results"
    )

    # Storage configuration
    save_drift_results: bool = Field(
        default=True,
        description="Save drift detection results to file"
    )

    drift_results_path: str = Field(
        default="monitoring/drift_results.json",
        description="Path to save drift detection results"
    )

