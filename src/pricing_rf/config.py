"""Configuration management using Pydantic settings.

This module centralizes all runtime configuration for the pricing RF project
using `pydantic-settings.BaseSettings`. Each settings class can be overridden
via environment variables with a prefix, enabling 12-factor friendly configs
across local dev, CI, and production.

Overview
--------
- `DataConfig`     → file paths, schema columns, feature lists
- `TrainingConfig` → temporal split ranges, Optuna/compute knobs, primary metric
- `ServiceConfig`  → serving host/port, MLflow registry, optional auth
- `Config`         → thin aggregator that instantiates all three

Environment variables
---------------------
Values are read from env with the given prefixes (uppercased field names):
- `DATA_`…      e.g., `DATA_RAW_DATA_PATH=/mnt/raw.csv`
- `TRAIN_`…     e.g., `TRAIN_N_TRIALS=200`
- `SERVICE_`…   e.g., `SERVICE_PORT=8080`

Example
-------
>>> cfg = Config()
>>> cfg.data.raw_data_path
PosixPath('data/raw.csv')
>>> cfg.training.n_trials
100
>>> cfg.service.mlflow_tracking_uri
'http://localhost:5000'

Notes
-----
- This module does not set an `.env` file automatically. If you want dotenv
  support, configure it in your application entrypoint or switch to
  `SettingsConfigDict(env_file=".env")` in each settings class.
"""

from pathlib import Path
from typing import List, Optional

from pydantic import Field

try:
    from pydantic_settings import BaseSettings
except ImportError:
    try:
        # Fallback for older pydantic versions
        from pydantic import BaseSettings
    except ImportError:
        # If both fail, create a minimal BaseSettings class
        from pydantic import BaseModel
        class BaseSettings(BaseModel):
            class Config:
                env_prefix = ""


class DataConfig(BaseSettings):
    """Data-layer configuration.

    Fields
    ------
    raw_data_path :
        Path to raw source data (CSV/Parquet supported by loaders).
    processed_data_path :
        Path to processed/feature-ready dataset.
    time_column :
        Name of the timestamp column used for ordering/splits.
    target_column :
        Name of the supervised learning target (e.g., price).
    feature_columns :
        Explicit list of model feature columns (optional; can be inferred upstream).
    categorical_features :
        List of categorical feature names (for encoders/monitoring).
    numerical_features :
        List of numerical feature names (for scalers/monitoring).

    Environment
    -----------
    Uses prefix `DATA_` (e.g., `DATA_TIME_COLUMN=as_of`).
    """
    raw_data_path: Path = Field(default="data/raw.csv", description="Path to raw data")
    processed_data_path: Path = Field(default="data/processed.parquet", description="Path to processed data")
    time_column: str = Field(default="timestamp", description="Name of time column")
    target_column: str = Field(default="price", description="Name of target column")
    feature_columns: List[str] = Field(default_factory=list, description="List of feature columns")
    categorical_features: List[str] = Field(default_factory=list, description="List of categorical features")
    numerical_features: List[str] = Field(default_factory=list, description="List of numerical features")
    
    class Config:
        """Settings behavior: environment variable prefix."""
        env_prefix = "DATA_"


class TrainingConfig(BaseSettings):
    """Training & experimentation configuration.

    Temporal splits (optional)
    --------------------------
    train_start_date, train_end_date :
        Boundaries for training window (ISO-8601 strings).
    val_start_date, val_end_date :
        Boundaries for validation window.
    test_start_date, test_end_date :
        Boundaries for test window.

    Optimization & compute
    ----------------------
    n_trials :
        Optuna trials for hyperparameter search.
    n_jobs :
        Parallel workers for sklearn where applicable (-1 uses all cores).
    random_state :
        Seed for reproducibility.

    Metrics
    -------
    primary_metric :
        Name of the metric to optimize/report as primary (e.g., "wape").

    Environment
    -----------
    Uses prefix `TRAIN_` (e.g., `TRAIN_N_TRIALS=200`).
    """
    # Data splitting
    train_start_date: Optional[str] = Field(default=None, description="Training start date")
    train_end_date: Optional[str] = Field(default=None, description="Training end date")
    val_start_date: Optional[str] = Field(default=None, description="Validation start date")
    val_end_date: Optional[str] = Field(default=None, description="Validation end date")
    test_start_date: Optional[str] = Field(default=None, description="Test start date")
    test_end_date: Optional[str] = Field(default=None, description="Test end date")
    
    # Model parameters
    n_trials: int = Field(default=100, description="Number of Optuna trials")
    n_jobs: int = Field(default=-1, description="Number of parallel jobs")
    random_state: int = Field(default=42, description="Random state for reproducibility")
    
    # Metrics
    primary_metric: str = Field(default="wape", description="Primary metric to optimize")
    
    class Config:
        """Settings behavior: environment variable prefix."""
        env_prefix = "TRAIN_"


class ServiceConfig(BaseSettings):
    """Service/runtime configuration for the FastAPI app.

    Serving
    -------
    model_name :
        Registered MLflow model name to load.
    model_version :
        Target model version (e.g., "latest" or an integer as string).
    host, port :
        Bind address and port for the API server.
    debug :
        Enable verbose logging and debug behavior.

    MLflow
    ------
    mlflow_tracking_uri :
        Tracking server URI (http(s) or local file path).

    Authentication (optional)
    -------------------------
    enable_auth :
        Toggle auth middleware/dependencies.
    jwt_secret :
        Secret used to validate/issue JWTs (when `enable_auth` is True).
    allowed_ips :
        Optional allowlist of IPs if implementing IP filtering.

    Environment
    -----------
    Uses prefix `SERVICE_` (e.g., `SERVICE_PORT=8080`).
    """
    model_name: str = Field(default="pricing-rf", description="MLflow model name")
    model_version: str = Field(default="latest", description="MLflow model version")
    host: str = Field(default="0.0.0.0", description="Service host")
    port: int = Field(default=8000, description="Service port")
    debug: bool = Field(default=False, description="Debug mode")
    
    # MLflow
    mlflow_tracking_uri: str = Field(default="http://localhost:5000", description="MLflow tracking URI")
    
    # Authentication (optional)
    enable_auth: bool = Field(default=False, description="Enable authentication")
    jwt_secret: Optional[str] = Field(default=None, description="JWT secret key")
    allowed_ips: List[str] = Field(default_factory=list, description="Allowed IP addresses")
    
    class Config:
        """Settings behavior: environment variable prefix."""
        env_prefix = "SERVICE_"


class Config:
    """Aggregator for all settings domains.

    Instantiates:
      - `self.data`     : DataConfig
      - `self.training` : TrainingConfig
      - `self.service`  : ServiceConfig

    Usage
    -----
    >>> cfg = Config()
    >>> cfg.service.port
    8000
    >>> cfg.data.time_column
    'timestamp'
    """
    def __init__(self):
        self.data = DataConfig()
        self.training = TrainingConfig()
        self.service = ServiceConfig()

