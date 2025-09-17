"""Utility helpers for the pricing RF package.

This module bundles small, reusable functions for I/O, logging, dataframe
validation, lightweight EDA, formatting, and a basic (heuristic) drift check.
They are intentionally dependency-light and safe to call from notebooks,
training scripts, or the FastAPI service.

Highlights
----------
- **Logging setup**: single-call helper to configure console + file logging.
- **Data validation**: assert required columns and datetime coercion.
- **Filesystem**: idempotent directory creation for outputs/artifacts.
- **Numerics**: safe division with NaN/inf handling.
- **Outliers**: quick IQR-based row filter.
- **EDA**: correlation matrix on selected columns.
- **Drift (heuristic)**: simple mean-shift detector (for quick checks only).
- **Formatting**: currency and percentage string helpers.
- **Memory**: human-readable dataframe memory consumption.

Notes
-----
- The `detect_data_drift` function is *not* a statistical test; it flags
  relative mean shifts exceeding a fixed threshold. For production drift
  monitoring, prefer the Alibi Detect–based detectors in the monitoring
  subsystem.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure root logging with console and file handlers.

    Parameters
    ----------
    level : str, default "INFO"
        Log level name (e.g., "DEBUG", "INFO", "WARNING", "ERROR").

    Returns
    -------
    logging.Logger
        A module-level logger instance.

    Notes
    -----
    - Uses `logging.basicConfig` with both `StreamHandler` and `FileHandler`
      (`pricing_rf.log`). Repeated calls in the same process may attach
      duplicate handlers depending on the environment.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('pricing_rf.log')
        ]
    )
    return logging.getLogger(__name__)


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    time_column: Optional[str] = None
) -> bool:
    """Validate presence of required columns and (optionally) a datetime field.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    required_columns : List[str]
        Column names that must be present.
    time_column : Optional[str], default None
        If provided and present in `df`, attempts to coerce it to datetime.

    Returns
    -------
    bool
        True if validation passes; raises `ValueError` otherwise.

    Raises
    ------
    ValueError
        If required columns are missing or the time column cannot be parsed.

    Notes
    -----
    - This function does not mutate `df`; it only attempts conversion to
      validate the field type.
    """
    # Check required columns exist
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check time column if specified
    if time_column and time_column in df.columns:
        try:
            pd.to_datetime(df[time_column])
        except Exception as e:
            raise ValueError(f"Time column '{time_column}' cannot be converted to datetime: {e}")
    
    return True


def create_directory_structure(base_path: str, subdirs: List[str]) -> None:
    """Create a directory tree under `base_path` (idempotent).

    Parameters
    ----------
    base_path : str
        Base directory where subdirectories will be created.
    subdirs : List[str]
        Relative subdirectory names to create.

    Returns
    -------
    None
    """
    base = Path(base_path)
    for subdir in subdirs:
        (base / subdir).mkdir(parents=True, exist_ok=True)


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, default: float = 0.0) -> np.ndarray:
    """Divide two arrays elementwise while suppressing inf/NaN.

    Parameters
    ----------
    numerator : np.ndarray
        Numerator values.
    denominator : np.ndarray
        Denominator values.
    default : float, default 0.0
        Value to substitute where division is undefined or non-finite.

    Returns
    -------
    np.ndarray
        Array of division results with non-finite values replaced by `default`.

    Notes
    -----
    - Uses `np.errstate(divide='ignore', invalid='ignore')` to suppress warnings.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(numerator, denominator)
        result = np.where(np.isfinite(result), result, default)
    return result


def remove_outliers_iqr(df: pd.DataFrame, columns: List[str], factor: float = 1.5) -> pd.DataFrame:
    """Row-filter outliers using the IQR rule per selected columns.

    For each column, keeps rows within `[Q1 - factor*IQR, Q3 + factor*IQR]`.
    Rows outside the band for any listed column are dropped.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    columns : List[str]
        Columns on which to apply the IQR filter.
    factor : float, default 1.5
        Multiplier for the IQR distance from Q1/Q3.

    Returns
    -------
    pd.DataFrame
        A filtered copy of `df`.

    Notes
    -----
    - This is a simple univariate rule and may remove many rows if columns are
      heavy-tailed. Consider domain-specific trimming or robust models.
    """
    df_clean = df.copy()
    
    for col in columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            df_clean = df_clean[
                (df_clean[col] >= lower_bound) & 
                (df_clean[col] <= upper_bound)
            ]
    
    return df_clean


def calculate_correlation_matrix(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Compute a Pearson correlation matrix for selected columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    columns : List[str]
        Column names to include in the correlation calculation.

    Returns
    -------
    pd.DataFrame
        Square correlation matrix (`len(columns)` × `len(columns)`).
    """
    return df[columns].corr()


def detect_data_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    columns: List[str],
    threshold: float = 0.1
) -> Dict[str, bool]:
    """Heuristic drift flagging based on relative mean shift.

    For each column, computes `abs(mean_current - mean_ref) / abs(mean_ref)`
    and compares it to `threshold`.

    Parameters
    ----------
    reference_data : pd.DataFrame
        Baseline dataset.
    current_data : pd.DataFrame
        Dataset to compare against the baseline.
    columns : List[str]
        Columns to evaluate.
    threshold : float, default 0.1
        Relative mean difference above which drift is flagged (e.g., 0.1 → 10%).

    Returns
    -------
    Dict[str, bool]
        Mapping `column -> drift_flag`.

    Caveats
    -------
    - Not a statistical test; insensitive to distributional shape changes,
      variance, or multimodality. Prefer KS/MMD or other detectors for
      production use.
    """
    drift_detected = {}
    
    for col in columns:
        if col in reference_data.columns and col in current_data.columns:
            ref_mean = reference_data[col].mean()
            curr_mean = current_data[col].mean()
            
            # Simple drift detection based on mean difference
            drift = abs(ref_mean - curr_mean) / ref_mean if ref_mean != 0 else 0
            drift_detected[col] = drift > threshold
    
    return drift_detected


def format_currency(value: float, currency: str = "USD") -> str:
    """Format a numeric value as a currency string.

    Parameters
    ----------
    value : float
        Amount to format.
    currency : str, default "USD"
        Currency code used for suffix/prefix formatting. Only "USD" is
        special-cased; others are appended as a suffix.

    Returns
    -------
    str
        Formatted currency string (e.g., "$1,234.56" or "1,234.56 EUR").

    Notes
    -----
    - Not locale-aware; for i18n/locale formatting, prefer `babel` or Python's
      `locale` module.
    """
    if currency == "USD":
        return f"${value:,.2f}"
    else:
        return f"{value:,.2f} {currency}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a numeric value as a percentage string.

    Parameters
    ----------
    value : float
        Percentage value **already on the 0–100 scale** (e.g., `7.5` → "7.50%").
        If your value is a fraction (0–1), multiply by 100 first.
    decimals : int, default 2
        Number of decimal places.

    Returns
    -------
    str
        Percentage string with a trailing `%` sign.
    """
    return f"{value:.{decimals}f}%"


def get_memory_usage(df: pd.DataFrame) -> Dict[str, str]:
    """Report dataframe memory footprint in megabytes.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    Dict[str, str]
        Dictionary with:
          - `total_memory_mb`: overall usage as a string with "MB" suffix
          - `per_column`: mapping of column → memory usage string

    Notes
    -----
    - Uses `deep=True` to include object dtype internals.
    """
    memory_usage = df.memory_usage(deep=True)
    total_memory = memory_usage.sum()
    
    return {
        'total_memory_mb': f"{total_memory / 1024**2:.2f} MB",
        'per_column': {col: f"{usage / 1024**2:.2f} MB" 
                      for col, usage in memory_usage.items()}
    }
