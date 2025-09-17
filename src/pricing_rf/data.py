"""Data loading, cleaning, and time-based splitting utilities.

This module provides minimal, dependency-light helpers for getting tabular,
time-indexed datasets ready for pricing model experiments. It focuses on:

- Fast file loading (CSV/Parquet).
- Basic cleaning (datetime coercion and simple missing-value handling).
- Temporal splits by **ratio** or **explicit date ranges**.
- Cross-validation index generation using `TimeSeriesSplit`.

Guiding principles
------------------
- **Time order first:** All splits are performed after sorting by the declared
  `time_column` in ascending order.
- **No mutation:** Functions return new objects and avoid mutating inputs.
- **Explicitness:** Date-based splits use inclusive lower bounds and exclusive
  upper bounds to avoid overlap.

Caveats
-------
- This module does not perform feature scaling, target leakage checks across
  multiple entities, or advanced outlier treatment.
- If your data contains multiple entities (e.g., card_id/SKU), consider
  per-entity sorting and splitting to avoid cross-entity leakage.
"""

import pandas as pd
from datetime import datetime
from typing import Tuple, Optional, List
from sklearn.model_selection import TimeSeriesSplit
import numpy as np


def load_data(file_path: str) -> pd.DataFrame:
    """Load a dataset from CSV or Parquet.

    Parameters
    ----------
    file_path : str
        Path to a `.csv` or `.parquet` file.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.

    Notes
    -----
    - File type is inferred from the file extension. Parquet is preferred for
      speed and types; CSV is a convenient default.
    """
    if file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    else:
        return pd.read_csv(file_path)


def clean_data(
    df: pd.DataFrame,
    time_column: str = 'timestamp',
    target_column: str = 'price'
) -> pd.DataFrame:
    """Coerce time, drop missing targets, and fill feature NaNs.

    Steps
    -----
    1) Convert `time_column` to datetime (if present).
    2) Drop rows with missing `target_column`.
    3) For remaining columns:
       - Numeric NaNs → column median
       - Categorical/object NaNs → column mode (or `"Unknown"` if no mode)

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    time_column : str, default 'timestamp'
        Name of the datetime column to coerce, if present.
    target_column : str, default 'price'
        Name of the target; rows with missing values are dropped.

    Returns
    -------
    pd.DataFrame
        Cleaned copy of `df`.

    Notes
    -----
    - This function does **not** remove outliers; use a dedicated utility if
      needed (e.g., IQR trimming) prior to training/evaluation.
    """
    df_clean = df.copy()
    
    # Convert time column to datetime
    if time_column in df_clean.columns:
        df_clean[time_column] = pd.to_datetime(df_clean[time_column])
    
    # Remove rows with missing target values
    df_clean = df_clean.dropna(subset=[target_column])
    
    # Handle missing values in features
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
    categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
    
    # Fill numerical missing values with median
    for col in numerical_cols:
        if col != target_column:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Fill categorical missing values with mode
    for col in categorical_cols:
        if col != time_column:
            df_clean[col] = df_clean[col].fillna(
                df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
            )
    
    return df_clean


def time_based_split(
    df: pd.DataFrame,
    time_column: str = 'timestamp',
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a dataset into train/val/test by **chronological ratios**.

    The dataframe is sorted by `time_column` ascending, then split by index
    positions according to the provided ratios.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    time_column : str, default 'timestamp'
        Name of the datetime column used for ordering.
    train_ratio, val_ratio, test_ratio : float
        Fractions that should sum to 1.0 (not strictly enforced here).

    Returns
    -------
    (train_df, val_df, test_df) : Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Non-overlapping temporal splits.

    Notes
    -----
    - Endpoints are index-based after sorting; if ratios do not sum to 1.0,
      the remainder will spill into the last split.
    - For multi-entity data, consider grouping and splitting within entity.
    """
    df_sorted = df.sort_values(time_column).reset_index(drop=True)
    
    n_samples = len(df_sorted)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    train_df = df_sorted[:train_end]
    val_df = df_sorted[train_end:val_end]
    test_df = df_sorted[val_end:]
    
    return train_df, val_df, test_df


def date_based_split(
    df: pd.DataFrame,
    time_column: str = 'timestamp',
    train_start: Optional[str] = None,
    train_end: Optional[str] = None,
    val_start: Optional[str] = None,
    val_end: Optional[str] = None,
    test_start: Optional[str] = None,
    test_end: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a dataset into train/val/test by **explicit date ranges**.

    All ranges use inclusive lower bounds and **exclusive** upper bounds:
    `[start, end)`. Missing ranges yield empty dataframes for that split.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    time_column : str, default 'timestamp'
        Name of the datetime column used for ordering and filtering.
    train_start, train_end, val_start, val_end, test_start, test_end : Optional[str]
        ISO-8601 parsable boundaries (e.g., "2024-01-01").

    Returns
    -------
    (train_df, val_df, test_df) : Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Date-bounded temporal splits.

    Notes
    -----
    - The dataframe is always sorted by `time_column` prior to filtering.
    - If a split’s bounds are not provided, the corresponding dataframe will be
      empty (except train, which defaults to the full range unless narrowed).
    """
    df_sorted = df.sort_values(time_column).reset_index(drop=True)
    
    train_df = df_sorted.copy()
    val_df = pd.DataFrame()
    test_df = pd.DataFrame()
    
    if train_start:
        train_df = train_df[train_df[time_column] >= pd.to_datetime(train_start)]
    if train_end:
        train_df = train_df[train_df[time_column] < pd.to_datetime(train_end)]
    
    if val_start and val_end:
        val_df = df_sorted[
            (df_sorted[time_column] >= pd.to_datetime(val_start)) &
            (df_sorted[time_column] < pd.to_datetime(val_end))
        ]
    
    if test_start and test_end:
        test_df = df_sorted[
            (df_sorted[time_column] >= pd.to_datetime(test_start)) &
            (df_sorted[time_column] < pd.to_datetime(test_end))
        ]
    
    return train_df, val_df, test_df


def get_time_series_cv_splits(
    df: pd.DataFrame,
    time_column: str = 'timestamp',
    n_splits: int = 5
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate expanding-window cross-validation splits.

    Wraps scikit-learn’s `TimeSeriesSplit` to produce index pairs suitable for
    temporal model validation. The dataframe is sorted by `time_column`
    ascending before computing splits.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    time_column : str, default 'timestamp'
        Name of the datetime column used for ordering.
    n_splits : int, default 5
        Number of validation folds.

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        List of `(train_idx, val_idx)` index arrays referring to rows in the
        **sorted** dataframe.

    Notes
    -----
    - These are **indices**, not sliced dataframes; apply them to your
      time-sorted arrays/dataframes for training/evaluation.
    """
    df_sorted = df.sort_values(time_column).reset_index(drop=True)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    
    for train_idx, val_idx in tscv.split(df_sorted):
        splits.append((train_idx, val_idx))
    
    return splits
