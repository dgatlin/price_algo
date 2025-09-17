"""Feature engineering and preprocessing utilities for pricing models.

This module defines a lightweight, composable feature pipeline for tabular,
time-aware data. It provides:

- A `ColumnTransformer`-based preprocessing pipeline with standard scaling for
  numeric columns and one-hot encoding for categorical columns.
- Convenience functions to enrich datasets with **time-derived**, **lagged**,
  **rolling-window**, and **price-change** features.
- A helper that composes all feature generators in a sensible default order.

⚠️ Important notes
------------------
- **Time order:** All time-based functions assume the input is **sorted by
  `time_column` in ascending order**. Sort your data beforehand.
- **Entity leakage:** The lag/rolling functions operate on the **entire frame**.
  If you have multiple entities (e.g., SKUs/cards), compute features **per
  entity** (e.g., `df.groupby("card_id").apply(...)`) to avoid cross-entity
  leakage.
- **NaNs from lags/rolls:** Early rows will contain NaNs (insufficient history).
  Decide on an imputation or row-dropping strategy downstream.
- **Remainder policy:** The transformer drops unspecified columns
  (`remainder='drop'`). Be explicit in `categorical_features` and
  `numerical_features`.
- **Datetime to numeric:** `timestamp_numeric` encodes seconds since epoch. Be
  mindful of timezone handling and pandas dtype conversions across versions.

Typical usage
-------------
pre = create_feature_pipeline(categorical, numerical)
X = pre.fit_transform(df_features)
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from typing import List, Optional


def create_feature_pipeline(
    categorical_features: List[str],
    numerical_features: List[str],
    target_encoding: bool = False
) -> ColumnTransformer:
    """Build a preprocessing pipeline for model-ready matrices.

    Applies:
      - `StandardScaler` to numeric columns.
      - `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` to
        categorical columns.

    Parameters
    ----------
    categorical_features
        Column names to one-hot encode.
    numerical_features
        Column names to scale.
    target_encoding
        Reserved flag for future use (not applied in this implementation).

    Returns
    -------
    ColumnTransformer
        A transformer that outputs a dense numeric array suitable for
        scikit-learn estimators.

    Notes
    -----
    - Columns not listed in `categorical_features` or `numerical_features`
      are dropped (see `remainder='drop'`).
    - For large cardinality categoricals, consider hashing or target encoding
      (not implemented here).
    """
    # Numerical features pipeline
    numerical_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    # Categorical features pipeline
    categorical_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='drop'
    )

    return preprocessor


def add_time_features(df: pd.DataFrame, time_column: str = 'timestamp') -> pd.DataFrame:
    """Derive calendar and cyclical features from a timestamp column.

    Adds the following columns when `time_column` exists:
      - `year`, `month`, `day`, `dayofweek`, `dayofyear`, ISO `week`, `quarter`
      - Cyclical encodings: `month_sin`, `month_cos`, `dayofweek_sin`, `dayofweek_cos`
      - `timestamp_numeric`: seconds since epoch (int)

    Parameters
    ----------
    df
        Input DataFrame.
    time_column
        Name of the datetime column to expand.

    Returns
    -------
    pd.DataFrame
        Copy of `df` with time-derived features appended.

    Notes
    -----
    - Ensure `df[time_column]` is timezone-aware/naive consistently.
    - Pandas versions differ in datetime→int casting behavior; this function
      uses integer seconds (`// 10**9`) for a compact numeric representation.
    """
    df_features = df.copy()

    if time_column in df_features.columns:
        df_features[time_column] = pd.to_datetime(df_features[time_column])

        # Extract time components
        df_features['year'] = df_features[time_column].dt.year
        df_features['month'] = df_features[time_column].dt.month
        df_features['day'] = df_features[time_column].dt.day
        df_features['dayofweek'] = df_features[time_column].dt.dayofweek
        df_features['dayofyear'] = df_features[time_column].dt.dayofyear
        df_features['week'] = df_features[time_column].dt.isocalendar().week
        df_features['quarter'] = df_features[time_column].dt.quarter

        # Cyclical encoding for time features
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
        df_features['dayofweek_sin'] = np.sin(2 * np.pi * df_features['dayofweek'] / 7)
        df_features['dayofweek_cos'] = np.cos(2 * np.pi * df_features['dayofweek'] / 7)

        # Time since epoch (seconds)
        # Depending on pandas version, .view("int64") may be preferred.
        df_features['timestamp_numeric'] = df_features[time_column].astype('int64') // 10**9

    return df_features


def add_lag_features(
    df: pd.DataFrame,
    target_column: str = 'price',
    lag_periods: List[int] = [1, 7, 30]
) -> pd.DataFrame:
    """Create lagged versions of the target.

    For each `lag` in `lag_periods`, adds `{target_column}_lag_{lag}` as a
    shifted copy of the target.

    Parameters
    ----------
    df
        Input DataFrame (assumed sorted by time within each entity).
    target_column
        Name of the target to lag.
    lag_periods
        List of integer lags (in rows) to compute.

    Returns
    -------
    pd.DataFrame
        Copy of `df` with lag columns appended.

    Notes
    -----
    - If your data has multiple entities, apply this per entity to avoid
      leakage (e.g., `df.groupby("card_id", group_keys=False).apply(add_lag_features, ...)`).
    - Lags introduce NaNs for early rows; handle downstream.
    """
    df_lags = df.copy()

    for lag in lag_periods:
        df_lags[f'{target_column}_lag_{lag}'] = df_lags[target_column].shift(lag)

    return df_lags


def add_rolling_features(
    df: pd.DataFrame,
    target_column: str = 'price',
    windows: List[int] = [7, 30, 90]
) -> pd.DataFrame:
    """Add rolling window summary statistics over the target.

    For each `window` in `windows`, computes:
      - Rolling mean, std, min, and max with `min_periods=1`.

    Parameters
    ----------
    df
        Input DataFrame (assumed sorted by time within each entity).
    target_column
        Name of the target to summarize.
    windows
        List of window sizes (in rows) to compute.

    Returns
    -------
    pd.DataFrame
        Copy of `df` with rolling summary columns appended.

    Notes
    -----
    - For multi-entity data, compute per entity via `groupby`.
    - `min_periods=1` emits early estimates with limited history; consider
      larger `min_periods` for stricter stability.
    """
    df_rolling = df.copy()

    for window in windows:
        df_rolling[f'{target_column}_rolling_mean_{window}'] = (
            df_rolling[target_column].rolling(window=window, min_periods=1).mean()
        )
        df_rolling[f'{target_column}_rolling_std_{window}'] = (
            df_rolling[target_column].rolling(window=window, min_periods=1).std()
        )
        df_rolling[f'{target_column}_rolling_min_{window}'] = (
            df_rolling[target_column].rolling(window=window, min_periods=1).min()
        )
        df_rolling[f'{target_column}_rolling_max_{window}'] = (
            df_rolling[target_column].rolling(window=window, min_periods=1).max()
        )

    return df_rolling


def add_price_change_features(
    df: pd.DataFrame,
    target_column: str = 'price',
    periods: List[int] = [1, 7, 30]
) -> pd.DataFrame:
    """Compute point and percentage changes of the target over prior periods.

    For each `period` in `periods`, adds:
      - `{target}_pct_change_{period}` via `pct_change(periods=period)`
      - `{target}_diff_{period}` via `diff(periods=period)`

    Parameters
    ----------
    df
        Input DataFrame (assumed sorted by time within each entity).
    target_column
        Name of the target to difference.
    periods
        Lookback steps (in rows) to compare against.

    Returns
    -------
    pd.DataFrame
        Copy of `df` with change features appended.

    Notes
    -----
    - NaNs are introduced for early rows; handle downstream.
    - Be mindful of division-by-zero in percentage changes when the
      baseline value is zero or near-zero.
    """
    df_changes = df.copy()

    for period in periods:
        df_changes[f'{target_column}_pct_change_{period}'] = (
            df_changes[target_column].pct_change(periods=period)
        )
        df_changes[f'{target_column}_diff_{period}'] = (
            df_changes[target_column].diff(periods=period)
        )

    return df_changes


def create_all_features(
    df: pd.DataFrame,
    time_column: str = 'timestamp',
    target_column: str = 'price',
    categorical_features: List[str] = None,
    numerical_features: List[str] = None
) -> pd.DataFrame:
    """Compose time, lag, rolling, and change features in a default sequence.

    This is a convenience helper that calls:
      1) `add_time_features`
      2) `add_lag_features`
      3) `add_rolling_features`
      4) `add_price_change_features`

    Parameters
    ----------
    df
        Input DataFrame. Ensure it is **time-sorted** (and entity-sorted,
        if applicable) to avoid leakage.
    time_column
        Datetime column used by `add_time_features`.
    target_column
        Target column used by lag/rolling/change functions.
    categorical_features, numerical_features
        Reserved parameters for parity with other builders; not used here.

    Returns
    -------
    pd.DataFrame
        Copy of `df` with all derived features appended.

    Notes
    -----
    - This function does not impute NaNs created by lags/rolls; perform
      cleaning downstream as appropriate for your modeling task.
    """
    if categorical_features is None:
        categorical_features = []
    if numerical_features is None:
        numerical_features = []

    # Add time features
    df_features = add_time_features(df, time_column)

    # Add lag features
    df_features = add_lag_features(df_features, target_column)

    # Add rolling features
    df_features = add_rolling_features(df_features, target_column)

    # Add price change features
    df_features = add_price_change_features(df_features, target_column)

    return df_features
