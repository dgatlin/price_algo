"""Custom evaluation metrics for pricing models.

This module implements a compact set of metrics tailored for price prediction
and monitoring. It includes absolute/relative error measures, tail-focused
error, quantile (pinball) loss, direction-of-change accuracy, and tolerance
(hit rate) style metrics, plus a convenience aggregator.

Conventions
-----------
- Inputs are 1D numpy arrays of equal length: `y_true.shape == y_pred.shape == (n,)`.
- Metrics that are percentages return values on the **0-100 scale**:
  WAPE, MAPE, sMAPE, directional_accuracy, and hit_rate.
- MAE/MSE/RMSE are returned in the **original units** of the target.

Caveats
-------
- Division by zero: MAPE and hit_rate divide by `y_true`; if your data can
  contain zeros or near-zeros, handle those upstream or switch to WAPE/sMAPE.
- sMAPE's denominator includes `|y_true| + |y_pred|`; rows where both are 0
  result in a 0/0 term—pandas/numpy will yield `nan` which propagates to the
  mean. Consider cleaning or adding an epsilon when appropriate.
- `tail_mae` defines the tail by the **percentile of `y_true`**, not `y_pred`.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import List, Optional


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error (WAPE).

    Definition
    ----------
    WAPE = sum(|y_true - y_pred|) / sum(|y_true|) * 100

    Properties
    ----------
    - Scale-independent and robust when magnitudes vary widely.
    - Undefined if `sum(|y_true|) == 0` (handle upstream if possible).

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        True and predicted values.

    Returns
    -------
    float
        WAPE as a percentage (0-100 scale).
    """
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (MAPE).

    Definition
    ----------
    MAPE = mean(|y_true - y_pred| / |y_true|) * 100

    Notes
    -----
    - Sensitive to zeros/near-zeros in `y_true` (division by zero).
      Prefer WAPE or sMAPE if zeros are present.

    Parameters
    ----------
    y_true, y_pred : np.ndarray

    Returns
    -------
    float
        MAPE as a percentage (0-100 scale).
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (sMAPE).

    Definition
    ----------
    sMAPE = mean( 2 * |y_true - y_pred| / (|y_true| + |y_pred|) ) * 100

    Notes
    -----
    - Bounded in [0, 200].
    - If both `|y_true|` and `|y_pred|` are zero for an item, the term is 0/0
      (`nan`). Clean or add an epsilon upstream if needed.

    Parameters
    ----------
    y_true, y_pred : np.ndarray

    Returns
    -------
    float
        sMAPE as a percentage (0–100 scale; sometimes up to 200 for large errors).
    """
    return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100


def tail_mae(y_true: np.ndarray, y_pred: np.ndarray, percentile: float = 90) -> float:
    """Mean Absolute Error computed on the high-value tail.

    The tail is defined by `y_true >= percentile(y_true, percentile)`.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
    percentile : float, default 90
        Percentile threshold (0–100). Example: 90 → top 10% by `y_true`.

    Returns
    -------
    float
        MAE on the tail in the original units of the target.
    """
    threshold = np.percentile(y_true, percentile)
    tail_mask = y_true >= threshold

    if np.sum(tail_mask) == 0:
        return 0.0

    return mean_absolute_error(y_true[tail_mask], y_pred[tail_mask])


def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float = 0.5) -> float:
    """Quantile (pinball) loss for quantile regression.

    Definition
    ----------
    For residual `e = y_true - y_pred` and quantile `q` in (0,1):
      L_q = mean( max(q * e, (q - 1) * e) )

    Parameters
    ----------
    y_true, y_pred : np.ndarray
    quantile : float, default 0.5
        Target quantile (e.g., 0.1, 0.5, 0.9).

    Returns
    -------
    float
        Mean pinball loss (lower is better).
    """
    error = y_true - y_pred
    return np.mean(np.maximum(quantile * error, (quantile - 1) * error))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Direction-of-change accuracy (% of correct up/down predictions).

    Computes the sign of first differences for `y_true` and `y_pred` and
    compares them element-wise.

    Parameters
    ----------
    y_true, y_pred : np.ndarray

    Returns
    -------
    float
        Percentage (0-100) of periods where `sign(Δy_pred) == sign(Δy_true)`.
        Returns 0.0 if fewer than 2 observations are provided.
    """
    if len(y_true) < 2:
        return 0.0

    true_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0

    return np.mean(true_direction == pred_direction) * 100


def hit_rate(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.05) -> float:
    """Hit rate within a relative error tolerance.

    Definition
    ----------
    Hit = 1 if |y_true - y_pred| / |y_true| <= threshold, else 0

    Parameters
    ----------
    y_true, y_pred : np.ndarray
    threshold : float, default 0.05
        Relative tolerance (e.g., 0.05 → within ±5% of actual).

    Returns
    -------
    float
        Percentage (0-100) of predictions within the tolerance band.

    Notes
    -----
    - Sensitive to zeros in `y_true`. Consider WAPE-based thresholds if zeros
      are common.
    """
    relative_error = np.abs((y_true - y_pred) / y_true)
    return np.mean(relative_error <= threshold) * 100


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Aggregate a suite of metrics for quick model assessment.

    Includes:
      - `mae`, `mse`, `rmse` (original units)
      - `wape`, `mape`, `smape` (percentages)
      - `tail_mae_90`, `tail_mae_95` (original units, high-value focus)
      - `directional_accuracy` (percentage)
      - `hit_rate_5pct`, `hit_rate_10pct` (percentages within ±5% / ±10%)

    Parameters
    ----------
    y_true, y_pred : np.ndarray

    Returns
    -------
    dict
        Dictionary of metric name → value.
    """
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'wape': wape(y_true, y_pred),
        'mape': mape(y_true, y_pred),
        'smape': smape(y_true, y_pred),
        'tail_mae_90': tail_mae(y_true, y_pred, percentile=90),
        'tail_mae_95': tail_mae(y_true, y_pred, percentile=95),
        'directional_accuracy': directional_accuracy(y_true, y_pred),
        'hit_rate_5pct': hit_rate(y_true, y_pred, threshold=0.05),
        'hit_rate_10pct': hit_rate(y_true, y_pred, threshold=0.10),
    }

    return metrics