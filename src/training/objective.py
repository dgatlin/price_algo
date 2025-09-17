"""Optuna objective functions for Random Forest hyperparameter optimization.

This module exposes factory functions that return Optuna-compatible objective
callables for both **single holdout** and **time-series cross-validation**
workflows. Each objective constructs a `RandomForestRegressor` with parameters
sampled from the trial, fits it, and returns an error metric to be **minimized**
(e.g., WAPE/MAE/tail MAE).

Usage
-----
# Holdout validation
objective = create_objective(X_train, y_train, X_val, y_val, primary_metric="wape")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

# Time-series CV
splits = get_time_series_cv_splits(df, n_splits=5)  # elsewhere in your code
objective_cv = create_objective_with_cv(X, y, splits, primary_metric="wape")
study = optuna.create_study(direction="minimize")
study.optimize(objective_cv, n_trials=100)

Metrics
-------
- "wape"     → Weighted Absolute Percentage Error (percentage; lower is better)
- "mae"      → Mean Absolute Error (original units; lower is better)
- "tail_mae" → MAE computed on the high-value tail (e.g., top 10%; lower is better)

Notes
-----
- Randomness: `random_state=42` is fixed for reproducibility within trials.
- Parallelism: `n_jobs=-1` uses all cores; adjust to control CPU usage.
- Search space: conservative bounds; tune as needed for your domain.
- Pruning: Not enabled here; you may integrate Optuna pruners if you add
  intermediate metrics (`trial.report(...)` + `trial.should_prune()`).
"""

import optuna
import numpy as np
from sklearn.metrics import mean_absolute_error
from typing import Tuple, Callable, List

from pricing_rf.model import build_rf
from pricing_rf.metrics import wape, tail_mae


def create_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    primary_metric: str = 'wape'
) -> Callable:
    """Return an Optuna objective for single holdout validation.

    The returned callable trains a `RandomForestRegressor` per trial with
    sampled hyperparameters, scores it on the validation set using the chosen
    metric, and returns the scalar error to be **minimized**.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training features and target.
    X_val, y_val : np.ndarray
        Validation features and target.
    primary_metric : {'wape', 'mae', 'tail_mae'}, default 'wape'
        Error metric minimized by Optuna.

    Returns
    -------
    Callable[[optuna.Trial], float]
        Objective function compatible with `study.optimize(...)`.

    Raises
    ------
    ValueError
        If `primary_metric` is not one of the supported options.
    """
    def objective(trial: optuna.Trial) -> float:
        """Optuna objective for a single trial."""
        # Define hyperparameter search space
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': 42,
            'n_jobs': -1
        }

        # Train model
        model = build_rf(**params)
        model.fit(X_train, y_train)

        # Validate
        y_pred = model.predict(X_val)

        # Compute metric (lower is better)
        if primary_metric == 'wape':
            score = wape(y_val, y_pred)
        elif primary_metric == 'mae':
            score = mean_absolute_error(y_val, y_pred)
        elif primary_metric == 'tail_mae':
            score = tail_mae(y_val, y_pred, percentile=90)
        else:
            raise ValueError(f"Unknown primary metric: {primary_metric}")

        return score

    return objective


def create_objective_with_cv(
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    primary_metric: str = 'wape'
) -> Callable:
    """Return an Optuna objective that averages error across CV folds.

    For each trial, the function trains and evaluates a model on each provided
    `(train_idx, val_idx)` split and returns the **mean** validation error,
    which Optuna minimizes.

    Parameters
    ----------
    X, y : np.ndarray
        Full dataset features and target arrays.
    cv_splits : list of (np.ndarray, np.ndarray)
        Sequence of `(train_idx, val_idx)` index pairs (e.g., from
        `TimeSeriesSplit`). Indices refer to rows of `X`/`y`.
    primary_metric : {'wape', 'mae', 'tail_mae'}, default 'wape'
        Error metric minimized by Optuna.

    Returns
    -------
    Callable[[optuna.Trial], float]
        Objective function compatible with `study.optimize(...)`.

    Raises
    ------
    ValueError
        If `primary_metric` is not one of the supported options.
    """
    def objective(trial: optuna.Trial) -> float:
        """Optuna objective computing mean CV error for a single trial."""
        # Define hyperparameter search space
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': 42,
            'n_jobs': -1
        }

        cv_scores = []

        for train_idx, val_idx in cv_splits:
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X[val_idx]
            y_val_fold = y[val_idx]

            # Train model
            model = build_rf(**params)
            model.fit(X_train_fold, y_train_fold)

            # Validate
            y_pred = model.predict(X_val_fold)

            # Compute metric (lower is better)
            if primary_metric == 'wape':
                score = wape(y_val_fold, y_pred)
            elif primary_metric == 'mae':
                score = mean_absolute_error(y_val_fold, y_pred)
            elif primary_metric == 'tail_mae':
                score = tail_mae(y_val_fold, y_pred, percentile=90)
            else:
                raise ValueError(f"Unknown primary metric: {primary_metric}")

            cv_scores.append(score)

        return np.mean(cv_scores)

    return objective
