"""Random Forest builders and (placeholder) quantile wrappers.

This module provides:
- A convenience factory for `sklearn.ensemble.RandomForestRegressor`.
- A lightweight wrapper that *mimics* quantile prediction by training one RF
  per requested quantile (structure only).
- Utilities for cross-validated evaluation, feature importance, and model I/O.

⚠️ Important on "quantile" behavior
-----------------------------------
Scikit-learn's `RandomForestRegressor` does **not** optimize a quantile loss.
The `QuantileRandomForest` class below trains independent RF point estimators
(one per requested quantile), but each is still minimizing MSE under the hood.
As a result, different "quantile" models may yield very similar predictions.

For **true** conditional quantile estimation, consider:
- Gradient Boosting with quantile loss (e.g. `HistGradientBoostingRegressor`
  or `GradientBoostingRegressor` with `loss="quantile"`).
- Quantile Regression Forests from `scikit-garden` (a separate package).
- Conformal prediction on top of a point estimator.

Typical usage
-------------
rf = build_rf(n_estimators=600, max_depth=12, max_features="sqrt")
rf.fit(X_train, y_train)

metrics = evaluate_rf_model(rf, X_valid, y_valid, cv=5)
fi = get_feature_importance(rf, feature_names)

save_model(rf, "rf.joblib")
rf2 = load_model("rf.joblib")
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from typing import Dict, Any, Optional, Tuple
import joblib


def build_rf(
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str = "sqrt",
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomForestRegressor:
    """Construct a `RandomForestRegressor` with common defaults.

    Parameters
    ----------
    n_estimators
        Number of trees in the forest.
    max_depth
        Maximum tree depth. `None` grows nodes until all leaves are pure.
    min_samples_split
        Minimum samples required to split an internal node.
    min_samples_leaf
        Minimum samples required to be at a leaf node.
    max_features
        Number of features to consider at each split (e.g., "sqrt", "log2").
    random_state
        RNG seed for reproducibility.
    n_jobs
        Parallelism for `fit`/`predict` (`-1` uses all cores).

    Returns
    -------
    RandomForestRegressor
        An unfitted scikit-learn Random Forest regressor.
    """
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return rf


def build_quantile_rf(
    quantile: float = 0.5,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str = "sqrt",
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomForestRegressor:
    """Construct a Random Forest used as a *placeholder* for quantile modeling.

    Notes
    -----
    This function returns a standard `RandomForestRegressor`. Scikit-learn
    does not provide a quantile loss for RF; therefore, this "quantile" RF
    is still a point estimator (MSE). It is intended to preserve the API
    shape when training separate models per quantile.

    For true quantiles, prefer Gradient Boosting with `loss="quantile"` or
    Quantile Regression Forests (scikit-garden).

    Parameters
    ----------
    quantile
        Target quantile in [0, 1]; recorded for bookkeeping only.
    n_estimators, max_depth, min_samples_split, min_samples_leaf,
    max_features, random_state, n_jobs
        See `build_rf`.

    Returns
    -------
    RandomForestRegressor
        An unfitted RF regressor (point estimator).
    """
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return rf


class QuantileRandomForest:
    """Multi-model wrapper that trains one RF per requested quantile.

    ⚠️ Limitation: Each underlying model is a standard RF point estimator
    (MSE). This wrapper does **not** change the loss to a quantile objective.

    Parameters
    ----------
    quantiles
        List of quantiles to "model" (e.g., [0.1, 0.5, 0.9]).
    **rf_params
        Keyword arguments forwarded to `build_quantile_rf` (e.g., depth, trees).

    Attributes
    ----------
    models : Dict[float, RandomForestRegressor]
        Mapping from quantile -> fitted RF model.
    """

    def __init__(self, quantiles: list = [0.1, 0.5, 0.9], **rf_params):
        self.quantiles = quantiles
        self.rf_params = rf_params
        self.models = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit one RF per configured quantile.

        Parameters
        ----------
        X
            Feature matrix of shape (n_samples, n_features).
        y
            Target vector of shape (n_samples,).

        Notes
        -----
        All models are trained with the same MSE objective; differences across
        "quantiles" may be minor. Use gradient boosting with quantile loss or
        conformal methods for calibrated intervals.
        """
        for q in self.quantiles:
            rf = build_quantile_rf(quantile=q, **self.rf_params)
            rf.fit(X, y)
            self.models[q] = rf

    def predict(self, X: np.ndarray) -> Dict[float, np.ndarray]:
        """Predict all configured quantiles.

        Parameters
        ----------
        X
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        Dict[float, np.ndarray]
            Mapping quantile -> predictions of shape (n_samples,).
        """
        predictions = {}
        for q, model in self.models.items():
            predictions[q] = model.predict(X)
        return predictions

    def predict_quantile(self, X: np.ndarray, quantile: float) -> np.ndarray:
        """Predict a single quantile.

        Parameters
        ----------
        X
            Feature matrix of shape (n_samples, n_features).
        quantile
            The requested quantile (must have been fitted).

        Returns
        -------
        np.ndarray
            Predictions of shape (n_samples,).

        Raises
        ------
        ValueError
            If the requested quantile has not been fitted.
        """
        if quantile not in self.models:
            raise ValueError(f"Quantile {quantile} not fitted")
        return self.models[quantile].predict(X)


def evaluate_rf_model(
    model: RandomForestRegressor,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
) -> Dict[str, float]:
    """Evaluate a Random Forest with CV and aggregate metrics.

    Workflow
    --------
    1) Compute cross-validated MAE (negative sign corrected).
    2) Fit on the provided `X, y` to get in-sample predictions.
    3) Compute additional metrics via `pricing_rf.metrics.evaluate_model`.

    Parameters
    ----------
    model
        An (unfitted or fitted) `RandomForestRegressor`.
    X, y
        Features and target.
    cv
        Number of cross-validation folds for `cross_val_score`.

    Returns
    -------
    Dict[str, float]
        Metrics dictionary including keys from `evaluate_model` plus:
        - `cv_mae_mean`: mean MAE across folds (positive value).
        - `cv_mae_std`:  std dev of MAE across folds.
    """
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")

    # Fit and compute additional metrics on full data (in-sample)
    model.fit(X, y)
    y_pred = model.predict(X)

    from .metrics import evaluate_model  # local import to avoid circular deps
    metrics = evaluate_model(y, y_pred)

    metrics["cv_mae_mean"] = -cv_scores.mean()
    metrics["cv_mae_std"] = cv_scores.std()
    return metrics


def get_feature_importance(model: RandomForestRegressor, feature_names: list) -> Dict[str, float]:
    """Return a sorted mapping of feature importances.

    Parameters
    ----------
    model
        A fitted `RandomForestRegressor`.
    feature_names
        List of feature names aligned with the model input order.

    Returns
    -------
    Dict[str, float]
        Dictionary `{feature_name: importance}`, sorted descending by importance.
    """
    importance_dict = dict(zip(feature_names, model.feature_importances_))
    return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))


def save_model(model: RandomForestRegressor, filepath: str) -> None:
    """Persist a trained model to disk using `joblib`."""
    joblib.dump(model, filepath)


def load_model(filepath: str) -> RandomForestRegressor:
    """Load a previously saved model from disk (via `joblib.load`)."""
    return joblib.load(filepath)
