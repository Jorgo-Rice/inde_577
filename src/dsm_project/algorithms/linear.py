"""Transparent batch-gradient linear regression teaching implementation."""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin


class BatchLinearRegressor(RegressorMixin, BaseEstimator):
    """Fit an affine least-squares model using batch gradient descent.

    Parameters
    ----------
    learning_rate
        Positive gradient-descent step size.
    max_iterations
        Positive number of parameter updates.
    tolerance
        Nonnegative gradient-norm stopping threshold.

    Notes
    -----
    This transparent implementation is intended for mathematical verification,
    not as a claim of production superiority over mature numerical libraries.
    Scale features before fitting when their units differ substantially.
    """

    def __init__(
        self,
        *,
        learning_rate: float = 0.02,
        max_iterations: int = 5_000,
        tolerance: float = 1e-8,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if tolerance < 0:
            raise ValueError("tolerance must be nonnegative")
        self.learning_rate = float(learning_rate)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)

    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self:
        """Fit the affine model and retain a reproducible objective trace."""

        array = np.asarray(features, dtype=float)
        target_array = np.asarray(targets, dtype=float)
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError("features must be a nonempty two-dimensional array")
        if target_array.shape != (array.shape[0],):
            raise ValueError("targets must contain one value per observation")
        if not np.isfinite(array).all() or not np.isfinite(target_array).all():
            raise ValueError("features and targets must be finite")

        sample_count, feature_count = array.shape
        weights = np.zeros(feature_count, dtype=float)
        bias = 0.0
        losses: list[float] = []
        for _ in range(self.max_iterations):
            residual = array @ weights + bias - target_array
            losses.append(float(0.5 * np.mean(residual**2)))
            weight_gradient = array.T @ residual / sample_count
            bias_gradient = float(residual.mean())
            gradient_norm = float(
                np.sqrt(weight_gradient @ weight_gradient + bias_gradient**2)
            )
            if gradient_norm <= self.tolerance:
                break
            weights -= self.learning_rate * weight_gradient
            bias -= self.learning_rate * bias_gradient

        self.coef_ = weights
        self.intercept_ = bias
        self.loss_history_ = np.asarray(losses)
        self.n_features_in_ = feature_count
        return self

    def predict(self, features: ArrayLike) -> NDArray[np.float64]:
        """Predict the continuous response for each input row."""

        if not hasattr(self, "coef_"):
            raise RuntimeError("fit must be called before predict")
        array = np.asarray(features, dtype=float)
        if array.ndim != 2 or array.shape[1] != self.n_features_in_:
            raise ValueError("features have the wrong shape for this fitted model")
        if not np.isfinite(array).all():
            raise ValueError("features must be finite")
        return array @ self.coef_ + self.intercept_
