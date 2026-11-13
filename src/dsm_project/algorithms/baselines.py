"""Simple baselines that every complex candidate must earn the right to beat."""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validated_rows(features: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(features, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("features must be a nonempty two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError("features must contain only finite values")
    return array


class MeanRegressor:
    """Predict the training-target mean for every observation."""

    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self:
        """Learn the mean after validating row alignment."""

        array = _validated_rows(features)
        target_array = np.asarray(targets, dtype=float)
        if target_array.shape != (array.shape[0],):
            raise ValueError("targets must contain one value per observation")
        if not np.isfinite(target_array).all():
            raise ValueError("targets must contain only finite values")
        self.mean_ = float(target_array.mean())
        self.n_features_in_ = array.shape[1]
        return self

    def predict(self, features: ArrayLike) -> NDArray[np.float64]:
        """Return one mean prediction per validated row."""

        if not hasattr(self, "mean_"):
            raise RuntimeError("fit must be called before predict")
        array = _validated_rows(features)
        if array.shape[1] != self.n_features_in_:
            raise ValueError("prediction feature count differs from fitting")
        return np.full(array.shape[0], self.mean_, dtype=float)


class MajorityClassifier:
    """Predict the most frequent training label for every observation."""

    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self:
        """Learn the modal label using deterministic tie breaking."""

        array = _validated_rows(features)
        target_array = np.asarray(targets)
        if target_array.shape != (array.shape[0],):
            raise ValueError("targets must contain one label per observation")
        labels, counts = np.unique(target_array, return_counts=True)
        self.label_ = labels[int(np.argmax(counts))]
        self.n_features_in_ = array.shape[1]
        return self

    def predict(self, features: ArrayLike) -> NDArray[np.object_]:
        """Return one modal-label prediction per validated row."""

        if not hasattr(self, "label_"):
            raise RuntimeError("fit must be called before predict")
        array = _validated_rows(features)
        if array.shape[1] != self.n_features_in_:
            raise ValueError("prediction feature count differs from fitting")
        return np.full(array.shape[0], self.label_)
