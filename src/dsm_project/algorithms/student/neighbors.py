"""k-nearest-neighbors classification written from first principles.

Mathematical contract
---------------------
Given training pairs :math:`(x_j, y_j)` and a query point :math:`x`, let
:math:`N_k(x)` be the indices of the ``k`` training points closest to ``x``
under a distance :math:`d`. The classifier predicts

.. math::

    \\hat{y}(x) = \\arg\\max_{c} \\sum_{j \\in N_k(x)} w_j \\, \\mathbb{1}[y_j = c],

where :math:`w_j = 1` for uniform voting and :math:`w_j = 1 / d(x, x_j)` for
distance weighting. "Fitting" only stores the training set, so all cost is paid
at prediction time: :math:`O(m n d)` to score ``m`` queries against ``n``
stored rows in ``d`` dimensions.

Tie-breaking is deterministic. Equidistant neighbors are ordered by training
row index (a stable sort), and tied votes go to the smallest label in
``classes_``.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin

_METRICS = ("euclidean", "manhattan")
_WEIGHTS = ("uniform", "distance")
#: Upper bound on query-by-training-by-feature differences held in memory at
#: once (about 64 MB of float64).
_BLOCK_ELEMENTS = 8_000_000


def pairwise_distances(
    queries: NDArray[np.float64],
    references: NDArray[np.float64],
    metric: str = "euclidean",
) -> NDArray[np.float64]:
    """Return the ``(m, n)`` matrix of distances between two row sets."""

    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {_METRICS}")
    differences = queries[:, np.newaxis, :] - references[np.newaxis, :, :]
    if metric == "euclidean":
        return np.sqrt(np.sum(differences**2, axis=2))
    return np.sum(np.abs(differences), axis=2)


class KNearestNeighborsClassifier(ClassifierMixin, BaseEstimator):
    """Classify each query by a vote among its ``k`` nearest training rows.

    Parameters
    ----------
    n_neighbors
        Positive number of neighbors ``k`` that vote.
    metric
        ``"euclidean"`` or ``"manhattan"`` distance.
    weights
        ``"uniform"`` gives each neighbor one vote; ``"distance"`` weights
        votes by inverse distance. Exact duplicates of a query (distance zero)
        then decide the vote on their own.

    Attributes
    ----------
    classes_ : ndarray of shape (K,)
        Sorted class labels seen during ``fit``.
    n_features_in_ : int
        Number of features seen during ``fit``.

    Notes
    -----
    Distances are only meaningful when features share comparable units, so
    standardize features (inside a pipeline, after the train/test split)
    before fitting.
    """

    def __init__(
        self,
        *,
        n_neighbors: int = 5,
        metric: str = "euclidean",
        weights: str = "uniform",
    ) -> None:
        if int(n_neighbors) != n_neighbors or n_neighbors <= 0:
            raise ValueError("n_neighbors must be a positive integer")
        if metric not in _METRICS:
            raise ValueError(f"metric must be one of {_METRICS}")
        if weights not in _WEIGHTS:
            raise ValueError(f"weights must be one of {_WEIGHTS}")
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.weights = weights

    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self:
        """Store a validated copy of the training data."""

        array = _validated_features(features)
        labels = np.asarray(targets)
        if labels.shape != (array.shape[0],):
            raise ValueError("targets must contain one label per observation")
        if self.n_neighbors > array.shape[0]:
            raise ValueError("n_neighbors cannot exceed the number of training rows")
        self.classes_, self._encoded_targets = np.unique(labels, return_inverse=True)
        self._training_features = array.copy()
        self.n_features_in_ = array.shape[1]
        return self

    def kneighbors(
        self, features: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
        """Return distances to and training indices of each query's neighbors."""

        if not hasattr(self, "_training_features"):
            raise RuntimeError("fit must be called before prediction")
        array = _validated_features(features)
        if array.shape[1] != self.n_features_in_:
            raise ValueError("features have the wrong shape for this fitted model")
        # Score queries in blocks so memory stays O(block * n * d) rather than
        # O(m * n * d); results are identical to scoring all rows at once.
        block = max(1, _BLOCK_ELEMENTS // max(1, self._training_features.size))
        all_distances, all_indices = [], []
        for start in range(0, array.shape[0], block):
            distances = pairwise_distances(
                array[start : start + block], self._training_features, self.metric
            )
            order = np.argsort(distances, axis=1, kind="stable")[:, : self.n_neighbors]
            all_distances.append(np.take_along_axis(distances, order, axis=1))
            all_indices.append(order)
        return np.vstack(all_distances), np.vstack(all_indices)

    def predict_proba(self, features: ArrayLike) -> NDArray[np.float64]:
        """Return normalized (weighted) vote shares; columns follow ``classes_``."""

        distances, indices = self.kneighbors(features)
        neighbor_classes = self._encoded_targets[indices]
        if self.weights == "uniform":
            vote_weights = np.ones_like(distances)
        else:
            exact = distances == 0.0
            with np.errstate(divide="ignore"):
                vote_weights = np.where(exact, 0.0, 1.0 / distances)
            has_exact = exact.any(axis=1)
            vote_weights[has_exact] = exact[has_exact].astype(float)

        votes = np.zeros((distances.shape[0], self.classes_.size), dtype=float)
        rows = np.repeat(np.arange(distances.shape[0]), self.n_neighbors)
        np.add.at(votes, (rows, neighbor_classes.ravel()), vote_weights.ravel())
        return votes / votes.sum(axis=1, keepdims=True)

    def predict(self, features: ArrayLike) -> NDArray[np.generic]:
        """Return the winning label for each query row."""

        probabilities = self.predict_proba(features)
        return self.classes_[np.argmax(probabilities, axis=1)]


def _validated_features(features: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(features, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("features must be a nonempty two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError("features must contain only finite values")
    return array
