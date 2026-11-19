"""Unsupervised learning: k-means clustering and principal component analysis.

k-means
-------
Given rows :math:`x_1, \\dots, x_n` and a number of clusters :math:`k`, k-means
looks for centers :math:`\\mu_1, \\dots, \\mu_k` that minimize the **inertia**,
the total squared distance from each row to its nearest center:

.. math::

    J(\\mu) = \\sum_{i=1}^{n} \\min_{j} \\lVert x_i - \\mu_j \\rVert^2 .

Lloyd's algorithm alternates two steps that can only lower :math:`J`: assign
each row to its nearest center, then move each center to the mean of its rows.
It stops when no row changes cluster (or the centers stop moving). Because it
finds a *local* minimum, it is restarted ``n_init`` times from k-means++
starting centers (each new center drawn with probability proportional to its
squared distance from the centers already chosen) and the run with the lowest
inertia is kept.

PCA
---
PCA finds orthogonal directions of maximum variance. With the centered data
matrix :math:`X_c = X - \\bar x` and its thin singular value decomposition
:math:`X_c = U S V^\\top`, the rows of :math:`V^\\top` are the principal
directions, :math:`X_c V` are the coordinates (scores), and the variance
explained by component :math:`j` is :math:`s_j^2 / (n - 1)`. Keeping the first
:math:`m` components gives the best rank-:math:`m` approximation of
:math:`X_c` in squared error. Signs of components are arbitrary; they are fixed
so the largest-magnitude entry of each component is positive, scikit-learn's
convention.
"""

from __future__ import annotations

from numbers import Integral
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin


def _validated(features: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(features, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("features must be a nonempty two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError("features must contain only finite values")
    return array


def squared_distances(
    rows: NDArray[np.float64], centers: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return the ``(n, k)`` matrix of squared Euclidean distances."""

    distances = (
        np.sum(rows**2, axis=1)[:, None]
        - 2.0 * rows @ centers.T
        + np.sum(centers**2, axis=1)[None, :]
    )
    return np.maximum(distances, 0.0)  # remove tiny negatives from rounding


def kmeans_plus_plus(
    rows: NDArray[np.float64], k: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Choose ``k`` starting centers with the k-means++ rule."""

    centers = [rows[rng.integers(rows.shape[0])]]
    closest = squared_distances(rows, np.asarray(centers))[:, 0]
    for _ in range(1, k):
        total = closest.sum()
        if total <= 0:  # every row coincides with a chosen center
            choice = rng.integers(rows.shape[0])
        else:
            choice = rng.choice(rows.shape[0], p=closest / total)
        centers.append(rows[choice])
        closest = np.minimum(closest, squared_distances(rows, rows[[choice]])[:, 0])
    return np.asarray(centers)


class KMeans(ClusterMixin, BaseEstimator):
    """k-means clustering with Lloyd's algorithm and k-means++ restarts.

    Parameters
    ----------
    n_clusters
        Number of clusters :math:`k`.
    n_init
        Number of restarts from different k-means++ starts; the lowest-inertia
        run is kept. Ignored (treated as 1) when ``init`` is an array.
    max_iter
        Maximum Lloyd iterations per restart.
    tol
        Stop when the total squared movement of the centers falls to
        ``tol`` times the mean feature variance (scikit-learn's convention);
        ``0`` runs until no row changes cluster.
    init
        ``"k-means++"`` or an array of starting centers, shape ``(k, d)``.
    random_state
        Seed for the k-means++ starts.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (k, d)
    labels_ : ndarray of shape (n,)
    inertia_ : float
        Sum of squared distances from rows to their assigned center.
    n_iter_ : int
        Lloyd iterations used by the kept run.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        *,
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        init: str | ArrayLike = "k-means++",
        random_state: int | None = None,
    ) -> None:
        if not isinstance(n_clusters, Integral) or n_clusters < 1:
            raise ValueError("n_clusters must be a positive integer")
        if not isinstance(n_init, Integral) or n_init < 1:
            raise ValueError("n_init must be a positive integer")
        if not isinstance(max_iter, Integral) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if tol < 0:
            raise ValueError("tol must be nonnegative")
        if isinstance(init, str) and init != "k-means++":
            raise ValueError("init must be 'k-means++' or an array of centers")
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state

    def fit(self, features: ArrayLike, targets: object = None) -> Self:
        """Run ``n_init`` restarts of Lloyd's algorithm and keep the best."""

        rows = _validated(features)
        if self.n_clusters > rows.shape[0]:
            raise ValueError("n_clusters cannot exceed the number of rows")
        rng = np.random.default_rng(self.random_state)
        threshold = self.tol * float(np.mean(np.var(rows, axis=0)))
        fixed_start = not isinstance(self.init, str)
        if fixed_start:
            start = np.asarray(self.init, dtype=float)
            if start.shape != (self.n_clusters, rows.shape[1]):
                raise ValueError(
                    "init centers must have shape (n_clusters, n_features)"
                )

        best = None
        for _ in range(1 if fixed_start else self.n_init):
            centers = (
                start.copy()
                if fixed_start
                else kmeans_plus_plus(rows, self.n_clusters, rng)
            )
            result = self._lloyd(rows, centers, threshold)
            if best is None or result[2] < best[2]:
                best = result
        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = best
        self.n_features_in_ = rows.shape[1]
        return self

    def _lloyd(
        self, rows: NDArray[np.float64], centers: NDArray[np.float64], threshold: float
    ) -> tuple[NDArray[np.float64], NDArray[np.intp], float, int]:
        labels = np.full(rows.shape[0], -1)
        iterations = 0
        for iterations in range(1, self.max_iter + 1):  # noqa: B007
            distances = squared_distances(rows, centers)
            new_labels = np.argmin(distances, axis=1)
            new_centers = centers.copy()
            for j in range(self.n_clusters):
                members = new_labels == j
                if members.any():
                    new_centers[j] = rows[members].mean(axis=0)
                else:
                    # Empty cluster: restart it at the row farthest from its center.
                    farthest = int(
                        np.argmax(distances[np.arange(rows.shape[0]), new_labels])
                    )
                    new_centers[j] = rows[farthest]
                    new_labels[farthest] = j
            shift = float(np.sum((new_centers - centers) ** 2))
            unchanged = np.array_equal(new_labels, labels)
            centers, labels = new_centers, new_labels
            if unchanged or shift <= threshold:
                break
        # Final assignment to the final centers, so labels and inertia agree.
        distances = squared_distances(rows, centers)
        labels = np.argmin(distances, axis=1)
        inertia = float(distances[np.arange(rows.shape[0]), labels].sum())
        return centers, labels, inertia, iterations

    def _check(self, features: ArrayLike) -> NDArray[np.float64]:
        if not hasattr(self, "cluster_centers_"):
            raise RuntimeError("fit must be called before predict")
        rows = _validated(features)
        if rows.shape[1] != self.n_features_in_:
            raise ValueError("features have the wrong shape for this fitted model")
        return rows

    def predict(self, features: ArrayLike) -> NDArray[np.intp]:
        """Assign each row to its nearest fitted center."""

        rows = self._check(features)
        return np.argmin(squared_distances(rows, self.cluster_centers_), axis=1)

    def transform(self, features: ArrayLike) -> NDArray[np.float64]:
        """Return Euclidean distances from each row to every center."""

        rows = self._check(features)
        return np.sqrt(squared_distances(rows, self.cluster_centers_))

    def score(self, features: ArrayLike, targets: object = None) -> float:
        """Negative inertia of ``features`` under the fitted centers."""

        rows = self._check(features)
        return -float(
            np.min(squared_distances(rows, self.cluster_centers_), axis=1).sum()
        )


class PCA(TransformerMixin, BaseEstimator):
    """Principal component analysis by singular value decomposition.

    Parameters
    ----------
    n_components
        Number of components to keep; ``None`` keeps ``min(n, d)``.

    Attributes
    ----------
    mean_ : ndarray of shape (d,)
    components_ : ndarray of shape (m, d)
        Orthonormal principal directions, largest variance first.
    explained_variance_ : ndarray of shape (m,)
        Variance of the data along each component, :math:`s_j^2 / (n - 1)`.
    explained_variance_ratio_ : ndarray of shape (m,)
        Share of the total variance explained by each component.
    singular_values_ : ndarray of shape (m,)
    """

    def __init__(self, n_components: int | None = None) -> None:
        if n_components is not None and (
            not isinstance(n_components, Integral) or n_components < 1
        ):
            raise ValueError("n_components must be a positive integer or None")
        self.n_components = n_components

    def fit(self, features: ArrayLike, targets: object = None) -> Self:
        """Center the data and compute its principal directions."""

        rows = _validated(features)
        n, d = rows.shape
        if n < 2:
            raise ValueError("PCA needs at least two rows")
        limit = min(n, d)
        keep = limit if self.n_components is None else self.n_components
        if keep > limit:
            raise ValueError("n_components cannot exceed min(n_samples, n_features)")

        self.mean_ = rows.mean(axis=0)
        _, singular, vt = np.linalg.svd(rows - self.mean_, full_matrices=False)
        # Deterministic signs: largest-magnitude entry of each component positive.
        signs = np.sign(vt[np.arange(vt.shape[0]), np.argmax(np.abs(vt), axis=1)])
        signs[signs == 0] = 1.0
        vt = vt * signs[:, None]

        variance = singular**2 / (n - 1)
        total = variance.sum()
        self.components_ = vt[:keep]
        self.singular_values_ = singular[:keep]
        self.explained_variance_ = variance[:keep]
        self.explained_variance_ratio_ = (
            variance[:keep] / total if total > 0 else np.zeros(keep)
        )
        self.n_components_ = keep
        self.n_features_in_ = d
        return self

    def transform(self, features: ArrayLike) -> NDArray[np.float64]:
        """Project rows onto the kept components (the PCA scores)."""

        if not hasattr(self, "components_"):
            raise RuntimeError("fit must be called before transform")
        rows = _validated(features)
        if rows.shape[1] != self.n_features_in_:
            raise ValueError("features have the wrong shape for this fitted model")
        return (rows - self.mean_) @ self.components_.T

    def inverse_transform(self, scores: ArrayLike) -> NDArray[np.float64]:
        """Map scores back to the original feature space (a reconstruction)."""

        if not hasattr(self, "components_"):
            raise RuntimeError("fit must be called before inverse_transform")
        values = np.asarray(scores, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_components_:
            raise ValueError("scores must have one column per kept component")
        return values @ self.components_ + self.mean_
