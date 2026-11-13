"""Multinomial logistic (softmax) regression fitted by batch gradient descent.

Mathematical contract
---------------------
Let :math:`X \\in \\mathbb{R}^{n \\times d}` hold one observation per row and let
:math:`Y \\in \\{0, 1\\}^{n \\times K}` be the one-hot encoding of ``K`` class
labels. The model has weights :math:`W \\in \\mathbb{R}^{d \\times K}` and
intercepts :math:`b \\in \\mathbb{R}^{K}` and predicts class probabilities

.. math::

    P = \\operatorname{softmax}(X W + \\mathbf{1} b^\\top),
    \\qquad P_{ik} = \\frac{e^{z_{ik}}}{\\sum_j e^{z_{ij}}}.

Fitting minimizes the mean cross-entropy plus an L2 penalty on ``W`` (the
intercepts are not penalized):

.. math::

    J(W, b) = -\\frac{1}{n} \\sum_{i,k} Y_{ik} \\log P_{ik}
              + \\frac{\\lambda}{2} \\lVert W \\rVert_F^2 .

Its gradients are

.. math::

    \\nabla_W J = \\frac{1}{n} X^\\top (P - Y) + \\lambda W,
    \\qquad
    \\nabla_b J = \\frac{1}{n} \\mathbf{1}^\\top (P - Y).

``J`` is convex, so batch gradient descent with a small enough step size
converges to a global minimizer. With ``l2_penalty > 0`` the minimizer is
unique, which makes a differential comparison with scikit-learn's
``LogisticRegression(C=1 / (n * l2_penalty))`` meaningful.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin


def softmax(scores: ArrayLike) -> NDArray[np.float64]:
    """Return row-wise softmax probabilities computed in a numerically stable way.

    Subtracting each row's maximum leaves the result unchanged mathematically
    but prevents ``exp`` from overflowing for large scores.
    """

    array = np.asarray(scores, dtype=float)
    if array.ndim != 2:
        raise ValueError("scores must be a two-dimensional array")
    shifted = array - array.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def cross_entropy_objective(
    weights: NDArray[np.float64],
    intercepts: NDArray[np.float64],
    features: NDArray[np.float64],
    one_hot_targets: NDArray[np.float64],
    l2_penalty: float = 0.0,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate the penalized mean cross-entropy and its exact gradients.

    Parameters
    ----------
    weights
        Weight matrix with shape ``(d, K)``.
    intercepts
        Intercept vector with shape ``(K,)``.
    features
        Design matrix with shape ``(n, d)``.
    one_hot_targets
        One-hot label matrix with shape ``(n, K)``.
    l2_penalty
        Nonnegative ridge strength applied to ``weights`` only.

    Returns
    -------
    tuple
        ``(objective, weight_gradient, intercept_gradient)``.
    """

    scores = features @ weights + intercepts
    # log-sum-exp keeps log(P) finite even when one probability underflows.
    row_max = scores.max(axis=1, keepdims=True)
    log_normalizer = row_max + np.log(
        np.exp(scores - row_max).sum(axis=1, keepdims=True)
    )
    log_probabilities = scores - log_normalizer
    sample_count = features.shape[0]
    data_loss = -float(np.sum(one_hot_targets * log_probabilities)) / sample_count
    penalty = 0.5 * l2_penalty * float(np.sum(weights**2))

    residual = np.exp(log_probabilities) - one_hot_targets
    weight_gradient = features.T @ residual / sample_count + l2_penalty * weights
    intercept_gradient = residual.mean(axis=0)
    return data_loss + penalty, weight_gradient, intercept_gradient


class SoftmaxRegression(ClassifierMixin, BaseEstimator):
    """Multiclass logistic regression fitted by batch gradient descent.

    Parameters
    ----------
    learning_rate
        Positive gradient-descent step size.
    max_iterations
        Positive maximum number of parameter updates.
    tolerance
        Nonnegative stopping threshold on the Euclidean norm of the full
        gradient.
    l2_penalty
        Nonnegative ridge strength :math:`\\lambda` applied to the weights.

    Attributes
    ----------
    classes_ : ndarray of shape (K,)
        Sorted class labels seen during ``fit``.
    coef_ : ndarray of shape (d, K)
        Fitted weight matrix ``W``.
    intercept_ : ndarray of shape (K,)
        Fitted intercepts ``b``.
    loss_history_ : ndarray
        Objective value before each update; useful for convergence plots.
    n_iter_ : int
        Number of updates actually performed.
    converged_ : bool
        Whether the gradient-norm tolerance was reached.

    Raises
    ------
    FloatingPointError
        From ``fit`` when the objective becomes non-finite, which signals that
        the step size is too large for the curvature of the problem.

    Notes
    -----
    Scale features before fitting: gradient descent with one global step size
    converges slowly when features have very different units. A full-batch
    iteration costs :math:`O(n d K)`.
    """

    def __init__(
        self,
        *,
        learning_rate: float = 0.1,
        max_iterations: int = 5_000,
        tolerance: float = 1e-6,
        l2_penalty: float = 0.0,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if tolerance < 0:
            raise ValueError("tolerance must be nonnegative")
        if l2_penalty < 0:
            raise ValueError("l2_penalty must be nonnegative")
        self.learning_rate = float(learning_rate)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.l2_penalty = float(l2_penalty)

    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self:
        """Fit weights and intercepts and record the objective trace."""

        array = _validated_features(features)
        labels = np.asarray(targets)
        if labels.shape != (array.shape[0],):
            raise ValueError("targets must contain one label per observation")
        classes, encoded = np.unique(labels, return_inverse=True)
        if classes.size < 2:
            raise ValueError("at least two distinct classes are required")

        sample_count, feature_count = array.shape
        one_hot = np.zeros((sample_count, classes.size), dtype=float)
        one_hot[np.arange(sample_count), encoded] = 1.0

        weights = np.zeros((feature_count, classes.size), dtype=float)
        intercepts = np.zeros(classes.size, dtype=float)
        losses: list[float] = []
        converged = False
        iterations = 0
        for iterations in range(self.max_iterations):  # noqa: B007
            with np.errstate(over="ignore", invalid="ignore"):
                loss, weight_gradient, intercept_gradient = cross_entropy_objective(
                    weights, intercepts, array, one_hot, self.l2_penalty
                )
            if not np.isfinite(loss):
                raise FloatingPointError(
                    "gradient descent diverged; reduce learning_rate or scale "
                    "the features"
                )
            losses.append(loss)
            gradient_norm = float(
                np.sqrt(np.sum(weight_gradient**2) + np.sum(intercept_gradient**2))
            )
            if gradient_norm <= self.tolerance:
                converged = True
                break
            weights -= self.learning_rate * weight_gradient
            intercepts -= self.learning_rate * intercept_gradient
        else:
            iterations = self.max_iterations

        self.classes_ = classes
        self.coef_ = weights
        self.intercept_ = intercepts
        self.loss_history_ = np.asarray(losses)
        self.n_iter_ = iterations
        self.converged_ = converged
        self.n_features_in_ = feature_count
        return self

    def decision_function(self, features: ArrayLike) -> NDArray[np.float64]:
        """Return the linear class scores ``X W + b``."""

        if not hasattr(self, "coef_"):
            raise RuntimeError("fit must be called before prediction")
        array = _validated_features(features)
        if array.shape[1] != self.n_features_in_:
            raise ValueError("features have the wrong shape for this fitted model")
        return array @ self.coef_ + self.intercept_

    def predict_proba(self, features: ArrayLike) -> NDArray[np.float64]:
        """Return class probabilities; columns follow ``classes_``."""

        return softmax(self.decision_function(features))

    def predict(self, features: ArrayLike) -> NDArray[np.generic]:
        """Return the most probable class label for each row."""

        scores = self.decision_function(features)
        return self.classes_[np.argmax(scores, axis=1)]


def _validated_features(features: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(features, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("features must be a nonempty two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError("features must contain only finite values")
    return array
