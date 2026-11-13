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

Fitting minimizes the (optionally weighted) mean cross-entropy plus an L2
penalty on ``W`` (the intercepts are not penalized). With sample weights
:math:`s_i > 0` and :math:`S = \\sum_i s_i`:

.. math::

    J(W, b) = -\\frac{1}{S} \\sum_{i,k} s_i Y_{ik} \\log P_{ik}
              + \\frac{\\lambda}{2} \\lVert W \\rVert_F^2 .

Its gradients are

.. math::

    \\nabla_W J = \\frac{1}{S} X^\\top \\operatorname{diag}(s) (P - Y) + \\lambda W,
    \\qquad
    \\nabla_b J = \\frac{1}{S} s^\\top (P - Y).

Unweighted fitting is the case :math:`s_i = 1`. Class weighting sets
:math:`s_i` from the label of row ``i`` so that a rare class (such as churned
users) contributes as much to the objective as a common one.

``J`` is convex, so batch gradient descent with a small enough step size
converges to a global minimizer. With ``l2_penalty > 0`` the minimizer is
unique, which makes a differential comparison with scikit-learn's
``LogisticRegression(C=1 / (S * l2_penalty))`` meaningful for three or more
classes. For two classes scikit-learn fits a single weight vector
:math:`w = w_1 - w_0`; at the softmax optimum :math:`w_0 = -w_1`, so the ridge
term equals :math:`(\\lambda / 4) \\lVert w \\rVert^2` and the matching setting
is ``C = 2 / (S * l2_penalty)``. Predicted probabilities are identical.
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
    sample_weights: NDArray[np.float64] | None = None,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate the penalized, weighted mean cross-entropy and its gradients.

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
    sample_weights
        Optional positive weight per row, shape ``(n,)``; ``None`` means equal
        weights.

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
    if sample_weights is None:
        sample_weights = np.ones(features.shape[0])
    total_weight = float(sample_weights.sum())
    row_losses = -np.sum(one_hot_targets * log_probabilities, axis=1)
    data_loss = float(sample_weights @ row_losses) / total_weight
    penalty = 0.5 * l2_penalty * float(np.sum(weights**2))

    residual = (np.exp(log_probabilities) - one_hot_targets) * sample_weights[:, None]
    weight_gradient = features.T @ residual / total_weight + l2_penalty * weights
    intercept_gradient = residual.sum(axis=0) / total_weight
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
    class_weight
        ``None`` (all rows weigh 1), ``"balanced"`` (row weight
        :math:`n / (K n_k)` for a row of class ``k``, so every class carries
        equal total weight), or a mapping from label to a positive weight;
        labels missing from the mapping get weight 1.

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
        class_weight: str | dict | None = None,
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
        if isinstance(class_weight, dict):
            if any(not weight > 0 for weight in class_weight.values()):
                raise ValueError("class weights must be positive")
        elif class_weight not in (None, "balanced"):
            raise ValueError("class_weight must be None, 'balanced', or a dict")
        self.l2_penalty = float(l2_penalty)
        self.class_weight = class_weight

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
        sample_weights = self._row_weights(classes, encoded)

        weights = np.zeros((feature_count, classes.size), dtype=float)
        intercepts = np.zeros(classes.size, dtype=float)
        losses: list[float] = []
        converged = False
        iterations = 0
        for iterations in range(self.max_iterations):  # noqa: B007
            with np.errstate(over="ignore", invalid="ignore"):
                loss, weight_gradient, intercept_gradient = cross_entropy_objective(
                    weights, intercepts, array, one_hot, self.l2_penalty,
                    sample_weights,
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

    def _row_weights(
        self, classes: NDArray[np.generic], encoded: NDArray[np.intp]
    ) -> NDArray[np.float64]:
        if self.class_weight is None:
            return np.ones(encoded.size)
        if self.class_weight == "balanced":
            counts = np.bincount(encoded, minlength=classes.size)
            per_class = encoded.size / (classes.size * counts)
        else:
            per_class = np.array(
                [float(self.class_weight.get(label, 1.0)) for label in classes.tolist()]
            )
        return per_class[encoded]

    def class_scores(self, features: ArrayLike) -> NDArray[np.float64]:
        """Return the linear score of every class, ``X W + b``, shape ``(n, K)``."""

        if not hasattr(self, "coef_"):
            raise RuntimeError("fit must be called before prediction")
        array = _validated_features(features)
        if array.shape[1] != self.n_features_in_:
            raise ValueError("features have the wrong shape for this fitted model")
        return array @ self.coef_ + self.intercept_

    def decision_function(self, features: ArrayLike) -> NDArray[np.float64]:
        """Return scores in scikit-learn's convention.

        For two classes this is one log-odds score per row,
        :math:`z_1 - z_0 = \\log(p_1 / p_0)`, shape ``(n,)``; scikit-learn's
        ranking metrics (average precision, ROC-AUC) require that shape. For
        three or more classes it is the ``(n, K)`` matrix of class scores.
        """

        scores = self.class_scores(features)
        if scores.shape[1] == 2:
            return scores[:, 1] - scores[:, 0]
        return scores

    def predict_proba(self, features: ArrayLike) -> NDArray[np.float64]:
        """Return class probabilities; columns follow ``classes_``."""

        return softmax(self.class_scores(features))

    def predict(self, features: ArrayLike) -> NDArray[np.generic]:
        """Return the most probable class label for each row."""

        scores = self.class_scores(features)
        return self.classes_[np.argmax(scores, axis=1)]


def _validated_features(features: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(features, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("features must be a nonempty two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError("features must contain only finite values")
    return array
