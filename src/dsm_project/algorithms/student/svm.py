"""Linear support vector machine trained by subgradient descent.

Mathematical contract
---------------------
Labels are mapped to :math:`y_i \\in \\{-1, +1\\}` (the larger label is
:math:`+1`). The model scores a row with :math:`f(x) = w^\\top x + b` and
minimizes the regularized **hinge loss**

.. math::

    J(w, b) = \\frac{\\lambda}{2} \\lVert w \\rVert^2
              + \\frac{1}{S} \\sum_{i=1}^{n}
                s_i \\max\\bigl(0,\\; 1 - y_i f(x_i)\\bigr),

with optional row weights :math:`s_i` and :math:`S = \\sum_i s_i`. The hinge is
zero for rows classified correctly *with margin* (:math:`y_i f(x_i) \\ge 1`),
so only rows on or inside the margin, the support vectors, shape the
solution. The intercept :math:`b` is not penalized.

The hinge is not differentiable at the margin, so training uses a
**subgradient**:

.. math::

    g_w = \\lambda w - \\frac{1}{S}\\sum_{i \\in M} s_i y_i x_i, \\qquad
    g_b = -\\frac{1}{S}\\sum_{i \\in M} s_i y_i,

where :math:`M` is the set of rows with :math:`y_i f(x_i) < 1`. Subgradient
steps do not decrease :math:`J` every time, so the step size shrinks as
:math:`\\eta_t = \\eta_0 / \\sqrt{t}` and the iterate with the lowest objective
is kept. :math:`J` is convex, so the best iterate approaches the optimum.

With :math:`\\lambda = 1 / (C n)` this is the objective scikit-learn's
``SVC(kernel="linear", C=C)`` minimizes (unweighted case), which is how the
implementation is verified.

An SVM gives scores, not probabilities. ``decision_function`` is the right
input for ranking metrics such as average precision.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin


def hinge_objective(
    weights: NDArray[np.float64],
    intercept: float,
    features: NDArray[np.float64],
    signed_targets: NDArray[np.float64],
    l2_penalty: float,
    row_weights: NDArray[np.float64] | None = None,
) -> tuple[float, NDArray[np.float64], float]:
    """Return the objective and one subgradient ``(J, g_w, g_b)``."""

    if row_weights is None:
        row_weights = np.ones(features.shape[0])
    total = float(row_weights.sum())
    margins = signed_targets * (features @ weights + intercept)
    violated = margins < 1.0
    hinge = float(row_weights @ np.maximum(0.0, 1.0 - margins)) / total
    objective = 0.5 * l2_penalty * float(weights @ weights) + hinge
    pull = row_weights[violated] * signed_targets[violated]
    gradient_w = l2_penalty * weights - features[violated].T @ pull / total
    gradient_b = -float(pull.sum()) / total
    return objective, gradient_w, gradient_b


class LinearSVM(ClassifierMixin, BaseEstimator):
    """Binary linear SVM (hinge loss, L2 penalty) fitted by subgradient descent.

    Parameters
    ----------
    l2_penalty
        Positive ridge strength :math:`\\lambda`; smaller values allow a
        narrower margin with fewer violations.
    learning_rate
        Initial step size :math:`\\eta_0`.
    max_iterations
        Number of full-batch subgradient steps.
    class_weight
        ``None``, ``"balanced"`` (row weight :math:`n / (2 n_k)`), or a mapping
        from label to positive weight.

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
    coef_ : ndarray of shape (d,)
    intercept_ : float
    objective_history_ : ndarray
        Objective at every step (it is not monotone; the best is kept).
    best_iteration_ : int
    """

    def __init__(
        self,
        *,
        l2_penalty: float = 0.01,
        learning_rate: float = 1.0,
        max_iterations: int = 2_000,
        class_weight: str | dict | None = None,
    ) -> None:
        if not l2_penalty > 0:
            raise ValueError("l2_penalty must be positive")
        if not learning_rate > 0:
            raise ValueError("learning_rate must be positive")
        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        if isinstance(class_weight, dict):
            if any(not weight > 0 for weight in class_weight.values()):
                raise ValueError("class weights must be positive")
        elif class_weight not in (None, "balanced"):
            raise ValueError("class_weight must be None, 'balanced', or a dict")
        self.l2_penalty = l2_penalty
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.class_weight = class_weight

    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self:
        """Run subgradient descent and keep the lowest-objective iterate."""

        rows = _validated(features)
        labels = np.asarray(targets)
        if labels.shape != (rows.shape[0],):
            raise ValueError("targets must contain one label per observation")
        classes, encoded = np.unique(labels, return_inverse=True)
        if classes.size != 2:
            raise ValueError("LinearSVM needs exactly two classes")
        signed = np.where(encoded == 1, 1.0, -1.0)
        row_weights = self._row_weights(classes, encoded)

        weights = np.zeros(rows.shape[1])
        intercept = 0.0
        best = (np.inf, weights.copy(), intercept, 0)
        history = []
        for step in range(1, self.max_iterations + 1):
            objective, gradient_w, gradient_b = hinge_objective(
                weights, intercept, rows, signed, self.l2_penalty, row_weights
            )
            history.append(objective)
            if objective < best[0]:
                best = (objective, weights.copy(), intercept, step - 1)
            rate = self.learning_rate / np.sqrt(step)
            weights = weights - rate * gradient_w
            intercept = intercept - rate * gradient_b
        final = hinge_objective(
            weights, intercept, rows, signed, self.l2_penalty, row_weights
        )
        if final[0] < best[0]:
            best = (final[0], weights, intercept, self.max_iterations)

        self.objective_, self.coef_, self.intercept_, self.best_iteration_ = best
        self.objective_history_ = np.asarray(history)
        self.classes_ = classes
        self.n_features_in_ = rows.shape[1]
        return self

    def _row_weights(
        self, classes: NDArray[np.generic], encoded: NDArray[np.intp]
    ) -> NDArray[np.float64]:
        if self.class_weight is None:
            return np.ones(encoded.size)
        if self.class_weight == "balanced":
            counts = np.bincount(encoded, minlength=2)
            return (encoded.size / (2 * counts))[encoded]
        per_class = np.array(
            [float(self.class_weight.get(c, 1.0)) for c in classes.tolist()]
        )
        return per_class[encoded]

    def decision_function(self, features: ArrayLike) -> NDArray[np.float64]:
        """Signed score :math:`w^\\top x + b`; positive favors ``classes_[1]``."""

        if not hasattr(self, "coef_"):
            raise RuntimeError("fit must be called before prediction")
        rows = _validated(features)
        if rows.shape[1] != self.n_features_in_:
            raise ValueError("features have the wrong shape for this fitted model")
        return rows @ self.coef_ + self.intercept_

    def predict(self, features: ArrayLike) -> NDArray[np.generic]:
        """Return ``classes_[1]`` where the score is positive, else ``classes_[0]``."""

        scores = self.decision_function(features)  # checks the model is fitted
        return self.classes_[(scores > 0).astype(int)]


def _validated(features: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(features, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("features must be a nonempty two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError("features must contain only finite values")
    return array
