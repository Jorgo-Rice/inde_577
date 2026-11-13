"""Tree ensembles: bagged random forests and gradient boosting.

Random forest (bagging + feature sampling)
------------------------------------------
Each of :math:`B` trees is grown on a bootstrap sample (``n`` rows drawn with
replacement) and examines only a random subset of features at each node. The
forest predicts the average of the trees' class probabilities,

.. math::

    \\hat p(x) = \\frac{1}{B} \\sum_{b=1}^{B} \\hat p_b(x).

If each tree's prediction has variance :math:`\\sigma^2` and two trees'
predictions have correlation :math:`\\rho`, the average has variance
:math:`\\rho\\sigma^2 + (1-\\rho)\\sigma^2 / B`. Bootstrapping and feature
sampling lower :math:`\\rho`, so averaging removes much of a deep tree's
variance while keeping its low bias. Rows left out of a tree's bootstrap
sample ("out-of-bag", about 37% of rows) give a free validation estimate.

Gradient boosting for binary log-loss
-------------------------------------
Boosting builds an additive score :math:`F(x)` in stages. With
:math:`p = \\sigma(F) = 1/(1+e^{-F})` and labels :math:`y \\in \\{0, 1\\}`, the
log-loss :math:`L = -[y\\log p + (1-y)\\log(1-p)]` has negative gradient
:math:`r = y - p` with respect to :math:`F`. Each stage:

1. computes the residuals :math:`r_i = y_i - p_i`;
2. fits a small regression tree to them (squared error);
3. replaces each leaf's value with one Newton step on the log-loss,
   :math:`\\gamma = \\sum_{\\text{leaf}} r_i / \\sum_{\\text{leaf}} p_i(1-p_i)`;
4. updates :math:`F \\leftarrow F + \\eta\\,\\gamma_{\\text{leaf}(x)}` with learning
   rate :math:`\\eta`.

The starting score is the log-odds of the training churn rate. This is the
algorithm scikit-learn's ``GradientBoostingClassifier`` implements, which
makes exact differential testing possible.
"""

from __future__ import annotations

from collections.abc import Iterator
from numbers import Integral, Real
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin

from dsm_project.algorithms.student.trees import (
    ClassificationTree,
    RegressionTree,
    _class_row_weights,
    _validated_features,
)


def _positive_int(value: object, name: str) -> None:
    if not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


class RandomForest(ClassifierMixin, BaseEstimator):
    """Random forest classifier built from student ``ClassificationTree``s.

    Parameters
    ----------
    n_estimators
        Number of trees :math:`B`.
    max_depth, min_samples_split, min_samples_leaf, criterion
        Passed to every tree.
    max_features
        Features examined per node (``"sqrt"`` by default, the usual choice for
        classification forests); ``None`` gives plain bagging.
    bootstrap
        Draw a bootstrap sample for each tree; ``False`` fits every tree on all
        rows (diversity then comes only from feature sampling).
    class_weight
        ``None``, ``"balanced"`` (computed once on the full training labels), or
        a label-to-weight mapping.
    random_state
        Seed for bootstrap samples and per-tree feature sampling.

    Attributes
    ----------
    estimators_ : list of ClassificationTree
    classes_, n_features_in_, feature_importances_
        Importances are the mean of the trees' normalized importances.
    oob_proba_ : ndarray of shape (n, K)
        Out-of-bag class probabilities for each training row (``NaN`` for a
        row that was in every bootstrap sample). Only when ``bootstrap=True``.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | float | str | None = "sqrt",
        bootstrap: bool = True,
        class_weight: str | dict | None = None,
        random_state: int | None = None,
    ) -> None:
        _positive_int(n_estimators, "n_estimators")
        # Validate the tree settings once, up front, with a throwaway tree.
        ClassificationTree(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            max_features=max_features,
        )
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self:
        """Grow ``n_estimators`` trees on bootstrap samples."""

        array = _validated_features(features)
        labels = np.asarray(targets)
        if labels.shape != (array.shape[0],):
            raise ValueError("targets must contain one label per observation")
        classes, encoded = np.unique(labels, return_inverse=True)
        rows = array.shape[0]
        class_weights = _class_row_weights(self.class_weight, classes, encoded)
        rng = np.random.default_rng(self.random_state)

        self.estimators_ = []
        oob_sum = np.zeros((rows, classes.size))
        oob_count = np.zeros(rows)
        for _ in range(self.n_estimators):
            tree = ClassificationTree(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=int(rng.integers(2**32 - 1)),
            )
            if self.bootstrap:
                counts = np.bincount(rng.integers(0, rows, rows), minlength=rows)
            else:
                counts = np.ones(rows, dtype=int)
            used = counts > 0
            # Rows drawn k times enter once with weight k, as in scikit-learn.
            tree.fit(
                array[used],
                encoded[used],
                sample_weight=class_weights[used] * counts[used],
            )
            self.estimators_.append(tree)
            if self.bootstrap and (~used).any():
                oob_sum[~used] += self._tree_proba(tree, array[~used], classes.size)
                oob_count[~used] += 1

        self.classes_ = classes
        self.n_features_in_ = array.shape[1]
        self.feature_importances_ = np.mean(
            [tree.feature_importances_ for tree in self.estimators_], axis=0
        )
        if self.bootstrap:
            with np.errstate(invalid="ignore"):
                self.oob_proba_ = oob_sum / oob_count[:, None]
        return self

    @staticmethod
    def _tree_proba(
        tree: ClassificationTree, features: NDArray[np.float64], n_classes: int
    ) -> NDArray[np.float64]:
        """Tree probabilities in the forest's class order.

        A bootstrap sample can miss a rare class entirely; that tree then knows
        fewer classes, so its columns are placed by their encoded label.
        """

        probabilities = np.zeros((features.shape[0], n_classes))
        probabilities[:, tree.classes_] = tree.predict_proba(features)
        return probabilities

    def predict_proba(self, features: ArrayLike) -> NDArray[np.float64]:
        """Average the trees' class probabilities (columns follow ``classes_``)."""

        if not hasattr(self, "estimators_"):
            raise RuntimeError("fit must be called before prediction")
        array = _validated_features(features)
        if array.shape[1] != self.n_features_in_:
            raise ValueError("features have the wrong shape for this fitted model")
        total = np.zeros((array.shape[0], self.classes_.size))
        for tree in self.estimators_:
            total += self._tree_proba(tree, array, self.classes_.size)
        return total / len(self.estimators_)

    def predict(self, features: ArrayLike) -> NDArray[np.generic]:
        """Return the class with the highest averaged probability."""

        return self.classes_[np.argmax(self.predict_proba(features), axis=1)]


class GradientBoostingBinaryClassifier(ClassifierMixin, BaseEstimator):
    """Gradient-boosted regression trees for two-class log-loss.

    Parameters
    ----------
    n_estimators
        Number of boosting stages (trees).
    learning_rate
        Shrinkage :math:`\\eta \\in (0, 1]` applied to every stage.
    max_depth, min_samples_leaf
        Size limits for each stage's regression tree (shallow trees are the
        norm: depth 3 allows three-way interactions).
    subsample
        Fraction of rows, drawn without replacement, used to grow each stage
        (stochastic gradient boosting); ``1.0`` uses all rows.
    random_state
        Seed for subsampling.

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
    init_score_ : float
        Starting log-odds :math:`\\log(\\bar y / (1 - \\bar y))`.
    estimators_ : list of RegressionTree
        One tree per stage, with Newton-step leaf values.
    train_loss_ : ndarray of shape (n_estimators,)
        Mean training log-loss after each stage.
    feature_importances_ : ndarray
        Mean of the stage trees' normalized importances.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        _positive_int(n_estimators, "n_estimators")
        if not isinstance(learning_rate, Real) or not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must lie in (0, 1]")
        if not isinstance(subsample, Real) or not 0 < subsample <= 1:
            raise ValueError("subsample must lie in (0, 1]")
        RegressionTree(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self:
        """Run ``n_estimators`` boosting stages."""

        array = _validated_features(features)
        labels = np.asarray(targets)
        if labels.shape != (array.shape[0],):
            raise ValueError("targets must contain one label per observation")
        classes, encoded = np.unique(labels, return_inverse=True)
        if classes.size != 2:
            raise ValueError(
                "GradientBoostingBinaryClassifier needs exactly two classes"
            )
        y = encoded.astype(float)
        rows = array.shape[0]
        rng = np.random.default_rng(self.random_state)

        rate = y.mean()
        self.init_score_ = float(np.log(rate / (1 - rate)))
        scores = np.full(rows, self.init_score_)
        self.estimators_ = []
        losses = []
        sample_size = max(1, int(round(self.subsample * rows)))
        for _ in range(self.n_estimators):
            probability = _sigmoid(scores)
            residual = y - probability
            if sample_size < rows:
                in_bag = np.sort(rng.choice(rows, size=sample_size, replace=False))
            else:
                in_bag = np.arange(rows)
            tree = RegressionTree(
                max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf
            ).fit(array[in_bag], residual[in_bag])
            # Newton step per leaf: sum(r) / sum(p(1 - p)) over in-bag rows.
            leaves = tree.apply(array[in_bag])
            numerator = np.bincount(
                leaves, weights=residual[in_bag], minlength=tree.tree_["left"].size
            )
            denominator = np.bincount(
                leaves,
                weights=(probability * (1 - probability))[in_bag],
                minlength=tree.tree_["left"].size,
            )
            is_leaf = tree.tree_["left"] == -1
            step = np.zeros_like(numerator)
            usable = is_leaf & (np.abs(denominator) >= 1e-150)
            step[usable] = numerator[usable] / denominator[usable]
            tree.tree_["value"][is_leaf, 0] = step[is_leaf]

            scores += self.learning_rate * tree.predict(array)
            self.estimators_.append(tree)
            losses.append(_log_loss(y, scores))

        self.classes_ = classes
        self.n_features_in_ = array.shape[1]
        self.train_loss_ = np.asarray(losses)
        self.feature_importances_ = np.mean(
            [tree.feature_importances_ for tree in self.estimators_], axis=0
        )
        return self

    def _check(self, features: ArrayLike) -> NDArray[np.float64]:
        if not hasattr(self, "estimators_"):
            raise RuntimeError("fit must be called before prediction")
        array = _validated_features(features)
        if array.shape[1] != self.n_features_in_:
            raise ValueError("features have the wrong shape for this fitted model")
        return array

    def decision_function(self, features: ArrayLike) -> NDArray[np.float64]:
        """Return the boosted log-odds score :math:`F(x)`, shape ``(n,)``."""

        array = self._check(features)
        scores = np.full(array.shape[0], self.init_score_)
        for tree in self.estimators_:
            scores += self.learning_rate * tree.predict(array)
        return scores

    def staged_predict_proba(
        self, features: ArrayLike
    ) -> Iterator[NDArray[np.float64]]:
        """Yield class probabilities after each stage (for learning curves)."""

        array = self._check(features)
        scores = np.full(array.shape[0], self.init_score_)
        for tree in self.estimators_:
            scores += self.learning_rate * tree.predict(array)
            probability = _sigmoid(scores)
            yield np.column_stack([1 - probability, probability])

    def predict_proba(self, features: ArrayLike) -> NDArray[np.float64]:
        """Return ``[P(classes_[0]), P(classes_[1])]`` for each row."""

        probability = _sigmoid(self.decision_function(features))
        return np.column_stack([1 - probability, probability])

    def predict(self, features: ArrayLike) -> NDArray[np.generic]:
        """Return the more probable class."""

        return self.classes_[(self.decision_function(features) > 0).astype(int)]

    def stump_contribution(
        self, feature: int, values: ArrayLike
    ) -> NDArray[np.float64]:
        """Return feature ``feature``'s additive log-odds term at ``values``.

        With ``max_depth=1`` every stage is a stump on a single feature, so the
        model is exactly additive: :math:`F(x) = F_0 + \\sum_j f_j(x_j)`. This
        returns :math:`f_j` evaluated at each value, so the curve can be
        plotted and compared with a linear model's straight line. Summing the
        contributions of all features and adding ``init_score_`` reproduces
        ``decision_function``.
        """

        if not hasattr(self, "estimators_"):
            raise RuntimeError("fit must be called first")
        if self.max_depth != 1:
            raise ValueError("stump contributions are defined only for max_depth=1")
        if not 0 <= feature < self.n_features_in_:
            raise ValueError("feature index out of range")
        points = np.asarray(values, dtype=float).ravel()
        contribution = np.zeros(points.size)
        for tree in self.estimators_:
            nodes = tree.tree_
            if nodes["left"][0] == -1:  # a stage that could not split adds a constant
                continue
            if nodes["feature"][0] != feature:
                continue
            leaf = np.where(
                points <= nodes["threshold"][0], nodes["left"][0], nodes["right"][0]
            )
            contribution += self.learning_rate * nodes["value"][leaf, 0]
        return contribution


def _sigmoid(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.5 * (1.0 + np.tanh(0.5 * scores))  # overflow-free logistic


def _log_loss(y: NDArray[np.float64], scores: NDArray[np.float64]) -> float:
    # -[y log p + (1 - y) log(1 - p)] = log(1 + e^F) - y F, computed stably.
    return float(np.mean(np.logaddexp(0.0, scores) - y * scores))
