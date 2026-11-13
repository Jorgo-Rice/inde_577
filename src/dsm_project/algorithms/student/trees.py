"""CART classification tree grown by greedy impurity reduction.

Mathematical contract
---------------------
A node holding rows with (weighted) class proportions :math:`p_1, \\dots, p_K`
has impurity

.. math::

    G = 1 - \\sum_k p_k^2 \\quad\\text{(Gini)}, \\qquad
    H = -\\sum_k p_k \\log_2 p_k \\quad\\text{(entropy)}.

A split sends rows with :math:`x_j \\le t` left and the rest right. Among all
features :math:`j` and thresholds :math:`t` (midpoints between consecutive
distinct values), the tree picks the split with the largest impurity decrease

.. math::

    \\Delta = I(\\text{parent})
              - \\frac{N_L}{N} I(\\text{left}) - \\frac{N_R}{N} I(\\text{right}),

where :math:`N`, :math:`N_L`, :math:`N_R` are (weighted) row counts. It then
recurses on both children until a stopping rule holds. A leaf predicts its
weighted class proportions, so ``predict_proba`` returns them and ``predict``
returns the most probable class.

Searching one feature costs :math:`O(n \\log n)` (sort once, then cumulative
class counts give every candidate split), so one level of the tree costs
:math:`O(d\\, n \\log n)`.

Conventions match scikit-learn so differential tests are meaningful:
``x <= threshold`` goes left, thresholds are midpoints, ties in gain keep the
first feature examined, and ``min_impurity_decrease`` is weighted by the
node's share of all rows.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin

_LEAF = -1
_CRITERIA = ("gini", "entropy")


def gini_impurity(class_counts: ArrayLike) -> float:
    """Gini impurity ``1 - sum(p_k^2)`` from (weighted) class counts."""

    counts = np.asarray(class_counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        raise ValueError("class counts must have a positive total")
    proportions = counts / total
    return float(1.0 - proportions @ proportions)


def entropy_impurity(class_counts: ArrayLike) -> float:
    """Base-2 entropy ``-sum(p_k log2 p_k)`` from (weighted) class counts."""

    counts = np.asarray(class_counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        raise ValueError("class counts must have a positive total")
    proportions = counts[counts > 0] / total
    return float(-(proportions * np.log2(proportions)).sum())


def _impurity_rows(counts: NDArray[np.float64], criterion: str) -> NDArray[np.float64]:
    """Row-wise impurity for a matrix of class counts (one candidate per row)."""

    totals = counts.sum(axis=1, keepdims=True)
    proportions = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    if criterion == "gini":
        return 1.0 - np.sum(proportions**2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.where(proportions > 0, np.log2(proportions), 0.0)
    return -np.sum(proportions * logs, axis=1)


def best_split(
    features: NDArray[np.float64],
    weighted_one_hot: NDArray[np.float64],
    *,
    criterion: str = "gini",
    min_samples_leaf: int = 1,
    feature_indices: NDArray[np.intp] | None = None,
) -> tuple[int, float, float] | None:
    """Return ``(feature, threshold, weighted_child_impurity)`` for the best split.

    Parameters
    ----------
    features
        Node rows, shape ``(n, d)``.
    weighted_one_hot
        One-hot class matrix multiplied by each row's sample weight, ``(n, K)``.
    criterion
        ``"gini"`` or ``"entropy"``.
    min_samples_leaf
        Minimum *unweighted* number of rows on each side.
    feature_indices
        Features to search, in order; all features when ``None``.

    Returns
    -------
    tuple or None
        ``None`` when no split leaves ``min_samples_leaf`` rows on both sides
        of two distinct feature values. ``weighted_child_impurity`` is
        :math:`(N_L I_L + N_R I_R) / N`.
    """

    rows = features.shape[0]
    if feature_indices is None:
        feature_indices = np.arange(features.shape[1])
    total_weight = weighted_one_hot.sum()
    best: tuple[int, float, float] | None = None
    best_impurity = np.inf
    for feature in feature_indices:
        order = np.argsort(features[:, feature], kind="stable")
        values = features[order, feature]
        left_counts = np.cumsum(weighted_one_hot[order], axis=0)[:-1]
        right_counts = weighted_one_hot.sum(axis=0) - left_counts
        left_sizes = np.arange(1, rows)
        valid = (
            (values[:-1] < values[1:])
            & (left_sizes >= min_samples_leaf)
            & (rows - left_sizes >= min_samples_leaf)
        )
        if not valid.any():
            continue
        left_weight = left_counts.sum(axis=1)
        right_weight = right_counts.sum(axis=1)
        child = (
            left_weight * _impurity_rows(left_counts, criterion)
            + right_weight * _impurity_rows(right_counts, criterion)
        ) / total_weight
        child[~valid] = np.inf
        position = int(np.argmin(child))
        if child[position] < best_impurity:
            best_impurity = float(child[position])
            threshold = (values[position] + values[position + 1]) / 2.0
            if threshold == values[position + 1]:  # adjacent floats: keep x <= t left
                threshold = values[position]
            best = (int(feature), float(threshold), best_impurity)
    return best


class ClassificationTree(ClassifierMixin, BaseEstimator):
    """Binary-split classification tree (CART) with Gini or entropy impurity.

    Parameters
    ----------
    criterion
        ``"gini"`` or ``"entropy"``.
    max_depth
        Maximum depth (root has depth 0); ``None`` grows until other rules stop.
    min_samples_split
        Smallest node (in rows) that may be split; at least 2.
    min_samples_leaf
        Smallest number of rows allowed in each child; at least 1.
    min_impurity_decrease
        Split only when :math:`\\frac{N_t}{N}\\Delta` reaches this value, where
        :math:`N_t / N` is the node's share of all (weighted) training rows.
    class_weight
        ``None``, ``"balanced"`` (weight :math:`n / (K n_k)`), or a mapping from
        label to positive weight. Weights change both impurities and leaf
        probabilities.
    max_features
        Features examined at each node: ``None`` (all), an integer, a fraction
        in ``(0, 1]``, or ``"sqrt"``. Values below all features make the tree
        random, which random forests rely on.
    random_state
        Seed for the per-node feature sample when ``max_features`` is set.

    Attributes
    ----------
    classes_, n_features_in_
        Labels and feature count seen during ``fit``.
    feature_importances_ : ndarray of shape (d,)
        Total weighted impurity decrease contributed by each feature,
        normalized to sum to one (all zeros for a single-leaf tree).
    tree_ : dict of ndarrays
        Node arrays ``left``, ``right`` (``-1`` at leaves), ``feature``,
        ``threshold``, ``value`` (class probabilities), ``n_samples``,
        ``weighted_n_samples``, and ``impurity``. Node 0 is the root.
    """

    def __init__(
        self,
        *,
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        class_weight: str | dict | None = None,
        max_features: int | float | str | None = None,
        random_state: int | None = None,
    ) -> None:
        if criterion not in _CRITERIA:
            raise ValueError(f"criterion must be one of {_CRITERIA}")
        if max_depth is not None and (
            not isinstance(max_depth, Integral) or max_depth < 0
        ):
            raise ValueError("max_depth must be a nonnegative integer or None")
        if not isinstance(min_samples_split, Integral) or min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer of at least 2")
        if not isinstance(min_samples_leaf, Integral) or min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be an integer of at least 1")
        if min_impurity_decrease < 0:
            raise ValueError("min_impurity_decrease must be nonnegative")
        if isinstance(class_weight, dict):
            if any(not weight > 0 for weight in class_weight.values()):
                raise ValueError("class weights must be positive")
        elif class_weight not in (None, "balanced"):
            raise ValueError("class_weight must be None, 'balanced', or a dict")
        if not (
            max_features is None
            or max_features == "sqrt"
            or (isinstance(max_features, Integral) and max_features >= 1)
            or (isinstance(max_features, Real) and 0 < max_features <= 1)
        ):
            raise ValueError(
                "max_features must be None, 'sqrt', a positive int, or in (0, 1]"
            )
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.max_features = max_features
        self.random_state = random_state

    # ------------------------------------------------------------------ fitting

    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self:
        """Grow the tree depth-first from the root."""

        array = _validated_features(features)
        labels = np.asarray(targets)
        if labels.shape != (array.shape[0],):
            raise ValueError("targets must contain one label per observation")
        classes, encoded = np.unique(labels, return_inverse=True)
        rows, feature_count = array.shape
        weights = _class_row_weights(self.class_weight, classes, encoded)
        weighted_one_hot = np.zeros((rows, classes.size))
        weighted_one_hot[np.arange(rows), encoded] = weights
        total_weight = float(weights.sum())
        features_per_node = self._features_per_node(feature_count)
        rng = np.random.default_rng(self.random_state)

        nodes: dict[str, list] = {
            key: []
            for key in (
                "left",
                "right",
                "feature",
                "threshold",
                "value",
                "n_samples",
                "weighted_n_samples",
                "impurity",
            )
        }
        importances = np.zeros(feature_count)

        def add_node(row_index: NDArray[np.intp]) -> int:
            counts = weighted_one_hot[row_index].sum(axis=0)
            node_weight = float(counts.sum())
            impurity = float(_impurity_rows(counts[None, :], self.criterion)[0])
            for key, value in (
                ("left", _LEAF),
                ("right", _LEAF),
                ("feature", _LEAF),
                ("threshold", np.nan),
                ("value", counts / node_weight),
                ("n_samples", row_index.size),
                ("weighted_n_samples", node_weight),
                ("impurity", impurity),
            ):
                nodes[key].append(value)
            return len(nodes["left"]) - 1

        stack = [(add_node(np.arange(rows)), np.arange(rows), 0)]
        while stack:
            node, row_index, depth = stack.pop()
            impurity = nodes["impurity"][node]
            if (
                (self.max_depth is not None and depth >= self.max_depth)
                or row_index.size < self.min_samples_split
                or row_index.size < 2 * self.min_samples_leaf
                or impurity <= 1e-12
            ):
                continue
            candidates = (
                np.arange(feature_count)
                if features_per_node == feature_count
                else rng.choice(feature_count, size=features_per_node, replace=False)
            )
            split = best_split(
                array[row_index],
                weighted_one_hot[row_index],
                criterion=self.criterion,
                min_samples_leaf=self.min_samples_leaf,
                feature_indices=candidates,
            )
            if split is None:
                continue
            feature, threshold, child_impurity = split
            node_weight = nodes["weighted_n_samples"][node]
            decrease = node_weight / total_weight * (impurity - child_impurity)
            # Same tolerance as scikit-learn: rounding can make a zero gain
            # look like -1e-17, which must not block a valid split.
            if decrease + np.finfo(float).eps < self.min_impurity_decrease:
                continue

            goes_left = array[row_index, feature] <= threshold
            left = add_node(row_index[goes_left])
            right = add_node(row_index[~goes_left])
            nodes["left"][node], nodes["right"][node] = left, right
            nodes["feature"][node], nodes["threshold"][node] = feature, threshold
            importances[feature] += node_weight * (impurity - child_impurity)
            # Push right first so the left subtree is numbered first (pre-order).
            stack.append((right, row_index[~goes_left], depth + 1))
            stack.append((left, row_index[goes_left], depth + 1))

        self.tree_ = {
            "left": np.asarray(nodes["left"], dtype=np.intp),
            "right": np.asarray(nodes["right"], dtype=np.intp),
            "feature": np.asarray(nodes["feature"], dtype=np.intp),
            "threshold": np.asarray(nodes["threshold"], dtype=float),
            "value": np.vstack(nodes["value"]),
            "n_samples": np.asarray(nodes["n_samples"], dtype=np.intp),
            "weighted_n_samples": np.asarray(nodes["weighted_n_samples"], dtype=float),
            "impurity": np.asarray(nodes["impurity"], dtype=float),
        }
        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else importances
        self.classes_ = classes
        self.n_features_in_ = feature_count
        return self

    def _features_per_node(self, feature_count: int) -> int:
        if self.max_features is None:
            return feature_count
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(feature_count)))
        if isinstance(self.max_features, Integral):
            if self.max_features > feature_count:
                raise ValueError("max_features cannot exceed the number of features")
            return int(self.max_features)
        return max(1, int(self.max_features * feature_count))

    # --------------------------------------------------------------- prediction

    def apply(self, features: ArrayLike) -> NDArray[np.intp]:
        """Return the index of the leaf each row reaches."""

        if not hasattr(self, "tree_"):
            raise RuntimeError("fit must be called before prediction")
        array = _validated_features(features)
        if array.shape[1] != self.n_features_in_:
            raise ValueError("features have the wrong shape for this fitted model")
        tree = self.tree_
        node = np.zeros(array.shape[0], dtype=np.intp)
        active = tree["left"][node] != _LEAF
        while active.any():
            rows = np.flatnonzero(active)
            current = node[rows]
            goes_left = (
                array[rows, tree["feature"][current]] <= tree["threshold"][current]
            )
            node[rows] = np.where(
                goes_left, tree["left"][current], tree["right"][current]
            )
            active = tree["left"][node] != _LEAF
        return node

    def predict_proba(self, features: ArrayLike) -> NDArray[np.float64]:
        """Return the class proportions of each row's leaf (columns = ``classes_``)."""

        leaves = self.apply(features)  # validates input and fitted state
        return self.tree_["value"][leaves]

    def predict(self, features: ArrayLike) -> NDArray[np.generic]:
        """Return the most probable class of each row's leaf."""

        probabilities = self.predict_proba(features)
        return self.classes_[np.argmax(probabilities, axis=1)]

    # ----------------------------------------------------------- inspection

    def get_depth(self) -> int:
        """Return the length of the longest root-to-leaf path."""

        if not hasattr(self, "tree_"):
            raise RuntimeError("fit must be called first")
        depth = np.zeros(self.tree_["left"].size, dtype=int)
        for node in range(depth.size):  # parents precede children (pre-order)
            for child in (self.tree_["left"][node], self.tree_["right"][node]):
                if child != _LEAF:
                    depth[child] = depth[node] + 1
        return int(depth.max())

    def get_n_leaves(self) -> int:
        """Return the number of leaves."""

        if not hasattr(self, "tree_"):
            raise RuntimeError("fit must be called first")
        return int(np.sum(self.tree_["left"] == _LEAF))

    def export_text(
        self,
        feature_names: list[str] | tuple[str, ...] | None = None,
        decimals: int = 3,
    ) -> str:
        """Return the fitted rules as indented ``if feature <= t`` text."""

        if not hasattr(self, "tree_"):
            raise RuntimeError("fit must be called first")
        names = (
            list(feature_names)
            if feature_names is not None
            else [f"x[{j}]" for j in range(self.n_features_in_)]
        )
        tree, lines = self.tree_, []

        def visit(node: int, indent: str) -> None:
            if tree["left"][node] == _LEAF:
                probabilities = ", ".join(
                    f"{label}: {p:.{decimals}f}"
                    for label, p in zip(
                        self.classes_.tolist(), tree["value"][node], strict=True
                    )
                )
                lines.append(
                    f"{indent}leaf (n={tree['n_samples'][node]}): {probabilities}"
                )
                return
            name, threshold = names[tree["feature"][node]], tree["threshold"][node]
            lines.append(f"{indent}if {name} <= {threshold:.{decimals}f}:")
            visit(tree["left"][node], indent + "    ")
            lines.append(f"{indent}else:  # {name} > {threshold:.{decimals}f}")
            visit(tree["right"][node], indent + "    ")

        visit(0, "")
        return "\n".join(lines)


def _class_row_weights(
    class_weight: str | dict | None,
    classes: NDArray[np.generic],
    encoded: NDArray[np.intp],
) -> NDArray[np.float64]:
    if class_weight is None:
        return np.ones(encoded.size)
    if class_weight == "balanced":
        counts = np.bincount(encoded, minlength=classes.size)
        per_class = encoded.size / (classes.size * counts)
    else:
        per_class = np.array(
            [float(class_weight.get(label, 1.0)) for label in classes.tolist()]
        )
    return per_class[encoded]


def _validated_features(features: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(features, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("features must be a nonempty two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError("features must contain only finite values")
    return array
