"""CART decision trees grown by greedy impurity reduction.

Mathematical contract
---------------------
**Classification.** A node holding rows with (weighted) class proportions
:math:`p_1, \\dots, p_K` has impurity

.. math::

    G = 1 - \\sum_k p_k^2 \\quad\\text{(Gini)}, \\qquad
    H = -\\sum_k p_k \\log_2 p_k \\quad\\text{(entropy)}.

**Regression.** A node holding targets :math:`y_i` with weights :math:`w_i`
has impurity equal to its weighted variance (mean squared error about the
node mean :math:`\\bar y`):

.. math::

    V = \\frac{\\sum_i w_i (y_i - \\bar y)^2}{\\sum_i w_i}.

A split sends rows with :math:`x_j \\le t` left and the rest right. Among all
features :math:`j` and thresholds :math:`t` (midpoints between consecutive
distinct values), the tree picks the split with the largest impurity decrease

.. math::

    \\Delta = I(\\text{parent})
              - \\frac{N_L}{N} I(\\text{left}) - \\frac{N_R}{N} I(\\text{right}),

where :math:`N`, :math:`N_L`, :math:`N_R` are (weighted) row counts, then
recurses on both children until a stopping rule holds. A classification leaf
predicts its weighted class proportions; a regression leaf predicts its
weighted mean.

Searching one feature costs :math:`O(n \\log n)` (sort once, then cumulative
sums give every candidate split), so one level of the tree costs
:math:`O(d\\, n \\log n)`.

Conventions match scikit-learn so differential tests are meaningful:
``x <= threshold`` goes left, thresholds are midpoints, ties in gain keep the
first feature examined, and ``min_impurity_decrease`` is weighted by the
node's share of all rows.
"""

from __future__ import annotations

from collections.abc import Callable
from numbers import Integral, Real
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

_LEAF = -1
_CRITERIA = ("gini", "entropy")

SplitFinder = Callable[
    [NDArray[np.intp], NDArray[np.intp]], tuple[int, float, float] | None
]
NodeSummary = Callable[[NDArray[np.intp]], tuple[NDArray[np.float64], float, float]]


# --------------------------------------------------------------------------
# Impurities and split search
# --------------------------------------------------------------------------


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


def _scan_feature(
    values: NDArray[np.float64],
    child_impurity: NDArray[np.float64],
    min_samples_leaf: int,
) -> tuple[int, float, float] | None:
    """Pick the best valid cut position for one sorted feature.

    ``child_impurity[i]`` is the impurity of cutting after sorted row ``i``.
    Returns ``(position, threshold, impurity)`` or ``None``.
    """

    rows = values.size
    left_sizes = np.arange(1, rows)
    valid = (
        (values[:-1] < values[1:])
        & (left_sizes >= min_samples_leaf)
        & (rows - left_sizes >= min_samples_leaf)
    )
    if not valid.any():
        return None
    child = np.where(valid, child_impurity, np.inf)
    position = int(np.argmin(child))
    threshold = (values[position] + values[position + 1]) / 2.0
    if threshold == values[position + 1]:  # adjacent floats: keep x <= t left
        threshold = values[position]
    return position, float(threshold), float(child[position])


def best_split(
    features: NDArray[np.float64],
    weighted_one_hot: NDArray[np.float64],
    *,
    criterion: str = "gini",
    min_samples_leaf: int = 1,
    feature_indices: NDArray[np.intp] | None = None,
) -> tuple[int, float, float] | None:
    """Return ``(feature, threshold, weighted_child_impurity)`` for classification.

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

    if feature_indices is None:
        feature_indices = np.arange(features.shape[1])
    total_weight = weighted_one_hot.sum()
    totals = weighted_one_hot.sum(axis=0)
    best: tuple[int, float, float] | None = None
    for feature in feature_indices:
        order = np.argsort(features[:, feature], kind="stable")
        left_counts = np.cumsum(weighted_one_hot[order], axis=0)[:-1]
        right_counts = totals - left_counts
        child = (
            left_counts.sum(axis=1) * _impurity_rows(left_counts, criterion)
            + right_counts.sum(axis=1) * _impurity_rows(right_counts, criterion)
        ) / total_weight
        found = _scan_feature(features[order, feature], child, min_samples_leaf)
        if found is not None and (best is None or found[2] < best[2]):
            best = (int(feature), found[1], found[2])
    return best


def best_regression_split(
    features: NDArray[np.float64],
    targets: NDArray[np.float64],
    weights: NDArray[np.float64],
    *,
    min_samples_leaf: int = 1,
    feature_indices: NDArray[np.intp] | None = None,
) -> tuple[int, float, float] | None:
    """Return ``(feature, threshold, weighted_child_variance)`` for regression.

    For each candidate cut the child sums of squared errors come from running
    sums: :math:`\\text{SSE} = \\sum w y^2 - (\\sum w y)^2 / \\sum w`. The
    returned impurity is :math:`(\\text{SSE}_L + \\text{SSE}_R) / \\sum w`.
    """

    if feature_indices is None:
        feature_indices = np.arange(features.shape[1])
    weighted = weights * targets
    weighted_square = weighted * targets
    total_w, total_wy, total_wy2 = weights.sum(), weighted.sum(), weighted_square.sum()
    best: tuple[int, float, float] | None = None
    for feature in feature_indices:
        order = np.argsort(features[:, feature], kind="stable")
        left_w = np.cumsum(weights[order])[:-1]
        left_wy = np.cumsum(weighted[order])[:-1]
        left_wy2 = np.cumsum(weighted_square[order])[:-1]
        right_w = total_w - left_w
        right_wy = total_wy - left_wy
        right_wy2 = total_wy2 - left_wy2
        with np.errstate(divide="ignore", invalid="ignore"):
            sse = (left_wy2 - left_wy**2 / left_w) + (right_wy2 - right_wy**2 / right_w)
        found = _scan_feature(features[order, feature], sse / total_w, min_samples_leaf)
        if found is not None and (best is None or found[2] < best[2]):
            best = (int(feature), found[1], found[2])
    return best


# --------------------------------------------------------------------------
# Shared tree machinery
# --------------------------------------------------------------------------


class _BaseTree(BaseEstimator):
    """Growth, traversal, and inspection shared by both tree types."""

    def _validate_common(
        self,
        max_depth: int | None,
        min_samples_split: int,
        min_samples_leaf: int,
        min_impurity_decrease: float,
        max_features: int | float | str | None,
    ) -> None:
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
        if not (
            max_features is None
            or max_features == "sqrt"
            or (isinstance(max_features, Integral) and max_features >= 1)
            or (isinstance(max_features, Real) and 0 < max_features <= 1)
        ):
            raise ValueError(
                "max_features must be None, 'sqrt', a positive int, or in (0, 1]"
            )

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

    def _grow(
        self,
        array: NDArray[np.float64],
        total_weight: float,
        summarize: NodeSummary,
        find_split: SplitFinder,
    ) -> None:
        """Grow depth-first and store node arrays in ``tree_``.

        ``summarize(rows)`` returns ``(leaf value, node weight, impurity)``;
        ``find_split(rows, candidate_features)`` returns a split or ``None``.
        """

        rows, feature_count = array.shape
        features_per_node = self._features_per_node(feature_count)
        rng = np.random.default_rng(self.random_state)
        keys = (
            "left", "right", "feature", "threshold", "value",
            "n_samples", "weighted_n_samples", "impurity",
        )  # fmt: skip
        nodes: dict[str, list] = {key: [] for key in keys}
        importances = np.zeros(feature_count)

        def add_node(row_index: NDArray[np.intp]) -> int:
            value, weight, impurity = summarize(row_index)
            for key, item in zip(
                keys,
                (_LEAF, _LEAF, _LEAF, np.nan, value, row_index.size, weight, impurity),
                strict=True,
            ):
                nodes[key].append(item)
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
            split = find_split(row_index, candidates)
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
            # Children are numbered when created, so every parent precedes its
            # children (get_depth relies on this); push right first so the left
            # subtree is grown first.
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
        self.n_features_in_ = feature_count

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

    def _leaf_text(self, node: int, decimals: int) -> str:
        raise NotImplementedError

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
                lines.append(
                    f"{indent}leaf (n={tree['n_samples'][node]}): "
                    f"{self._leaf_text(node, decimals)}"
                )
                return
            name, threshold = names[tree["feature"][node]], tree["threshold"][node]
            lines.append(f"{indent}if {name} <= {threshold:.{decimals}f}:")
            visit(tree["left"][node], indent + "    ")
            lines.append(f"{indent}else:  # {name} > {threshold:.{decimals}f}")
            visit(tree["right"][node], indent + "    ")

        visit(0, "")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Classification tree
# --------------------------------------------------------------------------


class ClassificationTree(ClassifierMixin, _BaseTree):
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
        self._validate_common(
            max_depth, min_samples_split, min_samples_leaf,
            min_impurity_decrease, max_features,
        )  # fmt: skip
        if isinstance(class_weight, dict):
            if any(not weight > 0 for weight in class_weight.values()):
                raise ValueError("class weights must be positive")
        elif class_weight not in (None, "balanced"):
            raise ValueError("class_weight must be None, 'balanced', or a dict")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.max_features = max_features
        self.random_state = random_state

    def fit(
        self,
        features: ArrayLike,
        targets: ArrayLike,
        sample_weight: ArrayLike | None = None,
    ) -> Self:
        """Grow the tree depth-first from the root.

        ``sample_weight`` multiplies the class weights row by row; a random
        forest passes bootstrap counts here.
        """

        array = _validated_features(features)
        labels = np.asarray(targets)
        if labels.shape != (array.shape[0],):
            raise ValueError("targets must contain one label per observation")
        classes, encoded = np.unique(labels, return_inverse=True)
        rows = array.shape[0]
        weights = _class_row_weights(self.class_weight, classes, encoded)
        weights = weights * _validated_sample_weight(sample_weight, rows)
        weighted_one_hot = np.zeros((rows, classes.size))
        weighted_one_hot[np.arange(rows), encoded] = weights
        criterion = self.criterion

        def summarize(row_index: NDArray[np.intp]):
            counts = weighted_one_hot[row_index].sum(axis=0)
            weight = float(counts.sum())
            impurity = float(_impurity_rows(counts[None, :], criterion)[0])
            return counts / weight, weight, impurity

        def find_split(row_index: NDArray[np.intp], candidates: NDArray[np.intp]):
            return best_split(
                array[row_index],
                weighted_one_hot[row_index],
                criterion=criterion,
                min_samples_leaf=self.min_samples_leaf,
                feature_indices=candidates,
            )

        self._grow(array, float(weights.sum()), summarize, find_split)
        self.classes_ = classes
        return self

    def predict_proba(self, features: ArrayLike) -> NDArray[np.float64]:
        """Return the class proportions of each row's leaf (columns = ``classes_``)."""

        leaves = self.apply(features)  # validates input and fitted state
        return self.tree_["value"][leaves]

    def predict(self, features: ArrayLike) -> NDArray[np.generic]:
        """Return the most probable class of each row's leaf."""

        probabilities = self.predict_proba(features)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def _leaf_text(self, node: int, decimals: int) -> str:
        return ", ".join(
            f"{label}: {p:.{decimals}f}"
            for label, p in zip(
                self.classes_.tolist(), self.tree_["value"][node], strict=True
            )
        )


# --------------------------------------------------------------------------
# Regression tree
# --------------------------------------------------------------------------


class RegressionTree(RegressorMixin, _BaseTree):
    """Binary-split regression tree (CART) minimizing squared error.

    Each leaf predicts the weighted mean of its training targets. Gradient
    boosting uses this tree as its base learner and then *replaces* the leaf
    values with its own Newton step (see ``GradientBoostingBinaryClassifier``).

    Parameters
    ----------
    max_depth, min_samples_split, min_samples_leaf, min_impurity_decrease,
    max_features, random_state
        As for :class:`ClassificationTree`; impurity is the weighted variance.

    Attributes
    ----------
    n_features_in_, feature_importances_, tree_
        As for :class:`ClassificationTree`, except ``tree_["value"]`` holds one
        predicted value per node, shape ``(nodes, 1)``.
    """

    def __init__(
        self,
        *,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        max_features: int | float | str | None = None,
        random_state: int | None = None,
    ) -> None:
        self._validate_common(
            max_depth, min_samples_split, min_samples_leaf,
            min_impurity_decrease, max_features,
        )  # fmt: skip
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.random_state = random_state

    def fit(
        self,
        features: ArrayLike,
        targets: ArrayLike,
        sample_weight: ArrayLike | None = None,
    ) -> Self:
        """Grow the tree to reduce weighted squared error."""

        array = _validated_features(features)
        values = np.asarray(targets, dtype=float)
        if values.shape != (array.shape[0],):
            raise ValueError("targets must contain one value per observation")
        if not np.isfinite(values).all():
            raise ValueError("targets must be finite")
        weights = _validated_sample_weight(sample_weight, array.shape[0])

        def summarize(row_index: NDArray[np.intp]):
            w, y = weights[row_index], values[row_index]
            weight = float(w.sum())
            mean = float(w @ y) / weight
            variance = max(float(w @ (y - mean) ** 2) / weight, 0.0)
            return np.array([mean]), weight, variance

        def find_split(row_index: NDArray[np.intp], candidates: NDArray[np.intp]):
            return best_regression_split(
                array[row_index],
                values[row_index],
                weights[row_index],
                min_samples_leaf=self.min_samples_leaf,
                feature_indices=candidates,
            )

        self._grow(array, float(weights.sum()), summarize, find_split)
        return self

    def predict(self, features: ArrayLike) -> NDArray[np.float64]:
        """Return the value of each row's leaf."""

        leaves = self.apply(features)
        return self.tree_["value"][leaves, 0]

    def _leaf_text(self, node: int, decimals: int) -> str:
        return f"value {self.tree_['value'][node, 0]:.{decimals}f}"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


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


def _validated_sample_weight(
    sample_weight: ArrayLike | None, rows: int
) -> NDArray[np.float64]:
    if sample_weight is None:
        return np.ones(rows)
    weights = np.asarray(sample_weight, dtype=float)
    if weights.shape != (rows,):
        raise ValueError("sample_weight must contain one weight per observation")
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("sample_weight must be finite, nonnegative, and not all zero")
    return weights


def _validated_features(features: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(features, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("features must be a nonempty two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError("features must contain only finite values")
    return array
