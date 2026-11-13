"""Mathematical, failure-mode, and differential tests for ClassificationTree."""

import numpy as np
import pytest
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from dsm_project.algorithms import ClassificationTree, create_student_algorithm
from dsm_project.algorithms.student.trees import (
    best_split,
    entropy_impurity,
    gini_impurity,
)

# --------------------------------------------------------------------------
# Impurity and split search
# --------------------------------------------------------------------------


def test_impurities_match_hand_computation() -> None:
    assert gini_impurity([5, 5]) == pytest.approx(0.5)
    assert gini_impurity([10, 0]) == 0.0
    assert gini_impurity([1, 1, 1]) == pytest.approx(2 / 3)
    assert entropy_impurity([5, 5]) == pytest.approx(1.0)
    assert entropy_impurity([4, 0]) == 0.0
    assert entropy_impurity([1, 1, 1, 1]) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="positive total"):
        gini_impurity([0, 0])


def test_best_split_finds_the_hand_computed_threshold() -> None:
    # Feature 0 separates the classes perfectly between 2 and 10, so the
    # midpoint 6 gives zero child impurity; feature 1 is noise.
    features = np.array([[1.0, 5.0], [2.0, 1.0], [10.0, 4.0], [11.0, 2.0]])
    one_hot = np.eye(2)[[0, 0, 1, 1]]
    feature, threshold, child_impurity = best_split(features, one_hot)

    assert (feature, threshold, child_impurity) == (0, 6.0, 0.0)


def test_best_split_respects_min_samples_leaf_and_constant_features() -> None:
    features = np.array([[0.0], [1.0], [2.0], [3.0]])
    one_hot = np.eye(2)[[0, 1, 1, 1]]
    # The pure split (after the first row) would leave a 1-row leaf.
    _, threshold, _ = best_split(features, one_hot, min_samples_leaf=2)
    assert threshold == 1.5

    assert best_split(np.ones((4, 1)), one_hot) is None


def test_best_split_uses_weighted_counts() -> None:
    # Unweighted, cutting at 1.5 isolates the single class-1 row at x = 1? No:
    # the weight of the two class-1 rows decides which side is purer.
    features = np.array([[0.0], [1.0], [2.0], [3.0]])
    labels = np.array([0, 1, 0, 1])
    plain = best_split(features, np.eye(2)[labels])
    heavy = best_split(features, np.eye(2)[labels] * np.array([[1], [1], [1], [10]]))

    assert plain is not None and heavy is not None
    assert heavy[1] == 2.5  # isolate the heavy class-1 row at x = 3
    assert heavy[2] < plain[2] or heavy[1] != plain[1]


# --------------------------------------------------------------------------
# Fitted trees
# --------------------------------------------------------------------------


def test_xor_needs_depth_two_and_is_then_learned_exactly() -> None:
    # No single axis-aligned split helps on XOR, but two levels separate it.
    features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 5, dtype=float)
    labels = np.array([0, 1, 1, 0] * 5)
    stump = ClassificationTree(max_depth=1).fit(features, labels)
    tree = ClassificationTree(max_depth=2).fit(features, labels)

    assert stump.score(features, labels) == pytest.approx(0.5)
    assert tree.score(features, labels) == 1.0
    assert tree.get_depth() == 2 and tree.get_n_leaves() == 4


def test_unlimited_tree_memorizes_distinct_rows_and_leaves_are_pure() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(60, 3))
    labels = rng.integers(0, 3, size=60)
    tree = ClassificationTree().fit(features, labels)

    np.testing.assert_array_equal(tree.predict(features), labels)
    leaves = tree.tree_["left"] == -1
    assert np.all(tree.tree_["impurity"][leaves] == 0)


def test_probabilities_are_leaf_class_frequencies() -> None:
    features = np.array([[0.0], [0.1], [0.2], [5.0], [5.1], [5.2]])
    labels = np.array(["a", "a", "b", "b", "b", "b"])
    # min_samples_leaf=3 rules out the pure split at 0.15 and forces 2.6.
    tree = ClassificationTree(max_depth=1, min_samples_leaf=3).fit(features, labels)

    assert tree.tree_["threshold"][0] == pytest.approx(2.6)
    np.testing.assert_allclose(tree.predict_proba([[0.05]]), [[2 / 3, 1 / 3]])
    np.testing.assert_allclose(tree.predict_proba([[9.0]]), [[0.0, 1.0]])
    np.testing.assert_allclose(tree.predict_proba(features).sum(axis=1), 1.0)


def test_stopping_rules_limit_growth() -> None:
    rng = np.random.default_rng(1)
    features = rng.normal(size=(200, 4))
    labels = (features[:, 0] + rng.normal(scale=1.0, size=200) > 0).astype(int)

    full = ClassificationTree().fit(features, labels)
    shallow = ClassificationTree(max_depth=3).fit(features, labels)
    leafy = ClassificationTree(min_samples_leaf=20).fit(features, labels)
    split_limited = ClassificationTree(min_samples_split=80).fit(features, labels)
    decrease_limited = ClassificationTree(min_impurity_decrease=0.01).fit(
        features, labels
    )

    assert shallow.get_depth() <= 3
    assert leafy.tree_["n_samples"][leafy.tree_["left"] == -1].min() >= 20
    internal = split_limited.tree_["left"] != -1
    assert split_limited.tree_["n_samples"][internal].min() >= 80
    for limited in (shallow, leafy, split_limited, decrease_limited):
        assert limited.get_n_leaves() < full.get_n_leaves()


def test_root_only_tree_predicts_the_majority() -> None:
    tree = ClassificationTree(max_depth=0).fit([[0.0], [1.0], [2.0]], [1, 1, 0])

    assert tree.get_n_leaves() == 1 and tree.get_depth() == 0
    np.testing.assert_allclose(tree.predict_proba([[5.0]]), [[1 / 3, 2 / 3]])
    np.testing.assert_array_equal(tree.feature_importances_, [0.0])


def test_feature_importances_identify_the_signal() -> None:
    rng = np.random.default_rng(2)
    features = rng.normal(size=(300, 3))
    labels = (features[:, 1] > 0.3).astype(int)
    tree = ClassificationTree(max_depth=3).fit(features, labels)

    assert tree.feature_importances_.sum() == pytest.approx(1.0)
    assert np.argmax(tree.feature_importances_) == 1


def test_export_text_shows_rules_and_leaves() -> None:
    tree = ClassificationTree(max_depth=1).fit(
        [[0.0], [1.0], [4.0], [5.0]], [0, 0, 1, 1]
    )
    text = tree.export_text(["activity_days"])

    assert "if activity_days <= 2.500:" in text
    assert "else:  # activity_days > 2.500" in text
    assert text.count("leaf (n=2)") == 2


def test_max_features_is_reproducible_and_random_across_seeds() -> None:
    features, labels = load_wine(return_X_y=True)
    first = ClassificationTree(max_features="sqrt", random_state=3).fit(
        features, labels
    )
    again = ClassificationTree(max_features="sqrt", random_state=3).fit(
        features, labels
    )
    other = ClassificationTree(max_features="sqrt", random_state=4).fit(
        features, labels
    )

    np.testing.assert_array_equal(first.tree_["feature"], again.tree_["feature"])
    assert not np.array_equal(first.tree_["feature"], other.tree_["feature"])


def test_balanced_class_weight_changes_leaf_probabilities() -> None:
    rng = np.random.default_rng(4)
    features = rng.normal(size=(300, 2))
    labels = (features[:, 0] + rng.normal(scale=1.5, size=300) > 1.8).astype(int)
    plain = ClassificationTree(max_depth=2).fit(features, labels)
    balanced = ClassificationTree(max_depth=2, class_weight="balanced").fit(
        features, labels
    )

    assert labels.mean() < 0.25
    assert (
        balanced.predict_proba(features)[:, 1].mean()
        > plain.predict_proba(features)[:, 1].mean()
    )


# --------------------------------------------------------------------------
# Differential agreement with scikit-learn
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parameters",
    [
        {"max_depth": 3},
        {"max_depth": 5, "criterion": "entropy"},
        {"max_depth": 4, "min_impurity_decrease": 0.005},
        {"max_depth": 4, "class_weight": "balanced"},
    ],
)
def test_tree_agrees_with_trusted_tree(parameters) -> None:
    # Continuous random features make exact ties in split gain essentially
    # impossible, so both implementations must grow the same tree.
    rng = np.random.default_rng(438)
    features = rng.normal(size=(400, 5))
    score = features[:, 0] - 0.8 * features[:, 2] * features[:, 3]
    labels = (score + rng.normal(scale=0.7, size=400) > 0.4).astype(int)
    train_x, test_x, train_y, _ = train_test_split(
        features, labels, test_size=0.3, random_state=0
    )
    student = ClassificationTree(**parameters).fit(train_x, train_y)
    trusted = DecisionTreeClassifier(random_state=0, **parameters).fit(train_x, train_y)

    if "class_weight" not in parameters:
        # With balanced weights scikit-learn's running sums can leave a
        # pure node with impurity ~1e-15 > its tolerance, so it splits a
        # pure node into two pure children. Ours computes exactly 0 and stops.
        # Shapes then differ by such no-op splits; predictions do not.
        assert student.get_n_leaves() == trusted.get_n_leaves()
        assert student.get_depth() == trusted.get_depth()
    np.testing.assert_allclose(
        student.predict_proba(test_x), trusted.predict_proba(test_x), atol=1e-12
    )
    np.testing.assert_allclose(
        student.feature_importances_, trusted.feature_importances_, atol=1e-12
    )


def test_every_split_is_optimal_even_when_ties_make_trees_differ() -> None:
    # With min_samples_leaf=7, small nodes can have exact ties between
    # features; scikit-learn breaks them in random feature order, so tree
    # shapes may legitimately differ. Instead verify by brute force that each
    # chosen split attains the minimum child impurity among all valid splits.
    rng = np.random.default_rng(438)
    features = rng.normal(size=(280, 5))
    labels = (features[:, 0] + rng.normal(scale=0.7, size=280) > 0.4).astype(int)
    tree = ClassificationTree(min_samples_leaf=7).fit(features, labels)
    one_hot = np.eye(2)[labels]

    def child_impurity(rows, feature, threshold):
        left = features[rows, feature] <= threshold
        total = 0.0
        for side in (left, ~left):
            counts = one_hot[rows][side].sum(axis=0)
            total += counts.sum() * gini_impurity(counts)
        return total / len(rows), left.sum(), (~left).sum()

    def visit(node, rows):
        if tree.tree_["left"][node] == -1:
            return
        feature, threshold = tree.tree_["feature"][node], tree.tree_["threshold"][node]
        chosen, n_left, n_right = child_impurity(rows, feature, threshold)
        assert min(n_left, n_right) >= 7
        for j in range(features.shape[1]):
            values = np.unique(features[rows, j])
            for t in (values[:-1] + values[1:]) / 2:
                impurity, a, b = child_impurity(rows, j, t)
                if min(a, b) >= 7:
                    assert chosen <= impurity + 1e-12
        left = features[rows, feature] <= threshold
        visit(tree.tree_["left"][node], rows[left])
        visit(tree.tree_["right"][node], rows[~left])

    visit(0, np.arange(len(labels)))


def test_tree_agrees_with_trusted_tree_on_wine_multiclass() -> None:
    features, labels = load_wine(return_X_y=True)
    student = ClassificationTree(max_depth=3).fit(features, labels)
    trusted = DecisionTreeClassifier(max_depth=3, random_state=0).fit(features, labels)

    np.testing.assert_allclose(
        student.predict_proba(features), trusted.predict_proba(features)
    )


# --------------------------------------------------------------------------
# Failure modes
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parameters",
    [
        {"criterion": "mse"},
        {"max_depth": -1},
        {"max_depth": 2.5},
        {"min_samples_split": 1},
        {"min_samples_leaf": 0},
        {"min_impurity_decrease": -0.1},
        {"class_weight": "auto"},
        {"class_weight": {1: 0.0}},
        {"max_features": 0},
        {"max_features": 1.5},
        {"max_features": "log"},
    ],
)
def test_tree_rejects_invalid_hyperparameters(parameters) -> None:
    with pytest.raises(ValueError):
        ClassificationTree(**parameters)


def test_tree_rejects_invalid_data_and_early_use() -> None:
    tree = ClassificationTree()
    with pytest.raises(RuntimeError, match="fit must be called"):
        tree.predict([[0.0]])
    with pytest.raises(RuntimeError, match="fit must be called"):
        tree.get_depth()
    with pytest.raises(ValueError, match="one label per observation"):
        tree.fit([[0.0], [1.0]], [0])
    with pytest.raises(ValueError, match="finite"):
        tree.fit([[np.nan], [1.0]], [0, 1])
    with pytest.raises(ValueError, match="cannot exceed"):
        ClassificationTree(max_features=3).fit([[0.0, 1.0], [1.0, 0.0]], [0, 1])

    tree.fit([[0.0, 1.0], [1.0, 0.0]], [0, 1])
    with pytest.raises(ValueError, match="wrong shape"):
        tree.predict([[1.0]])


def test_registry_builds_the_tree() -> None:
    tree = create_student_algorithm("decision_tree", max_depth=2)
    assert isinstance(tree, ClassificationTree) and tree.max_depth == 2
