"""Tests for the regression tree, random forest, and gradient boosting."""

import numpy as np
import pytest
from sklearn.datasets import load_wine
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor

from dsm_project.algorithms import (
    ClassificationTree,
    GradientBoostingBinaryClassifier,
    RandomForest,
    RegressionTree,
    create_student_algorithm,
)
from dsm_project.algorithms.student.trees import best_regression_split


def _float32_exact(array: np.ndarray) -> np.ndarray:
    # scikit-learn trees store features as float32; data that is exactly
    # representable in float32 makes both implementations see identical values.
    return np.asarray(array, dtype=np.float32).astype(float)


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(438)
    features = _float32_exact(rng.normal(size=(400, 5)))
    score = features[:, 0] + features[:, 1] * features[:, 2] - 0.5 * features[:, 3]
    labels = (score + rng.normal(scale=0.8, size=400) > 0.6).astype(int)
    return train_test_split(features, labels, test_size=0.3, random_state=0)


# --------------------------------------------------------------------------
# Regression tree
# --------------------------------------------------------------------------


def test_regression_split_matches_hand_computation() -> None:
    # Targets jump from about 1 to about 10 between x = 2 and x = 3.
    features = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    targets = np.array([1.0, 1.0, 1.0, 10.0, 10.0])
    feature, threshold, child = best_regression_split(features, targets, np.ones(5))

    assert (feature, threshold, child) == (0, 2.5, 0.0)


def test_regression_tree_leaves_predict_weighted_means() -> None:
    # Weighted SSE of the three cuts (weights 1, 3, 1, 1):
    #   x <= 0.5: 60.8;  x <= 3.0: 3 + 2 = 5;  x <= 5.5: 36.8 + 0 = 36.8.
    # So the tree cuts at 3.0 (midpoint of 1 and 5); leaf means are
    # (2 + 3*4) / 4 = 3.5 and (10 + 12) / 2 = 11.
    features = np.array([[0.0], [1.0], [5.0], [6.0]])
    targets = np.array([2.0, 4.0, 10.0, 12.0])
    tree = RegressionTree(max_depth=1).fit(
        features, targets, sample_weight=[1, 3, 1, 1]
    )

    assert tree.tree_["threshold"][0] == 3.0
    np.testing.assert_allclose(tree.predict([[0.5], [5.5]]), [3.5, 11.0])
    assert tree.get_n_leaves() == 2
    assert "leaf (n=2): value 3.500" in tree.export_text()


@pytest.mark.parametrize(
    "parameters",
    [{"max_depth": 3}, {"max_depth": 6, "min_samples_leaf": 5}, {"max_depth": None}],
)
def test_regression_tree_agrees_with_trusted_tree(parameters) -> None:
    rng = np.random.default_rng(7)
    features = _float32_exact(rng.normal(size=(300, 4)))
    targets = features[:, 0] ** 2 + features[:, 1] + rng.normal(size=300)
    student = RegressionTree(**parameters).fit(features, targets)
    trusted = DecisionTreeRegressor(random_state=0, **parameters).fit(features, targets)

    assert student.get_n_leaves() == trusted.get_n_leaves()
    np.testing.assert_allclose(
        student.predict(features), trusted.predict(features), atol=1e-12
    )
    if parameters.get("min_samples_leaf", 1) >= 5:
        # A 2-row node is split equally well by every feature that separates
        # the two rows; scikit-learn breaks that tie at random, so the credit
        # (importance) can go to a different feature. Larger leaves rule out
        # such ties, and then importances must agree too.
        np.testing.assert_allclose(
            student.feature_importances_, trusted.feature_importances_, atol=1e-10
        )


def test_regression_tree_rejects_bad_targets_and_weights() -> None:
    tree = RegressionTree()
    with pytest.raises(ValueError, match="one value per observation"):
        tree.fit([[0.0], [1.0]], [1.0])
    with pytest.raises(ValueError, match="finite"):
        tree.fit([[0.0], [1.0]], [1.0, np.inf])
    with pytest.raises(ValueError, match="sample_weight"):
        tree.fit([[0.0], [1.0]], [1.0, 2.0], sample_weight=[1.0, -1.0])


# --------------------------------------------------------------------------
# Random forest
# --------------------------------------------------------------------------


def test_one_tree_without_randomness_equals_a_single_tree(binary_data) -> None:
    train_x, test_x, train_y, _ = binary_data
    forest = RandomForest(n_estimators=1, bootstrap=False, max_features=None).fit(
        train_x, train_y
    )
    tree = ClassificationTree().fit(train_x, train_y)

    np.testing.assert_array_equal(
        forest.predict_proba(test_x), tree.predict_proba(test_x)
    )


def test_forest_probability_is_the_mean_of_tree_probabilities(binary_data) -> None:
    train_x, test_x, train_y, _ = binary_data
    forest = RandomForest(n_estimators=7, random_state=1).fit(train_x, train_y)
    manual = np.mean(
        [tree.predict_proba(test_x) for tree in forest.estimators_], axis=0
    )

    np.testing.assert_allclose(forest.predict_proba(test_x), manual)
    assert len({tree.random_state for tree in forest.estimators_}) == 7


def test_forest_is_reproducible_and_seed_dependent(binary_data) -> None:
    train_x, test_x, train_y, _ = binary_data
    first = RandomForest(n_estimators=10, random_state=3).fit(train_x, train_y)
    again = RandomForest(n_estimators=10, random_state=3).fit(train_x, train_y)
    other = RandomForest(n_estimators=10, random_state=4).fit(train_x, train_y)

    np.testing.assert_array_equal(
        first.predict_proba(test_x), again.predict_proba(test_x)
    )
    assert not np.array_equal(first.predict_proba(test_x), other.predict_proba(test_x))


def test_averaging_reduces_variance_across_seeds(binary_data) -> None:
    # The core claim of bagging: an average of B trees varies less (across
    # random seeds) than a single randomized tree does.
    train_x, test_x, train_y, _ = binary_data

    def spread(n_estimators):
        predictions = [
            RandomForest(n_estimators=n_estimators, random_state=seed)
            .fit(train_x, train_y)
            .predict_proba(test_x)[:, 1]
            for seed in range(8)
        ]
        return np.mean(np.var(predictions, axis=0))

    assert spread(25) < 0.25 * spread(1)


def test_out_of_bag_rows_were_not_used_by_their_trees(binary_data) -> None:
    train_x, _, train_y, _ = binary_data
    forest = RandomForest(n_estimators=40, random_state=2).fit(train_x, train_y)

    assert forest.oob_proba_.shape == (len(train_y), 2)
    assert not np.isnan(forest.oob_proba_).any()  # 40 trees leave every row out
    np.testing.assert_allclose(forest.oob_proba_.sum(axis=1), 1.0)
    # In-bag predictions are optimistic; out-of-bag ones are honest.
    in_bag_auc = roc_auc_score(train_y, forest.predict_proba(train_x)[:, 1])
    oob_auc = roc_auc_score(train_y, forest.oob_proba_[:, 1])
    assert oob_auc < in_bag_auc


def test_forest_handles_bootstraps_that_miss_a_class() -> None:
    features = np.arange(30, dtype=float).reshape(-1, 1)
    labels = np.array([0] * 28 + [1, 2])  # two rare classes
    forest = RandomForest(n_estimators=20, random_state=0).fit(features, labels)

    probabilities = forest.predict_proba(features)
    assert probabilities.shape == (30, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_forest_performance_is_comparable_to_trusted_forest() -> None:
    # Different random number generators make exact agreement impossible, so
    # compare cross-validated accuracy on the same folds instead.
    features, labels = load_wine(return_X_y=True)
    student = cross_val_score(
        RandomForest(n_estimators=100, random_state=0), features, labels
    )
    trusted = cross_val_score(
        RandomForestClassifier(n_estimators=100, random_state=0), features, labels
    )

    assert abs(student.mean() - trusted.mean()) < 0.03
    assert student.mean() > 0.93


# --------------------------------------------------------------------------
# Gradient boosting
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parameters",
    [
        {"n_estimators": 50},
        {"n_estimators": 80, "learning_rate": 0.3, "max_depth": 2},
        {"n_estimators": 40, "max_depth": 4, "min_samples_leaf": 10},
    ],
)
def test_boosting_agrees_with_trusted_boosting(binary_data, parameters) -> None:
    train_x, test_x, train_y, _ = binary_data
    student = GradientBoostingBinaryClassifier(**parameters).fit(train_x, train_y)
    trusted = GradientBoostingClassifier(random_state=0, **parameters).fit(
        train_x, train_y
    )

    # Same algorithm, so the fit on the training rows must match exactly.
    np.testing.assert_allclose(
        student.decision_function(train_x),
        trusted.decision_function(train_x),
        atol=1e-9,
    )
    # scikit-learn reports binomial deviance, which is twice the mean log-loss.
    np.testing.assert_allclose(student.train_loss_ * 2, trusted.train_score_, rtol=1e-9)
    if parameters.get("min_samples_leaf", 1) >= 10:
        # With tiny leaves, equally good splits (ties) can route unseen points
        # differently; with 10+ rows per leaf ties do not occur, so new points
        # must be scored identically as well.
        np.testing.assert_allclose(
            student.predict_proba(test_x), trusted.predict_proba(test_x), atol=1e-12
        )


def test_boosting_starts_at_prior_log_odds_and_lowers_training_loss(binary_data):
    train_x, _, train_y, _ = binary_data
    model = GradientBoostingBinaryClassifier(n_estimators=60).fit(train_x, train_y)
    rate = train_y.mean()

    assert model.init_score_ == pytest.approx(np.log(rate / (1 - rate)))
    assert np.all(np.diff(model.train_loss_) <= 1e-12)  # each stage helps on train
    prior_loss = -(rate * np.log(rate) + (1 - rate) * np.log(1 - rate))
    assert model.train_loss_[-1] < 0.8 * prior_loss


def test_newton_leaf_values_match_hand_computation() -> None:
    # One stage, depth 1, from prior p = 0.5: residuals are +-0.5 and
    # p(1-p) = 0.25, so a pure leaf of positives gets 0.5k / 0.25k = 2.
    features = np.array([[0.0], [1.0], [5.0], [6.0]])
    labels = np.array([0, 0, 1, 1])
    model = GradientBoostingBinaryClassifier(
        n_estimators=1, learning_rate=1.0, max_depth=1
    ).fit(features, labels)

    np.testing.assert_allclose(model.decision_function(features), [-2, -2, 2, 2])


def test_staged_probabilities_end_at_final_probabilities(binary_data) -> None:
    train_x, test_x, train_y, _ = binary_data
    model = GradientBoostingBinaryClassifier(n_estimators=15).fit(train_x, train_y)
    stages = list(model.staged_predict_proba(test_x))

    assert len(stages) == 15
    np.testing.assert_allclose(stages[-1], model.predict_proba(test_x))


def test_subsampling_is_reproducible_and_changes_the_fit(binary_data) -> None:
    train_x, test_x, train_y, _ = binary_data
    full = GradientBoostingBinaryClassifier(n_estimators=20).fit(train_x, train_y)
    sub_a = GradientBoostingBinaryClassifier(
        n_estimators=20, subsample=0.6, random_state=1
    ).fit(train_x, train_y)
    sub_b = GradientBoostingBinaryClassifier(
        n_estimators=20, subsample=0.6, random_state=1
    ).fit(train_x, train_y)

    np.testing.assert_array_equal(
        sub_a.predict_proba(test_x), sub_b.predict_proba(test_x)
    )
    assert not np.allclose(sub_a.predict_proba(test_x), full.predict_proba(test_x))


def test_boosting_works_with_sklearn_ranking_scorers(binary_data) -> None:
    train_x, _, train_y, _ = binary_data
    scores = cross_val_score(
        GradientBoostingBinaryClassifier(n_estimators=10),
        train_x,
        train_y,
        scoring="average_precision",
    )
    assert np.isfinite(scores).all()


# --------------------------------------------------------------------------
# Failure modes and registry
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("factory", "parameters"),
    [
        (RandomForest, {"n_estimators": 0}),
        (RandomForest, {"max_features": "log"}),
        (RandomForest, {"min_samples_leaf": 0}),
        (GradientBoostingBinaryClassifier, {"n_estimators": 0}),
        (GradientBoostingBinaryClassifier, {"learning_rate": 0.0}),
        (GradientBoostingBinaryClassifier, {"learning_rate": 1.5}),
        (GradientBoostingBinaryClassifier, {"subsample": 0.0}),
        (GradientBoostingBinaryClassifier, {"max_depth": -1}),
    ],
)
def test_ensembles_reject_invalid_hyperparameters(factory, parameters) -> None:
    with pytest.raises(ValueError):
        factory(**parameters)


def test_ensembles_reject_bad_use() -> None:
    for model in (RandomForest(n_estimators=2), GradientBoostingBinaryClassifier()):
        with pytest.raises(RuntimeError, match="fit must be called"):
            model.predict_proba([[0.0]])
    with pytest.raises(ValueError, match="exactly two classes"):
        GradientBoostingBinaryClassifier().fit([[0.0], [1.0], [2.0]], [0, 1, 2])

    forest = RandomForest(n_estimators=2, random_state=0).fit(
        [[0.0, 1.0], [1.0, 0.0]], [0, 1]
    )
    with pytest.raises(ValueError, match="wrong shape"):
        forest.predict([[1.0]])


def test_registry_builds_the_ensembles() -> None:
    assert isinstance(create_student_algorithm("random_forest"), RandomForest)
    assert isinstance(
        create_student_algorithm("gradient_boosting", n_estimators=5),
        GradientBoostingBinaryClassifier,
    )


def test_stump_contributions_add_up_to_the_decision_function(binary_data) -> None:
    train_x, test_x, train_y, _ = binary_data
    model = GradientBoostingBinaryClassifier(n_estimators=40, max_depth=1).fit(
        train_x, train_y
    )
    total = model.init_score_ + sum(
        model.stump_contribution(j, test_x[:, j]) for j in range(test_x.shape[1])
    )

    np.testing.assert_allclose(total, model.decision_function(test_x), atol=1e-12)


def test_stump_contribution_requires_stumps_and_valid_feature(binary_data) -> None:
    train_x, _, train_y, _ = binary_data
    deep = GradientBoostingBinaryClassifier(n_estimators=3, max_depth=2).fit(
        train_x, train_y
    )
    with pytest.raises(ValueError, match="max_depth=1"):
        deep.stump_contribution(0, [0.0])
    stumps = GradientBoostingBinaryClassifier(n_estimators=3, max_depth=1).fit(
        train_x, train_y
    )
    with pytest.raises(ValueError, match="out of range"):
        stumps.stump_contribution(99, [0.0])
