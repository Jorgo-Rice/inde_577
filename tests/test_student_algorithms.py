"""Mathematical, failure-mode, and differential tests for student algorithms.

Each estimator is checked three ways:

1. hand-computable or analytic cases whose answer is known in advance;
2. invalid inputs, use before fitting, and shape boundaries; and
3. differential agreement with a trusted scikit-learn implementation whose
   objective and conventions match.
"""

import numpy as np
import pytest
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dsm_project.algorithms import (
    KNearestNeighborsClassifier,
    SoftmaxRegression,
    create_student_algorithm,
)
from dsm_project.algorithms.student.logistic import cross_entropy_objective, softmax
from dsm_project.algorithms.student.neighbors import pairwise_distances

# --------------------------------------------------------------------------
# Softmax regression
# --------------------------------------------------------------------------


def test_softmax_rows_are_probabilities_and_resist_overflow() -> None:
    probabilities = softmax([[1000.0, 1000.0], [0.0, np.log(3.0)]])

    np.testing.assert_allclose(probabilities.sum(axis=1), [1.0, 1.0])
    np.testing.assert_allclose(probabilities[0], [0.5, 0.5])
    np.testing.assert_allclose(probabilities[1], [0.25, 0.75])


def test_objective_at_zero_parameters_equals_log_class_count() -> None:
    # With W = 0 and b = 0 every class has probability 1/K, so the mean
    # cross-entropy is exactly log K regardless of the data.
    rng = np.random.default_rng(0)
    features = rng.normal(size=(10, 3))
    one_hot = np.eye(4)[rng.integers(0, 4, size=10)]
    loss, _, _ = cross_entropy_objective(
        np.zeros((3, 4)), np.zeros(4), features, one_hot
    )

    assert loss == pytest.approx(np.log(4.0))


def test_objective_gradient_matches_finite_differences() -> None:
    rng = np.random.default_rng(577)
    features = rng.normal(size=(15, 3))
    one_hot = np.eye(3)[rng.integers(0, 3, size=15)]
    weights = rng.normal(size=(3, 3))
    intercepts = rng.normal(size=3)
    penalty = 0.3

    _, weight_gradient, intercept_gradient = cross_entropy_objective(
        weights, intercepts, features, one_hot, penalty
    )

    step = 1e-6
    numeric_weights = np.zeros_like(weights)
    for index in np.ndindex(weights.shape):
        shift = np.zeros_like(weights)
        shift[index] = step
        upper, _, _ = cross_entropy_objective(
            weights + shift, intercepts, features, one_hot, penalty
        )
        lower, _, _ = cross_entropy_objective(
            weights - shift, intercepts, features, one_hot, penalty
        )
        numeric_weights[index] = (upper - lower) / (2 * step)
    numeric_intercepts = np.zeros_like(intercepts)
    for k in range(intercepts.size):
        shift = np.zeros_like(intercepts)
        shift[k] = step
        upper, _, _ = cross_entropy_objective(
            weights, intercepts + shift, features, one_hot, penalty
        )
        lower, _, _ = cross_entropy_objective(
            weights, intercepts - shift, features, one_hot, penalty
        )
        numeric_intercepts[k] = (upper - lower) / (2 * step)

    np.testing.assert_allclose(weight_gradient, numeric_weights, atol=1e-7)
    np.testing.assert_allclose(intercept_gradient, numeric_intercepts, atol=1e-7)


def test_softmax_regression_separates_well_separated_classes() -> None:
    features = np.array([[-3.0], [-2.0], [-2.5], [2.0], [3.0], [2.5]])
    targets = np.array(["low", "low", "low", "high", "high", "high"])
    model = SoftmaxRegression(learning_rate=0.5, max_iterations=2_000).fit(
        features, targets
    )

    np.testing.assert_array_equal(model.predict(features), targets)
    np.testing.assert_array_equal(model.classes_, ["high", "low"])
    assert model.loss_history_[-1] < model.loss_history_[0]
    np.testing.assert_allclose(model.predict_proba(features).sum(axis=1), 1.0)


def test_softmax_regression_intercept_only_recovers_class_frequencies() -> None:
    # A constant feature duplicates the intercept, so any ridge penalty drives
    # W to zero and the optimum sets softmax(b) to the class frequencies.
    features = np.ones((10, 1))
    targets = np.array([0] * 2 + [1] * 3 + [2] * 5)
    model = SoftmaxRegression(
        learning_rate=0.5, max_iterations=20_000, tolerance=1e-10, l2_penalty=1.0
    ).fit(features, targets)

    np.testing.assert_allclose(
        model.predict_proba(features[:1])[0], [0.2, 0.3, 0.5], atol=1e-6
    )
    assert model.converged_


def test_softmax_regression_agrees_with_trusted_classifier() -> None:
    features, targets = load_wine(return_X_y=True)
    scaled = StandardScaler().fit_transform(features)
    penalty = 0.05
    student = SoftmaxRegression(
        learning_rate=0.5, max_iterations=20_000, tolerance=1e-9, l2_penalty=penalty
    ).fit(scaled, targets)
    # sklearn minimizes C * sum(loss) + ||W||^2 / 2; dividing by C * n shows
    # this equals our mean-loss objective when C = 1 / (n * penalty).
    trusted = LogisticRegression(
        C=1.0 / (len(targets) * penalty), tol=1e-12, max_iter=10_000
    ).fit(scaled, targets)

    assert student.converged_
    np.testing.assert_allclose(
        student.predict_proba(scaled), trusted.predict_proba(scaled), atol=1e-5
    )
    np.testing.assert_array_equal(student.predict(scaled), trusted.predict(scaled))


@pytest.mark.parametrize(
    "parameters",
    [
        {"learning_rate": 0.0},
        {"max_iterations": 0},
        {"tolerance": -1.0},
        {"l2_penalty": -0.1},
    ],
)
def test_softmax_regression_rejects_invalid_hyperparameters(parameters) -> None:
    with pytest.raises(ValueError):
        SoftmaxRegression(**parameters)


def test_softmax_regression_rejects_invalid_data_and_early_use() -> None:
    model = SoftmaxRegression()
    with pytest.raises(RuntimeError, match="fit must be called"):
        model.predict([[1.0]])
    with pytest.raises(ValueError, match="two distinct classes"):
        model.fit([[1.0], [2.0]], [1, 1])
    with pytest.raises(ValueError, match="one label per observation"):
        model.fit([[1.0], [2.0]], [0, 1, 1])
    with pytest.raises(ValueError, match="finite"):
        model.fit([[np.nan], [2.0]], [0, 1])

    model.fit([[0.0, 1.0], [1.0, 0.0]], [0, 1])
    with pytest.raises(ValueError, match="wrong shape"):
        model.predict([[1.0]])


def test_softmax_regression_reports_divergence_instead_of_returning_nan() -> None:
    # The penalty alone contributes curvature 1000, so a step of 0.5 overshoots.
    with pytest.raises(FloatingPointError, match="diverged"):
        SoftmaxRegression(learning_rate=0.5, l2_penalty=1e3).fit(
            [[1.0], [2.0], [3.0]], [0, 1, 1]
        )


def test_softmax_regression_is_deterministic() -> None:
    features, targets = load_wine(return_X_y=True)
    scaled = StandardScaler().fit_transform(features)
    first = SoftmaxRegression(max_iterations=200).fit(scaled, targets)
    second = SoftmaxRegression(max_iterations=200).fit(scaled, targets)

    np.testing.assert_array_equal(first.coef_, second.coef_)


# --------------------------------------------------------------------------
# k-nearest neighbors
# --------------------------------------------------------------------------


def test_pairwise_distances_match_hand_computation() -> None:
    queries = np.array([[0.0, 0.0]])
    references = np.array([[3.0, 4.0], [1.0, -1.0]])

    np.testing.assert_allclose(
        pairwise_distances(queries, references, "euclidean"), [[5.0, np.sqrt(2.0)]]
    )
    np.testing.assert_allclose(
        pairwise_distances(queries, references, "manhattan"), [[7.0, 2.0]]
    )


def test_knn_hand_computable_vote() -> None:
    # On a number line the three nearest training points to x = 0.9 are
    # 1.0 ("b"), 0.0 ("a"), and 2.0 ("b"), so "b" wins two votes to one.
    features = np.array([[0.0], [1.0], [2.0], [10.0]])
    targets = np.array(["a", "b", "b", "a"])
    model = KNearestNeighborsClassifier(n_neighbors=3).fit(features, targets)

    distances, indices = model.kneighbors([[0.9]])
    np.testing.assert_array_equal(indices, [[1, 0, 2]])
    np.testing.assert_allclose(distances, [[0.1, 0.9, 1.1]])
    np.testing.assert_array_equal(model.predict([[0.9]]), ["b"])
    np.testing.assert_allclose(model.predict_proba([[0.9]]), [[1 / 3, 2 / 3]])


def test_knn_one_neighbor_memorizes_training_data() -> None:
    rng = np.random.default_rng(1)
    features = rng.normal(size=(30, 4))
    targets = rng.integers(0, 3, size=30)
    model = KNearestNeighborsClassifier(n_neighbors=1).fit(features, targets)

    np.testing.assert_array_equal(model.predict(features), targets)


def test_knn_distance_weighting_can_overturn_majority() -> None:
    # Uniform voting: two distant "a" points beat one close "b".
    # Inverse-distance voting: 1/0.1 = 10 for "b" beats 1/4 + 1/5 for "a".
    features = np.array([[0.1], [4.0], [-5.0]])
    targets = np.array(["b", "a", "a"])
    query = [[0.0]]

    uniform = KNearestNeighborsClassifier(n_neighbors=3).fit(features, targets)
    weighted = KNearestNeighborsClassifier(n_neighbors=3, weights="distance").fit(
        features, targets
    )
    np.testing.assert_array_equal(uniform.predict(query), ["a"])
    np.testing.assert_array_equal(weighted.predict(query), ["b"])


def test_knn_distance_weighting_handles_exact_matches() -> None:
    features = np.array([[0.0], [0.0], [1.0]])
    targets = np.array([1, 1, 0])
    model = KNearestNeighborsClassifier(n_neighbors=3, weights="distance").fit(
        features, targets
    )

    np.testing.assert_allclose(model.predict_proba([[0.0]]), [[0.0, 1.0]])


def test_knn_tied_votes_go_to_smallest_label() -> None:
    features = np.array([[-1.0], [1.0]])
    model = KNearestNeighborsClassifier(n_neighbors=2).fit(features, [5, 3])

    np.testing.assert_array_equal(model.predict([[0.0]]), [3])


@pytest.mark.parametrize(
    ("n_neighbors", "metric", "weights"),
    [
        (1, "euclidean", "uniform"),
        (7, "euclidean", "uniform"),
        (7, "manhattan", "uniform"),
        (5, "euclidean", "distance"),
    ],
)
def test_knn_agrees_with_trusted_classifier(n_neighbors, metric, weights) -> None:
    features, targets = load_wine(return_X_y=True)
    train_x, test_x, train_y, _ = train_test_split(
        features, targets, test_size=0.3, random_state=438, stratify=targets
    )
    student = make_pipeline(
        StandardScaler(),
        KNearestNeighborsClassifier(
            n_neighbors=n_neighbors, metric=metric, weights=weights
        ),
    ).fit(train_x, train_y)
    trusted = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(
            n_neighbors=n_neighbors, metric=metric, weights=weights,
            algorithm="brute",
        ),
    ).fit(train_x, train_y)

    np.testing.assert_allclose(
        student.predict_proba(test_x), trusted.predict_proba(test_x), atol=1e-12
    )
    np.testing.assert_array_equal(student.predict(test_x), trusted.predict(test_x))


@pytest.mark.parametrize(
    "parameters",
    [
        {"n_neighbors": 0},
        {"n_neighbors": 2.5},
        {"metric": "cosine"},
        {"weights": "rank"},
    ],
)
def test_knn_rejects_invalid_hyperparameters(parameters) -> None:
    with pytest.raises(ValueError):
        KNearestNeighborsClassifier(**parameters)


def test_knn_rejects_invalid_data_and_early_use() -> None:
    model = KNearestNeighborsClassifier(n_neighbors=3)
    with pytest.raises(RuntimeError, match="fit must be called"):
        model.predict([[1.0]])
    with pytest.raises(ValueError, match="cannot exceed"):
        model.fit([[1.0], [2.0]], [0, 1])
    with pytest.raises(ValueError, match="finite"):
        model.fit([[np.inf], [2.0], [3.0]], [0, 1, 1])

    model.fit([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], [0, 1, 1])
    with pytest.raises(ValueError, match="wrong shape"):
        model.predict([[1.0]])


# --------------------------------------------------------------------------
# Registry integration
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["softmax_regression", "k_nearest_neighbors"])
def test_registered_classifiers_run_through_the_registry(name) -> None:
    features, targets = load_wine(return_X_y=True)
    estimator = make_pipeline(StandardScaler(), create_student_algorithm(name))
    estimator.fit(features, targets)

    assert estimator.score(features, targets) > 0.9
