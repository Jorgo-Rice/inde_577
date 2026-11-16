"""Tests for the student linear SVM and neural network."""

import numpy as np
import pytest
from sklearn.datasets import load_wine, make_moons
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from dsm_project.algorithms import (
    LinearSVM,
    NeuralNetworkClassifier,
    create_student_algorithm,
)
from dsm_project.algorithms.student.neural_network import forward, loss_and_gradients
from dsm_project.algorithms.student.svm import hinge_objective


@pytest.fixture
def linear_data():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(300, 3))
    score = features @ np.array([1.5, -1.0, 0.5]) + rng.normal(scale=0.8, size=300)
    return features, (score > 0.3).astype(int)


# --------------------------------------------------------------------------
# Linear SVM
# --------------------------------------------------------------------------


def test_hinge_objective_and_subgradient_by_hand() -> None:
    # w = 1, b = 0 on x = (-1, 0.5, 2) with y = (-1, +1, +1):
    # margins are 1, 0.5, 2, so only the middle row violates (hinge 0.5).
    features = np.array([[-1.0], [0.5], [2.0]])
    signed = np.array([-1.0, 1.0, 1.0])
    objective, grad_w, grad_b = hinge_objective(
        np.array([1.0]), 0.0, features, signed, 0.2
    )

    assert objective == pytest.approx(0.5 * 0.2 * 1.0 + 0.5 / 3)
    np.testing.assert_allclose(grad_w, [0.2 - 0.5 / 3])
    assert grad_b == pytest.approx(-1 / 3)


def test_svm_finds_the_maximum_margin_on_two_points() -> None:
    # Points at -1 and +1: the widest zero-hinge separator is w = 1, b = 0.
    model = LinearSVM(l2_penalty=0.01, max_iterations=20_000).fit(
        [[-1.0], [1.0]], [0, 1]
    )
    assert model.coef_[0] == pytest.approx(1.0, abs=0.02)
    assert model.intercept_ == pytest.approx(0.0, abs=0.02)
    np.testing.assert_array_equal(model.predict([[-3.0], [0.2]]), [0, 1])


def test_svm_reaches_the_trusted_optimum(linear_data) -> None:
    features, labels = linear_data
    penalty = 0.01
    student = LinearSVM(l2_penalty=penalty, max_iterations=10_000).fit(features, labels)
    # SVC minimizes 0.5||w||^2 + C * sum(hinge); dividing by C * n gives our
    # objective when C = 1 / (n * penalty). Its intercept is unpenalized too.
    trusted = SVC(kernel="linear", C=1 / (len(labels) * penalty)).fit(features, labels)
    signed = np.where(labels == 1, 1.0, -1.0)
    trusted_objective = hinge_objective(
        trusted.coef_[0], trusted.intercept_[0], features, signed, penalty
    )[0]

    assert student.objective_ <= trusted_objective * (1 + 1e-4)
    np.testing.assert_allclose(student.coef_, trusted.coef_[0], atol=0.01)
    assert np.mean(student.predict(features) == trusted.predict(features)) > 0.99


def test_svm_keeps_the_best_iterate(linear_data) -> None:
    features, labels = linear_data
    model = LinearSVM(max_iterations=500).fit(features, labels)
    # Subgradient steps can raise the objective; the kept iterate is the best seen.
    assert model.objective_ <= model.objective_history_.min() + 1e-12
    assert np.any(np.diff(model.objective_history_) > 0)  # not monotone
    assert model.objective_history_[0] == pytest.approx(1.0)  # w = 0, b = 0: hinge 1


def test_svm_decision_function_works_with_ranking_scorers(linear_data) -> None:
    features, labels = linear_data
    model = LinearSVM().fit(features, labels)
    assert model.decision_function(features).shape == (len(labels),)
    scores = cross_val_score(LinearSVM(), features, labels, scoring="average_precision")
    assert np.isfinite(scores).all() and scores.mean() > 0.8


def test_balanced_svm_flags_more_of_the_rare_class() -> None:
    rng = np.random.default_rng(3)
    features = rng.normal(size=(400, 2))
    labels = (features[:, 0] + rng.normal(scale=1.0, size=400) > 1.5).astype(int)
    plain = LinearSVM().fit(features, labels)
    balanced = LinearSVM(class_weight="balanced").fit(features, labels)
    assert labels.mean() < 0.2
    assert balanced.predict(features).sum() > plain.predict(features).sum()


@pytest.mark.parametrize(
    "parameters",
    [
        {"l2_penalty": 0.0},
        {"learning_rate": -1.0},
        {"max_iterations": 0},
        {"class_weight": "auto"},
        {"class_weight": {1: 0.0}},
    ],  # fmt: skip
)
def test_svm_rejects_invalid_hyperparameters(parameters) -> None:
    with pytest.raises(ValueError):
        LinearSVM(**parameters)


def test_svm_rejects_bad_use() -> None:
    with pytest.raises(RuntimeError, match="fit must be called"):
        LinearSVM().predict([[0.0]])
    with pytest.raises(ValueError, match="exactly two classes"):
        LinearSVM().fit([[0.0], [1.0], [2.0]], [0, 1, 2])
    model = LinearSVM(max_iterations=5).fit([[0.0, 1.0], [1.0, 0.0]], [0, 1])
    with pytest.raises(ValueError, match="wrong shape"):
        model.predict([[1.0]])


# --------------------------------------------------------------------------
# Neural network
# --------------------------------------------------------------------------


def test_backpropagation_matches_finite_differences() -> None:
    rng = np.random.default_rng(1)
    features = rng.normal(size=(8, 3))
    one_hot = np.eye(3)[rng.integers(0, 3, size=8)]
    parameters = [(rng.normal(size=(3, 5)), rng.normal(size=5)),
                  (rng.normal(size=(5, 4)), rng.normal(size=4)),
                  (rng.normal(size=(4, 3)), rng.normal(size=3))]  # fmt: skip
    _, gradients = loss_and_gradients(parameters, features, one_hot, alpha=0.1)

    step = 1e-6
    for layer, (weights, bias) in enumerate(parameters):
        for array, analytic in (
            (weights, gradients[layer][0]),
            (bias, gradients[layer][1]),
        ):
            numeric = np.zeros_like(array)
            for index in np.ndindex(array.shape):
                original = array[index]
                array[index] = original + step
                up, _ = loss_and_gradients(parameters, features, one_hot, alpha=0.1)
                array[index] = original - step
                down, _ = loss_and_gradients(parameters, features, one_hot, alpha=0.1)
                array[index] = original
                numeric[index] = (up - down) / (2 * step)
            np.testing.assert_allclose(analytic, numeric, atol=1e-6)


def test_forward_pass_outputs_probabilities() -> None:
    rng = np.random.default_rng(2)
    parameters = [
        (rng.normal(size=(2, 4)), np.zeros(4)),
        (rng.normal(size=(4, 3)), np.zeros(3)),
    ]
    pre, activations = forward(parameters, rng.normal(size=(5, 2)))
    np.testing.assert_allclose(activations[-1].sum(axis=1), 1.0)
    assert (activations[1] >= 0).all() and len(pre) == 2


def test_network_learns_xor_which_no_linear_model_can() -> None:
    features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 25, dtype=float)
    labels = np.array([0, 1, 1, 0] * 25)
    network = NeuralNetworkClassifier((8,), learning_rate=0.05, max_epochs=300,
                                      random_state=0).fit(features, labels)  # fmt: skip
    linear = LinearSVM().fit(features, labels)
    assert network.score(features, labels) == 1.0
    assert linear.score(features, labels) <= 0.75


def test_network_loss_falls_and_training_is_reproducible() -> None:
    features, labels = make_moons(300, noise=0.2, random_state=0)
    first = NeuralNetworkClassifier((16,), random_state=4).fit(features, labels)
    second = NeuralNetworkClassifier((16,), random_state=4).fit(features, labels)
    assert first.loss_curve_[-1] < 0.5 * first.loss_curve_[0]
    np.testing.assert_array_equal(
        first.predict_proba(features), second.predict_proba(features)
    )


def test_network_performance_is_comparable_to_trusted_mlp() -> None:
    features, labels = load_wine(return_X_y=True)
    features = StandardScaler().fit_transform(features)
    student = cross_val_score(
        NeuralNetworkClassifier((16,), random_state=0), features, labels
    )
    trusted = cross_val_score(
        MLPClassifier((16,), random_state=0, max_iter=1000), features, labels
    )
    assert student.mean() > 0.93
    assert abs(student.mean() - trusted.mean()) < 0.04


def test_network_binary_scores_work_with_ranking_scorers() -> None:
    features, labels = make_moons(200, noise=0.3, random_state=1)
    model = NeuralNetworkClassifier((8,), random_state=0).fit(features, labels)
    probabilities = model.predict_proba(features)
    np.testing.assert_allclose(
        model.decision_function(features),
        np.log(probabilities[:, 1] / probabilities[:, 0]),
        rtol=1e-6,
        atol=1e-8,
    )
    scores = cross_val_score(
        NeuralNetworkClassifier((8,), random_state=0),
        features,
        labels,
        scoring="average_precision",
    )
    assert np.isfinite(scores).all()


@pytest.mark.parametrize(
    "parameters",
    [
        {"hidden_layer_sizes": ()},
        {"hidden_layer_sizes": (0,)},
        {"alpha": -1.0},
        {"learning_rate": 0.0},
        {"batch_size": 0},
        {"max_epochs": 0},
        {"tol": -1.0},
    ],  # fmt: skip
)
def test_network_rejects_invalid_hyperparameters(parameters) -> None:
    with pytest.raises(ValueError):
        NeuralNetworkClassifier(**parameters)


def test_network_rejects_bad_use() -> None:
    with pytest.raises(RuntimeError, match="fit must be called"):
        NeuralNetworkClassifier().predict_proba([[0.0]])
    with pytest.raises(ValueError, match="two distinct classes"):
        NeuralNetworkClassifier().fit([[0.0], [1.0]], [1, 1])
    model = NeuralNetworkClassifier((4,), max_epochs=2, random_state=0).fit(
        [[0.0, 1.0], [1.0, 0.0]], [0, 1]
    )
    with pytest.raises(ValueError, match="wrong shape"):
        model.predict([[1.0]])


def test_registry_builds_both_models() -> None:
    assert isinstance(create_student_algorithm("linear_svm"), LinearSVM)
    network = create_student_algorithm("neural_network", hidden_layer_sizes=(8,))
    assert isinstance(network, NeuralNetworkClassifier)
