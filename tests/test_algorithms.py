"""Mathematical and failure-mode tests for project-owned algorithms."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dsm_project.algorithms import (
    BatchLinearRegressor,
    MajorityClassifier,
    MeanRegressor,
    create_student_algorithm,
    registered_student_algorithms,
)


def test_baselines_learn_only_training_target_summaries() -> None:
    features = [[0.0], [1.0], [2.0]]

    regressor = MeanRegressor().fit(features, [1.0, 4.0, 7.0])
    classifier = MajorityClassifier().fit(features, ["a", "a", "b"])

    np.testing.assert_allclose(regressor.predict([[10.0], [20.0]]), [4.0, 4.0])
    np.testing.assert_array_equal(classifier.predict([[10.0]]), ["a"])


def test_batch_linear_regression_recovers_an_affine_rule() -> None:
    features = np.linspace(-1.0, 1.0, 80).reshape(-1, 1)
    targets = 1.25 + 2.5 * features[:, 0]
    model = BatchLinearRegressor(learning_rate=0.1, tolerance=1e-10).fit(
        features, targets
    )

    np.testing.assert_allclose(model.coef_, [2.5], atol=1e-5)
    assert model.intercept_ == pytest.approx(1.25, abs=1e-5)
    assert model.loss_history_[-1] < model.loss_history_[0]


def test_batch_linear_regression_agrees_with_trusted_solver() -> None:
    rng = np.random.default_rng(438)
    features = rng.normal(size=(120, 3))
    targets = features @ np.array([0.7, -1.2, 2.1]) + 0.4
    transparent = BatchLinearRegressor(learning_rate=0.08).fit(features, targets)
    trusted = LinearRegression().fit(features, targets)

    np.testing.assert_allclose(
        transparent.predict(features), trusted.predict(features), atol=1e-4
    )


def test_estimators_reject_use_before_fit() -> None:
    with pytest.raises(RuntimeError, match="fit must be called"):
        MeanRegressor().predict([[1.0]])


def test_student_registry_exposes_only_explicit_implementations() -> None:
    assert registered_student_algorithms() == ("batch_linear_regression",)
    estimator = create_student_algorithm(
        "batch_linear_regression", learning_rate=0.05
    )
    assert isinstance(estimator, BatchLinearRegressor)

    with pytest.raises(ValueError, match="Unknown student algorithm"):
        create_student_algorithm("logistic_regression")


def test_batch_linear_regressor_composes_with_sklearn_pipeline() -> None:
    features = np.arange(12, dtype=float).reshape(-1, 1)
    targets = 2.0 * features[:, 0] - 1.0
    pipeline = make_pipeline(
        StandardScaler(),
        BatchLinearRegressor(learning_rate=0.05),
    ).fit(features, targets)

    assert pipeline.predict(features).shape == targets.shape
    assert pipeline.get_params()["batchlinearregressor__learning_rate"] == 0.05
