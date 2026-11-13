"""Integration tests for supervised and unsupervised experiment paths."""

import numpy as np

from dsm_project.config import load_config
from dsm_project.contracts import TaskKind
from dsm_project.data import load_project_dataset
from dsm_project.evaluation import comparison_table
from dsm_project.experiments import (
    available_models,
    evaluate_supervised_estimator,
    fit_supervised_candidate,
    run_clustering_example,
    run_pca_example,
)


def configured_example():
    config = load_config("configs/example.toml")
    return config, load_project_dataset(config)


def test_supervised_reference_paths_produce_finite_evidence() -> None:
    config, dataset = configured_example()
    regression = fit_supervised_candidate(
        dataset, config, task_name="regression", model_name="linear_regression"
    )
    classification = fit_supervised_candidate(
        dataset,
        config,
        task_name="classification",
        model_name="logistic_regression",
    )

    assert np.isfinite(regression.metric_value)
    assert 0.0 <= classification.metric_value <= 1.0
    assert regression.train_rows + regression.test_rows == len(dataset.observations)
    assert len(regression.observed) == regression.test_rows
    assert len(regression.predictions) == regression.test_rows
    assert len(regression.holdout_features) == regression.test_rows
    assert regression.model_name == "reference:linear_regression"


def test_student_and_reference_estimators_share_the_evaluation_boundary() -> None:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from dsm_project.algorithms import create_student_algorithm

    config, dataset = configured_example()
    estimator = make_pipeline(
        StandardScaler(),
        create_student_algorithm("batch_linear_regression", learning_rate=0.05),
    )
    result = evaluate_supervised_estimator(
        dataset,
        config,
        task_name="regression",
        estimator_name="student:batch_linear_regression",
        estimator=estimator,
    )

    assert result.model_name == "student:batch_linear_regression"
    assert np.isfinite(result.metric_value)


def test_every_declared_classification_family_runs() -> None:
    config, dataset = configured_example()
    table = comparison_table(
        dataset,
        config,
        task_name="classification",
        model_names=available_models(TaskKind.CLASSIFICATION),
    )

    assert set(table["model"]) == {
        f"reference:{name}"
        for name in available_models(TaskKind.CLASSIFICATION)
    }
    assert table["value"].between(0.0, 1.0).all()


def test_unsupervised_paths_do_not_require_targets_for_fitting() -> None:
    config, dataset = configured_example()
    clustering = run_clustering_example(dataset, config)
    representation = run_pca_example(dataset)

    assert len(clustering.labels) == len(dataset.observations)
    assert -1.0 <= clustering.stability_ari <= 1.0
    assert representation.coordinates.shape == (len(dataset.observations), 2)
    assert representation.components.shape == (2, len(representation.feature_names))
    assert 0.0 < representation.explained_variance_ratio.sum() <= 1.0
