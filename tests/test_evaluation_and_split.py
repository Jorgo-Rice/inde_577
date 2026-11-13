"""Tests for the shared holdout boundary and bootstrap uncertainty."""

from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from dsm_project.config import load_config
from dsm_project.data import load_project_dataset
from dsm_project.evaluation import bootstrap_metric_interval
from dsm_project.experiments import split_supervised_task

ROOT = Path(__file__).parents[1]


def test_split_is_reproducible_disjoint_and_stratified() -> None:
    config = load_config(ROOT / "configs/project.toml")
    dataset = load_project_dataset(config)
    first = split_supervised_task(dataset, config, task_name="classification")
    second = split_supervised_task(dataset, config, task_name="classification")

    assert first.test_features.index.equals(second.test_features.index)
    assert set(first.train_features.index).isdisjoint(first.test_features.index)
    assert len(first.train_features) + len(first.test_features) == len(
        dataset.observations
    )
    overall = dataset.observations["cultivar"].value_counts(normalize=True)
    held_out = first.test_targets.value_counts(normalize=True)
    np.testing.assert_allclose(held_out.sort_index(), overall.sort_index(), atol=0.03)


def test_split_rejects_unknown_and_unsupervised_tasks() -> None:
    config = load_config(ROOT / "configs/project.toml")
    dataset = load_project_dataset(config)
    with pytest.raises(ValueError, match="Unknown task"):
        split_supervised_task(dataset, config, task_name="nope")
    with pytest.raises(ValueError, match="unsupervised"):
        split_supervised_task(dataset, config, task_name="unsupervised")


def test_bootstrap_interval_is_degenerate_for_perfect_predictions() -> None:
    labels = np.array([0, 1, 1, 0, 1])
    interval = bootstrap_metric_interval(labels, labels, accuracy_score)

    assert interval.estimate == interval.lower == interval.upper == 1.0


def test_bootstrap_interval_brackets_estimate_and_is_reproducible() -> None:
    rng = np.random.default_rng(0)
    observed = rng.integers(0, 2, size=200)
    predicted = np.where(rng.random(200) < 0.8, observed, 1 - observed)
    first = bootstrap_metric_interval(observed, predicted, accuracy_score)
    second = bootstrap_metric_interval(observed, predicted, accuracy_score)

    assert first.lower < first.estimate < first.upper
    assert first == second
    # Normal-approximation half-width for p ~ 0.8, n = 200 is about 0.055.
    assert 0.08 < first.upper - first.lower < 0.14


@pytest.mark.parametrize(
    "keywords", [{"confidence": 1.0}, {"confidence": 0.0}, {"resamples": 0}]
)
def test_bootstrap_rejects_invalid_settings(keywords) -> None:
    with pytest.raises(ValueError):
        bootstrap_metric_interval([0, 1], [0, 1], accuracy_score, **keywords)


def test_bootstrap_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="aligned"):
        bootstrap_metric_interval([0, 1, 1], [0, 1], accuracy_score)
