"""Tests for the promoted churn model and its versioned bundle."""

from pathlib import Path

import numpy as np
import pytest

from dsm_project.artifacts import load_bundle, save_bundle
from dsm_project.churn_model import (
    DEFAULT_ECONOMICS,
    MODEL_VERSION,
    build_churn_estimator,
    train_churn_model,
)
from dsm_project.config import load_config
from dsm_project.data import load_project_dataset
from dsm_project.evaluation import OutreachEconomics
from dsm_project.experiments import evaluate_supervised_estimator

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "configs/waze_fixture.toml"


def test_bundle_records_version_threshold_and_test_results(tmp_path) -> None:
    bundle_dir = train_churn_model(FIXTURE, tmp_path / "bundle")
    bundle = load_bundle(bundle_dir)
    metadata = bundle.metadata

    assert metadata["model_version"] == MODEL_VERSION
    assert metadata["decision_threshold"] == pytest.approx(0.25)
    assert metadata["threshold_policy"]["save_rate"] == DEFAULT_ECONOMICS.save_rate
    assert metadata["positive_class"] == "churned"
    assert len(metadata["feature_names"]) == 11
    for key in (
        "average_precision",
        "recall_at_threshold",
        "brier_score",
        "share_flagged",
    ):
        assert 0.0 <= metadata["test_metrics"][key] <= 1.0
    assert metadata["source_revision"]
    assert metadata["selection_record"].startswith("notebooks/08")


def test_saved_model_flags_exactly_at_the_threshold(tmp_path) -> None:
    bundle = load_bundle(train_churn_model(FIXTURE, tmp_path / "bundle"))
    dataset = load_project_dataset(load_config(FIXTURE))
    features = dataset.observations.loc[:, bundle.metadata["feature_names"]]
    probability = bundle.estimator.predict_proba(features)[:, 1]

    flagged = bundle.estimator.predict(features)
    np.testing.assert_array_equal(flagged, (probability >= 0.25).astype(int))


def test_custom_economics_change_the_threshold(tmp_path) -> None:
    generous = OutreachEconomics(
        cost_per_contact=1.0, value_per_saved_user=10.0, save_rate=0.5
    )
    bundle = load_bundle(train_churn_model(FIXTURE, tmp_path / "b", economics=generous))
    assert bundle.metadata["decision_threshold"] == pytest.approx(0.2)


def test_saving_is_reproducible(tmp_path) -> None:
    first = load_bundle(train_churn_model(FIXTURE, tmp_path / "one"))
    second = load_bundle(train_churn_model(FIXTURE, tmp_path / "two"))
    assert first.metadata["test_metrics"] == second.metadata["test_metrics"]


def test_extra_metadata_cannot_overwrite_core_fields(tmp_path) -> None:
    config = load_config(FIXTURE)
    dataset = load_project_dataset(config)
    result = evaluate_supervised_estimator(
        dataset, config, task_name="classification", estimator_name="x",
        estimator=build_churn_estimator(0.25),
    )  # fmt: skip
    with pytest.raises(ValueError, match="may not overwrite"):
        save_bundle(
            result, dataset, config, tmp_path, extra_metadata={"model_name": "y"}
        )
