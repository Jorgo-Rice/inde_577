"""Tests for the Waze data boundary, using the committed constructed fixture.

The real CSV is not in the repository, so CI exercises the same adapter and
validation code on data/fixtures/waze_fixture.csv. One test additionally
checks the real file when it is present locally.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dsm_project.config import load_config
from dsm_project.contracts import TaskKind
from dsm_project.data import load_project_dataset
from dsm_project.data.waze import (
    FEATURE_COLUMNS,
    WazeChurnAdapter,
    prepare_waze_observations,
    validate_waze_frame,
)
from dsm_project.experiments import split_supervised_task

ROOT = Path(__file__).parents[1]
FIXTURE_CONFIG = ROOT / "configs/waze_fixture.toml"
REAL_CONFIG = ROOT / "configs/waze.toml"


@pytest.fixture
def raw() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data/fixtures/waze_fixture.csv")


def test_fixture_loads_through_the_registered_adapter() -> None:
    config = load_config(FIXTURE_CONFIG)
    dataset = load_project_dataset(config)
    classification = dataset.tasks["classification"]

    assert classification.kind is TaskKind.CLASSIFICATION
    assert classification.target_column == "churned"
    assert classification.feature_columns == FEATURE_COLUMNS
    assert "label" not in classification.feature_columns
    assert dataset.tasks["unsupervised"].target_column is None
    assert "regression" not in dataset.tasks


def test_preparation_drops_unlabeled_rows_and_encodes_columns(raw) -> None:
    observations, report = prepare_waze_observations(raw)

    assert report.raw_rows == len(raw)
    assert report.unlabeled_rows == raw["label"].isna().sum() == 6
    assert report.modeled_rows == len(observations) == len(raw) - 6
    assert observations["label"].notna().all()
    np.testing.assert_array_equal(
        observations["churned"], (observations["label"] == "churned").astype(int)
    )
    np.testing.assert_array_equal(
        observations["is_iphone"], (observations["device"] == "iPhone").astype(int)
    )
    assert report.churn_rate == pytest.approx(observations["churned"].mean())
    assert report.total_below_month_sessions == 1  # constructed on purpose


def test_preparation_does_not_mutate_the_raw_frame(raw) -> None:
    before = raw.copy()
    prepare_waze_observations(raw)
    pd.testing.assert_frame_equal(raw, before)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda f: f.drop(columns="drives"), "missing columns"),
        (lambda f: f.assign(ID=0), "unique"),
        (lambda f: f.assign(sessions=f["sessions"] * -1 - 1), "non-negative"),
        (lambda f: f.assign(sessions=np.nan), "finite"),
        (lambda f: f.assign(label="left"), "Unknown churn labels"),
        (lambda f: f.assign(device="Windows"), "Unknown devices"),
        (lambda f: f.assign(activity_days=40, driving_days=0), "days in one month"),
        (lambda f: f.assign(driving_days=f["activity_days"] + 1), "driving_days"),
        (lambda f: f.assign(drives=f["sessions"] + 1), "drives cannot exceed"),
    ],
)
def test_validation_rejects_structural_defects(raw, mutate, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_waze_frame(mutate(raw.copy()))


def test_missing_file_and_checksum_mismatch_are_reported(tmp_path) -> None:
    config = load_config(FIXTURE_CONFIG)
    missing = replace(config, dataset={**config.dataset, "path": "nope.csv"})
    with pytest.raises(FileNotFoundError, match="Waze data not found"):
        WazeChurnAdapter().load(missing)

    wrong = replace(config, dataset={**config.dataset, "sha256": "0" * 64})
    with pytest.raises(ValueError, match="sha256"):
        WazeChurnAdapter().load(wrong)


def test_split_is_stratified_on_churn() -> None:
    config = load_config(FIXTURE_CONFIG)
    dataset = load_project_dataset(config)
    split = split_supervised_task(dataset, config, task_name="classification")

    assert abs(split.test_targets.mean() - split.train_targets.mean()) < 0.06
    assert set(split.train_features.columns) == set(FEATURE_COLUMNS)


@pytest.mark.skipif(
    not (ROOT / "data/local/waze_dataset.csv").is_file(),
    reason="real Waze CSV is not present (it is not committed)",
)
def test_real_dataset_matches_the_dataset_card() -> None:
    dataset, report = WazeChurnAdapter().load_with_report(load_config(REAL_CONFIG))

    assert report.raw_rows == 14_999
    assert report.unlabeled_rows == 700
    assert report.modeled_rows == len(dataset.observations) == 14_299
    assert report.churn_rate == pytest.approx(0.1774, abs=1e-4)
