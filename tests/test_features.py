"""Tests for the engineered Waze features."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dsm_project.algorithms import create_student_algorithm
from dsm_project.config import load_config
from dsm_project.data import load_project_dataset
from dsm_project.data.waze import FEATURE_COLUMNS
from dsm_project.features import (
    ENGINEERED_FEATURES,
    WazeFeatureBuilder,
    engineer_features,
)

ROOT = Path(__file__).parents[1]


def _user(**overrides) -> dict:
    row = {
        "sessions": 40, "drives": 30, "total_sessions": 200.0,
        "n_days_after_onboarding": 100, "total_navigations_fav1": 10,
        "total_navigations_fav2": 30, "driven_km_drives": 600.0,
        "duration_minutes_drives": 300.0, "activity_days": 20, "driving_days": 10,
        "is_iphone": 1,
    }  # fmt: skip
    row.update(overrides)
    return row


def test_hand_computed_user() -> None:
    features = engineer_features(pd.DataFrame([_user()])).iloc[0]

    assert features["share_sessions_last_month"] == pytest.approx(40 / 200)
    assert features["lifetime_sessions_per_day"] == pytest.approx(200 / 100)
    assert features["km_per_driving_day"] == pytest.approx(600 / 10)
    assert features["km_per_drive"] == pytest.approx(600 / 30)
    assert features["drives_per_driving_day"] == pytest.approx(30 / 10)
    assert features["sessions_per_active_day"] == pytest.approx(40 / 20)
    assert features["total_favorite_navigations"] == 40
    assert features["share_sessions_to_favorites"] == pytest.approx(40 / 200)
    assert features["professional_driver"] == 0  # 30 drives < 60
    assert features["heavy_driver"] == 0
    assert features["low_driving_frequency"] == 0
    assert features["high_driving_frequency"] == 0
    assert features["high_recent_usage"] == 0


def test_flag_thresholds_are_inclusive_where_documented() -> None:
    rows = pd.DataFrame(
        [
            _user(drives=60, driving_days=15),  # professional: 60+ and 15+
            _user(drives=201, sessions=250),  # heavy: more than 200
            _user(drives=200, sessions=250),  # 200 is not heavy
            _user(driving_days=5),  # low: 5 or fewer
            _user(driving_days=20),  # high: 20 or more
            _user(sessions=120, total_sessions=200.0),  # recent: 60%
        ]
    )
    features = engineer_features(rows)

    assert features["professional_driver"].tolist()[0] == 1
    assert features["heavy_driver"].tolist()[1:3] == [1, 0]
    assert features["low_driving_frequency"].tolist()[3] == 1
    assert features["high_driving_frequency"].tolist()[4] == 1
    assert features["high_recent_usage"].tolist()[5] == 1


def test_zero_denominators_give_zero_not_inf() -> None:
    idle = _user(sessions=0, drives=0, driving_days=0, activity_days=0,
                 driven_km_drives=0.0, total_sessions=0.0)  # fmt: skip
    features = engineer_features(pd.DataFrame([idle]))

    assert np.isfinite(features.to_numpy(dtype=float)).all()
    for column in ("km_per_driving_day", "km_per_drive", "drives_per_driving_day",
                   "sessions_per_active_day", "share_sessions_last_month"):  # fmt: skip
        assert features.loc[0, column] == 0.0


def test_output_columns_types_and_index() -> None:
    frame = pd.DataFrame([_user(), _user(drives=250, sessions=300)], index=[7, 9])
    features = engineer_features(frame)

    assert list(features.columns) == list(ENGINEERED_FEATURES)
    assert list(features.index) == [7, 9]
    flags = [c for c, m in ENGINEERED_FEATURES.items() if m.startswith("1 if")]
    assert set(np.unique(features[flags])) <= {0, 1}


def test_missing_raw_columns_are_rejected() -> None:
    with pytest.raises(ValueError, match="missing raw Waze columns"):
        engineer_features(pd.DataFrame([_user()]).drop(columns="drives"))


def test_builder_is_stateless_and_does_not_mutate_input() -> None:
    frame = pd.DataFrame([_user(), _user(drives=0, driving_days=0)])
    before = frame.copy()
    first = WazeFeatureBuilder().fit(frame).transform(frame)
    second = WazeFeatureBuilder().fit(frame.iloc[[1]]).transform(frame)

    pd.testing.assert_frame_equal(frame, before)
    pd.testing.assert_frame_equal(first, second)  # fitting data does not matter


def test_builder_output_names_and_options() -> None:
    frame = pd.DataFrame([_user()])
    full = WazeFeatureBuilder().fit(frame)
    only = WazeFeatureBuilder(keep_raw=False, include=("heavy_driver",)).fit(frame)

    assert list(full.transform(frame).columns) == list(full.get_feature_names_out())
    assert list(full.get_feature_names_out()) == [
        *FEATURE_COLUMNS,
        *ENGINEERED_FEATURES,
    ]
    assert list(only.transform(frame).columns) == ["heavy_driver"]
    with pytest.raises(ValueError, match="unknown engineered"):
        WazeFeatureBuilder(include=("km_per_hour",))
    with pytest.raises(TypeError, match="DataFrame"):
        WazeFeatureBuilder().fit(frame.to_numpy())
    with pytest.raises(RuntimeError, match="fit must be called"):
        WazeFeatureBuilder().transform(frame)


def test_builder_runs_inside_a_model_pipeline_on_the_fixture() -> None:
    config = load_config(ROOT / "configs/waze_fixture.toml")
    dataset = load_project_dataset(config)
    features = dataset.observations.loc[:, list(FEATURE_COLUMNS)]
    model = make_pipeline(
        WazeFeatureBuilder(),
        StandardScaler(),
        create_student_algorithm("softmax_regression", l2_penalty=0.1),
    ).fit(features, dataset.observations["churned"])

    probabilities = model.predict_proba(features)
    assert probabilities.shape == (len(features), 2)
    assert model[-1].n_features_in_ == len(FEATURE_COLUMNS) + len(ENGINEERED_FEATURES)
