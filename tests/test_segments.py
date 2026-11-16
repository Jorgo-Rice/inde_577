"""Tests for the named behavior segments."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dsm_project.config import load_config
from dsm_project.data import load_project_dataset
from dsm_project.data.waze import USAGE_COLUMNS
from dsm_project.segments import THREE_SEGMENT_NAMES, UserSegmenter

ROOT = Path(__file__).parents[1]


def _users(rng, n, *, activity, sessions):
    """Construct users around a given activity level and session volume."""

    frame = pd.DataFrame({column: rng.uniform(50, 60, n) for column in USAGE_COLUMNS})
    frame["activity_days"] = np.clip(rng.normal(activity, 1.0, n), 0, 31)
    frame["driving_days"] = np.clip(frame["activity_days"] - 2, 0, 31)
    frame["sessions"] = rng.normal(sessions, sessions * 0.05, n)
    frame["drives"] = frame["sessions"] * 0.8
    return frame


@pytest.fixture
def three_groups():
    rng = np.random.default_rng(0)
    groups = {
        "Occasional users": _users(rng, 60, activity=2, sessions=50),
        "Steady light users": _users(rng, 60, activity=20, sessions=10),
        "Steady heavy users": _users(rng, 60, activity=20, sessions=120),
    }
    frame = pd.concat(groups.values(), ignore_index=True)
    truth = np.repeat(list(groups), 60)
    return frame, truth


def test_three_segments_get_names_from_their_behavior(three_groups) -> None:
    frame, truth = three_groups
    segmenter = UserSegmenter(3, random_state=0).fit(frame)

    assert sorted(segmenter.segment_names_) == sorted(THREE_SEGMENT_NAMES)
    assert (segmenter.predict(frame).to_numpy() == truth).mean() > 0.98
    profile = segmenter.profile_
    assert profile.loc["Occasional users", "activity_days"] < 5
    assert (
        profile.loc["Steady light users", "sessions"]
        < profile.loc["Steady heavy users", "sessions"]
    )


def test_names_do_not_depend_on_cluster_numbering(three_groups) -> None:
    frame, _ = three_groups
    names = [
        UserSegmenter(3, random_state=seed).fit(frame).predict(frame)
        for seed in range(4)
    ]
    for other in names[1:]:
        pd.testing.assert_series_equal(names[0], other)


def test_other_segment_counts_use_ordered_generic_names(three_groups) -> None:
    frame, _ = three_groups
    segmenter = UserSegmenter(4, random_state=0).fit(frame)
    assert sorted(segmenter.segment_names_) == [f"Segment {i}" for i in range(1, 5)]
    first = segmenter.profile_.loc["Segment 1", "activity_days"]
    assert first == segmenter.profile_["activity_days"].min()


def test_predict_keeps_the_index_and_ignores_extra_columns(three_groups) -> None:
    frame, _ = three_groups
    segmenter = UserSegmenter(3, random_state=0).fit(frame)
    subset = frame.iloc[[5, 70, 150]].assign(is_iphone=1, churned=0)
    subset.index = ["a", "b", "c"]
    result = segmenter.predict(subset)
    assert list(result.index) == ["a", "b", "c"] and result.name == "segment"


def test_segmenter_runs_on_the_fixture_without_labels() -> None:
    dataset = load_project_dataset(load_config(ROOT / "configs/waze_fixture.toml"))
    usage = dataset.observations.drop(columns=["churned", "label"])
    segments = UserSegmenter(3, random_state=0).fit(usage).predict(usage)
    assert set(segments) <= set(THREE_SEGMENT_NAMES)
    assert len(segments) == len(usage)


def test_segmenter_rejects_bad_input(three_groups) -> None:
    frame, _ = three_groups
    with pytest.raises(ValueError):
        UserSegmenter(1)
    with pytest.raises(ValueError, match="missing usage columns"):
        UserSegmenter().fit(frame.drop(columns="drives"))
    with pytest.raises(ValueError, match="non-negative"):
        UserSegmenter().fit(frame.assign(sessions=-1.0))
    with pytest.raises(RuntimeError, match="fit must be called"):
        UserSegmenter().predict(frame)
