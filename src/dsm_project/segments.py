"""Behavior segments for Waze users, found without the churn label.

The segmenter log-transforms the ten usage measurements (they have long right
tails), standardizes them, and clusters users with the student
:class:`~dsm_project.algorithms.KMeans`. Churn labels are never used to form
the segments; they are only used afterwards to *audit* how churn differs.

Segment names are derived from the fitted centers, so they do not depend on
the arbitrary cluster numbering:

- With three segments, the cluster with the fewest active days is
  ``"Occasional users"``; of the other two, the one with fewer sessions is
  ``"Steady light users"`` and the other ``"Steady heavy users"``.
- With any other number of segments, names are ``"Segment 1"`` ... in order of
  median active days, then median sessions.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from dsm_project.algorithms.student.unsupervised import KMeans
from dsm_project.data.waze import USAGE_COLUMNS

THREE_SEGMENT_NAMES = ("Occasional users", "Steady light users", "Steady heavy users")


class UserSegmenter(BaseEstimator):
    """Fit behavior segments on usage data and assign named segments to users.

    Parameters
    ----------
    n_segments
        Number of k-means clusters.
    random_state
        Seed for the k-means++ restarts.

    Attributes
    ----------
    scaler_ : StandardScaler
        Fitted on ``log1p`` of the usage columns.
    kmeans_ : KMeans
        Student k-means fitted on the standardized values.
    segment_names_ : list of str
        Name of each k-means cluster, indexed by cluster number.
    profile_ : DataFrame
        Median usage per named segment (original units), for explanation.
    """

    def __init__(self, n_segments: int = 3, random_state: int | None = 438) -> None:
        if not isinstance(n_segments, int) or n_segments < 2:
            raise ValueError("n_segments must be an integer of at least 2")
        self.n_segments = n_segments
        self.random_state = random_state

    @staticmethod
    def _usage(frame: pd.DataFrame) -> pd.DataFrame:
        missing = set(USAGE_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"missing usage columns: {sorted(missing)}")
        usage = frame.loc[:, list(USAGE_COLUMNS)].astype(float)
        if (usage < 0).to_numpy().any():
            raise ValueError("usage values must be non-negative")
        return usage

    def fit(self, frame: pd.DataFrame, targets: object = None) -> Self:
        """Learn the scaling and the segment centers from usage data only."""

        usage = self._usage(frame)
        logged = np.log1p(usage)
        self.scaler_ = StandardScaler().fit(logged)
        self.kmeans_ = KMeans(self.n_segments, random_state=self.random_state).fit(
            self.scaler_.transform(logged)
        )

        medians = usage.groupby(self.kmeans_.labels_).median()
        self.segment_names_ = self._name_segments(medians)
        self.profile_ = medians.rename(index=dict(enumerate(self.segment_names_)))
        self.profile_.index.name = "segment"
        return self

    def _name_segments(self, medians: pd.DataFrame) -> list[str]:
        names = [""] * self.n_segments
        if self.n_segments == 3:
            occasional = int(medians["activity_days"].idxmin())
            others = medians.drop(index=occasional).sort_values("sessions").index
            for cluster, name in zip(
                [occasional, *others], THREE_SEGMENT_NAMES, strict=True
            ):
                names[int(cluster)] = name
        else:
            order = medians.sort_values(["activity_days", "sessions"]).index
            for rank, cluster in enumerate(order, start=1):
                names[int(cluster)] = f"Segment {rank}"
        return names

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        """Return the segment name of each user."""

        if not hasattr(self, "kmeans_"):
            raise RuntimeError("fit must be called before predict")
        logged = np.log1p(self._usage(frame))
        clusters = self.kmeans_.predict(self.scaler_.transform(logged))
        return pd.Series(
            np.asarray(self.segment_names_, dtype=object)[clusters],
            index=frame.index,
            name="segment",
        )
