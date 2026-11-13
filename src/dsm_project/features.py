"""Engineered behavioral features for the Waze churn data.

These features build on my earlier exploratory analysis of the Waze data,
which found that **consistency of use matters more than intensity** and that
some very heavy drivers churn. They turn the raw monthly and lifetime counts
into rates, shares, and flags that describe *how* someone uses Waze.

Design choices
--------------
- **Stateless.** Every feature is computed from the row itself, never from
  other users, so the transformer learns nothing during ``fit`` and cannot leak
  information across the train/test split.
- **Zero denominators give 0.** A user with no driving days has no
  "km per driving day"; 0 is the natural value and matches the earlier
  analysis. Ratios never produce ``inf`` or ``NaN``.
- **Excluded:** km per hour of driving. About a quarter of users imply speeds
  above 200 km/h, so the ratio mostly measures data errors (dataset card).
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from dsm_project.data.waze import FEATURE_COLUMNS

#: Name -> one-line meaning, in output order. The README of the feature set.
ENGINEERED_FEATURES: dict[str, str] = {
    "share_sessions_last_month": "sessions this month / lifetime sessions",
    "lifetime_sessions_per_day": "lifetime sessions / days since onboarding",
    "km_per_driving_day": "km driven this month / driving days",
    "km_per_drive": "km driven this month / drives",
    "drives_per_driving_day": "drives / driving days",
    "sessions_per_active_day": "sessions / active days",
    "total_favorite_navigations": "navigations to favorite places 1 + 2",
    "share_sessions_to_favorites": "favorite navigations / lifetime sessions",
    "professional_driver": "1 if 60+ drives and 15+ driving days this month",
    "heavy_driver": "1 if more than 200 drives this month",
    "low_driving_frequency": "1 if 5 or fewer driving days this month",
    "high_driving_frequency": "1 if 20 or more driving days this month",
    "high_recent_usage": "1 if 60%+ of lifetime sessions happened this month",
}


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide, returning 0 where the denominator is 0."""

    numerator = numerator.astype(float)
    denominator = denominator.astype(float)
    result = np.divide(
        numerator.to_numpy(),
        denominator.to_numpy(),
        out=np.zeros(len(numerator)),
        where=denominator.to_numpy() != 0,
    )
    return pd.Series(result, index=numerator.index)


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the engineered columns for each row (raw columns not included)."""

    missing = set(FEATURE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"missing raw Waze columns: {sorted(missing)}")
    f = frame
    share_last_month = _safe_ratio(f["sessions"], f["total_sessions"])
    favorites = f["total_navigations_fav1"] + f["total_navigations_fav2"]
    features = pd.DataFrame(
        {
            "share_sessions_last_month": share_last_month,
            "lifetime_sessions_per_day": _safe_ratio(
                f["total_sessions"], f["n_days_after_onboarding"]
            ),
            "km_per_driving_day": _safe_ratio(f["driven_km_drives"], f["driving_days"]),
            "km_per_drive": _safe_ratio(f["driven_km_drives"], f["drives"]),
            "drives_per_driving_day": _safe_ratio(f["drives"], f["driving_days"]),
            "sessions_per_active_day": _safe_ratio(f["sessions"], f["activity_days"]),
            "total_favorite_navigations": favorites.astype(float),
            "share_sessions_to_favorites": _safe_ratio(favorites, f["total_sessions"]),
            "professional_driver": ((f["drives"] >= 60) & (f["driving_days"] >= 15)),
            "heavy_driver": f["drives"] > 200,
            "low_driving_frequency": f["driving_days"] <= 5,
            "high_driving_frequency": f["driving_days"] >= 20,
            "high_recent_usage": share_last_month >= 0.6,
        },
        index=frame.index,
    )
    flags = [
        name
        for name, meaning in ENGINEERED_FEATURES.items()
        if meaning.startswith("1 if")
    ]
    features[flags] = features[flags].astype(int)
    return features[list(ENGINEERED_FEATURES)]


class WazeFeatureBuilder(TransformerMixin, BaseEstimator):
    """Append (or select) engineered behavioral features in a scikit-learn pipeline.

    Parameters
    ----------
    keep_raw
        Keep the 11 raw columns and append the engineered ones (default), or
        return only the engineered ones.
    include
        Optional subset of :data:`ENGINEERED_FEATURES` names to add.

    Notes
    -----
    Input must be a DataFrame with the raw ``FEATURE_COLUMNS``; the output is a
    DataFrame, so column names survive into later pipeline steps.
    """

    def __init__(
        self, *, keep_raw: bool = True, include: tuple[str, ...] | None = None
    ):
        if include is not None:
            unknown = set(include) - set(ENGINEERED_FEATURES)
            if unknown:
                raise ValueError(f"unknown engineered features: {sorted(unknown)}")
        self.keep_raw = keep_raw
        self.include = include

    def fit(self, features: pd.DataFrame, targets: object = None) -> Self:
        """Check the input columns; nothing is learned (the transform is stateless)."""

        if not isinstance(features, pd.DataFrame):
            raise TypeError("WazeFeatureBuilder expects a pandas DataFrame")
        engineer_features(features.head(1))  # validates the raw columns
        self.n_features_in_ = features.shape[1]
        self.feature_names_in_ = np.asarray(features.columns, dtype=object)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return raw columns (optionally) plus the selected engineered columns."""

        if not hasattr(self, "feature_names_in_"):
            raise RuntimeError("fit must be called before transform")
        if not isinstance(features, pd.DataFrame):
            raise TypeError("WazeFeatureBuilder expects a pandas DataFrame")
        engineered = engineer_features(features)
        if self.include is not None:
            engineered = engineered[list(self.include)]
        if not self.keep_raw:
            return engineered
        return pd.concat([features.loc[:, list(FEATURE_COLUMNS)], engineered], axis=1)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Names of the output columns, in order."""

        engineered = (
            list(self.include)
            if self.include is not None
            else list(ENGINEERED_FEATURES)
        )
        raw = list(FEATURE_COLUMNS) if self.keep_raw else []
        return np.asarray(raw + engineered, dtype=object)
