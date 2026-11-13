"""Validated boundary for the Waze user-churn dataset.

One row describes one Waze app user. Month-level counts (``sessions``,
``drives``, ``driven_km_drives``, ``duration_minutes_drives``,
``activity_days``, ``driving_days``) summarize the same month in which the
churn ``label`` was recorded; ``total_sessions``, ``n_days_after_onboarding``,
and the favorite-place navigation counts cover the user's whole tenure.

Because activity and churn are measured over the same month, a model trained
on these columns partly *observes* disengagement as it happens. It can rank
current users by risk, but it is not evidence of how early churn can be
anticipated. See ``docs/DATASET_CARD.md``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dsm_project.config import ProjectConfig
from dsm_project.contracts import (
    DatasetProvenance,
    ProjectDataset,
    TaskDefinition,
    TaskKind,
)

IDENTIFIER = "ID"
LABEL = "label"
TARGET = "churned"
LABEL_VALUES = ("churned", "retained")
DEVICE_VALUES = ("Android", "iPhone")

#: Non-negative numeric usage measurements present in the raw file.
USAGE_COLUMNS: tuple[str, ...] = (
    "sessions",
    "drives",
    "total_sessions",
    "n_days_after_onboarding",
    "total_navigations_fav1",
    "total_navigations_fav2",
    "driven_km_drives",
    "duration_minutes_drives",
    "activity_days",
    "driving_days",
)
RAW_COLUMNS: tuple[str, ...] = (IDENTIFIER, LABEL, *USAGE_COLUMNS, "device")

#: Model inputs after preparation: usage columns plus a device indicator.
FEATURE_COLUMNS: tuple[str, ...] = (*USAGE_COLUMNS, "is_iphone")

#: Driving speeds above this (km per hour of driving) are physically implausible.
IMPLAUSIBLE_SPEED_KMH = 200.0


@dataclass(frozen=True)
class WazeDataReport:
    """Counts of rows removed or flagged at the data boundary."""

    raw_rows: int
    unlabeled_rows: int
    modeled_rows: int
    churn_rate: float
    total_below_month_sessions: int
    implausible_speed_rows: int


def validate_waze_frame(frame: pd.DataFrame) -> None:
    """Reject structural defects in the raw Waze table.

    Raises
    ------
    ValueError
        If columns are missing, identifiers repeat, values are negative or
        non-finite, categories are unknown, or day counts are inconsistent.
    """

    missing = set(RAW_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Waze data is missing columns: {sorted(missing)}")
    if frame[IDENTIFIER].isna().any() or not frame[IDENTIFIER].is_unique:
        raise ValueError("Waze user IDs must be present and unique")

    usage = frame.loc[:, USAGE_COLUMNS]
    if not all(pd.api.types.is_numeric_dtype(usage[c]) for c in USAGE_COLUMNS):
        raise ValueError("Waze usage columns must be numeric")
    values = usage.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Waze usage columns must be finite and not missing")
    if (values < 0).any():
        raise ValueError("Waze usage columns must be non-negative")

    labels = set(frame[LABEL].dropna().unique())
    if not labels <= set(LABEL_VALUES):
        raise ValueError(f"Unknown churn labels: {sorted(labels - set(LABEL_VALUES))}")
    unknown_devices = set(frame["device"].unique()) - set(DEVICE_VALUES)
    if unknown_devices:
        raise ValueError(f"Unknown devices: {sorted(map(str, unknown_devices))}")

    for column in ("activity_days", "driving_days"):
        if (frame[column] > 31).any():
            raise ValueError(f"{column} cannot exceed the days in one month")
    if (frame["driving_days"] > frame["activity_days"]).any():
        raise ValueError("driving_days cannot exceed activity_days")
    if (frame["drives"] > frame["sessions"]).any():
        raise ValueError("drives cannot exceed sessions")


def prepare_waze_observations(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, WazeDataReport]:
    """Validate, drop unlabeled users, and encode the target and device.

    Returns
    -------
    observations
        Labeled rows with ``churned`` (1 = churned) and ``is_iphone`` columns.
    report
        What was removed or flagged, for the dataset card and notebooks.
    """

    validate_waze_frame(frame)
    labeled = frame.loc[frame[LABEL].notna()].copy()
    labeled[TARGET] = (labeled[LABEL] == "churned").astype(int)
    labeled["is_iphone"] = (labeled["device"] == "iPhone").astype(int)

    hours = labeled["duration_minutes_drives"] / 60.0
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = labeled["driven_km_drives"] / hours
    report = WazeDataReport(
        raw_rows=len(frame),
        unlabeled_rows=int(frame[LABEL].isna().sum()),
        modeled_rows=len(labeled),
        churn_rate=float(labeled[TARGET].mean()) if len(labeled) else float("nan"),
        total_below_month_sessions=int(
            (labeled["total_sessions"] < labeled["sessions"]).sum()
        ),
        implausible_speed_rows=int((speed > IMPLAUSIBLE_SPEED_KMH).sum()),
    )
    return labeled.reset_index(drop=True), report


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class WazeChurnAdapter:
    """Load the configured Waze CSV into a validated project dataset."""

    def load(self, config: ProjectConfig) -> ProjectDataset:
        """Read, verify, validate, and declare the Waze tasks."""

        dataset, _ = self.load_with_report(config)
        return dataset

    def load_with_report(
        self, config: ProjectConfig
    ) -> tuple[ProjectDataset, WazeDataReport]:
        """Like :meth:`load`, also returning the boundary report."""

        path = config.root / str(config.dataset.get("path", ""))
        if not path.is_file():
            raise FileNotFoundError(
                f"Waze data not found at {path}. Place the CSV there; "
                "see docs/DATASET_CARD.md for how to obtain it."
            )
        expected = str(config.dataset.get("sha256", "")).strip()
        if expected and file_sha256(path) != expected:
            raise ValueError(
                f"{path.name} does not match the recorded sha256; the data "
                "version changed. Update the dataset card before re-running."
            )

        observations, report = prepare_waze_observations(pd.read_csv(path))
        tasks: dict[str, TaskDefinition] = {}
        if "classification" in config.tasks:
            tasks["classification"] = TaskDefinition(
                name=str(config.tasks["classification"]["name"]),
                kind=TaskKind.CLASSIFICATION,
                feature_columns=FEATURE_COLUMNS,
                target_column=TARGET,
                metric=str(config.tasks["classification"]["metric"]),
            )
        if "unsupervised" in config.tasks:
            tasks["unsupervised"] = TaskDefinition(
                name=str(config.tasks["unsupervised"]["name"]),
                kind=TaskKind.UNSUPERVISED,
                feature_columns=FEATURE_COLUMNS,
                target_column=None,
                metric=str(config.tasks["unsupervised"]["metric"]),
            )
        dataset = ProjectDataset(
            observations=observations,
            identifier=IDENTIFIER,
            tasks=tasks,
            provenance=DatasetProvenance(
                name=str(config.dataset["name"]),
                source=str(config.dataset.get("source", "undocumented")),
                license=str(config.dataset.get("license", "undocumented")),
                observation_unit=str(config.dataset["observation_unit"]),
            ),
        )
        dataset.validate()
        return dataset, report
