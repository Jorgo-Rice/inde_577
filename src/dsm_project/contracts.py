"""Scientific and software contracts shared by project components."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

import pandas as pd

from dsm_project.config import ProjectConfig


class TaskKind(StrEnum):
    """Supported forms of statistical learning evidence."""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    UNSUPERVISED = "unsupervised"


@dataclass(frozen=True)
class TaskDefinition:
    """Mathematical meaning and columns for one project task."""

    name: str
    kind: TaskKind
    feature_columns: tuple[str, ...]
    target_column: str | None
    metric: str


@dataclass(frozen=True)
class DatasetProvenance:
    """Minimum identity and scope attached to a dataset adapter."""

    name: str
    source: str
    license: str
    observation_unit: str


@dataclass(frozen=True)
class ProjectDataset:
    """Validated observations, task declarations, and provenance."""

    observations: pd.DataFrame
    identifier: str
    tasks: dict[str, TaskDefinition]
    provenance: DatasetProvenance

    def validate(self) -> None:
        """Reject structural defects before an experiment sees the data."""

        if self.observations.empty:
            raise ValueError("The dataset contains no observations")
        if self.identifier not in self.observations:
            raise ValueError(f"Missing identifier column: {self.identifier}")
        if self.observations[self.identifier].isna().any():
            raise ValueError("Observation identifiers may not be missing")
        if not self.observations[self.identifier].is_unique:
            raise ValueError("Observation identifiers must be unique")
        for task in self.tasks.values():
            missing = set(task.feature_columns) - set(self.observations.columns)
            if missing:
                raise ValueError(f"Task {task.name!r} has missing features: {missing}")
            if task.target_column and task.target_column not in self.observations:
                raise ValueError(
                    f"Task {task.name!r} has missing target {task.target_column!r}"
                )
            if task.target_column in task.feature_columns:
                raise ValueError(f"Task {task.name!r} includes its target as a feature")


@runtime_checkable
class DatasetAdapter(Protocol):
    """Boundary every example or student dataset must implement."""

    def load(self, config: ProjectConfig) -> ProjectDataset:
        """Load, declare, and validate a project dataset."""

        ...
