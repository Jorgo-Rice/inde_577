"""Configurable boundary for student-supplied tabular data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dsm_project.config import ProjectConfig
from dsm_project.contracts import (
    DatasetProvenance,
    ProjectDataset,
    TaskDefinition,
    TaskKind,
)


class CSVProjectAdapter:
    """Load a local CSV using explicit feature and task declarations."""

    def load(self, config: ProjectConfig) -> ProjectDataset:
        """Load the configured file without guessing scientific semantics."""

        relative_path = Path(str(config.dataset.get("path", "")))
        if not relative_path.as_posix():
            raise ValueError("dataset.path is required for the CSV adapter")
        data_path = config.root / relative_path
        if not data_path.is_file():
            raise FileNotFoundError(
                f"Student data not found: {data_path}. See START_HERE.md."
            )
        observations = pd.read_csv(data_path)
        feature_columns = tuple(config.dataset.get("feature_columns", ()))
        if not feature_columns:
            raise ValueError("dataset.feature_columns must be declared explicitly")

        tasks: dict[str, TaskDefinition] = {}
        for name, kind in (
            ("regression", TaskKind.REGRESSION),
            ("classification", TaskKind.CLASSIFICATION),
            ("unsupervised", TaskKind.UNSUPERVISED),
        ):
            if name not in config.tasks:
                continue
            declaration = config.tasks[name]
            target = (
                None
                if kind is TaskKind.UNSUPERVISED
                else declaration.get("target")
            )
            tasks[name] = TaskDefinition(
                name=str(declaration["name"]),
                kind=kind,
                feature_columns=feature_columns,
                target_column=str(target) if target else None,
                metric=str(declaration["metric"]),
            )

        dataset = ProjectDataset(
            observations=observations,
            identifier=str(config.dataset["identifier"]),
            tasks=tasks,
            provenance=DatasetProvenance(
                name=str(config.dataset["name"]),
                source=str(config.dataset.get("source", "undocumented")),
                license=str(config.dataset.get("license", "undocumented")),
                observation_unit=str(config.dataset["observation_unit"]),
            ),
        )
        dataset.validate()
        return dataset
