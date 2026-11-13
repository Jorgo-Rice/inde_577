"""Typed access to project and dataset configuration."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectConfig:
    """Validated configuration used by data and experiment boundaries."""

    path: Path
    project: dict[str, Any]
    dataset: dict[str, Any]
    tasks: dict[str, dict[str, Any]]
    split: dict[str, Any]

    @property
    def root(self) -> Path:
        """Return the repository root inferred from ``configs/<file>``."""

        return self.path.resolve().parent.parent


def load_config(path: str | Path = "configs/example.toml") -> ProjectConfig:
    """Load the TOML project configuration and validate required sections.

    Parameters
    ----------
    path
        Path to a project configuration file.

    Returns
    -------
    ProjectConfig
        Immutable access to the declared configuration.

    Raises
    ------
    FileNotFoundError
        If the configuration does not exist.
    ValueError
        If required sections or fields are absent.
    """

    config_path = Path(path)
    with config_path.open("rb") as stream:
        document = tomllib.load(stream)

    for section in ("project", "dataset", "tasks", "split"):
        if section not in document:
            raise ValueError(f"Configuration is missing [{section}]")
    for field in ("adapter", "name", "observation_unit", "identifier"):
        if not str(document["dataset"].get(field, "")).strip():
            raise ValueError(f"Configuration is missing dataset.{field}")

    fraction = float(document["split"].get("test_fraction", 0.0))
    if not 0.0 < fraction < 1.0:
        raise ValueError("split.test_fraction must lie strictly between zero and one")

    return ProjectConfig(
        path=config_path,
        project=dict(document["project"]),
        dataset=dict(document["dataset"]),
        tasks={name: dict(value) for name, value in document["tasks"].items()},
        split=dict(document["split"]),
    )
