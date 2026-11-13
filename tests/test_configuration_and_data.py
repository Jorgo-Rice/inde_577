"""Contracts for configuration and bring-your-own-data boundaries."""

from pathlib import Path

import pandas as pd
import pytest

from dsm_project.config import load_config
from dsm_project.contracts import TaskKind
from dsm_project.data import load_project_dataset


def test_example_configuration_loads_validated_real_data() -> None:
    config = load_config("configs/example.toml")
    dataset = load_project_dataset(config)

    assert len(dataset.observations) == 178
    assert dataset.observations[dataset.identifier].is_unique
    assert dataset.tasks["regression"].kind is TaskKind.REGRESSION
    assert dataset.tasks["classification"].kind is TaskKind.CLASSIFICATION
    assert dataset.tasks["unsupervised"].target_column is None


def test_csv_adapter_rejects_a_missing_student_file(tmp_path: Path) -> None:
    config_file = tmp_path / "configs" / "student.toml"
    config_file.parent.mkdir()
    config_file.write_text(
        """
[project]
name = "test"
intended_user = "test"
[dataset]
adapter = "csv"
name = "test"
source = "test"
license = "test"
path = "data/local/missing.csv"
observation_unit = "row"
identifier = "id"
feature_columns = ["x"]
[tasks.unsupervised]
name = "structure"
metric = "stability"
[split]
random_state = 1
test_fraction = 0.2
""".strip()
    )
    config = load_config(config_file)

    with pytest.raises(FileNotFoundError, match="Student data not found"):
        load_project_dataset(config)


def test_dataset_validation_detects_duplicate_identifiers() -> None:
    config = load_config("configs/example.toml")
    dataset = load_project_dataset(config)
    observations = dataset.observations.copy()
    observations.loc[1, dataset.identifier] = observations.loc[0, dataset.identifier]
    duplicate_dataset = type(dataset)(
        observations=pd.DataFrame(observations),
        identifier=dataset.identifier,
        tasks=dataset.tasks,
        provenance=dataset.provenance,
    )

    with pytest.raises(ValueError, match="must be unique"):
        duplicate_dataset.validate()
