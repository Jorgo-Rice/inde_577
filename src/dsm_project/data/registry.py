"""Resolve a declared dataset adapter at one controlled boundary."""

from __future__ import annotations

from dsm_project.config import ProjectConfig
from dsm_project.contracts import DatasetAdapter, ProjectDataset
from dsm_project.data.csv_adapter import CSVProjectAdapter
from dsm_project.data.example import ExampleWineAdapter


def load_project_dataset(config: ProjectConfig) -> ProjectDataset:
    """Load the dataset selected by configuration.

    New adapters should be registered explicitly and tested. Dynamic imports
    from arbitrary configuration are intentionally avoided.
    """

    adapters: dict[str, DatasetAdapter] = {
        "example_wine": ExampleWineAdapter(),
        "csv": CSVProjectAdapter(),
    }
    adapter_name = str(config.dataset["adapter"])
    try:
        adapter = adapters[adapter_name]
    except KeyError as error:
        raise ValueError(f"Unknown dataset adapter: {adapter_name}") from error
    return adapter.load(config)
