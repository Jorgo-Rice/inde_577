"""Offline reference adapter for the untouched template."""

from __future__ import annotations

from sklearn.datasets import load_wine

from dsm_project.config import ProjectConfig
from dsm_project.contracts import (
    DatasetProvenance,
    ProjectDataset,
    TaskDefinition,
    TaskKind,
)


class ExampleWineAdapter:
    """Load the public wine-chemistry example packaged with scikit-learn."""

    def load(self, config: ProjectConfig) -> ProjectDataset:
        """Return validated observations and three distinct learning tasks."""

        bunch = load_wine(as_frame=True)
        observations = bunch.frame.rename(columns={"target": "cultivar"}).copy()
        observations.insert(0, "observation_id", range(1, len(observations) + 1))

        chemistry = tuple(bunch.feature_names)
        regression_features = tuple(
            feature for feature in chemistry if feature != "alcohol"
        )
        tasks = {
            "regression": TaskDefinition(
                name=str(config.tasks["regression"]["name"]),
                kind=TaskKind.REGRESSION,
                feature_columns=regression_features,
                target_column="alcohol",
                metric=str(config.tasks["regression"]["metric"]),
            ),
            "classification": TaskDefinition(
                name=str(config.tasks["classification"]["name"]),
                kind=TaskKind.CLASSIFICATION,
                feature_columns=chemistry,
                target_column="cultivar",
                metric=str(config.tasks["classification"]["metric"]),
            ),
            "unsupervised": TaskDefinition(
                name=str(config.tasks["unsupervised"]["name"]),
                kind=TaskKind.UNSUPERVISED,
                feature_columns=chemistry,
                target_column=None,
                metric=str(config.tasks["unsupervised"]["metric"]),
            ),
        }
        dataset = ProjectDataset(
            observations=observations,
            identifier="observation_id",
            tasks=tasks,
            provenance=DatasetProvenance(
                name=str(config.dataset["name"]),
                source=str(config.dataset["source"]),
                license=str(config.dataset["license"]),
                observation_unit=str(config.dataset["observation_unit"]),
            ),
        )
        dataset.validate()
        return dataset
