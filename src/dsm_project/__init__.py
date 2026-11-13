"""Reusable package for the final machine-learning product."""

from dsm_project.config import ProjectConfig, load_config
from dsm_project.contracts import DatasetAdapter, ProjectDataset, TaskDefinition

__all__ = [
    "DatasetAdapter",
    "ProjectConfig",
    "ProjectDataset",
    "TaskDefinition",
    "load_config",
]

__version__ = "0.1.0"
