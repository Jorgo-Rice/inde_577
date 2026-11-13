"""Candidate comparison tables for exploratory example notebooks."""

from __future__ import annotations

import pandas as pd

from dsm_project.config import ProjectConfig
from dsm_project.contracts import ProjectDataset
from dsm_project.experiments.supervised import fit_supervised_candidate


def comparison_table(
    dataset: ProjectDataset,
    config: ProjectConfig,
    *,
    task_name: str,
    model_names: tuple[str, ...],
) -> pd.DataFrame:
    """Evaluate declared candidates on one shared instructional split.

    Notes
    -----
    This compact example is not a final model-selection procedure. Student
    projects must add validation or nested resampling, uncertainty, baselines,
    and an untouched final test boundary appropriate to their data.
    """

    rows = []
    for model_name in model_names:
        result = fit_supervised_candidate(
            dataset,
            config,
            task_name=task_name,
            model_name=model_name,
        )
        rows.append(
            {
                "model": result.model_name,
                "metric": result.metric_name,
                "value": result.metric_value,
                "train_rows": result.train_rows,
                "test_rows": result.test_rows,
            }
        )
    return pd.DataFrame(rows)
