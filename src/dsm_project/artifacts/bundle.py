"""Persist estimators together with the metadata needed to interpret them."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
from sklearn.base import BaseEstimator

from dsm_project.config import ProjectConfig
from dsm_project.contracts import ProjectDataset
from dsm_project.experiments.supervised import ExperimentResult


@dataclass(frozen=True)
class ModelBundle:
    """Loaded estimator plus the metadata defining its prediction contract."""

    estimator: BaseEstimator
    metadata: dict[str, Any]


def save_bundle(
    result: ExperimentResult,
    dataset: ProjectDataset,
    config: ProjectConfig,
    directory: str | Path,
) -> Path:
    """Write a model and human-readable metadata into one artifact directory."""

    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "model.joblib"
    metadata_path = output / "metadata.json"
    joblib.dump(result.estimator, model_path)
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "model_name": result.model_name,
        "task_name": result.task_name,
        "metric_name": result.metric_name,
        "metric_value": result.metric_value,
        "train_rows": result.train_rows,
        "test_rows": result.test_rows,
        "feature_names": list(result.feature_names),
        "dataset": asdict(dataset.provenance),
        "configuration": str(config.path),
        "python_version": platform.python_version(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return output


def load_bundle(directory: str | Path) -> ModelBundle:
    """Load and minimally validate one persisted model bundle."""

    source = Path(directory)
    model_path = source / "model.joblib"
    metadata_path = source / "metadata.json"
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Incomplete model bundle: {source}")
    metadata = json.loads(metadata_path.read_text())
    required = {"schema_version", "model_name", "task_name", "feature_names"}
    missing = required - set(metadata)
    if missing:
        raise ValueError(f"Model metadata is missing fields: {sorted(missing)}")
    estimator = joblib.load(model_path)
    return ModelBundle(estimator=estimator, metadata=metadata)
