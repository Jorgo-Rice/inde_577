"""Leakage-resistant supervised reference experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from dsm_project.config import ProjectConfig
from dsm_project.contracts import ProjectDataset, TaskKind


@dataclass(frozen=True)
class ExperimentResult:
    """Candidate model plus bounded held-out evidence.

    The holdout fields support diagnostics in executable reports without
    forcing notebooks to reproduce the package's split logic. They are not
    written into the promoted model artifact.
    """

    model_name: str
    task_name: str
    metric_name: str
    metric_value: float
    train_rows: int
    test_rows: int
    feature_names: tuple[str, ...]
    estimator: BaseEstimator
    holdout_features: pd.DataFrame
    observed: np.ndarray
    predictions: np.ndarray


def available_models(kind: TaskKind) -> tuple[str, ...]:
    """Return the stable reference catalog for a supervised task kind."""

    catalogs = {
        TaskKind.REGRESSION: (
            "linear_regression",
            "neighbors",
            "kernel_svm",
            "decision_tree",
            "random_forest",
            "gradient_boosting",
            "neural_network",
        ),
        TaskKind.CLASSIFICATION: (
            "logistic_regression",
            "linear_svm",
            "kernel_svm",
            "neighbors",
            "decision_tree",
            "random_forest",
            "gradient_boosting",
            "neural_network",
        ),
    }
    if kind not in catalogs:
        raise ValueError(f"No supervised catalog exists for {kind}")
    return catalogs[kind]


def _candidate(model_name: str, kind: TaskKind, seed: int) -> BaseEstimator:
    if kind is TaskKind.REGRESSION:
        models: dict[str, BaseEstimator] = {
            "linear_regression": LinearRegression(),
            "neighbors": KNeighborsRegressor(n_neighbors=7),
            "kernel_svm": SVR(C=5.0),
            "decision_tree": DecisionTreeRegressor(max_depth=4, random_state=seed),
            "random_forest": RandomForestRegressor(
                n_estimators=100, random_state=seed, n_jobs=1
            ),
            "gradient_boosting": GradientBoostingRegressor(random_state=seed),
            "neural_network": MLPRegressor(
                hidden_layer_sizes=(24, 12),
                max_iter=3_000,
                random_state=seed,
            ),
        }
    elif kind is TaskKind.CLASSIFICATION:
        models = {
            "logistic_regression": LogisticRegression(max_iter=3_000),
            "linear_svm": SVC(kernel="linear", C=1.0),
            "kernel_svm": SVC(kernel="rbf", C=2.0),
            "neighbors": KNeighborsClassifier(n_neighbors=7),
            "decision_tree": DecisionTreeClassifier(max_depth=4, random_state=seed),
            "random_forest": RandomForestClassifier(
                n_estimators=100, random_state=seed, n_jobs=1
            ),
            "gradient_boosting": GradientBoostingClassifier(random_state=seed),
            "neural_network": MLPClassifier(
                hidden_layer_sizes=(24, 12),
                max_iter=3_000,
                random_state=seed,
            ),
        }
    else:
        raise ValueError("A supervised candidate requires a supervised task")
    try:
        estimator = models[model_name]
    except KeyError as error:
        raise ValueError(f"Unknown {kind.value} model: {model_name}") from error
    return make_pipeline(StandardScaler(), estimator)


def evaluate_supervised_estimator(
    dataset: ProjectDataset,
    config: ProjectConfig,
    *,
    task_name: str,
    estimator_name: str,
    estimator: BaseEstimator,
) -> ExperimentResult:
    """Fit an estimator through the shared leakage-resistant example path.

    Student-owned estimators and trusted references use this same boundary so
    that data partitioning and metric calculation do not silently differ.
    """

    try:
        task = dataset.tasks[task_name]
    except KeyError as error:
        raise ValueError(f"Unknown task: {task_name}") from error
    if task.kind is TaskKind.UNSUPERVISED or task.target_column is None:
        raise ValueError("Use an unsupervised runner for this task")

    features = dataset.observations.loc[:, task.feature_columns]
    targets = dataset.observations.loc[:, task.target_column]
    seed = int(config.split.get("random_state", 438))
    stratify: Any = targets if task.kind is TaskKind.CLASSIFICATION else None
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        targets,
        test_size=float(config.split["test_fraction"]),
        random_state=seed,
        stratify=stratify,
    )
    estimator.fit(train_x, train_y)
    predictions = estimator.predict(test_x)
    if task.kind is TaskKind.REGRESSION:
        metric_name = "root_mean_squared_error"
        metric_value = float(root_mean_squared_error(test_y, predictions))
    else:
        metric_name = "balanced_accuracy"
        metric_value = float(balanced_accuracy_score(test_y, predictions))
    if not np.isfinite(metric_value):
        raise RuntimeError("Evaluation produced a non-finite metric")
    return ExperimentResult(
        model_name=estimator_name,
        task_name=task_name,
        metric_name=metric_name,
        metric_value=metric_value,
        train_rows=len(train_x),
        test_rows=len(test_x),
        feature_names=task.feature_columns,
        estimator=estimator,
        holdout_features=test_x.copy(),
        observed=np.asarray(test_y),
        predictions=np.asarray(predictions),
    )


def fit_supervised_candidate(
    dataset: ProjectDataset,
    config: ProjectConfig,
    *,
    task_name: str,
    model_name: str,
) -> ExperimentResult:
    """Fit one trusted reference candidate on the shared example split."""

    try:
        task = dataset.tasks[task_name]
    except KeyError as error:
        raise ValueError(f"Unknown task: {task_name}") from error
    seed = int(config.split.get("random_state", 438))
    estimator = _candidate(model_name, task.kind, seed)
    return evaluate_supervised_estimator(
        dataset,
        config,
        task_name=task_name,
        estimator_name=f"reference:{model_name}",
        estimator=estimator,
    )
