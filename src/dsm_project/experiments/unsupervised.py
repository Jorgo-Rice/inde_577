"""Unsupervised examples with evidence appropriate to unlabeled fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dsm_project.config import ProjectConfig
from dsm_project.contracts import ProjectDataset, TaskKind


@dataclass(frozen=True)
class ClusteringResult:
    """Cluster assignments and internal plus optional audit evidence."""

    labels: NDArray[np.int64]
    silhouette: float
    stability_ari: float
    external_audit_ari: float | None
    cluster_centers_scaled: NDArray[np.float64]
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class RepresentationResult:
    """Low-dimensional coordinates and retained-variance evidence."""

    coordinates: NDArray[np.float64]
    explained_variance_ratio: NDArray[np.float64]
    components: NDArray[np.float64]
    feature_names: tuple[str, ...]


def run_clustering_example(
    dataset: ProjectDataset,
    config: ProjectConfig,
    *,
    clusters: int = 3,
) -> ClusteringResult:
    """Fit clusters without labels and audit stability under another seed."""

    task = dataset.tasks["unsupervised"]
    if task.kind is not TaskKind.UNSUPERVISED:
        raise ValueError("The declared task is not unsupervised")
    features = dataset.observations.loc[:, task.feature_columns]
    scaled = StandardScaler().fit_transform(features)
    seed = int(config.split.get("random_state", 438))
    first_model = KMeans(n_clusters=clusters, n_init=20, random_state=seed)
    first = first_model.fit_predict(scaled)
    second = KMeans(
        n_clusters=clusters, n_init=20, random_state=seed + 1
    ).fit_predict(scaled)
    external = None
    if "cultivar" in dataset.observations:
        external = float(adjusted_rand_score(dataset.observations["cultivar"], first))
    return ClusteringResult(
        labels=np.asarray(first, dtype=np.int64),
        silhouette=float(silhouette_score(scaled, first)),
        stability_ari=float(adjusted_rand_score(first, second)),
        external_audit_ari=external,
        cluster_centers_scaled=np.asarray(first_model.cluster_centers_, dtype=float),
        feature_names=task.feature_columns,
    )


def run_pca_example(
    dataset: ProjectDataset,
    *,
    components: int = 2,
) -> RepresentationResult:
    """Fit a standardized PCA representation without outcome labels."""

    task = dataset.tasks["unsupervised"]
    features = dataset.observations.loc[:, task.feature_columns]
    pipeline = make_pipeline(StandardScaler(), PCA(n_components=components))
    coordinates = pipeline.fit_transform(features)
    pca = pipeline.named_steps["pca"]
    return RepresentationResult(
        coordinates=np.asarray(coordinates, dtype=float),
        explained_variance_ratio=np.asarray(pca.explained_variance_ratio_),
        components=np.asarray(pca.components_, dtype=float),
        feature_names=task.feature_columns,
    )
