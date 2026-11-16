"""Project-owned algorithms and explicit trusted-library adapters."""

from dsm_project.algorithms.baselines import MajorityClassifier, MeanRegressor
from dsm_project.algorithms.linear import BatchLinearRegressor
from dsm_project.algorithms.registry import (
    STUDENT_ALGORITHMS,
    create_student_algorithm,
    registered_student_algorithms,
)
from dsm_project.algorithms.student import (
    PCA,
    ClassificationTree,
    GradientBoostingBinaryClassifier,
    KMeans,
    KNearestNeighborsClassifier,
    LinearSVM,
    NeuralNetworkClassifier,
    RandomForest,
    RegressionTree,
    SoftmaxRegression,
)

__all__ = [
    "STUDENT_ALGORITHMS",
    "BatchLinearRegressor",
    "ClassificationTree",
    "GradientBoostingBinaryClassifier",
    "KMeans",
    "KNearestNeighborsClassifier",
    "LinearSVM",
    "MajorityClassifier",
    "MeanRegressor",
    "NeuralNetworkClassifier",
    "PCA",
    "RandomForest",
    "RegressionTree",
    "SoftmaxRegression",
    "create_student_algorithm",
    "registered_student_algorithms",
]
