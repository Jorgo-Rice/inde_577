"""Project-owned algorithms and explicit trusted-library adapters."""

from dsm_project.algorithms.baselines import MajorityClassifier, MeanRegressor
from dsm_project.algorithms.linear import BatchLinearRegressor
from dsm_project.algorithms.registry import (
    STUDENT_ALGORITHMS,
    create_student_algorithm,
    registered_student_algorithms,
)
from dsm_project.algorithms.student import (
    ClassificationTree,
    GradientBoostingBinaryClassifier,
    KNearestNeighborsClassifier,
    RandomForest,
    RegressionTree,
    SoftmaxRegression,
)

__all__ = [
    "STUDENT_ALGORITHMS",
    "BatchLinearRegressor",
    "ClassificationTree",
    "GradientBoostingBinaryClassifier",
    "KNearestNeighborsClassifier",
    "MajorityClassifier",
    "MeanRegressor",
    "RandomForest",
    "RegressionTree",
    "SoftmaxRegression",
    "create_student_algorithm",
    "registered_student_algorithms",
]
