"""Student-owned algorithm implementations.

Add course algorithms here, then expose them through
``dsm_project.algorithms.registry.STUDENT_ALGORITHMS``.
"""

from dsm_project.algorithms.student.ensembles import (
    GradientBoostingBinaryClassifier,
    RandomForest,
)
from dsm_project.algorithms.student.logistic import SoftmaxRegression
from dsm_project.algorithms.student.neighbors import KNearestNeighborsClassifier
from dsm_project.algorithms.student.trees import ClassificationTree, RegressionTree

__all__ = [
    "ClassificationTree",
    "GradientBoostingBinaryClassifier",
    "KNearestNeighborsClassifier",
    "RandomForest",
    "RegressionTree",
    "SoftmaxRegression",
]
