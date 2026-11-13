"""Student-owned algorithm implementations.

Add course algorithms here, then expose them through
``dsm_project.algorithms.registry.STUDENT_ALGORITHMS``.
"""

from dsm_project.algorithms.student.logistic import SoftmaxRegression
from dsm_project.algorithms.student.neighbors import KNearestNeighborsClassifier

__all__ = ["KNearestNeighborsClassifier", "SoftmaxRegression"]
