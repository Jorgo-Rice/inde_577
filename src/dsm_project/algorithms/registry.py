"""Explicit registry for student-owned algorithm implementations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from dsm_project.algorithms.linear import BatchLinearRegressor

EstimatorFactory = Callable[..., Any]

# The transparent linear regressor is the completed exemplar. Students add
# reviewed implementations here; trusted-library references do not belong in
# this mapping.
STUDENT_ALGORITHMS: dict[str, EstimatorFactory] = {
    "batch_linear_regression": BatchLinearRegressor,
}


def registered_student_algorithms() -> tuple[str, ...]:
    """Return stable public names for student-owned implementations."""

    return tuple(sorted(STUDENT_ALGORITHMS))


def create_student_algorithm(name: str, **parameters: Any) -> Any:
    """Construct a registered student estimator.

    Parameters
    ----------
    name
        Stable registry name recorded in configuration and model metadata.
    **parameters
        Explicit constructor parameters for the selected estimator.

    Raises
    ------
    ValueError
        If ``name`` has not been registered as student-owned code.
    """

    try:
        factory = STUDENT_ALGORITHMS[name]
    except KeyError as error:
        choices = ", ".join(registered_student_algorithms())
        message = f"Unknown student algorithm {name!r}; choose from: {choices}"
        raise ValueError(message) from error
    return factory(**parameters)
