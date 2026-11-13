"""Resampling-based uncertainty for metrics computed on a finite test set."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

Metric = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class MetricInterval:
    """Point estimate plus a percentile bootstrap interval."""

    estimate: float
    lower: float
    upper: float
    confidence: float
    resamples: int


def bootstrap_metric_interval(
    observed: ArrayLike,
    predicted: ArrayLike,
    metric: Metric,
    *,
    confidence: float = 0.95,
    resamples: int = 2_000,
    random_state: int = 438,
) -> MetricInterval:
    """Estimate a percentile bootstrap interval for ``metric(observed, predicted)``.

    Test rows are resampled with replacement and the metric is recomputed on
    each resample. The interval describes sampling variability of the *test
    set*; it does not capture variability from re-training the model.

    Parameters
    ----------
    observed, predicted
        Aligned one-dimensional arrays of true and predicted values.
    metric
        Function with signature ``metric(observed, predicted) -> float``.
    confidence
        Central coverage of the interval, strictly between zero and one.
    resamples
        Positive number of bootstrap resamples.
    random_state
        Seed for reproducible resampling.
    """

    truth = np.asarray(observed)
    guess = np.asarray(predicted)
    if truth.ndim != 1 or truth.shape != guess.shape or truth.size == 0:
        raise ValueError("observed and predicted must be aligned nonempty 1-D arrays")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if resamples <= 0:
        raise ValueError("resamples must be positive")

    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, truth.size, size=(resamples, truth.size))
    values = np.array([metric(truth[row], guess[row]) for row in indices])
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(values, [tail, 1.0 - tail])
    return MetricInterval(
        estimate=float(metric(truth, guess)),
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
        resamples=resamples,
    )
