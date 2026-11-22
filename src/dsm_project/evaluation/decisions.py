"""Turn churn probabilities into outreach decisions a product manager can defend.

Cost-based threshold
--------------------
Contacting a user costs ``c`` (message, incentive, attention). If the user
would have churned, the contact saves them with probability ``s`` (the *save
rate*), which is worth ``V`` (the value of keeping a user). For a user whose
churn probability is ``p``, the expected net value of contacting them is

.. math::

    \\text{EV}(p) = p \\cdot s \\cdot V - c,

which is positive exactly when :math:`p \\ge p^* = c / (s V)`, the
**break-even probability**. Flagging users above :math:`p^*` maximizes expected
value *if the probabilities are calibrated* (users scored 0.3 churn about 30%
of the time). That is why calibration is checked before trusting a cost-based
threshold. The inputs are business assumptions that a PM owns; the functions
here make their consequences explicit.

Calibration
-----------
A calibration (reliability) table groups users by predicted probability and
compares the mean prediction in each group with the observed churn rate. The
Brier score, the mean of :math:`(p_i - y_i)^2`, summarizes calibration and
sharpness in one number (lower is better; predicting the base rate for
everyone scores :math:`\\bar y (1 - \\bar y)`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class OutreachEconomics:
    """Business assumptions behind a cost-based outreach threshold.

    Parameters
    ----------
    cost_per_contact
        Cost of contacting one user (any consistent currency unit).
    value_per_saved_user
        Value of keeping one user who would otherwise have churned.
    save_rate
        Share of contacted would-be churners whom the outreach keeps, in (0, 1].
    """

    cost_per_contact: float
    value_per_saved_user: float
    save_rate: float

    def __post_init__(self) -> None:
        if not self.cost_per_contact > 0:
            raise ValueError("cost_per_contact must be positive")
        if not self.value_per_saved_user > 0:
            raise ValueError("value_per_saved_user must be positive")
        if not 0 < self.save_rate <= 1:
            raise ValueError("save_rate must lie in (0, 1]")

    @property
    def break_even_probability(self) -> float:
        """Churn probability above which contacting a user has positive value."""

        return self.cost_per_contact / (self.save_rate * self.value_per_saved_user)


def expected_outreach_value(
    observed: ArrayLike,
    scores: ArrayLike,
    threshold: float,
    economics: OutreachEconomics,
) -> float:
    """Net value per user of contacting everyone scored at or above ``threshold``.

    Uses the *observed* outcomes: each flagged churner contributes
    ``save_rate * value_per_saved_user`` and every flagged user costs
    ``cost_per_contact``. The total is divided by the number of users, so
    results are comparable across groups of different size.
    """

    truth = np.asarray(observed).astype(bool)
    values = np.asarray(scores, dtype=float)
    if truth.shape != values.shape or truth.ndim != 1 or truth.size == 0:
        raise ValueError("observed and scores must be aligned nonempty 1-D arrays")
    flagged = values >= threshold
    gain = (
        economics.save_rate * economics.value_per_saved_user * np.sum(flagged & truth)
    )
    cost = economics.cost_per_contact * np.sum(flagged)
    return float((gain - cost) / truth.size)


def best_value_threshold(
    observed: ArrayLike, scores: ArrayLike, economics: OutreachEconomics
) -> tuple[float, float]:
    """Return ``(threshold, value per user)`` maximizing observed outreach value.

    Every distinct score is tried as a threshold; ties in value go to the
    highest threshold (fewest contacts). Contacting no one (value 0) is always
    an option, returned as threshold ``inf``.
    """

    truth = np.asarray(observed).astype(bool)
    values = np.asarray(scores, dtype=float)
    if truth.shape != values.shape or truth.ndim != 1 or truth.size == 0:
        raise ValueError("observed and scores must be aligned nonempty 1-D arrays")
    order = np.argsort(-values, kind="stable")
    sorted_scores, sorted_truth = values[order], truth[order]
    per_user_gain = economics.save_rate * economics.value_per_saved_user
    running = np.cumsum(
        np.where(sorted_truth, per_user_gain, 0.0) - economics.cost_per_contact
    )
    last_of_tie = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    candidates, totals = sorted_scores[last_of_tie], running[last_of_tie]
    best = int(np.argmax(totals))
    if totals[best] <= 0:
        return float("inf"), 0.0
    return float(candidates[best]), float(totals[best] / truth.size)


def calibration_table(
    observed: ArrayLike, probabilities: ArrayLike, *, bins: int = 10
) -> pd.DataFrame:
    """Mean predicted probability versus observed rate in equal-size groups.

    Users are sorted by predicted probability and split into ``bins`` groups
    of (nearly) equal size, so every point on a reliability diagram rests on
    the same amount of data.
    """

    truth = np.asarray(observed, dtype=float)
    values = np.asarray(probabilities, dtype=float)
    if truth.shape != values.shape or truth.ndim != 1 or truth.size == 0:
        raise ValueError(
            "observed and probabilities must be aligned nonempty 1-D arrays"
        )
    if not np.all((values >= 0) & (values <= 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    if not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer")
    order = np.argsort(values, kind="stable")
    groups = np.array_split(order, min(bins, truth.size))
    return pd.DataFrame(
        {
            "mean predicted": [values[g].mean() for g in groups],
            "observed rate": [truth[g].mean() for g in groups],
            "users": [g.size for g in groups],
        },
        index=pd.RangeIndex(1, len(groups) + 1, name="group"),
    )


def brier_score(observed: ArrayLike, probabilities: ArrayLike) -> float:
    """Mean squared difference between predicted probability and outcome."""

    truth = np.asarray(observed, dtype=float)
    values = np.asarray(probabilities, dtype=float)
    if truth.shape != values.shape or truth.size == 0:
        raise ValueError("observed and probabilities must be aligned and nonempty")
    return float(np.mean((values - truth) ** 2))
