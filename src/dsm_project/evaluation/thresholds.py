"""Decision thresholds for binary classifiers on imbalanced data.

A probabilistic classifier ranks users by risk; a *threshold* turns that
ranking into an action (flag for outreach or not). The default of 0.5 is
rarely right when the positive class is rare or when missing a positive costs
more than a false alarm. The threshold is a hyperparameter: choose it on
out-of-fold development predictions, never on the final test set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, clone


@dataclass(frozen=True)
class ThresholdChoice:
    """The selected threshold and the development metrics it achieves."""

    threshold: float
    f_beta: float
    precision: float
    recall: float
    beta: float
    flagged_fraction: float


def precision_recall_at(
    observed: ArrayLike, scores: ArrayLike, threshold: float
) -> tuple[float, float]:
    """Return (precision, recall) when flagging rows with ``score >= threshold``.

    Precision is defined as 0 when nothing is flagged.
    """

    truth = np.asarray(observed).astype(bool)
    flagged = np.asarray(scores, dtype=float) >= threshold
    true_positives = float(np.sum(flagged & truth))
    precision = true_positives / flagged.sum() if flagged.any() else 0.0
    recall = true_positives / truth.sum() if truth.any() else 0.0
    return precision, recall


def f_beta(precision: float, recall: float, beta: float) -> float:
    """F-beta score; ``beta > 1`` weights recall more heavily than precision."""

    if precision == 0.0 and recall == 0.0:
        return 0.0
    b2 = beta**2
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def choose_threshold(
    observed: ArrayLike, scores: ArrayLike, *, beta: float = 2.0
) -> ThresholdChoice:
    """Pick the score threshold that maximizes F-beta.

    Every distinct score is a candidate threshold (flag ``score >= t``). Ties
    in F-beta go to the *highest* threshold, which flags the fewest users.

    Parameters
    ----------
    observed
        Binary truth, 1 for the positive (e.g. churned) class.
    scores
        Predicted probability or score of the positive class.
    beta
        Positive weight of recall relative to precision.
    """

    truth = np.asarray(observed)
    values = np.asarray(scores, dtype=float)
    if truth.ndim != 1 or truth.shape != values.shape or truth.size == 0:
        raise ValueError("observed and scores must be aligned nonempty 1-D arrays")
    if not set(np.unique(truth).tolist()) <= {0, 1}:
        raise ValueError("observed must be binary (0/1)")
    if not truth.astype(bool).any():
        raise ValueError("observed must contain at least one positive")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    if beta <= 0:
        raise ValueError("beta must be positive")

    # Sort scores descending; flagging the top j rows equals threshold = score_j.
    order = np.argsort(-values, kind="stable")
    sorted_scores = values[order]
    sorted_truth = truth[order].astype(float)
    true_positives = np.cumsum(sorted_truth)
    flagged = np.arange(1, values.size + 1)
    # Only the last row of each run of tied scores is a valid cut point.
    last_of_tie = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    tp = true_positives[last_of_tie]
    count = flagged[last_of_tie]
    candidates = sorted_scores[last_of_tie]
    precision = tp / count
    recall = tp / sorted_truth.sum()
    b2 = beta**2
    denominator = b2 * precision + recall
    scores_f = np.divide(
        (1 + b2) * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best = int(np.argmax(scores_f))  # first maximum = highest threshold
    return ThresholdChoice(
        threshold=float(candidates[best]),
        f_beta=float(scores_f[best]),
        precision=float(precision[best]),
        recall=float(recall[best]),
        beta=float(beta),
        flagged_fraction=float(count[best] / values.size),
    )


class ThresholdedClassifier(ClassifierMixin, BaseEstimator):
    """Wrap a binary probabilistic classifier with a fixed decision threshold.

    ``predict`` returns the positive class (the larger label, e.g. 1) when the
    wrapped estimator's positive-class probability is at least ``threshold``.

    Parameters
    ----------
    estimator
        Unfitted classifier with ``fit`` and ``predict_proba``; it is cloned
        when this wrapper is fitted.
    threshold
        Probability cut-off in ``[0, 1]``, chosen beforehand on development
        data (see :func:`choose_threshold`).
    """

    def __init__(self, estimator: BaseEstimator, threshold: float = 0.5) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must lie in [0, 1]")
        self.estimator = estimator
        self.threshold = threshold

    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self:
        """Fit a clone of the wrapped estimator."""

        labels = np.unique(np.asarray(targets))
        if labels.size != 2:
            raise ValueError("ThresholdedClassifier requires exactly two classes")
        self.estimator_ = clone(self.estimator).fit(features, targets)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, features: ArrayLike) -> NDArray[np.float64]:
        """Delegate probabilities to the fitted estimator."""

        if not hasattr(self, "estimator_"):
            raise RuntimeError("fit must be called before prediction")
        return self.estimator_.predict_proba(features)

    def predict(self, features: ArrayLike) -> NDArray[np.generic]:
        """Flag the positive class where its probability reaches the threshold."""

        positive = self.predict_proba(features)[:, 1] >= self.threshold
        return np.where(positive, self.classes_[1], self.classes_[0])
