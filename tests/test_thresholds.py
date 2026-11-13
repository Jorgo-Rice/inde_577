"""Tests for threshold selection and the thresholded classifier wrapper."""

import numpy as np
import pytest
from sklearn.metrics import fbeta_score, precision_score, recall_score

from dsm_project.algorithms import MajorityClassifier, SoftmaxRegression
from dsm_project.evaluation import (
    ThresholdedClassifier,
    choose_threshold,
    f_beta,
    precision_recall_at,
)


def test_precision_recall_at_hand_computed_case() -> None:
    observed = [1, 0, 1, 0, 1]
    scores = [0.9, 0.8, 0.4, 0.3, 0.1]
    # Flag score >= 0.4: rows 0, 1, 2 -> TP = 2, flagged = 3, positives = 3.
    precision, recall = precision_recall_at(observed, scores, 0.4)

    assert precision == pytest.approx(2 / 3)
    assert recall == pytest.approx(2 / 3)
    assert precision_recall_at(observed, scores, 0.95) == (0.0, 0.0)


def test_f_beta_matches_definition_and_trusted_metric() -> None:
    observed = np.array([1, 0, 1, 1, 0, 0, 1])
    predicted = np.array([1, 1, 0, 1, 0, 0, 0])
    precision = precision_score(observed, predicted)
    recall = recall_score(observed, predicted)

    for beta in (0.5, 1.0, 2.0):
        assert f_beta(precision, recall, beta) == pytest.approx(
            fbeta_score(observed, predicted, beta=beta)
        )
    assert f_beta(0.0, 0.0, 2.0) == 0.0


def test_choose_threshold_finds_brute_force_optimum() -> None:
    rng = np.random.default_rng(0)
    observed = (rng.random(300) < 0.2).astype(int)
    scores = np.round(0.4 * observed + rng.random(300) * 0.8, 2)  # many ties
    choice = choose_threshold(observed, scores, beta=2.0)

    best = max(
        f_beta(*precision_recall_at(observed, scores, t), 2.0)
        for t in np.unique(scores)
    )
    assert choice.f_beta == pytest.approx(best)
    precision, recall = precision_recall_at(observed, scores, choice.threshold)
    assert (choice.precision, choice.recall) == pytest.approx((precision, recall))
    assert choice.flagged_fraction == pytest.approx(np.mean(scores >= choice.threshold))


def test_choose_threshold_perfect_separation_and_tie_rule() -> None:
    choice = choose_threshold([0, 0, 1, 1], [0.1, 0.2, 0.7, 0.9])
    assert choice.threshold == 0.7
    assert choice.f_beta == pytest.approx(1.0)

    # Every threshold flags both positives; the highest threshold flags fewest.
    tie = choose_threshold([1, 1], [0.6, 0.6])
    assert tie.threshold == 0.6 and tie.flagged_fraction == 1.0


@pytest.mark.parametrize(
    ("observed", "scores", "keywords", "message"),
    [
        ([0, 0], [0.1, 0.2], {}, "positive"),
        ([0, 2], [0.1, 0.2], {}, "binary"),
        ([0, 1], [0.1], {}, "aligned"),
        ([0, 1], [0.1, np.nan], {}, "finite"),
        ([0, 1], [0.1, 0.2], {"beta": 0.0}, "beta"),
    ],
)
def test_choose_threshold_rejects_invalid_input(observed, scores, keywords, message):
    with pytest.raises(ValueError, match=message):
        choose_threshold(observed, scores, **keywords)


def test_thresholded_classifier_applies_cutoff_and_clones() -> None:
    rng = np.random.default_rng(2)
    features = rng.normal(size=(150, 2))
    targets = (features[:, 0] + 0.5 * rng.normal(size=150) > 0.9).astype(int)
    base = SoftmaxRegression(l2_penalty=0.1)
    low = ThresholdedClassifier(base, threshold=0.1).fit(features, targets)
    high = ThresholdedClassifier(base, threshold=0.9).fit(features, targets)

    probability = low.predict_proba(features)[:, 1]
    np.testing.assert_array_equal(
        low.predict(features), (probability >= 0.1).astype(int)
    )
    assert low.predict(features).sum() > high.predict(features).sum()
    assert not hasattr(base, "coef_")  # the wrapped template stays unfitted


def test_thresholded_classifier_rejects_bad_use() -> None:
    with pytest.raises(ValueError, match="threshold"):
        ThresholdedClassifier(MajorityClassifier(), threshold=1.5)
    with pytest.raises(RuntimeError, match="fit must be called"):
        ThresholdedClassifier(SoftmaxRegression()).predict([[0.0]])
    with pytest.raises(ValueError, match="exactly two"):
        ThresholdedClassifier(SoftmaxRegression()).fit([[0.0], [1.0], [2.0]], [0, 1, 2])
