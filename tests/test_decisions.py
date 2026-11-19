"""Tests for cost-based thresholds and calibration helpers."""

import numpy as np
import pytest
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from dsm_project.evaluation import (
    OutreachEconomics,
    best_value_threshold,
    brier_score,
    calibration_table,
    expected_outreach_value,
)

ECONOMICS = OutreachEconomics(
    cost_per_contact=2.0, value_per_saved_user=50.0, save_rate=0.2
)


def test_break_even_probability_by_hand() -> None:
    # Contact pays off when p * 0.2 * 50 >= 2, i.e. p >= 0.2.
    assert ECONOMICS.break_even_probability == pytest.approx(0.2)


@pytest.mark.parametrize(
    "arguments",
    [(0.0, 50.0, 0.2), (2.0, 0.0, 0.2), (2.0, 50.0, 0.0), (2.0, 50.0, 1.5)],
)
def test_economics_reject_invalid_assumptions(arguments) -> None:
    with pytest.raises(ValueError):
        OutreachEconomics(*arguments)


def test_expected_value_by_hand() -> None:
    observed = [1, 0, 1, 0, 0]
    scores = [0.9, 0.8, 0.4, 0.1, 0.05]
    # Threshold 0.4 flags 3 users (2 churners): 2 * 10 - 3 * 2 = 14 over 5 users.
    assert expected_outreach_value(observed, scores, 0.4, ECONOMICS) == pytest.approx(
        14 / 5
    )
    assert expected_outreach_value(observed, scores, 0.95, ECONOMICS) == 0.0


def test_best_value_threshold_matches_brute_force() -> None:
    rng = np.random.default_rng(0)
    probabilities = rng.uniform(size=500)
    observed = (rng.uniform(size=500) < probabilities).astype(int)  # calibrated
    threshold, value = best_value_threshold(observed, probabilities, ECONOMICS)

    brute = max(
        expected_outreach_value(observed, probabilities, t, ECONOMICS)
        for t in np.unique(probabilities)
    )
    assert value == pytest.approx(brute)
    assert expected_outreach_value(observed, probabilities, threshold, ECONOMICS) == (
        pytest.approx(value)
    )
    # For calibrated scores the empirical optimum sits near the break-even 0.2.
    assert abs(threshold - ECONOMICS.break_even_probability) < 0.08


def test_best_value_threshold_contacts_no_one_when_outreach_never_pays() -> None:
    expensive = OutreachEconomics(
        cost_per_contact=100.0, value_per_saved_user=10.0, save_rate=0.5
    )
    threshold, value = best_value_threshold([1, 0, 1], [0.9, 0.5, 0.8], expensive)
    assert threshold == float("inf") and value == 0.0


def test_calibration_table_matches_trusted_quantile_curve() -> None:
    rng = np.random.default_rng(1)
    probabilities = rng.uniform(size=1000)
    observed = (rng.uniform(size=1000) < probabilities).astype(int)
    table = calibration_table(observed, probabilities, bins=10)
    trusted_rate, trusted_mean = calibration_curve(
        observed, probabilities, n_bins=10, strategy="quantile"
    )

    assert table["users"].sum() == 1000 and table["users"].nunique() == 1
    np.testing.assert_allclose(table["mean predicted"], trusted_mean, atol=1e-12)
    np.testing.assert_allclose(table["observed rate"], trusted_rate, atol=1e-12)


def test_brier_score_matches_trusted_and_base_rate_identity() -> None:
    observed = np.array([1, 0, 0, 1, 0, 0, 0, 0])
    probabilities = np.array([0.8, 0.1, 0.3, 0.6, 0.2, 0.05, 0.4, 0.1])
    assert brier_score(observed, probabilities) == pytest.approx(
        brier_score_loss(observed, probabilities)
    )
    rate = observed.mean()
    assert brier_score(observed, np.full(8, rate)) == pytest.approx(rate * (1 - rate))


def test_decision_helpers_reject_bad_input() -> None:
    with pytest.raises(ValueError, match="aligned"):
        expected_outreach_value([1, 0], [0.5], 0.5, ECONOMICS)
    with pytest.raises(ValueError, match="aligned"):
        best_value_threshold([1], [0.5, 0.2], ECONOMICS)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        calibration_table([1, 0], [0.5, 1.5])
    with pytest.raises(ValueError, match="bins"):
        calibration_table([1, 0], [0.5, 0.2], bins=0)
    with pytest.raises(ValueError, match="aligned"):
        brier_score([1, 0], [0.5])
