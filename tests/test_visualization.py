"""Tests for the shared seaborn helpers (numbers exactly, drawings by structure)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from dsm_project.visualization import (  # noqa: E402
    OUTCOME_PALETTE,
    outcome_labels,
    plot_distributions_by_outcome,
    plot_rate_by_group,
    rate_by_group,
    set_theme,
    wilson_interval,
)


def test_wilson_interval_matches_hand_computation() -> None:
    # 5 of 10: center 0.5; half-width 1.96 * sqrt(0.025 + 0.0096) / 1.384 = 0.2634.
    low, high = wilson_interval(5, 10)
    assert float(low) == pytest.approx(0.2366, abs=1e-3)
    assert float(high) == pytest.approx(0.7634, abs=1e-3)


def test_wilson_interval_stays_inside_zero_one() -> None:
    low, high = wilson_interval(np.array([0, 10]), np.array([10, 10]))
    assert low[0] == pytest.approx(0.0, abs=1e-12) and 0 < high[0] < 0.35
    assert high[1] == pytest.approx(1.0, abs=1e-12) and 0.65 < low[1] < 1
    with pytest.raises(ValueError):
        wilson_interval(3, 2)


def test_rate_by_group_counts_rates_and_order() -> None:
    target = pd.Series([1, 0, 0, 1, 1, 0])
    groups = pd.Series(pd.Categorical(["b", "a", "a", "b", "a", "c"],
                                      categories=["c", "b", "a", "empty"]))  # fmt: skip
    table = rate_by_group(target, groups)

    assert list(table.index) == ["c", "b", "a"]  # category order kept, empty dropped
    assert table.loc["a", "users"] == 3 and table.loc["a", "churned"] == 1
    assert table.loc["b", "rate"] == 1.0
    assert (table["ci_low"] <= table["rate"]).all() and (
        table["rate"] <= table["ci_high"]
    ).all()


def test_outcome_labels_and_rejection() -> None:
    assert outcome_labels(np.array([0, 1])).tolist() == ["retained", "churned"]
    with pytest.raises(ValueError, match="binary"):
        outcome_labels(np.array([0, 2]))


def test_rate_plot_draws_one_bar_per_group() -> None:
    set_theme()
    table = rate_by_group(pd.Series([1, 0, 1, 0]), pd.Series(["x", "x", "y", "y"]))
    fig, ax = plt.subplots()
    plot_rate_by_group(table, overall_rate=0.5, ax=ax, xlabel="group", title="t")

    assert len(ax.patches) == 2
    assert ax.get_ylabel() == "churn rate (95% CI)"
    plt.close(fig)


@pytest.mark.parametrize("kind", ["ecdf", "kde", "box"])
def test_distribution_small_multiples(kind) -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"a": rng.gamma(2, size=80) + 0.1, "b": rng.normal(size=80)})
    target = pd.Series(rng.integers(0, 2, size=80))
    fig = plot_distributions_by_outcome(frame, ["a", "b"], target, kind=kind,
                                        columns=3, log_scale=["a"])  # fmt: skip

    visible = [ax for ax in fig.axes if ax.get_visible() and ax.get_title()]
    assert [ax.get_title() for ax in visible] == ["a", "b"]
    plt.close(fig)


def test_distribution_plot_rejects_bad_input() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ValueError, match="kind"):
        plot_distributions_by_outcome(frame, ["a"], pd.Series([0, 1]), kind="violin")
    with pytest.raises(ValueError, match="unknown features"):
        plot_distributions_by_outcome(frame, ["zzz"], pd.Series([0, 1]))
    assert set(OUTCOME_PALETTE) == {"retained", "churned"}
