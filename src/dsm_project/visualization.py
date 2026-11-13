"""Shared seaborn styling and churn-focused chart helpers.

Every notebook calls :func:`set_theme` once so figures share one look, and
uses :data:`OUTCOME_PALETTE` so "retained" and "churned" keep the same colors
everywhere. Helpers that compute numbers (for example :func:`rate_by_group`)
are separate from the helpers that draw them, so the numbers can be tested.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

#: Consistent colors for the two outcomes across all notebooks.
OUTCOME_PALETTE = {"retained": "#1b9e77", "churned": "#d95f02"}
OUTCOME_ORDER = ["retained", "churned"]


def set_theme() -> None:
    """Apply the project's seaborn theme (white grid, readable notebook text)."""

    sns.set_theme(context="notebook", style="whitegrid", palette="colorblind")
    plt.rcParams.update({"figure.dpi": 100, "axes.titleweight": "bold"})


def outcome_labels(target: pd.Series | np.ndarray) -> pd.Series:
    """Map a 0/1 churn target to readable ``"retained"`` / ``"churned"`` labels."""

    values = pd.Series(np.asarray(target))
    if not set(values.unique()) <= {0, 1}:
        raise ValueError("target must be binary (0 = retained, 1 = churned)")
    return values.map({0: "retained", 1: "churned"}).rename("outcome")


def wilson_interval(
    successes: np.ndarray | float, trials: np.ndarray | float, z: float = 1.96
) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for a binomial proportion (95% by default).

    Unlike the normal approximation, it stays inside [0, 1] and behaves well
    for small groups or rates near 0 or 1, which matters for churn rates of
    small segments.
    """

    k = np.asarray(successes, dtype=float)
    n = np.asarray(trials, dtype=float)
    if np.any(n <= 0) or np.any(k < 0) or np.any(k > n):
        raise ValueError("need 0 <= successes <= trials and trials > 0")
    p = k / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    half_width = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return center - half_width, center + half_width


def rate_by_group(target: pd.Series, groups: pd.Series) -> pd.DataFrame:
    """Churn rate, count, and 95% Wilson interval for each group.

    Parameters
    ----------
    target
        Binary churn indicator (1 = churned).
    groups
        Group label for each row (for example ``pd.cut`` bins). Categorical
        order is preserved; empty categories are dropped.

    Returns
    -------
    DataFrame
        Indexed by group, with columns ``users``, ``churned``, ``rate``,
        ``ci_low``, and ``ci_high``.
    """

    target = pd.Series(np.asarray(target), index=groups.index)
    table = target.groupby(groups, observed=True).agg(users="size", churned="sum")
    table = table[table["users"] > 0]
    table["rate"] = table["churned"] / table["users"]
    table["ci_low"], table["ci_high"] = wilson_interval(
        table["churned"], table["users"]
    )
    return table


def plot_rate_by_group(
    table: pd.DataFrame,
    *,
    overall_rate: float | None = None,
    ax: Axes | None = None,
    xlabel: str = "",
    title: str = "",
    color: str = OUTCOME_PALETTE["churned"],
) -> Axes:
    """Bar chart of churn rate by group with 95% interval whiskers."""

    ax = ax or plt.gca()
    labels = [str(label) for label in table.index]
    sns.barplot(x=labels, y=table["rate"].to_numpy(), color=color, ax=ax)
    ax.errorbar(
        np.arange(len(labels)),
        table["rate"],
        yerr=[table["rate"] - table["ci_low"], table["ci_high"] - table["rate"]],
        fmt="none",
        ecolor="black",
        capsize=3,
        linewidth=1,
    )
    if overall_rate is not None:
        ax.axhline(
            overall_rate, color="black", linestyle=":", label="overall churn rate"
        )
        ax.legend(loc="upper right")
    ax.set(xlabel=xlabel, ylabel="churn rate (95% CI)", title=title)
    return ax


def plot_distributions_by_outcome(
    frame: pd.DataFrame,
    features: Sequence[str],
    target: pd.Series,
    *,
    kind: str = "ecdf",
    columns: int = 3,
    log_scale: Sequence[str] = (),
) -> Figure:
    """Small multiples comparing each feature's distribution for the two outcomes.

    Parameters
    ----------
    kind
        ``"ecdf"`` (empirical CDF, no smoothing choices), ``"kde"`` (density,
        each outcome normalized separately), or ``"box"``.
    log_scale
        Features to draw on a log axis (heavy right tails).
    """

    if kind not in {"ecdf", "kde", "box"}:
        raise ValueError("kind must be 'ecdf', 'kde', or 'box'")
    missing = set(features) - set(frame.columns)
    if missing:
        raise ValueError(f"unknown features: {sorted(missing)}")
    data = frame.loc[:, list(features)].reset_index(drop=True)
    data["outcome"] = outcome_labels(target).to_numpy()
    rows = int(np.ceil(len(features) / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(4.6 * columns, 3.4 * rows), squeeze=False
    )
    for ax, feature in zip(axes.flat, features, strict=False):
        log = feature in log_scale
        if kind == "ecdf":
            sns.ecdfplot(
                data=data,
                x=feature,
                hue="outcome",
                hue_order=OUTCOME_ORDER,
                palette=OUTCOME_PALETTE,
                log_scale=log,
                ax=ax,
            )
        elif kind == "kde":
            sns.kdeplot(
                data=data,
                x=feature,
                hue="outcome",
                hue_order=OUTCOME_ORDER,
                palette=OUTCOME_PALETTE,
                common_norm=False,
                fill=True,
                alpha=0.3,
                log_scale=log,
                cut=0,
                ax=ax,
            )
        else:
            sns.boxplot(
                data=data,
                x="outcome",
                y=feature,
                order=OUTCOME_ORDER,
                hue="outcome",
                palette=OUTCOME_PALETTE,
                legend=False,
                log_scale=log,
                fliersize=1.5,
                ax=ax,
            )
        ax.set_title(feature)
    for ax in list(axes.flat)[len(features) :]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig
