"""Write data/fixtures/waze_fixture.csv: constructed rows with the Waze schema.

The rows are generated, not sampled from the real dataset, so they can be
committed regardless of the dataset's license. They exercise the schema,
unlabeled rows, zero-drive users, an inconsistent lifetime total, and a churn
signal strong enough for smoke tests. They are not evidence about real users.

Run from the repository root:  uv run python scripts/build_waze_fixture.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROWS = 120
OUTPUT = Path(__file__).parents[1] / "data/fixtures/waze_fixture.csv"


def build(rows: int = ROWS, seed: int = 577) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    churned = rng.random(rows) < 0.2
    activity_days = np.where(
        churned, rng.integers(0, 14, rows), rng.integers(8, 32, rows)
    ).clip(0, 31)
    driving_days = np.minimum(activity_days, rng.integers(0, 31, rows))
    sessions = rng.integers(0, 300, rows)
    drives = np.floor(sessions * rng.uniform(0.6, 1.0, rows)).astype(int)
    drives[:3] = 0  # users who opened the app but never drove
    total_sessions = sessions + rng.uniform(1, 600, rows)
    total_sessions[3] = sessions[3] * 0.5  # inconsistent lifetime total, kept
    km = rng.uniform(60, 9000, rows)
    minutes = km / rng.uniform(40, 110, rows) * 60
    frame = pd.DataFrame(
        {
            "ID": np.arange(rows),
            "label": np.where(churned, "churned", "retained").astype(object),
            "sessions": sessions,
            "drives": drives,
            "total_sessions": total_sessions.round(4),
            "n_days_after_onboarding": rng.integers(4, 3500, rows),
            "total_navigations_fav1": rng.integers(0, 400, rows),
            "total_navigations_fav2": rng.integers(0, 120, rows),
            "driven_km_drives": km.round(4),
            "duration_minutes_drives": minutes.round(4),
            "activity_days": activity_days,
            "driving_days": driving_days,
            "device": rng.choice(["iPhone", "Android"], rows, p=[0.65, 0.35]),
        }
    )
    frame.loc[rng.choice(rows, 6, replace=False), "label"] = np.nan
    return frame


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    build().to_csv(OUTPUT, index=False)
    print(f"wrote {OUTPUT}")
