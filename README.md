# INDE 577 Final Project — Data Science & Machine Learning

Final project for Rice University **INDE 577 / CMOR 438: Data Science & Machine
Learning** (Fall 2026, Dr. Randy R. Davila), built on the course
[project template](https://github.com/RandyRDavila/Data-Science-Machine-Learning-Project-Template).

**Author:** Jorgo Qirjaj

**Waze churn risk.** Which Waze users are about to churn, and how many of
them can a retention campaign of a given size reach? This project builds on
my earlier exploratory analysis and modeling of the Waze churn data and
develops it under this course's standards:
algorithms implemented from scratch as a tested Python package
(`src/dsm_project`), self-contained case-study notebooks (`notebooks/`), a
sealed test set, baselines, and uncertainty, served through a small
reproducible system (versioned artifact, API, logs, CI).

## Status — milestones 1, 2, 2A and 3 done

| Area | State |
| --- | --- |
| Product and data contract | [`docs/PRODUCT_CONTRACT.md`](docs/PRODUCT_CONTRACT.md), [`docs/DATASET_CARD.md`](docs/DATASET_CARD.md), [ADR 0002](docs/decisions/0002-waze-churn-dataset.md) |
| Data boundary | `WazeChurnAdapter`: checksum-pinned CSV, schema and range validation, 700 unlabeled users dropped; constructed CI fixture |
| Student algorithms | `SoftmaxRegression` (gradient descent, L2, class weighting), `KNearestNeighborsClassifier` (blocked search for large data), `ClassificationTree` and `RegressionTree` (CART with rule export), `RandomForest` (bagging, feature sampling, out-of-bag estimates), and `GradientBoostingBinaryClassifier` (log-loss boosting with Newton leaf steps) |
| Evaluation | Shared holdout split, bootstrap intervals, F-beta threshold selection, `ThresholdedClassifier` |
| Tests | Analytic cases, gradient checks, failure modes, differential tests against scikit-learn, data-boundary tests |
| Case studies | [02 — Waze churn: softmax vs k-NN](notebooks/02-waze-churn-softmax-vs-knn.ipynb); [03 — decision tree](notebooks/03-waze-churn-decision-tree.ipynb); [04 — ensembles](notebooks/04-waze-churn-ensembles.ipynb); [01 — wine](notebooks/01-wine-cultivar-softmax-vs-knn.ipynb) (algorithm verification on clean data) |
| Next | Engineered features, engagement segmentation (k-means, PCA), cross-family model selection, promotion into the API |

### Headline result (sealed test set, 3,575 users)

| Model | Average precision | Recall at frozen threshold | Precision | Users flagged |
| --- | --- | --- | --- | --- |
| Baseline (always "retained") | 0.177 | 0.00 | — | 0% |
| **Student softmax regression** | **0.386** | **0.855** (95% CI 0.83–0.88) | 0.27 | 56% |
| Student k-NN (k = 301) | 0.366 | 0.828 | 0.27 | 54% |

The student softmax model ranks users about twice as well as random. At a 20%
outreach budget it reaches about 45% of churners at about 40% precision.
It relies mostly on active days, driving days, and tenure, so it misses
nearly all churners who were still active on more than 20 days that month.
Activity and churn are measured in the same month, so these results show
disengagement in progress, not early warning.

**Decision tree (development data only, test set kept sealed).** The best
student tree (depth 4, at least 100 users per leaf) reaches cross-validated AP
about 0.35 and loses to softmax regression on all 15 paired folds (by about
0.04). Churn risk is close to additive in activity and tenure, so a single
tree's thresholds add variance without adding signal. Its depth-3 rules are
still a readable summary for a PM; the "inactive, newer, heavy driver" segment
churns at about 48%, but it reappears in only 17 of 30 bootstrap refits.

**Ensembles (development data only).** The best student random forest
(100-user leaves) reaches AP about 0.38 and the best gradient boosting (depth-1
stumps, 200 stages) about 0.39: both fix the single tree's variance, but
neither beats softmax regression. Boosted stumps tie it (higher on 10 of 15
paired folds, by 0.001 on average). Deeper boosting does worse, so the churn
signal is additive. The stumps' learned curves show modest bends (risk falls
fastest over the first ~10 active days; users with 200+ drives a month churn
at 26% vs 17%), useful leads for engineered features.

### Getting the data

The dataset is public and its use was approved by the instructor. The CSV is
not committed to this repository: download the Waze churn
dataset from Kaggle and save it as `data/local/waze_dataset.csv`; the loader
verifies its checksum. Without it, tests use a constructed fixture and skip
the real-data notebook.

## Quick start

Requires [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync --locked
uv run python scripts/setup_project.py   # installs the "DSM Project" kernel
uv run pytest -q                         # unit, differential, and notebook tests
uv run ruff check src tests scripts
```

Open the notebook in VS Code or Jupyter and select the **DSM Project** kernel.

Using the student algorithms directly:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from dsm_project.algorithms import create_student_algorithm

model = make_pipeline(
    StandardScaler(),
    create_student_algorithm("softmax_regression", l2_penalty=0.1),
)
model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test)
```

## Where things are

| Path | What it holds |
| --- | --- |
| `src/dsm_project/algorithms/student/` | Student-authored algorithms (`logistic.py`, `neighbors.py`) |
| `src/dsm_project/data/waze.py` | Waze data boundary: validation, cleaning, encoding |
| `src/dsm_project/algorithms/registry.py` | The registry that connects student code to experiments |
| `src/dsm_project/experiments/` | Shared split and evaluation boundary, scikit-learn reference models |
| `src/dsm_project/evaluation/` | Bootstrap uncertainty, threshold selection, comparison tables |
| `configs/waze.toml` | Project configuration: data path and checksum, tasks, split seed |
| `configs/waze_fixture.toml` | Same configuration pointed at the CI fixture |
| `configs/project.toml` | Wine verification configuration |
| `notebooks/` | Case-study notebooks (`00-…` is the course template) |
| `examples/` | Completed course reference notebooks (not my work) |
| `tests/` | Test suite; notebooks are executed in CI |
| `docs/` | Product contract, dataset/model cards, decisions, and the [original template guide](docs/TEMPLATE_GUIDE.md) |

## Roadmap

1. ~~Decision tree from scratch (lecture 14)~~ done: it does not recover
   active churners and does not beat the linear baseline.
2. ~~Ensembles (lecture 15)~~ done: random forest and gradient boosting
   (verified against scikit-learn) close the single tree's gap; boosted
   stumps tie the linear baseline.
3. Engineered features suggested by the boosted curves and my earlier
   analysis (heavy-driver flag, curved activity term, ratio features).
4. Engagement segmentation with k-means and PCA.
5. Model selection across families (lecture 16), then promotion into the
   API with input monitoring (lecture 17).
6. Model card, operations plan, tagged release, and presentation.

See [`ASSIGNMENT.md`](ASSIGNMENT.md), [`MILESTONES.md`](MILESTONES.md), and
[`PROJECT_CHECKLIST.md`](PROJECT_CHECKLIST.md) for the full course requirements.
