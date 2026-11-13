# INDE 577 Final Project — Data Science & Machine Learning

Final project for Rice University **INDE 577 / CMOR 438: Data Science & Machine
Learning** (Fall 2026, Dr. Randy R. Davila), built on the course
[project template](https://github.com/RandyRDavila/Data-Science-Machine-Learning-Project-Template).

**Author:** Jorgo Qirjaj

The project is one coherent machine-learning product: course algorithms
implemented from scratch as a tested Python package (`src/dsm_project`), used
in self-contained case-study notebooks (`notebooks/`) that answer a real
question on a chosen dataset, and served through a small reproducible system
(versioned artifact, API, logs, CI).

## Status — milestone 2A (first vertical slice)

| Area | State |
| --- | --- |
| Template verified locally | Done — all original tests pass |
| Student algorithms | **`SoftmaxRegression`** (multinomial logistic regression, batch gradient descent, L2) and **`KNearestNeighborsClassifier`** (Euclidean/Manhattan, uniform/distance voting), registered in the student registry |
| Tests | Analytic cases, finite-difference gradient check, failure modes, and differential tests against scikit-learn (`tests/test_student_algorithms.py`) |
| Evaluation helpers | Shared holdout split (`split_supervised_task`) and bootstrap confidence intervals (`bootstrap_metric_interval`) |
| Case study | [`notebooks/01-wine-cultivar-softmax-vs-knn.ipynb`](notebooks/01-wine-cultivar-softmax-vs-knn.ipynb): cross-validated model selection, one sealed test evaluation, failure analysis, and a promotion decision |
| Dataset | **Stand-in.** The UCI wine data are used to build and verify the pipeline. The project's own dataset is still to be chosen |
| API / artifact | Still serving the template's reference model; student model promotion is a later milestone |

### Headline result (stand-in data)

Both student classifiers reach **0.98 balanced accuracy** on 45 sealed test
wines (95% bootstrap CI ≈ 0.94–1.00) against a 0.33 majority baseline. Both
match their scikit-learn references exactly. The one error is the same sample
for every model, in a known chemical overlap between two cultivars. Softmax
regression gave that sample only 0.59 probability, which supports a
"confirm automatically or send to review" design.

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
| `src/dsm_project/algorithms/registry.py` | The registry that connects student code to experiments |
| `src/dsm_project/experiments/` | Shared split and evaluation boundary, scikit-learn reference models |
| `src/dsm_project/evaluation/` | Comparison tables and bootstrap uncertainty |
| `configs/project.toml` | Versioned project configuration (dataset, tasks, split seed) |
| `notebooks/` | Case-study notebooks (`00-…` is the course template) |
| `examples/` | Completed course reference notebooks (not my work) |
| `tests/` | Test suite; notebooks are executed in CI |
| `docs/` | Product contract, dataset/model cards, decisions, and the [original template guide](docs/TEMPLATE_GUIDE.md) |

## Roadmap

1. Choose the project dataset; write the adapter, dataset card, and CI fixture.
2. Re-run the classification case study on that data.
3. Add more course algorithms where they fit the data (e.g., linear
   regression study, decision tree, k-means and PCA for unsupervised structure).
4. Promote the selected student model through the artifact and API path, with
   monitoring.
5. Model card, operations plan, and a tagged release.

See [`ASSIGNMENT.md`](ASSIGNMENT.md), [`MILESTONES.md`](MILESTONES.md), and
[`PROJECT_CHECKLIST.md`](PROJECT_CHECKLIST.md) for the full course requirements.
