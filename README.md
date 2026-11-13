# Data Science and Machine Learning Project Template

This is the starting repository for final projects in Rice University
CMOR 438 / INDE 577. It is a small but complete machine-learning system that
students extend throughout the semester. The goal is not merely to fit models
in notebooks. The goal is to learn how mathematical ideas become tested,
reviewed, reproducible, deployable, and observable software.

> **The template supplies the production shell; your team supplies the
> scientific and product substance.** You will bring your own data, implement
> course algorithms, write tests, compare your work with trusted libraries,
> and promote reviewed code through an experiment-to-production workflow.

Read [`ASSIGNMENT.md`](ASSIGNMENT.md) before changing the template. It defines
the final product and, in particular, what students must write in each guided
algorithm or multi-algorithm case study.

## What you inherit and what you own

The untouched template already runs this vertical slice:

```text
configuration
    -> validated data adapter
    -> declared learning task
    -> reproducible reference experiment
    -> versioned model artifact
    -> prediction API and small browser client
    -> structured logs, tests, container, and CI
```

That working slice lets you observe production behavior before you change it.
It is scaffolding, not the final project.

| The template gives you | Your team must add or replace |
| --- | --- |
| Package, configuration, and test structure | A specific user, question, and product contract |
| A public offline wine-data example | Your own documented, lawful data boundary |
| Simple baselines and one transparent linear-regression exemplar | Applicable course algorithms implemented by your team |
| Clearly labeled scikit-learn reference experiments | Mathematical derivations and differential comparisons |
| Artifact save/load contract | A justified model-selection and promotion decision |
| FastAPI prediction service and minimal client | Product behavior useful to your intended user |
| Logging, CI, container, and release workflows | Domain-specific monitoring, failure tests, and operations evidence |

Calling a scikit-learn model does **not** satisfy an algorithm-implementation
requirement. Reference models are measuring instruments: use them to check and
compare student-authored code.

## Before you begin

Install these tools:

- [Git](https://git-scm.com/);
- [uv](https://docs.astral.sh/uv/);
- [VS Code](https://code.visualstudio.com/) with the Python and Jupyter
  extensions; and
- a GitHub account.

Docker is useful for the deployment milestone but is not required for the
first setup. All commands below run in macOS Terminal, Linux shells, Windows
PowerShell, and the VS Code terminal unless a platform-specific alternative is
shown.

## 1. Create your team's repository

1. Click **Use this template** near the top of this GitHub page.
2. Choose **Create a new repository**.
3. Select the owner agreed upon by the team and give the repository a
   descriptive product name.
4. Clone the newly created repository. Do not clone this template and push
   student work back to it.

```bash
git clone YOUR-NEW-REPOSITORY-URL
cd YOUR-NEW-REPOSITORY-NAME
```

Each teammate clones the same team repository. Do not create unrelated copies
for each teammate.

## 2. Build and verify the untouched environment

Run these commands from the repository root:

```bash
uv sync --locked
uv run python scripts/setup_project.py
uv run dsm-project verify
uv run ruff check src tests scripts
uv run pytest -q
```

What those commands do:

- `uv sync --locked` creates `.venv` and installs the exact locked dependency
  set. Do not activate the environment manually when using `uv run`.
- `setup_project.py` verifies imports and installs the **DSM Project** notebook
  kernel. It does not launch JupyterLab.
- `dsm-project verify` exercises data, supervised, and unsupervised reference
  paths.
- Ruff checks Python quality; pytest checks units, integration boundaries, the
  API, repository rules, and all example notebooks from clean kernels.

In VS Code, open the repository folder—not merely an individual notebook. When
opening a notebook, select the **DSM Project** kernel. Do not begin project work
until the untouched template passes locally.

## 3. Define the product before fitting a model

Open a small issue for the proposal, then complete these documents in a branch:

1. [`docs/PRODUCT_CONTRACT.md`](docs/PRODUCT_CONTRACT.md): intended user,
   supported decision or investigation, prohibited uses, task-algorithm matrix,
   evaluation rule, and system boundary.
2. [`docs/DATASET_CARD.md`](docs/DATASET_CARD.md): source, license, observation
   unit, sampling process, fields, targets, sensitive attributes, and limits.
3. [`docs/TEAM_OWNERSHIP.md`](docs/TEAM_OWNERSHIP.md): individual ownership and
   shared review responsibilities.
4. [`docs/decisions/0001-project-scope.md`](docs/decisions/0001-project-scope.md):
   the first durable engineering decision.

Be precise about what one row represents and what information exists at the
moment a prediction or analysis would be made. Many apparent modeling successes
are data leakage in disguise.

## 4. Bring your own data safely

The repository intentionally does not assume that student data can be
published.

Start with the cross-domain catalog in
[`docs/PUBLIC_DATA_SOURCES.md`](docs/PUBLIC_DATA_SOURCES.md). It links public
repositories in science, health, engineering, climate, economics, society,
transportation, geospatial analysis, and knowledge graphs, with access and
modeling cautions for each area.

1. Place local files under `data/local/`. Git ignores this directory.
2. Copy the student configuration example.

macOS, Linux, or Git Bash:

```bash
cp configs/student.toml.example configs/student.toml
```

Windows PowerShell:

```powershell
Copy-Item configs/student.toml.example configs/student.toml
```

3. Edit `configs/student.toml` with the dataset path, identifier, feature
   columns, task targets, metrics, and split rule.
4. Adapt `src/dsm_project/data/csv_adapter.py` or write a new adapter and
   register it explicitly in `src/dsm_project/data/registry.py`.
5. Add a small public, deidentified, or constructed fixture under
   `data/fixtures/` so CI can test schema and transformations without private
   data or credentials.
6. Add tests for missing fields, duplicate identifiers, invalid values, and
   leakage boundaries before training.

Never commit credentials, `.env`, private data, restricted data, or large raw
files. If data live in a database or object store, commit retrieval code and
metadata—not secret keys or an unauthorized copy. Read [`data/README.md`](data/README.md)
and [`SECURITY.md`](SECURITY.md) before opening the first data pull request.

## 5. Implement algorithms as production code

Read [`STUDENT_IMPLEMENTATION_GUIDE.md`](STUDENT_IMPLEMENTATION_GUIDE.md) before
writing an estimator. Student-owned algorithms belong in:

```text
src/dsm_project/algorithms/student/
```

For each applicable course algorithm:

1. Define the mathematical objective, inputs, fitted state, outputs,
   assumptions, and failure conditions.
2. Implement a typed and documented class in the student package. Use NumPy-
   style docstrings.
3. Add it to `STUDENT_ALGORITHMS` in
   `src/dsm_project/algorithms/registry.py`.
4. Test a hand-computable or analytic case.
5. Test invalid parameters, shape boundaries, use before fitting, numerical
   behavior, and reproducibility.
6. Compare with a trusted implementation when objectives and conventions
   match. A disagreement must be investigated and explained.
7. Drive the implementation through the configured data and experiment
   boundaries.
8. Use a notebook to teach and visualize the evidence, not to hide reusable
   implementation state.

The included `BatchLinearRegressor` is the standard to study first. It is a
transparent exemplar, not permission to copy and rename an implementation.

## 6. Build self-contained applied case studies

Follow the full
[`notebook case-study standard`](docs/NOTEBOOK_CASE_STUDY_STANDARD.md). The goal
is an accessible technical article backed by professional package code—not a
code dump and not an unreviewed production system living in notebook state.

The completed `examples/` notebooks cover the course families and execute in
CI:

- data and task contracts;
- linear and logistic regression;
- support vector machines and kernels;
- nearest neighbors;
- trees and ensembles;
- neural networks;
- model selection;
- clustering; and
- dimensionality reduction.

Create project notebooks under `notebooks/` from
[`00-guided-case-study-template.ipynb`](notebooks/00-guided-case-study-template.ipynb).
Each notebook must use data analysis and machine learning to answer one real
scientific, engineering, social, or operational question on the team's chosen
data. It may use one algorithm or a justified mixture of algorithms. Students
must make and defend the data, modeling, metric, diagnostic, and promotion
choices; the template guides the sequence without supplying those answers.

The notebook must define its question and notation, explain what each important
code cell is doing and why, interpret evidence in prose, identify failures, and
reach a bounded conclusion. It must restart successfully from top to bottom.
Training, validation, API, and monitoring behavior belongs under `src/` or
`scripts/`; the notebook imports that tested behavior and explains what it
reveals.

## 7. Work like a software team

Start each coherent change from an issue and a current `main` branch:

```bash
git switch main
git pull --ff-only
git switch -c team/short-purpose
```

Make the smallest coherent change, run focused tests while working, then run
the complete local gate:

```bash
uv run ruff check src tests scripts
uv run pytest -q
uv build
git status
```

Stage only files you understand:

```bash
git add PATHS-YOU-INTEND-TO-COMMIT
git commit -m "Describe the completed behavior"
git push -u origin team/short-purpose
```

Open a pull request, link the issue with `Closes #NUMBER`, complete the provided
checklist, and request a teammate review. The author explains the mathematics
and evidence; the reviewer checks scientific meaning, leakage, interfaces,
tests, security, and operational consequences. Merge only after CI and review
pass. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full contract.

## 8. Exercise the production path

The reference commands make the software behavior visible:

```bash
# Train and save a versioned example artifact.
uv run dsm-project train-example

# Start the API with that artifact.
uv run dsm-project serve
```

Open <http://127.0.0.1:8000> while the server is running. Useful interfaces are:

- `GET /health`: service and loaded-model identity;
- `GET /model`: artifact metadata;
- `POST /predict`: validated inference requests; and
- `GET /docs`: generated interactive API documentation.

Stop the server with `Ctrl+C`. The final project must replace or extend this
reference behavior so it serves the declared user and a reviewed student model.

When Docker is available:

```bash
docker compose up --build
```

This creates the same service in an isolated container. The GitHub container
workflow verifies that a clean image starts and satisfies the health contract.

## 9. Promote evidence, not merely a high score

Professional model selection separates at least three roles:

- training data fit candidate parameters;
- validation or cross-validation evidence guides design and selection; and
- final test data estimate performance once after the procedure is frozen.

Retain baselines and losing candidates. Report uncertainty, meaningful slices,
failure cases, computational cost, and task-appropriate metrics. For
classification, explain the consequences of each confusion-matrix error; for
unsupervised work, define stability, reconstruction, density, geometry, or
scientific utility rather than silently treating hidden labels as training
targets.

Promote a student model into the API only after its mathematical, unit,
differential, integration, and review gates pass. Save model identity,
configuration, feature order, data provenance, metrics, limitations, and a
rollback target with the artifact.

## 10. Operate and release the product

Complete [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) and
[`docs/OPERATIONS.md`](docs/OPERATIONS.md). Demonstrate that logs and metrics can
answer: What failed? For which request? Under which model and data contract?
What changed? Can the team reproduce and roll back the result?

A version tag such as `v0.1.0` triggers the release workflow only after the
team is ready. It rebuilds and verifies the project, creates source and wheel
artifacts with checksums, and publishes a GitHub release.

## Repository map

| Location | Responsibility |
| --- | --- |
| `configs/` | Versioned experiment and data-boundary declarations |
| `data/local/` | Ignored student data; never committed |
| `data/fixtures/` | Small CI-safe schema and failure examples |
| `examples/` | Completed course-topic case studies; reference only |
| `notebooks/` | Guided, self-contained student case studies on chosen data |
| `src/dsm_project/data/` | Adapters, validation, provenance, and tasks |
| `src/dsm_project/algorithms/student/` | Student-authored course algorithms |
| `src/dsm_project/algorithms/registry.py` | Explicit bridge from student code to the system |
| `src/dsm_project/experiments/` | Repeatable experiments and trusted references |
| `src/dsm_project/evaluation/` | Metrics and comparison evidence |
| `src/dsm_project/artifacts/` | Versioned model save/load contract |
| `src/dsm_project/service/` | API and minimal analytical client |
| `src/dsm_project/monitoring/` | Structured operational evidence |
| `scripts/` | Repeatable setup, training, and system checks |
| `tests/` | Unit, mathematical, integration, API, notebook, and contract tests |
| `docs/` | Product, data, model, ownership, decisions, and operations records |
| `.github/` | Issues, pull requests, dependency updates, CI, containers, and releases |

## Completion standard

Use [`PROJECT_CHECKLIST.md`](PROJECT_CHECKLIST.md) as the definition of done and
[`MILESTONES.md`](MILESTONES.md) as the recommended sequence. A strong project
is not the one with the most algorithms or highest isolated score. It is the
one whose mathematical claims, data boundaries, code, evidence, interfaces,
and operational behavior agree—and can be reproduced and reviewed by someone
outside the team.

## Common mistakes to avoid

- Running commands outside the repository root.
- Selecting a global Python kernel instead of **DSM Project**.
- Committing `data/local/`, credentials, generated artifacts, or `.env`.
- Implementing reusable behavior only in notebooks.
- Presenting a scikit-learn call as a student-authored algorithm.
- Scaling or selecting features before the train/test boundary is established.
- Repeatedly inspecting the final test set during development.
- Merging directly to `main` without an issue, review, and passing CI.
- Reporting one aggregate metric without baselines, uncertainty, slices, or
  failure analysis.
- Deploying a model without artifact identity, logs, monitoring, or rollback.

If setup fails, include the command, complete error, operating system,
`uv --version`, `uv run python --version`, and `uv run python -c "import sys;
print(sys.executable)"` in a bug issue. Do not solve environment problems by
installing unrelated packages globally.
